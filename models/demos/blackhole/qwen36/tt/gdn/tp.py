# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP Gated DeltaNet decode for Qwen3.5.

Shards by value heads (no cross-device comm in recurrence; all-reduce after row-parallel out proj).
Uses recurrent_gated_delta_rule_decode_ttnn. GDN norm is gated RMSNorm (weight, no +1), then SiLU(z).
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # sd keys may be raw (linear_attn.*), stripped, or bf16-remapped (qkv_proj + q/k/v_conv).
    P = "linear_attn." if any(k.startswith("linear_attn.") for k in sd) else ""

    def first_key(*names):
        for n in names:
            if (P + n) in sd:
                return sd[P + n]
        raise KeyError(f"none of {[P + n for n in names]} found in GDN state dict")

    # fused QKV+Z (column-parallel)
    qkv_w = first_key("in_proj_qkv.weight", "qkv_proj.weight")
    if (P + "conv1d.weight") in sd:
        conv1d_w = sd[P + "conv1d.weight"]
    else:  # bf16 remap: reassemble fused conv from q/k/v streams
        conv1d_w = torch.cat([sd[P + "q_conv.weight"], sd[P + "k_conv.weight"], sd[P + "v_conv.weight"]], dim=0)
    qkv_re = tpc.prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    fused = torch.cat(
        [
            torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
            for d in range(tp)
        ],
        dim=0,
    )
    tw = {}
    tw["qkvz"] = tpc.shard_w(
        fused, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("qkvz"), dtype=ttnn.bfloat8_b
    )
    # fused A+B (column-parallel)
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    ab = torch.cat(
        [torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0) for d in range(tp)],
        dim=0,
    )
    tw["ab"] = tpc.shard_w(
        ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=ttnn.bfloat8_b
    )
    # out proj (row-parallel)
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out"),
        dtype=ttnn.bfloat8_b,
    )
    # per-head params + conv taps
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    return tw


class TPGatedDeltaNet:
    """TP GDN: per-device value-head recurrence + all-reduce."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        # Pre-build chunk masks once (trace-safe; from_torch inside trace TT_FATALs).
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        self.conv_states = None
        self.rec_state = None
        # _stable_state: in-place state updates for decode trace (fixed buffer addresses).
        self._stable_state = False
        self.conv_carry = None  # cross-chunk prefill conv left-context
        self._zero_conv0 = None  # persistent zero for conv_states[0] (trace-safe)
        self._pending = []  # per-user states during batched per-user prefill

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 rec_state default (decode drift mitigation); QWEN35_GDN_STATE_BF16=1 for bf16.
        if os.environ.get("QWEN35_GDN_STATE_BF16") != "1":
            self.rec_state = ttnn.from_torch(
                torch.zeros(self.B, self.Nv, self.Dk, self.Dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None

    def reset_state_inplace(self):
        """Zero conv + rec state in place (preserves addresses for tracing)."""
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
            self._batched_conv_carry = None
        if self.conv_states is None:
            self.reset_state()
            return

        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        if self.conv_carry is None:
            self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
            self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        zc = z((1, self.B, self.qkv_dim_tp))
        for cs in self.conv_states:
            ttnn.copy(zc, cs)
        ttnn.deallocate(zc)
        zr = z((self.B, self.Nv, self.Dk, self.Dv))
        ttnn.copy(zr, self.rec_state)
        ttnn.deallocate(zr)
        zcc = z((1, self.K - 1, self.qkv_dim_tp))
        ttnn.copy(zcc, self.conv_carry)
        ttnn.deallocate(zcc)

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False):
        """Causal chunk-prefill from zero init. x: [1,1,T,dim].

        valid_len: real tokens (< T = right-pad masked). capture_state: write decode state.
        return_state: return (out, rec, conv) for assemble_batched_state; skip self.* writeback.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[1]
        # Pass raw valid_len (None for full chunk = trace-safe static slice, not from_torch one-hot).

        # Cross-chunk carry when _stable_state; return_state is always from scratch.
        carry = self._stable_state and not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, T, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, T, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, T, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, T, 2 * Nv))
        ttnn.deallocate(ab)

        # FIR conv + SiLU; conv_state = previous chunk's last K-1 inputs.
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=self.conv_carry if carry else None,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_len,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
        ttnn.deallocate(conv)
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=2)
        k = ttnn.repeat_interleave(k, rf, dim=2)

        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"]))), (1, T, Nv))
        ttnn.deallocate(a)

        o, final_state = chunk_gated_delta_rule_seq_adapter(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
        )
        B, D = 1, self.qkv_dim_tp
        captured = None
        if return_state:
            captured = (final_state, conv_new_state)
        else:
            if carry:
                ttnn.copy(final_state, self.rec_state)
                ttnn.deallocate(final_state)
                ttnn.copy(conv_new_state, self.conv_carry)
            else:
                self.rec_state = final_state
            # Finalize decode conv window (conv_states[1..K-1] = last K-1 real inputs).
            if capture_state:
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    zero = ttnn.from_torch(
                        torch.zeros(1, B, D, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
            ttnn.deallocate(conv_new_state)
        # gated RMSNorm + SiLU(z) gate
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if return_state:
            return out, captured[0], captured[1]
        return out

    def forward_prefill_collect(self, x, chunk_size=128, valid_len=None):
        """Per-user prefill; stash state for finalize_pending() → assemble_batched_state."""
        out, rec, conv = self.forward_prefill(x, chunk_size=chunk_size, valid_len=valid_len, return_state=True)
        self._pending.append((rec, conv))
        return out

    def finalize_pending(self):
        """Assemble collected per-user states into batched decode buffers."""
        assert self._pending, "finalize_pending called with no collected per-user states"
        rec_list = [r for (r, _) in self._pending]
        conv_list = [c for (_, c) in self._pending]
        self.assemble_batched_state(rec_list, conv_list)
        self._pending = []

    def assemble_batched_state(self, rec_list, conv_new_list):
        """Stitch B per-user (rec, conv) states into batched buffers (row u = user u).

        Under _stable_state: copy into fixed-address buffers; else assign.
        """
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        rec_batched = ttnn.concat(rec_list, dim=0)  # [B, Nv, Dk, Dv]
        conv_states = [
            ttnn.from_torch(
                torch.zeros(1, self.B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        ]
        for m in range(1, self.K):  # conv_states[m] row u = conv_new_list[u][:, m-1]
            rows = [
                ttnn.reshape(ttnn.slice(conv_new_list[u], (0, m - 1, 0), (1, m, D)), (1, 1, D)) for u in range(self.B)
            ]
            cs = ttnn.concat(rows, dim=1)  # [1, B, D]
            for r in rows:
                ttnn.deallocate(r)
            conv_states.append(cs)

        if self._stable_state and self.rec_state is not None:
            rec_src = (
                rec_batched
                if rec_batched.dtype == self.rec_state.dtype
                else ttnn.typecast(rec_batched, self.rec_state.dtype)
            )
            ttnn.copy(rec_src, self.rec_state)
            if rec_src is not rec_batched:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(rec_batched)
            for m in range(self.K):
                ttnn.copy(conv_states[m], self.conv_states[m])
                ttnn.deallocate(conv_states[m])
        else:
            self.rec_state = rec_batched
            self.conv_states = conv_states
        for t in rec_list:
            ttnn.deallocate(t)
        for t in conv_new_list:
            ttnn.deallocate(t)

    def forward_prefill_batched(self, x, chunk_size=128, valid_lens=None, carry=False):
        """Batched prefill: all B users in one pass. x: [B,T,dim]; valid_lens per-row masking.

        carry=True: chunk-outer carry from self.rec_state + _batched_conv_carry.
        Kernel cap: BH=B*Nv_tp <= ~32 (B<=4 at TP=4); larger B needs grouped launches.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[-3], x.shape[-2], x.shape[-1]))  # [.,B,T,dim] -> [B,T,dim]
        B, T = x.shape[0], x.shape[1]
        D = self.qkv_dim_tp

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (B, T, D))
        z = ttnn.slice(qkvz, (0, 0, D), (B, T, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (B, T, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (B, T, 2 * Nv))
        ttnn.deallocate(ab)

        # FIR conv; chunk-outer carry uses _batched_conv_carry as left-context.
        if carry and getattr(self, "_batched_conv_carry", None) is None:
            self._batched_conv_carry = ttnn.from_torch(
                torch.zeros(B, self.K - 1, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        conv_carry_in = self._batched_conv_carry if carry else None
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=conv_carry_in,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_lens,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (B, T, kd)), (B, T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (B, T, 2 * kd)), (B, T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (B, T, D)), (B, T, Nv, Dv))
        ttnn.deallocate(conv)
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=2)
        k = ttnn.repeat_interleave(k, rf, dim=2)

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"]))), (B, T, Nv))
        ttnn.deallocate(a)

        # Chunk-parallel recurrence over BH = B*Nv.
        o, final_state = chunk_gated_delta_rule_seq_adapter(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_lens,
        )

        # Write batched decode state (row u = user u).
        if self._stable_state and self.rec_state is not None:
            rec_src = (
                final_state
                if final_state.dtype == self.rec_state.dtype
                else ttnn.typecast(final_state, self.rec_state.dtype)
            )
            ttnn.copy(rec_src, self.rec_state)
            if rec_src is not final_state:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(final_state)
        else:
            self.rec_state = final_state
        # conv_states[0]=zero; conv_states[m] row u = conv_new_state[u, m-1].
        zero0 = ttnn.from_torch(
            torch.zeros(1, B, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        new_conv = [zero0]
        for m in range(1, self.K):
            cs = ttnn.reshape(ttnn.slice(conv_new_state, (0, m - 1, 0), (B, m, D)), (1, B, D))
            new_conv.append(cs)
        if carry:
            if conv_carry_in is not None:
                ttnn.deallocate(conv_carry_in)
            self._batched_conv_carry = conv_new_state
        else:
            ttnn.deallocate(conv_new_state)
        if self._stable_state and self.conv_states is not None:
            for m in range(self.K):
                ttnn.copy(new_conv[m], self.conv_states[m])
                ttnn.deallocate(new_conv[m])
        else:
            self.conv_states = new_conv

        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (B, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, B, T, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x):
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,B,dim]

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, B, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, B, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, B, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, B, 2 * Nv))
        ttnn.deallocate(ab)

        # conv1d shift-register + weighted sum + silu
        st = self.conv_states
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0])
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        # GQA expand Q/K; recurrence L2-norms internally
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (B, 1, Nv, Dk))
        k = ttnn.reshape(k, (B, 1, Nv, Dk))
        v = ttnn.reshape(v, (B, 1, Nv, Dv))

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"])))
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        # fp32 decode step default; QWEN35_GDN_DECODE_BF16=1 for bf16.
        o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q,
            k,
            v,
            beta,
            g,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        if self._stable_state:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self.rec_state = new_rec

        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)  # gated norm, no +1
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
