# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for Qwen3.5/3.6 Gated DeltaNet on Blackhole.

Tests (shared loaders/mesh from test_factory):
* test_gdn_tp         — decode PCC @ pos0 + pos1 shape/NaN
* test_gdn_tp_prefill — chunk-prefill vs step-by-step decode (T=128, internal consistency)

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
"""
import os

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    get_pcc_threshold,
    load_gdn_layer,
    model_path,
    parametrize_batch,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch()
def test_gdn_tp(mesh_device, B, reset_seeds, ensure_gc, request):
    """TP decode PCC @ pos0 vs torch ref; second step checks shape/NaN."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    out = gdn.forward_decode(x_tt)
    out_t = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()
    assert out_t.shape[-1] == args.dim and not torch.isnan(out_t).any() and out_t.abs().max() > 0

    # pos0 torch ref (full, unsharded)
    Nk, Nv, Dk, Dv = args.gdn_nk, args.gdn_nv, args.gdn_dk, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    xf = x[0, 0].float()
    qkv = xf @ sd["linear_attn.in_proj_qkv.weight"].float().T  # [B, 2*key_dim+value_dim]
    z = xf @ sd["linear_attn.in_proj_z.weight"].float().T
    b = xf @ sd["linear_attn.in_proj_b.weight"].float().T  # [B, Nv]
    tap3 = sd["linear_attn.conv1d.weight"].float()[:, 0, 3]  # [qkv_dim], newest-token tap
    conv = F.silu(qkv * tap3)
    q = conv[:, :key_dim].reshape(B, Nk, Dk)
    k = conv[:, key_dim : 2 * key_dim].reshape(B, Nk, Dk)
    v = conv[:, 2 * key_dim :].reshape(B, Nv, Dv)
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=1)
    k = k.repeat_interleave(rf, dim=1)
    q = F.normalize(q, dim=-1) * (Dk**-0.5)
    k = F.normalize(k, dim=-1)
    beta = torch.sigmoid(b)  # [B, Nv]
    qk = (q * k).sum(-1)  # [B, Nv]
    o = beta[..., None] * qk[..., None] * v  # [B, Nv, Dv]
    # gated RMSNorm (weight only, no +1)
    o_n = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * sd["linear_attn.norm.weight"].float()
    gated = (o_n * F.silu(z.reshape(B, Nv, Dv))).reshape(B, value_dim)
    ref = gated @ sd["linear_attn.out_proj.weight"].float().T  # [B, dim]

    passing, pcc = comp_pcc(ref, out_t, get_pcc_threshold(request))
    logger.info(f"GDN TP PCC (pos0) = {pcc}")
    assert passing, f"GDN TP PCC too low: {pcc}"

    x2 = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    out2 = gdn.forward_decode(x2)
    out2_t = ttnn.to_torch(out2, mesh_composer=tp_composer(mesh_device))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: GDN TP decode (pos0 PCC + pos1 shape/NaN)")


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch(batches=(8, 32))
def test_gdn_tp_peruser_state(mesh_device, B, reset_seeds, ensure_gc, request):
    """Per-user prefill state assembled into batched buffers must match B independent B=1 runs."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    # B=1 reference needs max_batch_size=1 (forward_decode keys shapes off self.B).
    args1 = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} B={B}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)
    T = 128  # one chunk-seq kernel chunk

    xp = [torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16) for _ in range(B)]
    xd = [torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16) for _ in range(B)]

    # Reference: B independent B=1 prefill + decode.
    ref_rows = []
    for u in range(B):
        g = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        g.reset_state()
        g.forward_prefill(replicate_to_device(mesh_device, xp[u]), chunk_size=T, capture_state=True)
        out_u = g.forward_decode(replicate_to_device(mesh_device, xd[u]))
        ref_rows.append(ttnn.to_torch(out_u, mesh_composer=comp)[0, 0, 0].float())

    # Batched: per-user prefill → assemble_batched_state → single decode.
    gb = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    rec_list, conv_list = [], []
    for u in range(B):
        _, rec_u, conv_u = gb.forward_prefill(replicate_to_device(mesh_device, xp[u]), chunk_size=T, return_state=True)
        rec_list.append(rec_u)
        conv_list.append(conv_u)
    gb.assemble_batched_state(rec_list, conv_list)
    x_dec = torch.cat(xd, dim=2)  # [1, 1, B, dim]
    out_b = gb.forward_decode(replicate_to_device(mesh_device, x_dec))
    out_t = ttnn.to_torch(out_b, mesh_composer=comp)  # [1, 1, B, dim]

    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref_rows[u], out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"per-user GDN state (B={B}) PCC min={worst:.5f} max={max(pccs):.5f}")
    bad = [(u, p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"users below PCC {thr}: {bad}"
    logger.info(f"PASSED: per-user GDN state (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
# B capped at (2,4): chunk-seq kernel L1-bound (BH=B*Nv_tp <= ~32 at TP=4).
@parametrize_batch(batches=(2, 4))
def test_gdn_tp_batched_prefill(mesh_device, B, reset_seeds, ensure_gc, request):
    """True batched prefill (forward_prefill_batched) must match B independent B=1 prefills."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    args1 = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} B={B}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)

    Tmax = 128
    lens = [Tmax - 8 * (u % 8) for u in range(B)]  # {72..128}
    xp = [torch.randn(1, 1, lens[u], args.dim, dtype=torch.bfloat16) for u in range(B)]
    xd = [torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16) for u in range(B)]

    # Reference: B independent B=1 prefill + decode.
    ref_rows = []
    for u in range(B):
        g = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        g.reset_state()
        g.forward_prefill(replicate_to_device(mesh_device, xp[u]), chunk_size=Tmax, capture_state=True)
        out_u = g.forward_decode(replicate_to_device(mesh_device, xd[u]))
        ref_rows.append(ttnn.to_torch(out_u, mesh_composer=comp)[0, 0, 0].float())

    # Batched: pad to Tmax, one forward_prefill_batched, then decode.
    gb = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gb.reset_state()
    x_pad = torch.zeros(B, Tmax, args.dim, dtype=torch.bfloat16)
    for u in range(B):
        x_pad[u, : lens[u], :] = xp[u][0, 0]
    gb.forward_prefill_batched(replicate_to_device(mesh_device, x_pad), chunk_size=Tmax, valid_lens=lens)
    x_dec = torch.cat(xd, dim=2)  # [1, 1, B, dim]
    out_b = gb.forward_decode(replicate_to_device(mesh_device, x_dec))
    out_t = ttnn.to_torch(out_b, mesh_composer=comp)  # [1, 1, B, dim]

    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref_rows[u], out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"batched GDN prefill (B={B}) PCC min={worst:.5f} max={max(pccs):.5f} lens={lens}")
    bad = [(u, lens[u], p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"users below PCC {thr}: {bad}"
    logger.info(f"PASSED: batched GDN prefill (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch(batches=(2,))
def test_gdn_tp_batched_prefill_chunked(mesh_device, B, reset_seeds, ensure_gc, request):
    """Two carried chunks (carry=True) must match single-shot batched prefill."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)

    C = 128
    T = 2 * C
    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
    xd = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)

    # Reference: single-shot batched prefill over full T.
    gref = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gref.reset_state()
    gref._stable_state = True
    gref.forward_prefill_batched(replicate_to_device(mesh_device, x.unsqueeze(0)), chunk_size=C)
    out_ref = ttnn.to_torch(gref.forward_decode(replicate_to_device(mesh_device, xd)), mesh_composer=comp)

    # Test: two carried chunks.
    g = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    g.reset_state()
    g._stable_state = True
    g.reset_state_inplace()
    g.forward_prefill_batched(replicate_to_device(mesh_device, x[:, :C].unsqueeze(0)), chunk_size=C, carry=True)
    g.forward_prefill_batched(replicate_to_device(mesh_device, x[:, C:].unsqueeze(0)), chunk_size=C, carry=True)
    out_t = ttnn.to_torch(g.forward_decode(replicate_to_device(mesh_device, xd)), mesh_composer=comp)

    thr = get_pcc_threshold(request, default=0.99)
    pccs = [compute_pcc(out_ref[0, 0, u].float(), out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"batched chunk-outer carry (B={B}) PCC min={worst:.5f} max={max(pccs):.5f}")
    assert worst >= thr, f"carry vs single-shot PCC {worst:.5f} < {thr}: {pccs}"
    logger.info(f"PASSED: batched chunk-outer GDN prefill carry (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_tp_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """Chunk-prefill vs step-by-step decode on T=128 (zero init, no external ref)."""
    os.environ.setdefault("HF_MODEL", model_path())
    T = 128
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} T={T}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    composer = tp_composer(mesh_device)

    gdn.reset_state()
    out_pf = gdn.forward_prefill(x_tt, chunk_size=128)
    pf = ttnn.to_torch(out_pf, mesh_composer=composer)[0, 0].float()  # [T, dim]

    gdn.reset_state()
    dec_rows = []
    for t in range(T):
        xt = replicate_to_device(mesh_device, x[:, :, t : t + 1, :])
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())  # [dim]
    dec = torch.stack(dec_rows, dim=0)  # [T, dim]

    passing, pcc = comp_pcc(dec, pf, get_pcc_threshold(request))
    logger.info(f"GDN TP PREFILL vs DECODE PCC (T={T}) = {pcc}")
    assert passing, f"GDN prefill/decode mismatch PCC: {pcc}"
