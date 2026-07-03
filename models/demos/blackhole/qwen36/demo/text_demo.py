# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5/3.6 E2E text gen on Blackhole (P150 / P150x4).

Parametrized prefill+decode: ISL 128-256k (single-user), batched B=8/B=32 (TP) up to 64k.

Run all:      pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s
Run 128:      pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"
Run batched:  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "b8"
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name

# MESH_DEVICE selects TP mesh: 27B default P150x4 (1,4); 9B single P150 (1,1). Multi-device needs FABRIC_1D.
_MESH_SHAPE = {"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 4))
_MULTI = _MESH_SHAPE != (1, 1)
# TP long-context prefill replays a per-chunk trace; needs trace_region_size (default 0). 256 MiB matches validated TP config.
_TP_TRACE_REGION_SIZE = 256 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": _TP_TRACE_REGION_SIZE} if _MULTI else {}
        ),
    }
]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen36/demo/sample_prompts"
SHARED_PROMPTS_DIR = "models/demos/llama3_70b_galaxy/demo/sample_prompts"


# seqlen → index in eval_frankenstein_long.json. Clips text to exceed target tokens (~4.3 chars/token).
_FRANKENSTEIN_CONFIGS = {
    8192: 0,  # ~16k tokens
    16384: 1,  # ~32k tokens
    32768: 1,
    65536: 2,  # ~70k tokens
    131072: 3,  # ~104k tokens (Frankenstein max)
    # 256k: index 4 = War and Peace clipped to ~256k tokens (test_demo_text guards actual_len).
    262144: 4,
}


def _load_and_cache_context(context_url, max_length=None):
    """Download text from URL, cache locally, clip to max_length."""
    cache_dir = Path(SAMPLE_PROMPTS_DIR) / ".context_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        context_text = cache_file.read_text()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        response = requests.get(context_url)
        response.raise_for_status()
        context_text = response.text
        cache_file.write_text(context_text)
        logger.info(f"Downloaded and cached context: {context_url}")

    if max_length:
        context_text = context_text[:max_length]
    return context_text


def _get_prompt(seqlen, tokenizer):
    """Load ~seqlen tokens, clipped not padded. Shared llama3_70b_galaxy prompts; no pad (logits from last token)."""
    # QWEN35_REF_PROMPT=1: match reference 27B 64k task (pg84.txt, chat template, no manual thinking seed).
    if os.environ.get("QWEN35_REF_PROMPT") and seqlen >= 4096:
        with open(f"{SHARED_PROMPTS_DIR}/input_data_long_64k.json") as f:
            rd = json.load(f)[0]
        context = _load_and_cache_context(rd["context"], rd.get("max_length"))
        instruction = rd["prompt"]
        sys_msg = "You are a helpful assistant."
        overhead = len(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": "\n\n" + instruction}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        context = tokenizer.decode(ctx_ids[: max(0, seqlen - overhead - 8)])
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": context + "\n\n" + instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :seqlen]

    if seqlen <= 128:
        path = f"{SHARED_PROMPTS_DIR}/input_data_questions_prefill_128.json"
        with open(path) as f:
            data = json.load(f)
        inputs = tokenizer(data[0]["prompt"], return_tensors="pt")
        return inputs["input_ids"][:, :seqlen]

    # Long (16k+): Frankenstein corpus; model continues raw text.
    if seqlen in _FRANKENSTEIN_CONFIGS:
        idx = _FRANKENSTEIN_CONFIGS[seqlen]
        path = f"{SAMPLE_PROMPTS_DIR}/eval_frankenstein_long.json"
        with open(path) as f:
            data = json.load(f)
        entry = data[idx]
        context = _load_and_cache_context(entry["context"], entry.get("max_length"))
        instruction = entry["prompt"]
        prefix = "<|im_start|>user\n"
        # Seed <think> for reasoning; long context dilutes suffix signal without it.
        # QWEN35_NO_THINK=1: empty thinking block (enable_thinking=False scaffold).
        if os.environ.get("QWEN35_NO_THINK"):
            suffix = (
                f"\n\nBased on the above text: {instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            suffix = f"\n\nBased on the above text: {instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        wrapper_ids = tokenizer(prefix + suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_context_tokens = seqlen - wrapper_ids.shape[1]
        context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][
            :, :max_context_tokens
        ]
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat([prefix_ids, context_ids, suffix_ids], dim=1)[:, :seqlen]

    # Medium (1k–8k): static prompt files.
    size_label = f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen)
    path = f"{SAMPLE_PROMPTS_DIR}/input_data_long_{size_label}.json"
    with open(path) as f:
        data = json.load(f)
    prompt_text = data[0]["prompt"]
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs["input_ids"][:, :seqlen]


def _warmup_prefill(model, device, token_ids):
    """Compile prefill programs (discarded). Caps at 4096 tokens to avoid non-paged L1 clashes at 8K+."""
    T = token_ids.shape[1]
    warmup_tokens = token_ids[:, : min(T, 4096)]
    warmup_len = warmup_tokens.shape[1]
    logger.info(f"Warmup prefill ({warmup_len} tokens) — compiling programs...")
    t0 = time.time()
    logits = model.prefill(warmup_tokens)
    ttnn.synchronize_device(device)

    compile_time = time.time() - t0
    logger.info(f"Warmup complete: {compile_time:.1f}s (programs now cached)")

    model.reset_state(batch_size=token_ids.shape[0])


BLOCK_SIZE = 64
PREFILL_CHUNK = 2048  # prompts padded to multiples of this
# 4096 blocks × 64 = 262144 (native context). _blocks_for sizes per seqlen.
MAX_BLOCK_BUDGET = 4096


def _blocks_for(seqlen, max_generated_tokens):
    """Block budget for padded prefill bucket + decode, capped at 256k, floored at 64 blocks (4k)."""
    bucket = ((seqlen + PREFILL_CHUNK - 1) // PREFILL_CHUNK) * PREFILL_CHUNK
    needed = bucket + max_generated_tokens
    blocks = max(64, (needed + BLOCK_SIZE - 1) // BLOCK_SIZE)
    # Chunked SDPA needs num_blocks % 8 == 0 (stick alignment).
    blocks = ((blocks + 7) // 8) * 8
    return min(MAX_BLOCK_BUDGET, blocks)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, use_trace, batch, repeat_batches",
    [
        pytest.param(128, 50, True, 1, 1, id="traced_128"),
        pytest.param(128, 50, False, 1, 1, id="paged_128"),
        pytest.param(4096, 100, True, 1, 1, id="traced_4k"),
        pytest.param(4096, 100, False, 1, 1, id="paged_4k"),
        pytest.param(8192, 500, True, 1, 1, id="traced_8k"),
        pytest.param(8192, 100, False, 1, 1, id="paged_8k"),
        pytest.param(16384, 100, True, 1, 1, id="traced_16k"),
        pytest.param(32768, 100, True, 1, 1, id="traced_32k"),
        pytest.param(65536, 500, True, 1, 1, id="traced_64k"),
        pytest.param(65536, 100, False, 1, 1, id="paged_64k"),
        pytest.param(131072, 100, True, 1, 1, id="traced_128k"),
        pytest.param(262144, 100, True, 1, 1, id="traced_256k"),
        pytest.param(128, 50, True, 1, 2, id="determinism_128"),
        # Batched decode (TP): shared paged KV + batched GDN state.
        pytest.param(128, 50, True, 8, 1, id="batched_128_b8"),
        pytest.param(128, 50, True, 32, 1, id="batched_128_b32"),
        # T>128 → prefill_chunked_peruser.
        pytest.param(4096, 50, True, 8, 1, id="batched_4k_b8"),
        pytest.param(4096, 50, True, 32, 1, id="batched_4k_b32"),
        # B=8 long-context ladder; paged KV ~1–8 GB/device at 8k–64k.
        pytest.param(8192, 50, True, 8, 1, id="batched_8k_b8"),
        pytest.param(16384, 50, True, 8, 1, id="batched_16k_b8"),
        pytest.param(32768, 50, True, 8, 1, id="batched_32k_b8"),
        # 64k TTFT exceeds default 300s pytest timeout.
        pytest.param(65536, 50, True, 8, 1, id="batched_64k_b8", marks=pytest.mark.timeout(900)),
    ],
)
def test_demo_text(
    mesh_device,
    seqlen,
    max_generated_tokens,
    use_trace,
    batch,
    repeat_batches,
):
    """E2E text generation: prefill + decode with perf validation."""
    from transformers import AutoTokenizer

    device = mesh_device
    if batch > 1 and not _MULTI:
        pytest.skip("batched decode is the TP (multi-device) path; run with MESH_DEVICE=P150x4")
    device.enable_program_cache()
    num_blocks = _blocks_for(seqlen, max_generated_tokens)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        # n_layers=4,  # uncomment for fast iteration; default uses 32-layer config
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    token_ids = _get_prompt(seqlen, tokenizer)
    # Prompt + generation must fit max_seq_len; floor to 128-multiple (GDN sub-chunk alignment).
    max_prompt_len = ((max_seq_len - max_generated_tokens) // 128) * 128
    if token_ids.shape[1] > max_prompt_len:
        token_ids = token_ids[:, :max_prompt_len]
    actual_len = token_ids.shape[1]
    # Fail loudly if 256k corpus (index 4) is too short; Frankenstein cases (0–3) cap at ~104k.
    if _FRANKENSTEIN_CONFIGS.get(seqlen) == 4:
        assert (
            actual_len >= 0.95 * seqlen
        ), f"prompt clipped to {actual_len} tokens, expected ~{seqlen} (corpus too short for seqlen={seqlen})"
    logger.info(
        f"Prompt: {actual_len} tokens (block budget: {num_blocks} blocks x {BLOCK_SIZE} = {max_seq_len} tokens)"
    )

    # TP: chunk-outer prefill + paged decode; GDN state + paged KV carry across chunks.
    if model.num_devices > 1 and batch > 1:
        # Replicate one prompt to B users; caller asserts identical decode.
        rows, perf = _run_tp_generation_batched(model, tokenizer, token_ids, max_generated_tokens, batch)
        text0 = tokenizer.decode(rows[0], skip_special_tokens=True)
        logger.info(
            f"[TP {model.num_devices}-dev B={batch}] ttft={perf['ttft_s']:.2f}s "
            f"per-user-decode={perf['decode_tok_s']:.2f} tok/s aggregate={perf['agg_tok_s']:.1f} tok/s"
        )
        logger.info(f"[TP B={batch}] GENERATED (row 0): {text0!r}")
        for u in range(batch):
            assert len(rows[u]) == max_generated_tokens, f"row {u}: {len(rows[u])} != {max_generated_tokens}"
            assert rows[u] == rows[0], f"row {u} diverged from row 0 (identical prompts must decode identically)"
        assert len(set(rows[0])) > 1, f"degenerate generation: {rows[0]}"
        return

    if model.num_devices > 1:
        if repeat_batches > 1:
            results = []
            for run in range(repeat_batches):
                gen, _ = _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks)
                results.append(gen)
                if run < repeat_batches - 1:
                    model.free_kv_caches()
            for i in range(1, repeat_batches):
                assert results[0] == results[i], (
                    f"Non-deterministic output between run 0 and run {i}.\n"
                    f"Run 0: {results[0]}\nRun {i}: {results[i]}"
                )
            return
        generated, perf = _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"[TP {model.num_devices}-dev] ttft={perf['ttft_s']:.2f}s decode={perf['decode_tok_s']:.2f} tok/s")
        logger.info(f"[TP] GENERATED: {text!r}")
        assert len(generated) == max_generated_tokens, f"{len(generated)} != {max_generated_tokens}"
        assert len(set(generated)) > 1, f"degenerate generation: {generated}"
        # Perf JSON for CI target check (validate_perf_targets.py); not asserted here.
        _save_tp_benchmark(perf, model, seqlen=seqlen, prompt_len=actual_len, num_generated=len(generated))
        return

    # Skip legacy warmup for short traced prompts (masked-bucket path compiles in capture).
    PREFILL_CHUNK = 2048
    t_compile = time.time()
    if not (use_trace and actual_len < PREFILL_CHUNK):
        _warmup_prefill(model, device, token_ids)
    t_compile = time.time() - t_compile

    if repeat_batches > 1:
        results = []
        for run in range(repeat_batches):
            if use_trace:
                gen, _ = _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks)
            else:
                gen, _ = _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks)
            results.append(gen)
            if run < repeat_batches - 1:
                model.free_kv_caches()
        for i in range(1, repeat_batches):
            assert results[0] == results[i], (
                f"Non-deterministic output between run 0 and run {i}.\n" f"Run 0: {results[0]}\nRun {i}: {results[i]}"
            )
        return

    if use_trace:
        generated, perf = _run_traced_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
            num_blocks,
        )
    else:
        generated, perf = _run_paged_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
            num_blocks,
        )

    perf["compile_time"] = t_compile
    text = tokenizer.decode(generated, skip_special_tokens=True)
    _log_results(perf, actual_len, len(generated), text)
    _assert_results(perf, actual_len, len(generated))


def _should_use_chunked_trace(model):
    """Use chunk-outer trace (one 2048-token chunk replayed) to stay under 4 GiB trace ceiling."""
    return any(
        (not layer.is_full_attention)
        and getattr(getattr(layer.attention, "weights", None), "use_chunk_seq_prefill", False)
        for layer in model.layers
    )


def _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks):
    """TP generation: traced chunk-outer prefill + paged decode. Returns (tokens, perf_dict).

    Captures one 2048-token chunk trace, replays per chunk with GDN/KV state carried in place.
    Eager fallback OOMs past ~7872 tokens. Mirrors test_model_tp_contract.py / qwen36_vllm.py.
    """
    vocab = model.args.vocab_size
    T = token_ids.shape[1]

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Page-table width must be multiple of 32 for flexible chunked SDPA.
    num_blocks = ((num_blocks + 31) // 32) * 32

    # Paged KV (n_local_kv_heads at TP) + GDN state reset.
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    CHUNK = 2048
    t_cap = time.time()
    profiler.start("compile_prefill")
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
    profiler.end("compile_prefill")
    logger.info(f"[TP] prefill chunk-trace captured in {time.time() - t_cap:.1f}s")

    t0 = time.time()
    profiler.start("inference_prefill")
    logits_dev = model.prefill_traced_chunked(token_ids[:, :T], page_table, actual_len=T)
    ttnn.synchronize_device(model.device)
    profiler.end("inference_prefill")
    ttft = time.time() - t0

    # Greedy default; QWEN35_TEMP>0 for sampling. QWEN35_REP_PENALTY / QWEN35_NO_REPEAT_NGRAM optional.
    _temp = float(os.environ.get("QWEN35_TEMP", "0") or 0)
    _rep_pen = float(os.environ.get("QWEN35_REP_PENALTY", "1.0") or 1.0)
    _no_repeat = int(os.environ.get("QWEN35_NO_REPEAT_NGRAM", "0") or 0)
    generated = []

    def _pick(vec):
        v = vec.float()
        if _rep_pen != 1.0 and generated:
            idx = torch.tensor(sorted(set(generated)))
            s = v[idx]
            v[idx] = torch.where(s > 0, s / _rep_pen, s * _rep_pen)
        if _no_repeat > 0 and len(generated) >= _no_repeat - 1:
            prefix = tuple(generated[-(_no_repeat - 1) :]) if _no_repeat > 1 else ()
            n = _no_repeat
            for i in range(len(generated) - n + 1):
                if tuple(generated[i : i + n - 1]) == prefix:
                    v[generated[i + n - 1]] = float("-inf")
        if _temp > 0:
            return int(torch.multinomial(torch.softmax(v / _temp, dim=-1), 1).item())
        return int(torch.argmax(v).item())

    # Logits replicated on mesh; gather one replica.
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    nxt = _pick(lt.reshape(-1, vocab)[0])
    generated.append(nxt)

    # Traced decode: snapshot/restore GDN around throwaway capture (capture advances state).
    # QWEN35_TP_DECODE_EAGER=1 forces eager loop.
    from models.tt_transformers.tt.common import copy_host_to_device

    mesh = model.mesh_device
    eager = os.environ.get("QWEN35_TP_DECODE_EAGER") == "1"

    def _read(out):
        return _pick(model.process_output_decode(out, B=1, S=1).reshape(-1)[:vocab])

    def _update(token, position):
        host = model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=page_table,
        )
        copy_host_to_device(host, device_tensors=dev)

    # Snapshot all TP ranks' GDN state; restore via ttnn.copy into same buffers.
    _gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def _snapshot_gdn():
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in _gdn
        ]

    def _restore_gdn(snap):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    # Persistent decode input buffers.
    dev = model.prepare_inputs_decode(
        torch.tensor([[nxt]], dtype=torch.int32),
        torch.tensor([T], dtype=torch.int32),
        page_table=page_table,
    )

    trace_id = None
    tt_logits = None
    profiler.start("compile_decode")
    if not eager:
        gdn_snap = _snapshot_gdn()
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        _restore_gdn(gdn_snap)
    profiler.end("compile_decode")

    pos = T
    decode_times = []
    profiler.start("inference_decode")
    while len(generated) < max_generated_tokens:
        _update(nxt, pos)
        t_step = time.time()
        if eager:
            tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        decode_times.append(time.time() - t_step)
        nxt = _read(tt_logits)
        generated.append(nxt)
        pos += 1
    if trace_id is not None:
        ttnn.release_trace(mesh, trace_id)
    profiler.end("inference_decode")

    # Drop first decode step (one-time costs) for steady-state throughput.
    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    profiler.end("run")
    return generated, {"ttft_s": ttft, "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0, "profiler": profiler}


def _run_tp_generation_batched(model, tokenizer, token_ids, max_generated_tokens, batch):
    """TP batched generation: per-user prefill into shared paged KV, then B-wide traced decode."""
    from models.tt_transformers.tt.common import copy_host_to_device

    B = batch
    vocab = model.args.vocab_size
    mesh = model.mesh_device
    T = token_ids.shape[1]

    # Per-user blocks; round up to multiple of 8 (SDPA stick alignment).
    bpu = max(8, -(-(T + max_generated_tokens) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    total_blocks = B * bpu
    kv_cache_shape = [total_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=B)
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])  # [B, bpu]

    # Prefill routes: T==128 traced bucket; T>128 prefill_chunked_peruser; T<128 prefill_paged_peruser.
    # QWEN_BATCHED_GROUPED=1 (default): group short prompts (T<=128) for ~4x TTFT at B=32.
    bucket = 128
    eager = os.environ.get("QWEN35_TP_PREFILL_EAGER") == "1"
    grouped_short = os.environ.get("QWEN_BATCHED_GROUPED", "1") != "0" and T <= 128
    use_traced_bucket = (T == bucket) and not eager and not grouped_short
    token_list = [token_ids[:, :T] for _ in range(B)]
    if use_traced_bucket:
        model.capture_prefill_trace_bucket(mesh, page_table[0:1].contiguous(), bucket=bucket)
    t0 = time.time()
    if grouped_short:
        pf_logits = model.prefill_paged_grouped(token_list, page_table, valid_lens=[T] * B, group_size=4)
    elif use_traced_bucket:
        pf_logits = model.prefill_traced_bucket_batched(token_list, page_table, valid_lens=[T] * B)
    elif T > bucket:
        pf_logits = model.prefill_chunked_peruser(token_list, page_table, valid_lens=[T] * B)
    else:
        pf_logits = model.prefill_paged_peruser(token_list, page_table, valid_lens=[T] * B)
    ttnn.synchronize_device(mesh)
    ttft = time.time() - t0
    if use_traced_bucket:
        model.release_prefill_trace_bucket()

    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def _pick(vec):
        return int(torch.argmax(vec.float()).item())

    nxt = [_pick(ttnn.to_torch(pf_logits[u], mesh_composer=comp0).reshape(-1, vocab)[0]) for u in range(B)]
    generated = [[nxt[u]] for u in range(B)]

    # Traced batched decode with GDN snapshot/restore around throwaway capture.
    eager = os.environ.get("QWEN35_TP_DECODE_EAGER") == "1"
    _gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def _snapshot_gdn():
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in _gdn
        ]

    def _restore_gdn(snap):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    def _update(tokens_row, positions):
        host = model.prepare_decode_inputs_host(
            torch.tensor(tokens_row, dtype=torch.int32).reshape(B, 1),
            torch.tensor(positions, dtype=torch.int32),
            page_table=page_table,
        )
        copy_host_to_device(host, device_tensors=dev)

    pos = [T] * B
    dev = model.prepare_inputs_decode(
        torch.tensor(nxt, dtype=torch.int32).reshape(B, 1),
        torch.tensor(pos, dtype=torch.int32),
        page_table=page_table,
    )

    trace_id, tt_logits = None, None
    if not eager:
        snap = _snapshot_gdn()
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        _restore_gdn(snap)

    # End-to-end decode timing (update + device + readback + select).
    decode_times = []
    while len(generated[0]) < max_generated_tokens:
        t_step = time.time()
        _update([generated[u][-1] for u in range(B)], pos)
        if eager:
            tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        logits_step = model.process_output_decode(tt_logits, B)  # [B, 1, vocab]
        for u in range(B):
            generated[u].append(_pick(logits_step[u, 0, :vocab]))
        pos = [p + 1 for p in pos]
        decode_times.append(time.time() - t_step)
    if trace_id is not None:
        ttnn.release_trace(mesh, trace_id)

    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    return generated, {
        "ttft_s": ttft,
        "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0,  # per step (all B users)
        "agg_tok_s": (B / avg) if avg > 0 else 0.0,
    }


def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Prefill + paged traced decode. Returns (tokens, perf_dict)."""
    T = token_ids.shape[1]

    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # Chunk-outer trace: one 2048-token chunk replayed per chunk (under 4 GiB ceiling).
    assert _should_use_chunked_trace(model), "chunk-seq GDN prefill must be enabled"
    chunk_size = 2048
    bucket_size = ((T + chunk_size - 1) // chunk_size) * chunk_size
    logger.info(f"Capturing prefill trace at bucket_size={bucket_size} (prompt {T} tokens, chunk-outer replay)...")
    t_cap = time.time()
    model.capture_prefill_trace_chunked(
        device, page_table, chunk_size=chunk_size, warmup_masked_buckets=(T < chunk_size)
    )
    logger.info(f"Prefill trace captured in {time.time() - t_cap:.1f}s")
    pad_len = bucket_size - T
    # Pad with last real token (not 0) to avoid corrupting DeltaNet recurrence state.
    last_token = token_ids[:, -1:].expand(1, pad_len) if pad_len > 0 else token_ids[:, :0]
    padded_token_ids = torch.cat([token_ids, last_token], dim=1)

    t0 = time.time()
    if T < chunk_size:
        logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=T)
    else:
        logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=T)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()
    gen = Generator([model], [model.args], device)

    # prime_decode_trace: GDN snapshot/restore so capture doesn't double-advance state.
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace

    prime_decode_trace(gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([T]), page_table)

    generated = [next_token]
    decode_times = []
    current_pos = T

    for i in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([current_pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)

        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), f"NaN in traced decode at step {i}"
        next_token = int(dl.argmax())
        generated.append(next_token)
        current_pos += 1

        if next_token == tokenizer.eos_token_id:
            break

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Prefill + paged decode (non-traced). Returns (tokens, perf_dict)."""
    T = token_ids.shape[1]

    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    t0 = time.time()
    logits = model.prefill_paged(token_ids, page_table)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in paged prefill logits"
    next_token = logits_torch.argmax().item()

    gen = Generator([model], [model.args], device)

    generated = [next_token]
    decode_times = []

    for i in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([T + i]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=False,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)

        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), f"NaN in paged decode logits at step {i}"
        next_token = int(dl.argmax())

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _save_tp_benchmark(perf, model, seqlen, prompt_len, num_generated):
    """Emit benchmark JSON for CI perf check (no-op outside CI). Uses nominal seqlen for target lookup."""
    profiler = perf["profiler"]
    ttft_s = perf["ttft_s"]
    decode_tok_s = perf["decode_tok_s"]
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "compile_decode": profiler.get_duration("compile_decode"),
        "prefill_t/s": (prompt_len / ttft_s) if ttft_s > 0 else 0.0,
        "prefill_time_to_token": ttft_s,
        "decode_t/s": decode_tok_s,
        "decode_t/s/u": decode_tok_s,
    }
    benchmark_data = create_benchmark_data(profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, {})
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name=model.args.base_model_name,
        ml_model_type="llm",
        device_name=determine_device_name(model.mesh_device),
        num_layers=model.args.n_layers,
        batch_size=1,
        input_sequence_length=seqlen,
        output_sequence_length=num_generated,
    )


def _log_results(perf, prompt_len, num_generated, text):
    ttft = perf["ttft"]
    avg_ms = perf["avg_decode_s"] * 1000
    tok_s = 1000.0 / avg_ms if avg_ms > 0 else 0
    compile_time = perf.get("compile_time", 0)

    logger.info("=" * 70)
    logger.info(f"  Compile (warmup):    {compile_time:.3f}s")
    logger.info(f"  Prefill {prompt_len} tokens:  TTFT = {ttft:.3f}s ({prompt_len / ttft:.0f} tok/s)")
    logger.info(f"  Decode:  {avg_ms:.1f}ms/token  ({tok_s:.1f} tok/s)")
    logger.info(f"  Generated {num_generated} tokens in {perf['decode_steps']} steps")
    logger.info(f"  Text: {text[:6000]}")
    logger.info("=" * 70)


def _assert_results(perf, prompt_len, num_generated):
    # Correctness only; perf targets checked by validate_perf_targets.py.
    assert num_generated >= 1, "Should generate at least 1 token"
