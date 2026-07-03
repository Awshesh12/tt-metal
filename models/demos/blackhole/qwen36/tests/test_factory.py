# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Qwen3.5/3.6 demo tests.

Provides model_path, layer weight loaders, PCC helpers/thresholds, mesh parametrization,
and TP tensor helpers. Heavy imports stay lazy so CPU-only tests collect fast.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import ttnn

# TP tests default 27B; single-device tests setdefault 9B.
_DEFAULT_HF_MODEL = "Qwen/Qwen3.6-27B"
_PCC_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "pcc_thresholds.json")

# Prefill buckets gated by --max-prefill (conftest); lengths above cap are skipped.
PREFILL_BUCKETS = [128, 1024, 2048]


def model_path():
    """Resolve HF model id/dir from HF_MODEL."""
    return os.path.expanduser(os.environ.get("HF_MODEL", _DEFAULT_HF_MODEL))


def load_layer_weights(ckpt_dir, layer_idx, names, search_prefix, out_prefix=""):
    """Dequantize selected layers.<i>.<search_prefix><name> weights into a dict.

    ``names`` may be leaf names or (name, has_weight_suffix) tuples (e.g. A_log, dt_bias).
    """
    from safetensors import safe_open

    from ..tt.tp_common import dequant_fp8_block

    ckpt_dir = Path(ckpt_dir)
    wm = json.load(open(ckpt_dir / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for entry in names:
        name, has_w = entry if isinstance(entry, tuple) else (entry, True)
        leaf = f"{search_prefix}{name}" + (".weight" if has_w else "")
        base = next(k for k in wm if k.endswith(f"layers.{layer_idx}.{leaf}"))
        with safe_open(str(ckpt_dir / wm[base]), framework="pt") as sf:
            w = sf.get_tensor(base)
            sk = base + "_scale_inv"
            if wm.get(sk):
                with safe_open(str(ckpt_dir / wm[sk]), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(sk))
            else:
                w = w.to(torch.bfloat16)
        out[f"{out_prefix}{name}" + (".weight" if has_w else "")] = w
    return out


def load_attn_layer(ckpt_dir, layer_idx):
    """Full-attention layer weights (q/k/v/o proj + q/k norm)."""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"],
        search_prefix="self_attn.",
    )


def load_gdn_layer(ckpt_dir, layer_idx):
    """Gated-DeltaNet layer weights (linear_attn.*)."""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "conv1d",
            ("A_log", False),
            ("dt_bias", False),
            "norm",
        ],
        search_prefix="linear_attn.",
        out_prefix="linear_attn.",
    )


def load_mlp_layer(ckpt_dir, layer_idx):
    """SwiGLU MLP weights (gate/up/down proj)."""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        ["gate_proj", "up_proj", "down_proj"],
        search_prefix="mlp.",
    )


def compute_pcc(a, b):
    """Pearson correlation between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


def compare_tensors(tt_tensor, torch_tensor, pcc_threshold=0.99):
    """Compare TT/torch tensor to reference; returns (passing, pcc)."""
    from loguru import logger

    from models.common.utility_functions import comp_pcc

    tt = tt_tensor if isinstance(tt_tensor, torch.Tensor) else ttnn.to_torch(tt_tensor)
    passing, pcc = comp_pcc(torch_tensor, tt, pcc_threshold)
    logger.info(f"PCC={pcc} (threshold={pcc_threshold}) [{'PASS' if passing else 'FAIL'}]")
    return passing, pcc


@lru_cache(maxsize=1)
def _load_pcc_thresholds():
    with open(_PCC_THRESHOLDS_PATH) as f:
        return json.load(f)


def get_pcc_threshold(request, default=0.99):
    """Per-test threshold from pcc_thresholds.json (keyed by function name). Unlisted tests use default."""
    table = _load_pcc_thresholds()
    func = getattr(request.node, "originalname", None) or request.node.name.split("[")[0]
    return table.get(func, default)


def _resolve_mesh_shape(max_tp=4):
    return {"P150": (1, 1), "P150x4": (1, 4)}.get(
        os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), max_tp))
    )


def parametrize_mesh_tp(max_tp=4):
    """Parametrize TP test over env-selected mesh + FABRIC_1D (P150→(1,1), P150x4→(1,4))."""
    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)(
            fn
        )
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


def parametrize_batch(batches=(1, 8, 32)):
    """Parametrize decode test over batch sizes B (power-of-two <= 32)."""
    return pytest.mark.parametrize("B", [pytest.param(b, id=f"B{b}") for b in batches])


def parametrize_mesh_only(max_tp=4):
    """Parametrize over mesh shape only (no FABRIC_1D); for non-CCL tests."""
    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        return pytest.mark.parametrize(
            "mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True
        )(fn)

    return decorator


def tp_composer(mesh_device):
    """ConcatMeshToTensor for TP outputs (dim=3 multi-device, dim=0 single)."""
    nd = mesh_device.get_num_devices()
    return ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)


def replicate_to_device(mesh_device, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Replicate torch tensor to all mesh devices."""
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
