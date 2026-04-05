# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared PyTorch device selection (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import sys

import torch


def preferred_torch_device_str() -> str:
    """Best available accelerator string for PyTorch: cuda:0, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def preferred_torch_device() -> torch.device:
    return torch.device(preferred_torch_device_str())


def get_available_onnx_execution_providers() -> list[str]:
    """Return ONNX Runtime execution providers or empty when ORT is unavailable."""
    try:
        import onnxruntime as ort
    except Exception:
        return []
    return ort.get_available_providers()


def resolve_onnx_providers(device: str = "cpu", available_providers: list[str] | None = None) -> list[str]:
    """Select ONNX Runtime providers in priority order for this host."""
    if available_providers is None:
        available_providers = get_available_onnx_execution_providers()

    if device != "gpu":
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available_providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def onnx_audio_features_device_str(available_providers: list[str] | None = None) -> str:
    """Return 'gpu' when CUDA or CoreML acceleration is available, else 'cpu'."""
    providers = resolve_onnx_providers(device="gpu", available_providers=available_providers)
    return "gpu" if providers[0] != "CPUExecutionProvider" else "cpu"


def empty_accelerator_cache() -> None:
    """Release cached memory on the active PyTorch accelerator, if any."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_mod = getattr(torch, "mps", None)
    if mps_mod is not None:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            empty = getattr(mps_mod, "empty_cache", None)
            if empty is not None:
                empty()
