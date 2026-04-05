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

from unittest import mock

import pytest

from openwakeword.torch_device import (
    empty_accelerator_cache,
    onnx_audio_features_device_str,
    resolve_onnx_providers,
    preferred_torch_device_str,
)


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda:0"),
        (True, False, "cuda:0"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_preferred_torch_device_str_priority(cuda, mps, expected):
    with mock.patch("torch.cuda.is_available", return_value=cuda):
        mps_backend = mock.Mock()
        mps_backend.is_available = mock.Mock(return_value=mps)
        with mock.patch("torch.backends.mps", mps_backend, create=True):
            assert preferred_torch_device_str() == expected


@pytest.mark.parametrize(
    "providers,expected",
    [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "gpu"),
        (["CoreMLExecutionProvider", "CPUExecutionProvider"], "gpu"),
        (["CPUExecutionProvider"], "cpu"),
    ],
)
def test_onnx_audio_features_device_str(providers, expected):
    assert onnx_audio_features_device_str(available_providers=providers) == expected


def test_resolve_onnx_providers_prefers_cuda():
    providers = resolve_onnx_providers(
        device="gpu",
        available_providers=["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_onnx_providers_coreml_on_mac():
    with mock.patch("sys.platform", "darwin"):
        providers = resolve_onnx_providers(
            device="gpu",
            available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_onnx_providers_coreml_not_used_off_mac():
    with mock.patch("sys.platform", "linux"):
        providers = resolve_onnx_providers(
            device="gpu",
            available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
    assert providers == ["CPUExecutionProvider"]


def test_empty_accelerator_cache_cpu_only_no_error():
    with mock.patch("torch.cuda.is_available", return_value=False):
        mps_backend = mock.Mock()
        mps_backend.is_available = mock.Mock(return_value=False)
        with mock.patch("torch.backends.mps", mps_backend, create=True):
            with mock.patch("torch.mps", None, create=True):
                empty_accelerator_cache()
