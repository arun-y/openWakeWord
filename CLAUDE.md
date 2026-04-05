# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenWakeWord is a wake word/phrase detection library built on frozen pre-trained speech embeddings. Custom wake word models are small DNNs/RNNs trained on top of these shared embeddings — enabling efficient multi-model inference.

## Commands

**Install for development (Linux requires `libspeexdsp-dev` first):**
```bash
pip install -e .[test]   # includes test dependencies
pip install -e .[full]   # includes training dependencies
```

**Run tests:**
```bash
pytest                          # full suite with coverage, flake8, mypy
pytest tests/test_model.py      # single test file
```

**Train a custom wake word model:**
```bash
# Edit examples/custom_model.yml, then:
python openwakeword/train.py --config examples/custom_model.yml
```

## Architecture

Three inference stages run in sequence per 80ms audio frame (1280 samples @ 16kHz):

```
Raw PCM Audio (16-bit, 16kHz)
  → [Speex noise suppression, optional]
  → AudioFeatures (openwakeword/utils.py)
       ├─ Melspectrogram ONNX model → 76×32 spectrogram
       └─ Google speech embedding model → 96-dim vector (shared, frozen)
  → Wake word classifier (per model, small DNN/RNN) → score [0-1]
  → [Custom verifier, optional] (logistic regression, speaker-specific)
  → [Silero VAD filter, optional]
  → Prediction output with debounce/patience filtering
```

**Key modules:**
- `openwakeword/model.py` — `Model` class: main inference API (`predict()`, `predict_clip()`, `reset()`)
- `openwakeword/utils.py` — `AudioFeatures`: melspectrogram + embedding extraction; supports ONNX and TFLite runtimes
- `openwakeword/train.py` — training pipeline: synthetic data → augmentation → DNN/RNN training → ONNX/TFLite export
- `openwakeword/data.py` — adversarial negative generation, audio augmentation, memory-mapped batch loading
- `openwakeword/vad.py` — Silero VAD wrapper
- `openwakeword/custom_verifier_model.py` — speaker-specific logistic regression verifier

**Model files:** Downloaded to `~/.cache/openwakeword/models/` (not bundled in PyPI). Format: `{name}_v{version}.{tflite|onnx}`.

**Runtimes:** Both ONNX (`onnxruntime`) and TFLite (`ai-edge-litert`) are supported. Select via `Model(inference_framework="tflite")`.

## Training Pipeline

Training uses 100% synthetic positive data generated via Piper TTS, augmented with room impulse responses (RIRs) and background noise. Negative data comes from pre-computed feature files (e.g., ACAV100M).

Key config parameters in `examples/custom_model.yml`:
- `target_phrase` — words/phrases to detect
- `n_samples` — synthetic positive samples (min 20,000, often 100,000+)
- `piper_sample_generator_path` — path to the [piper-sample-generator](https://github.com/dscripka/piper-sample-generator) fork
- `false_positive_validation_data_path` — `.npy` features for early stopping (dataset available on HuggingFace: `davidscripka/openwakeword_features`)
- `feature_data_files` — pre-computed negative feature `.npy` files
- `target_false_positives_per_hour` — controls auto training stop (default: 0.2)
- `layer_size` / `model_type` — DNN capacity (default: 32, `"dnn"`)

## Usage

```python
from openwakeword.model import Model

# Load specific models
oww = Model(wakeword_models=["alexa", "hey_mycroft"])

# Or load a custom trained model by path
oww = Model(wakeword_models=["path/to/my_model.tflite"])

# Predict on a chunk of audio (numpy array, 16-bit PCM)
predictions = oww.predict(audio_chunk)  # returns dict of {model_name: score}
```
