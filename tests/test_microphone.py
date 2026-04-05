import numpy as np
import pyaudio
from openwakeword.model import Model

MODEL_PATH = "/Users/aruny/Workspace/wakeupramu/wakeword-studio/models/ok_nova/output/ok_nova.onnx"

CHUNK_SIZE = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def main():
    oww = Model(wakeword_models=[MODEL_PATH])
    model_names = list(oww.models.keys())
    print("Loaded models:", oww.models.keys())
    print("Model details:", oww.models)
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print(f"Listening for wake word... (model: {model_names})")

    try:
        while True:
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            print(f"Chunk size: {len(audio_np)}")  # Should be 1280
            predictions = oww.predict(audio_np)
            print(
                f"Audio: {np.abs(audio_np).mean():.4f} | Score: {float(predictions[model_names[0]]):.4f}",
                end="\r",
            )

            for name in model_names:
                score = float(predictions[name])
                if score > 0.05:
                    print(f"\nWake word detected! Score: {score:.3f}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
