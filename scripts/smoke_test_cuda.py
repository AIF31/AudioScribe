from faster_whisper import WhisperModel


def main() -> None:
    model_name = "tiny"
    print("Loading faster-whisper model on CUDA...")
    WhisperModel(model_name, device="cuda", compute_type="float16")
    print("CUDA faster-whisper model loaded successfully:", model_name)


if __name__ == "__main__":
    main()
