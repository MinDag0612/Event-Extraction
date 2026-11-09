from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-3B"
MODEL_DIR = Path(__file__).resolve().parent


def main():
    print(f"Downloading: {MODEL_ID}")
    print(f"Destination: {MODEL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "model.safetensors.index.json"
        ],
    )

    print("\nModel downloaded successfully.")


if __name__ == "__main__":
    main()

