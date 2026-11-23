from datasets import load_dataset
from pathlib import Path

output_dir = Path("data/TinyStories")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:50000]",
)

output_file = output_dir / "train.txt"

with output_file.open("w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"].strip()

        if text:
            f.write(text)
            f.write("\n\n")

print(f"Dataset downloaded and saved to: {output_file}")
print(f"Total samples: {len(dataset):,}")
print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")