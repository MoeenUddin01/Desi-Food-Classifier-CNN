import shutil
import random
from pathlib import Path

def split_dataset(
    source_dir="data/raw/dataset",
    target_dir="data/processed",
    train_ratio=0.8,
    seed=42,
    image_extensions=(".jpg", ".jpeg", ".png", ".webp")
):
  
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        # checking file extensions and collecting images
        images = []
        for img in class_dir.iterdir():
            if img.suffix.lower() in image_extensions:
               images.append(img)

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create class folders
        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (test_dir / class_dir.name).mkdir(exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(img, train_dir / class_dir.name / img.name)

        for img in test_images:
            shutil.copy(img, test_dir / class_dir.name / img.name)

        print(f"{class_dir.name}: {len(train_images)} train, {len(test_images)} test")

    print("Dataset split completed.")


if __name__ == "__main__":
    split_dataset(
        source_dir="data/raw/dataset",
        target_dir="data/processed",
        train_ratio=0.8,
        seed=42
    )