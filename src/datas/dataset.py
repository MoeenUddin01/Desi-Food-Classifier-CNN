# from torchvision import datasets
# #from transforms import train_transform, test_transform
# from .transforms import train_transform, test_transform

# train_dataset = datasets.ImageFolder(
#     root='data/processed/train',
#     transform=train_transform
# )

# test_dataset = datasets.ImageFolder(
#     root='data/processed/test',
#     transform=test_transform
# )

from torchvision import datasets
# Use absolute import to avoid relative import errors
from src.datas.transforms import train_transform, test_transform

# Use Path to ensure cross-platform paths
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

train_dataset = datasets.ImageFolder(
    root=DATA_DIR / "train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=DATA_DIR / "test",
    transform=test_transform
)
