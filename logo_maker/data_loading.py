from pathlib import Path
from typing import Any

from PIL import Image
from torch import arange, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

pytorch_transforms = Any

FORWARD_TRANSFORMS = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

BACKWARD_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage()
])


def tensor_to_image(tensor: Tensor) -> Image.Image:
    return BACKWARD_TRANSFORMS(tensor)


class ImgFolderDataset(Dataset):
    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir
        self.transform = FORWARD_TRANSFORMS
        dir_content = img_dir.glob('**/*')
        self.img_file_locations = [file for file in dir_content if file.is_file() and file.suffix == ".png"]

    def __len__(self) -> int:
        return len(self.img_file_locations)

    def __getitem__(self, idx) -> Tensor:
        return self.transform(Image.open(self.img_file_locations[idx]).convert("L"))



# def loader_from_folder(img_dir: Path, n_samples: int | None = None, **kwargs):
#     dataset = ImgFolderDataset(img_dir)
#     if n_samples is not None:
#         if n_samples < len(dataset):
#             dataset = Subset(dataset, arange(n_samples))
#         else:
#             raise ValueError("Cannot draw {n_samples} samples from dataset with length {len(dataset)}.")
#     return DataLoader(dataset, **kwargs)
