from pathlib import Path
from typing import Any

from PIL import Image
from torch import arange, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

pytorch_transforms = Any

TRANSFORMS = transforms.Compose([
    transforms.ToTensor()
])
#
# TRANSFORMS = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     #transforms.GaussianBlur(5, sigma=(0.1, 0.5)),
#     transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0))
# ])

class ImgFolderDataset(Dataset):
    def __init__(self, img_dir: Path, transform: pytorch_transforms = TRANSFORMS) -> None:
        self.img_dir = img_dir
        self.transform = transform
        dir_content = img_dir.glob('**/*')
        self.img_file_locations = [file for file in dir_content if file.is_file() and file.suffix == ".png"]
        #self.tensors = [transforms.ToTensor()(Image.open(img_file)) for img_file in img_files]

    def __len__(self) -> int:
        return len(self.img_file_locations)

    def __getitem__(self, idx) -> Tensor:
        return self.transform(Image.open(self.img_file_locations[idx]))
        # if self.transform is not None:
        #     sample = self.transform(self.tensors[idx])
        # else:
        #     sample = self.tensors[idx]
        # return sample


# def loader_from_folder(img_dir: Path, n_samples: int | None = None, **kwargs):
#     dataset = ImgFolderDataset(img_dir)
#     if n_samples is not None:
#         if n_samples < len(dataset):
#             dataset = Subset(dataset, arange(n_samples))
#         else:
#             raise ValueError("Cannot draw {n_samples} samples from dataset with length {len(dataset)}.")
#     return DataLoader(dataset, **kwargs)
