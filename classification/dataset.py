import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import Tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_channel_padding_collate(
    batch_data: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    labels = []
    images = []
    for img, label in batch_data:
        labels.append(label)
        if img.shape[0] == 4:
            images.append(img)
        elif img.shape[0] == 1:
            images.append(img.expand([4, -1, -1]))
        elif img.shape[0] == 3:
            images.append(img[:1].expand([4, -1, -1]))
        else:
            raise ValueError("Input tensor has wrong number of channels.")

    return torch.stack(images), torch.stack(labels)


# Built-in Dataset class should be inherited from your custom datasets.
class PokemonDataset(Dataset):
    def __init__(self, path_to_data: str) -> None:
        # Initialize the inherited class with super().
        super().__init__()

        self.path_to_data = path_to_data
        self.images_list = os.listdir(os.path.join(self.path_to_data, "images"))
        self.metadata = pd.read_csv(os.path.join(self.path_to_data, "pokemon.csv"))
        # Create label encoding for the text labels to make them machine-understandable.
        self.metadata["label_code"] = pd.Categorical(self.metadata["Type1"]).codes

    def __getitem__(self, idx: Tensor) -> Tuple[Tensor, Tensor]:
        """Compulsory for iteration."""
        # Get the next image file name from the list.
        image_name = self.images_list[idx]
        # Create the path to image.
        path_to_image = os.path.join(self.path_to_data, "images", image_name)
        # Read image to a torch tensor. Since we are creating a new tensor here, move it to our preferred device.
        image = read_image(path_to_image).to(device)
        # Apply transformations.
        transformed_image = self.augmentation(self.preprocessing(image))
        # Pokemon name is the file name but without the extension .png, .jpg etc.
        pokemon_name = image_name.split(".")[0]
        # Find the pokemon from the csv and get its label code.
        label = self.metadata[self.metadata["Name"] == pokemon_name][
            "label_code"
        ].to_numpy()
        # Return. Since we are creating a tensor here (torch.from_numpy), we should move it to our preferred device.
        return transformed_image, torch.from_numpy(label).to(device).squeeze()

    def __len__(self) -> int:
        """Compulsory for iteration."""
        return len(self.images_list)

    # You can create other functions to simplify your code (__getitem__ function), such as get_label() or get_image()

    def preprocessing(self, img: Tensor) -> Tensor:
        # Optional
        return img

    def augmentation(self, img: Tensor) -> Tensor:
        # Optional
        return img
