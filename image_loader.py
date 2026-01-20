import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

"""
Expected Directory Structure for Image-to-Image Translation:

ROOT_DIRECTORY/
    ├── train/
    │   ├── A/              # Source domain images (e.g., young faces)
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── B/              # Target domain images (e.g., old faces)
    │       ├── img1.jpg
    │       ├── img2.jpg
    │       └── ...
    └── test/
        ├── A/              # Test images from source domain
        │   ├── img1.jpg
        │   └── ...
        └── B/              # Test images from target domain
            ├── img1.jpg
            └── ...
"""


class ImageTranslationDataset(Dataset):
    """
    Custom PyTorch Dataset for unpaired image-to-image translation tasks.

    This dataset loads images from two separate domains (A and B) and can handle
    both aligned (paired) and unaligned (unpaired) training scenarios.

    Args:
        root (str): Root directory path containing train/test folders
        transforms_ (list): List of torchvision transforms to apply to images
        unaligned (bool): If True, randomly pairs images from domains A and B
                         If False, uses index-based pairing (default: False)
        mode (str): Dataset split to use - either 'train' or 'test' (default: 'train')
    """

    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        # Compose the list of transformations into a single pipeline
        self.transform = transforms.Compose(transforms_)

        # Flag to determine pairing strategy between domains
        # Unaligned: suitable for unpaired translation (CycleGAN, UNIT)
        # Aligned: requires corresponding images in both domains
        self.unaligned = unaligned

        # Construct paths to source domain (A) images
        path_A = os.path.join(root, "%s/A" % mode)
        self.files_A = sorted(glob.glob(path_A + "/*.*"))

        # Construct paths to target domain (B) images
        path_B = os.path.join(root, "%s/B" % mode)
        self.files_B = sorted(glob.glob(path_B + "/*.*"))

    def __getitem__(self, index):
        """
        Retrieve a pair of images from domains A and B.

        Args:
            index (int): Index of the item to retrieve

        Returns:
            dict: Dictionary containing transformed images from both domains
                  Format: {"A": tensor_A, "B": tensor_B}
        """
        # Load image from domain A using modulo to prevent index overflow
        # This ensures cycling through dataset when index exceeds dataset size
        img_path_A = self.files_A[index % len(self.files_A)]
        image_A = Image.open(img_path_A)

        # Determine how to select the corresponding image from domain B
        if self.unaligned:
            # Random pairing: select any image from domain B
            # This is crucial for unpaired image translation where
            # there's no direct correspondence between domains
            random_idx = random.randint(0, len(self.files_B) - 1)
            img_path_B = self.files_B[random_idx]
            image_B = Image.open(img_path_B)
        else:
            # Index-based pairing: use the same index (with modulo wrapping)
            # Useful when datasets have implicit pairing or similar sizes
            img_path_B = self.files_B[index % len(self.files_B)]
            image_B = Image.open(img_path_B)

        # Apply preprocessing transformations to both images
        # Typically includes: resize, normalize, convert to tensor, augmentation
        transformed_A = self.transform(image_A)
        transformed_B = self.transform(image_B)

        # Return as dictionary for easy access in training loop
        return {"A": transformed_A, "B": transformed_B}

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Uses the maximum length between domains to ensure all images
        are eventually used during training, even if domain sizes differ.

        Returns:
            int: Number of samples (maximum of domain A and B sizes)
        """
        return max(len(self.files_A), len(self.files_B))