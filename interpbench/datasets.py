"""
datasets.py

Custom dataset loaders for datasets not directly supported by torchvision.

Supports
--------
- TinyImageNet-200
- CUB-200-2011
"""

from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset


# ============================================================
# Tiny ImageNet-200
# ============================================================

class TinyImageNet200(Dataset):
    """
    Tiny ImageNet-200 loader.

    Expected structure

    data/
        tiny-imagenet-200/
            wnids.txt
            train/
                n01443537/
                    images/
                        *.JPEG
            val/
                images/
                val_annotations.txt
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.train = train

        wnids = (self.root / "wnids.txt").read_text().splitlines()
        self.class_to_idx = {
            c: i
            for i, c in enumerate(wnids)
        }

        self.samples = []

        if train:

            train_root = self.root / "train"

            for cls in wnids:

                image_dir = train_root / cls / "images"

                for img in image_dir.glob("*.JPEG"):

                    self.samples.append(
                        (
                            img,
                            self.class_to_idx[cls],
                        )
                    )

        else:

            val_root = self.root / "val"

            annotations = {}

            with open(
                val_root / "val_annotations.txt"
            ) as f:

                for line in f:

                    tokens = line.strip().split("\t")

                    annotations[tokens[0]] = tokens[1]

            image_dir = val_root / "images"

            for img in image_dir.glob("*.JPEG"):

                cls = annotations[img.name]

                self.samples.append(
                    (
                        img,
                        self.class_to_idx[cls],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# CUB-200-2011
# ============================================================

class CUB200(Dataset):
    """
    CUB-200-2011 loader.

    Expected structure

    data/
        CUB_200_2011/
            images/
            classes.txt
            images.txt
            image_class_labels.txt
            train_test_split.txt
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform

        images_file = self.root / "images.txt"
        labels_file = self.root / "image_class_labels.txt"
        split_file = self.root / "train_test_split.txt"

        images = {}
        labels = {}
        splits = {}

        with open(images_file) as f:
            for line in f:
                idx, path = line.strip().split()
                images[int(idx)] = path

        with open(labels_file) as f:
            for line in f:
                idx, label = line.strip().split()
                labels[int(idx)] = int(label) - 1

        with open(split_file) as f:
            for line in f:
                idx, is_train = line.strip().split()
                splits[int(idx)] = bool(int(is_train))

        self.samples = []

        for idx in sorted(images.keys()):

            if splits[idx] != train:
                continue

            path = self.root / "images" / images[idx]

            self.samples.append(
                (
                    path,
                    labels[idx],
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label