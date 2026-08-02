"""
example.py

Example training script for InterpBench.

Supported Models
----------------
- ResNet-18
- VGG-16
- WideResNet-28-10
- ViT-B/16

Supported Datasets
------------------
- SVHN
- CIFAR10
- CIFAR100
- TinyImageNet200
- CUB200

Simply change MODEL_NAME, DATASET and METHOD below.
"""

import mair
import timm
import torchvision.datasets as tvdatasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from interpbench.datasets import (
    TinyImageNet200,
    CUB200,
)

from interpbench.trainers import (
    StandardTrainer,
    PGDTrainer,
)

import sys

array_id = int(sys.argv[1])
print(f"Running array task {array_id}")

RUN=array_id
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Available models
#
#   "ResNet-18"
#   "VGG-16"
#   "WideResNet-28-10"
#   "ViT-B/16"
#
MODEL_NAME = "ResNet-18"

# Available datasets
#
#   "SVHN"
#   "CIFAR10"
#   "CIFAR100"
#   "TinyImageNet200"
#   "CUB200"
#
DATASET = "CIFAR10"

# Available training methods
#
#   "STANDARD"
#   "PGD"
#
METHOD = "STANDARD"

BATCH_SIZE = 32
EPOCHS = 100

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

MODELS = {

    "ResNet-18": {
        "mair_name": "ResNet18",
    },

    "VGG-16": {
        "mair_name": "VGG16",
    },

    "WideResNet-28-10": {
        "mair_name": "WRN28-10",
    },
    "ViT-B/16": {
        "timm_name": "vit_base_patch16_224",
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

if MODEL_NAME not in MODELS:
    raise ValueError(
        f"Unknown model '{MODEL_NAME}'. "
        f"Available models: {list(MODELS.keys())}"
    )

# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

DATASETS = {

    "SVHN": {
        "classes": 10,
        "size": 32,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },

    "CIFAR10": {
        "classes": 10,
        "size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },

    "CIFAR100": {
        "classes": 100,
        "size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },

    "TinyImageNet200": {
        "classes": 200,
        "size": 64,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
    },

    "CUB200": {
        "classes": 200,
        "size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },

}

if DATASET not in DATASETS:
    raise ValueError(
        f"Unknown dataset '{DATASET}'. "
        f"Available datasets: {list(DATASETS.keys())}"
    )

cfg = DATASETS[DATASET]
N_CLASSES = cfg["classes"]

# ------------------------------------------------------------
# Image transform
# ------------------------------------------------------------

if MODEL_NAME == "ViT-B/16":

    input_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

else:

    input_size = cfg["size"]
    mean = cfg["mean"]
    std = cfg["std"]

transform = transforms.Compose([

    transforms.Resize(
        (input_size, input_size)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean,
        std,
    ),

])

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

if DATASET == "SVHN":

    trainset = tvdatasets.SVHN(
        root="./data",
        split="train",
        download=True,
        transform=transform,
    )

    testset = tvdatasets.SVHN(
        root="./data",
        split="test",
        download=True,
        transform=transform,
    )

elif DATASET == "CIFAR10":

    trainset = tvdatasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = tvdatasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

elif DATASET == "CIFAR100":

    trainset = tvdatasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = tvdatasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

elif DATASET == "TinyImageNet200":

    trainset = TinyImageNet200(
        root="./data/tiny-imagenet-200",
        train=True,
        transform=transform,
    )

    testset = TinyImageNet200(
        root="./data/tiny-imagenet-200",
        train=False,
        transform=transform,
    )

elif DATASET == "CUB200":

    trainset = CUB200(
        root="./data/CUB_200_2011",
        train=True,
        transform=transform,
    )

    testset = CUB200(
        root="./data/CUB_200_2011",
        train=False,
        transform=transform,
    )

else:

    raise RuntimeError("Unsupported dataset.")

# ------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model_cfg = MODELS[MODEL_NAME]

if "mair_name" in model_cfg:
    backend_name = model_cfg["mair_name"]
    model = mair.utils.load_model(
        model_name=backend_name,
        n_classes=N_CLASSES,
    )

else:
    # Find any "<backend>_name" entry (e.g. timm_name)
    backend_key = next(
        (k for k in model_cfg if k.endswith("_name")),
        None,
    )

    if backend_key is None:
        raise ValueError(
            f"No model backend specified for '{MODEL_NAME}'. "
            "Expected a key like 'mair_name' or 'timm_name'."
        )

    backend = backend_key[:-5]  # remove "_name"
    backend_name = model_cfg[backend_key]

    if backend == "timm":
        #ImageNet pretrained ViT-B/16 is the standard transfer-learning setup.
        model = timm.create_model(
            backend_name,
            pretrained=True,
            num_classes=N_CLASSES,
        )
        print("created {} model".format(backend))
    else:
        raise ValueError(
            f"Unsupported model backend '{backend}'."
        )

model = model.cuda()

rmodel = mair.RobModel(
    model,
    n_classes=N_CLASSES,
).cuda()
# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
'''
model = mair.utils.load_model(
    model_name=MODELS[MODEL_NAME]["mair_name"],
    n_classes=N_CLASSES,
).cuda()

rmodel = mair.RobModel(
    model,
    n_classes=N_CLASSES,
).cuda()
'''
# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

if METHOD == "STANDARD":

    trainer = StandardTrainer(
        rmodel,
        trainloader,
        testloader,
    )

elif METHOD == "PGD":

    trainer = PGDTrainer(
        rmodel,
        trainloader,
        testloader,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
    )

else:

    raise ValueError(
        f"Unknown method '{METHOD}'. "
        "Available methods: ['STANDARD', 'PGD']"
    )

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

report = trainer.fit(
    epochs=EPOCHS,
    benchmark_every=10,
    model=MODEL_NAME,
    dataset=DATASET,
    trainer=METHOD,
    run=RUN,
)

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

model_tag = (
    MODELS[MODEL_NAME].get(
        "mair_name",
        MODELS[MODEL_NAME].get("timm_name")
    )
)

filename = (
    f"{model_tag}_"
    f"{DATASET}_"
    f"{METHOD}_"
    f"{RUN}.report"
)

report.save(filename)

print(report)
print(f"\nSaved report to {filename}")
