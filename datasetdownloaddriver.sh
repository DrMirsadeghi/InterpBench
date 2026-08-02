#!/usr/bin/env bash

set -e

DATA_DIR="data"

mkdir -p "${DATA_DIR}"

###############################################
# TinyImageNet-200
###############################################

if [ ! -d "${DATA_DIR}/tiny-imagenet-200" ]; then

    echo "Downloading TinyImageNet-200..."

    wget \
        -O "${DATA_DIR}/tiny-imagenet-200.zip" \
        http://cs231n.stanford.edu/tiny-imagenet-200.zip

    unzip -q \
        "${DATA_DIR}/tiny-imagenet-200.zip" \
        -d "${DATA_DIR}"

    rm "${DATA_DIR}/tiny-imagenet-200.zip"

    echo "TinyImageNet-200 installed."

else

    echo "TinyImageNet-200 already exists."

fi

###############################################
# CUB-200-2011
###############################################

if [ ! -d "${DATA_DIR}/CUB_200_2011" ]; then

    echo "Downloading CUB-200-2011..."

    wget \
        -O "${DATA_DIR}/CUB_200_2011.tgz" \
        https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

    tar -xzf \
        "${DATA_DIR}/CUB_200_2011.tgz" \
        -C "${DATA_DIR}"

    rm "${DATA_DIR}/CUB_200_2011.tgz"

    wget \
        -O "${DATA_DIR}/CUB_200_2011/images.txt" \
        https://huggingface.co/datasets/pawlo2013/CUB/resolve/main/CUB_200_2011/images.txt

    echo "CUB-200-2011 installed."

else

    echo "CUB-200-2011 already exists."

fi

###############################################
# Verify
###############################################

echo
echo "Installed datasets:"

[ -d "${DATA_DIR}/tiny-imagenet-200" ] && \
    echo "  ✓ TinyImageNet-200"

[ -d "${DATA_DIR}/CUB_200_2011" ] && \
    echo "  ✓ CUB-200-2011"

echo
echo "Done."