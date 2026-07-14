"""Download and load CIFAR-10 and MNIST for the ZeroGrad example scripts.

CIFAR-10 is downloaded from a fast S3 mirror in image-directory format (PNG
files per class).  MNIST is downloaded from the standard IDX gzip format.
Both are cached in ``~/.cache/zerograd/`` so subsequent runs skip the download.

Pillow (PIL) is required for CIFAR-10 image loading; MNIST uses only NumPy.

Functions:
    load_cifar10() -> (x_train, y_train, x_test, y_test)
    load_mnist()   -> (x_train, y_train, x_test, y_test)

Images are returned flattened and normalized to ``[0, 1]`` as ``float32``;
labels are ``int32``.
"""

from __future__ import annotations

import gzip
import os
import struct
import tarfile
import urllib.request

import numpy as np

CACHE_DIR = os.path.expanduser("~/.cache/zerograd")

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────

# The Toronto mirror (cs.toronto.edu) is frequently throttled.  This S3 mirror
# hosts the same dataset in fast.ai's image-directory format (PNGs per class).
CIFAR_URL = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz"
CIFAR_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10, downloading if necessary.

    Returns (x_train, y_train, x_test, y_test) where images are flattened
    to (N, 3072) float32 in [0, 1] and labels are (N,) int32.
    """
    from PIL import Image

    archive = os.path.join(CACHE_DIR, "cifar-10-python.tar.gz")
    extracted = os.path.join(CACHE_DIR, "cifar10")

    if not os.path.exists(extracted):
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(archive):
            print(f"Downloading CIFAR-10 (~168 MB) from S3 mirror ...")
            urllib.request.urlretrieve(CIFAR_URL, archive)
        print("Extracting CIFAR-10 ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(CACHE_DIR)

    def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for class_idx, label in enumerate(CIFAR_LABELS):
            class_dir = os.path.join(extracted, split, label)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if not filename.endswith(".png"):
                    continue
                img = Image.open(os.path.join(class_dir, filename))
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr.flatten())
                labels.append(class_idx)
        return np.array(images), np.array(labels, dtype=np.int32)

    x_train, y_train = _load_split("train")
    x_test, y_test = _load_split("test")
    return x_train, y_train, x_test, y_test


# ── MNIST ─────────────────────────────────────────────────────────────────────

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, downloading if necessary.

    Returns (x_train, y_train, x_test, y_test) where images are flattened
    to (N, 784) float32 in [0, 1] and labels are (N,) int32.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    def _download(name: str) -> str:
        path = os.path.join(CACHE_DIR, MNIST_FILES[name])
        if not os.path.exists(path):
            url = f"{MNIST_BASE}/{MNIST_FILES[name]}"
            print(f"Downloading {name} from {url} ...")
            urllib.request.urlretrieve(url, path)
        return path

    def _load_images(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            _magic, n, rows, cols = struct.unpack(">4i", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0

    def _load_labels(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            _magic, n = struct.unpack(">2i", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int32)

    x_train = _load_images(_download("train_images"))
    y_train = _load_labels(_download("train_labels"))
    x_test = _load_images(_download("test_images"))
    y_test = _load_labels(_download("test_labels"))
    return x_train, y_train, x_test, y_test
