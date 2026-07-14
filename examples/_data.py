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
import socket
import struct
import tarfile
import time
import urllib.error
import urllib.request

import numpy as np

CACHE_DIR = os.path.expanduser("~/.cache/zerograd")


# ── Download helper: progress, timeout, retry (see issue #35) ────────────────

def _download(url: str, dest: str, description: str, *, retries: int = 3, timeout: float = 60.0) -> str:
    """Download ``url`` to ``dest`` with a progress bar, socket timeout, and retry.

    A transient network failure (URLError, socket timeout, connection error)
    is retried with exponential back-off. A partial file from a failed attempt
    is removed before retrying.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _download_once(url, dest, description, timeout)
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
            last_err = exc
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                print(f"  {description}: attempt {attempt}/{retries} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(
        f"failed to download {description} after {retries} attempts: {last_err}"
    )


def _download_once(url: str, dest: str, description: str, timeout: float) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        total_str = f"{total / (1024 * 1024):.1f}MB" if total else "?MB"
        downloaded = 0
        chunk = 64 * 1024
        t0 = time.time()
        last_print = 0.0
        with open(dest, "wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                now = time.time()
                # Throttle the progress line to ~2 updates per second.
                if now - last_print >= 0.5 or downloaded == total:
                    elapsed = now - t0
                    rate = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                    done = f"{downloaded / (1024 * 1024):.1f}MB"
                    print(f"\r  {description}: {done}/{total_str} ({rate:.1f}MB/s)", end="", flush=True)
                    last_print = now
        print()  # newline after the progress line
    return dest


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────

# The Toronto mirror (cs.toronto.edu) is frequently throttled.  This S3 mirror
# hosts the same dataset in fast.ai's image-directory format (PNGs per class).
CIFAR_URL = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz"
# The cached archive is named after the source format (a tarball of PNG image
# directories), not the Toronto pickle format (see issue #32).
CIFAR_ARCHIVE = "cifar10.tgz"
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

    archive = os.path.join(CACHE_DIR, CIFAR_ARCHIVE)
    extracted = os.path.join(CACHE_DIR, "cifar10")

    if not os.path.exists(extracted):
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(archive):
            print(f"Downloading CIFAR-10 (~168 MB) from S3 mirror ...")
            _download(CIFAR_URL, archive, "CIFAR-10")
        print("Extracting CIFAR-10 ...")
        with tarfile.open(archive, "r:gz") as tar:
            # Use the "data" filter to avoid path traversal and silence the
            # Python 3.12 deprecation warning (see issue #29).
            tar.extractall(CACHE_DIR, filter="data")

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

    def _download_file(name: str) -> str:
        path = os.path.join(CACHE_DIR, MNIST_FILES[name])
        if not os.path.exists(path):
            url = f"{MNIST_BASE}/{MNIST_FILES[name]}"
            _download(url, path, f"MNIST {name}")
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

    x_train = _load_images(_download_file("train_images"))
    y_train = _load_labels(_download_file("train_labels"))
    x_test = _load_images(_download_file("test_images"))
    y_test = _load_labels(_download_file("test_labels"))
    return x_train, y_train, x_test, y_test
