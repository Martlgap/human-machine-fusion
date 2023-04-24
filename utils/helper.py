import tensorflow as tf
import numpy as np
import os
import hashlib
import gdown
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlopen, HTTPError
from .data import load_verification_dataset


# Datasets
DATASETS = ["lfw.tfrecord", "calfw.tfrecord", "cplfw.tfrecord", "mlfw.tfrecord", "sllfw.tfrecord", "xqlfw.tfrecord", "base.zip"]

# Models
OCTUPLET_MODELS = ["ArcFaceOctupletLoss.tf.zip", "FaceTransformerOctupletLoss.pt"]

# Urls
BASE_URL = "https://github.com/Martlgap/human-machine-fusion/releases/download/v.1.0.0/"
OCTUPLET_URL = "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/"
PRODPOLY_URL = "https://drive.google.com/uc?id=1Y9250PK5aK5oxyrvWVqcQsVdfqjtFL53/TPAMI2020-PiNet.zip"

# File hashes
FILE_HASHES = {
    "lfw.tfrecord": "7f37bc11e7cd1d74f80a7a1f077b33cc246fedb699c5e71bebd568772ad47a48",
    "calfw.tfrecord": "f4f5ffc78a6156cfac75993ef2bdc35a44e7e60f61096c326627e8b8acb45e63",
    "cplfw.tfrecord": "25dd42dc7ee860e0147d83a9b57343f2a5937a3915b03d4554b59e70bcc002b3",
    "mlfw.tfrecord": "5b55052a36c163dcba790b4102b52b874d4049b777cb7ab652fa3f6bbac9f64c",
    "sllfw.tfrecord": "2acd1d63f5de1949f0a3b3b1c76a18d406dda8bea769f85bf45d1c8b29110330",
    "xqlfw.tfrecord": "5fa489c07a2056911ccd02e6abaa7a2ae27b20f8ab02445b3fe24124b33e6123",
    "base.zip": "b275bdce7794c2f3b553cd3b0422f91dfe13d1310cc6b4222bdcf689053e07d7",
    "ArcFaceOctupletLoss.tf.zip": "8603f374fd385081ce5ce80f5997e3363f4247c8bbad0b8de7fb26a80468eeea",
    "FaceTransformerOctupletLoss.pt": "f2c7cf1b074ecb17e546dc7043a835ad6944a56045c9675e8e1158817d662662",
    "TPAMI2020-PiNet.zip": "1048ef26510aa568f5ed047bfc5aafa1e428af2339cbc1a555d7dfe987261f87",
}


def download_models():
    print("Downloading models ...")
    for model in OCTUPLET_MODELS:
        get_file(OCTUPLET_URL + model, FILE_HASHES[model], folder="models")
    get_file(PRODPOLY_URL, FILE_HASHES["TPAMI2020-PiNet.zip"], folder="models", google_drive=True)
    print("Done!")
    return None


def download_datasets():
    print("Downloading datasets ...")
    for dataset in DATASETS:
        get_file(BASE_URL + dataset, FILE_HASHES[dataset], folder="datasets")
    print("Done!")
    return None


def extract_datasets():
    print("Extracting datasets ...")
    for dataset in DATASETS:
        extract("./datasets/" + dataset)
    print("Done!")
    return None


def extract(path) -> None:
    """Extracts the given tfrecord file to png images

    Args:
        path (_type_): Path to the tfrecord file
    """
    dataset = load_verification_dataset(path)
    dataset_path = path.split(".tfrecord", 1)[0]
    os.makedirs(dataset_path, exist_ok=True)
    for idx, (img1, img2, _) in tqdm(enumerate(dataset), total=6000):
        tf.io.write_file(os.path.join(dataset_path, f"{idx:04d}_0.png"), tf.image.encode_png(tf.cast(img1 * 255, tf.uint8)))
        tf.io.write_file(os.path.join(dataset_path, f"{idx:04d}_1.png"), tf.image.encode_png(tf.cast(img2 * 255, tf.uint8)))


def get_file(origin: str, file_hash: str, folder: str = "data", google_drive: bool = False) -> None:
    """Downloads a file from a URL if it not already in the cache. The file at indicated by origin is downloaded to the folder provided.

    Args:
        origin (str): URL of the file to download
        file_hash (str): Hash of the file to verify the data integrity
        folder (str, optional): Name of the Folder to store files. Defaults to "data".
        google_drive (bool, optional): Is the provided URL a GoogleDrive link? Defaults to False.

    Raises:
        Exception: If the download is not successful
    """
    tmp_file = os.path.join("./", folder, origin.split("/")[-1])
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    if not os.path.exists(tmp_file):
        download = True
    else:
        hasher = hashlib.sha256()
        with open(tmp_file, "rb") as file:
            for chunk in iter(lambda: file.read(65535), b""):
                hasher.update(chunk)
        if not hasher.hexdigest() == file_hash:
            print(
                "A local file was found, but it seems to be incomplete or outdated because the file hash does not "
                "match the original value of " + file_hash + " so data will be downloaded."
            )
            download = True
        else:
            download = False

    if download:
        if google_drive:
            gdown.download(origin.rsplit("/", 1)[0], tmp_file, quiet=False)
        else:
            try:
                response = urlopen(origin)
                with tqdm.wrapattr(
                    open(tmp_file, "wb"),
                    "write",
                    miniters=1,
                    desc="Downloading " + origin.split("/")[-1] + " to: " + tmp_file,
                    total=getattr(response, "length", None),
                ) as file:
                    for chunk in response:
                        file.write(chunk)
                    file.close()
            except HTTPError as e:
                raise Exception("Dataset fetching process failed. URL might be invalid or connection is lost.") from e

    if origin.endswith(".zip"):
        with ZipFile(tmp_file, "r") as zipObj:
            zipObj.extractall(os.path.dirname(tmp_file))


def get_hash(filepath: str) -> str:
    """Generates a sha256 hash of a file

    Args:
        filepath (str): Path to the file

    Returns:
        str: Hash of the file
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(65535), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
