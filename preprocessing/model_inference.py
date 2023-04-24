import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
import numpy as np
import pickle
from utils.data import DataSetLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from utils.models import ProdPoly, FaceTransformerOctupletLoss, ArcFaceOctupletLoss


DATASETS = [
    "../datasets/xqlfw.tfrecord",
    "../datasets/lfw.tfrecord",
    "../datasets/calfw.tfrecord",
    "../datasets/sllfw.tfrecord",
    "../datasets/cplfw.tfrecord",
    "../datasets/mlfw.tfrecord",
]

MODELS = ["ProdPoly", "FaceTrans", "ArcFace"]


for MODEL in tqdm(MODELS):
    distances = {elem: None for elem in DATASETS}

    # Initialize Model
    if MODEL == "ProdPoly":
        model = ProdPoly()
    elif MODEL == "FaceTrans":
        model = FaceTransformerOctupletLoss()
    elif MODEL == "ArcFace":
        model = ArcFaceOctupletLoss()
    else:
        raise ValueError(f"Unknown model {MODEL}.")

    for DATASET in tqdm(DATASETS):
        # Load Dataset
        dataset, labels = DataSetLoader(path=DATASET)()

        # Reshape Dataset
        dataset_flat = dataset.reshape(dataset.shape[0] * 2, *dataset.shape[2:])

        # Perform Batchwise Inference
        embeddings_flat = model(dataset_flat)

        # Reshape Embeddings
        embeddings = embeddings_flat.reshape(embeddings_flat.shape[0] // 2, 2, embeddings_flat.shape[-1])

        # Calculate Cosine Distances for all Pairs
        distances[DATASET] = paired_cosine_distances(embeddings[:, 0, :], embeddings[:, 1, :])

    # Destroy Model
    del model

    # Save Distances
    with open(f"../temp/distances_{MODEL}.pkl", "wb") as f:
        pickle.dump(distances, f)


# Generate Labels (300 genuine, 300 impostor, 300 genuine, 300 impostor, ...)
labels = [True] * 300 + [False] * 300
labels = labels * 10
labels = np.asarray(labels)

# Save Labels
with open("../temp/labels.pkl", "wb") as f:
    pickle.dump(labels, f)
