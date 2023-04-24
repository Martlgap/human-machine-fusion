import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from tqdm import tqdm
from utils.calculate_score import calculate_score, threshold_flip
from utils.calculate_score import ConfidenceScoreGenerator
from sklearn.model_selection import KFold


MODELS = ["FaceTrans", "ArcFace", "ProdPoly"]


# Method to calculate Accuracy
def calculate_accuracy(threshold: float, dist: np.array, issame: np.array) -> list:
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))
    actual_tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    actual_fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return actual_tpr, actual_fpr, acc


# Get Scores for Machine Predictions
def get_scores(distances: np.ndarray, labels: list, p0k: float) -> list:
    k_fold = KFold(n_splits=10, shuffle=False)
    sigmoid_parameters, thresholds = ConfidenceScoreGenerator(bins=2000, p0k=p0k).foldwise(distances, labels, k_folds=10)
    # print(f"Best Thresholds: {thresholds} and Parameters: {sigmoid_parameters}")
    scores, predictions = [], []
    for fold_idx, (_, test_set) in enumerate(k_fold.split(np.arange(len(distances)))):
        scores_raw = [calculate_score(sigmoid_parameters[fold_idx], e) for e in distances[test_set]]
        scores += threshold_flip(distances[test_set], scores_raw, thresholds[fold_idx]).tolist()
        predictions += np.less(distances[test_set], thresholds[fold_idx]).tolist()
    return scores, predictions


# Load Labels
with open("../temp/labels.pkl", "rb") as f:
    labels = pickle.load(f)


# Load Distances from different Face Recognition Models and Calculate Predictions and Scores
distances_raw = {}
predictions = {}
scores = {}

for model in tqdm(MODELS):
    # Load Distances
    with open(f"../temp/distances_{model}.pkl", "rb") as f:
        distances_raw[model] = pickle.load(f)

    # Make Score and Prediction Dictionary for each dataset
    scores[model] = {elem.split(".tfrecord")[0].split("/")[-1]: None for elem in distances_raw[model].keys()}
    predictions[model] = {elem.split(".tfrecord")[0].split("/")[-1]: None for elem in distances_raw[model].keys()}

    # Get Scores and Predictions
    for key, distances in tqdm(distances_raw[model].items(), leave=False):
        scores[model][key.split(".tfrecord")[0].split("/")[-1]], predictions[model][key.split(".tfrecord")[0].split("/")[-1]] = get_scores(
            distances, labels, -18
        )


# Save Scores
with open("../temp/scores.pkl", "wb") as f:
    pickle.dump([scores[model] for model in MODELS], f)


# Save Predictions
with open("../temp/predictions.pkl", "wb") as f:
    pickle.dump([predictions[model] for model in MODELS], f)
