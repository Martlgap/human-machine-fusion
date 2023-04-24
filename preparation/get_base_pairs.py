import warnings

warnings.filterwarnings("ignore")
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd


# Load Scores from Machines
with open("../temp/scores.pkl", "rb") as f:
    [scores_FaceTrans, scores_ArcFace, scores_ProdPoly] = pickle.load(f)
scores = {"FaceTrans": scores_FaceTrans, "ArcFace": scores_ArcFace, "ProdPoly": scores_ProdPoly}
df_scores_raw = pd.DataFrame(
    {(outerKey, innerKey): values for outerKey, innerDict in scores.items() for innerKey, values in innerDict.items()}
)


# Load Predictions from Machines
with open("../temp/predictions.pkl", "rb") as f:
    [predictions_FaceTrans, predictions_ArcFace, predictions_ProdPoly] = pickle.load(f)
predictions = {"FaceTrans": predictions_FaceTrans, "ArcFace": predictions_ArcFace, "ProdPoly": predictions_ProdPoly}
df_predictions = pd.DataFrame(
    {(outerKey, innerKey): values for outerKey, innerDict in predictions.items() for innerKey, values in innerDict.items()}
)


# Drop not used Datasets
NOT_USE = ["lfw", "sllfw"]
df_scores_raw = df_scores_raw.drop(columns=NOT_USE, level=1)
df_predictions = df_predictions.drop(columns=NOT_USE, level=1)


# Load Labels
with open("../temp/labels.pkl", "rb") as f:
    labels = pickle.load(f)


# Calculate Mean
df_scores = pd.concat({"MEAN": df_scores_raw.groupby(level=1, axis=1).mean()}, axis=1)


# Sort
df_sorted = df_scores.apply(lambda x: np.argsort(x))


# Make Binning and get thresholds for bins as indizes
NUM_BINS = 5
df_scorebins = df_scores.apply(lambda x: pd.cut(x, bins=np.linspace(0.4999, 1.0, NUM_BINS + 1)).value_counts()).cumsum()


# Get randomly 2 samples for each bin and each dataset. Also make sure that labels are equally distributed in the bins
base_pairs = {}
for dataset in tqdm(df_sorted["MEAN"].columns):
    if dataset in ["lfw", "sllfw"]:
        continue  # We do not use these datasets for the base pairs
    base_pairs[dataset] = []
    from_i = 0
    for idx, bin in enumerate(df_scorebins["MEAN"][dataset].index):
        to_i = df_scorebins["MEAN"][dataset][bin]
        if to_i - from_i <= 1:
            continue

        # Sample 1
        while True:
            random_idx1 = np.random.randint(from_i, to_i - 1)
            random_pair_id = df_sorted["MEAN"][dataset][random_idx1]
            if labels[random_pair_id]:
                base_pairs[dataset].append(random_pair_id)
                break

        # Sample 2
        while True:
            random_idx2 = np.random.randint(from_i, to_i - 1)
            random_pair_id = df_sorted["MEAN"][dataset][random_idx2]
            if not labels[random_pair_id] and not random_idx2 == random_idx1:
                base_pairs[dataset].append(random_pair_id)
                break

        # Set next bin border
        from_i = df_scorebins["MEAN"][dataset][bin]


# Write base pairs to file
with open("../temp/base_pairs.pkl", "wb") as f:
    pickle.dump(base_pairs, f)

"""
# Copy selected pairs to new folder
for dataset in base_pairs.keys():
    for pair_id in base_pairs[dataset]:
        folder = "/mnt/ssd2/datasets/base"
        file1 = os.path.join("/mnt/ssd2/datasets/", dataset, f"{pair_id:04d}_0.png")
        file2 = os.path.join("/mnt/ssd2/datasets/", dataset, f"{pair_id:04d}_1.png")
        os.system(f"cp {file1} {folder}")
        os.system(f"cp {file2} {folder}")
"""

# Get list of scores and labels
data = {"scores": [], "pair_ids": []}
for dataset in base_pairs.keys():
    data["scores"] += list(df_scores["MEAN"][dataset].values[base_pairs[dataset]])
    data["pair_ids"] += list(base_pairs[dataset])


# Sort by scores
sort_idx = np.argsort(data["scores"])[::-1]
for key in data.keys():
    data[key] = np.array(data[key])[sort_idx]


# Write to file
NUM_USERS = 60
with open("../survey_distribution/base.txt", "w") as f:
    for uid in range(1, NUM_USERS + 1):
        for idx in range(len(sort_idx)):
            f.write(f'{uid}\t{data["pair_ids"][idx]}\n')
