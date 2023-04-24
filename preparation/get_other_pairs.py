import warnings

warnings.filterwarnings("ignore")
import pickle
import os
import pandas as pd
import numpy as np
import random
import csv
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


# Calculate MIN of all models scores
df_scores = df_scores_raw.groupby(axis=1, level=1).min()


# Make Binning and get thresholds for bins as indizes
NUM_BINS = 10
df_scorebins_thresh = df_scores.apply(lambda x: pd.cut(x, bins=np.linspace(0.4999, 1.0, NUM_BINS + 1)).value_counts()).cumsum()


# Sort data according to scores
df_sorted = df_scores.apply(lambda x: np.argsort(x))


# Experiment 1
# Get randomly samples for each bin
PER_BIN = 30
exp1_sets = {}
for dataset in tqdm(df_sorted.columns):
    exp1_pairs = []
    from_i = 0
    for idx, bin in enumerate(df_scorebins_thresh[dataset].index):
        to_i = df_scorebins_thresh[dataset][bin]
        nrof_elements = 0
        while True:
            random_idx = np.random.randint(from_i, to_i - 1)
            random_pair_id = df_sorted[dataset][random_idx]
            if random_pair_id not in exp1_pairs:
                exp1_pairs.append(random_pair_id)
                nrof_elements += 1
            if nrof_elements == PER_BIN:
                break
        # Set next bin border
        from_i = df_scorebins_thresh[dataset][bin]
    exp1_sets[dataset] = set(exp1_pairs)


# Experiment 2
# Take all scores to 0.75 and randomly sample the lists
exp2_sets = {}
for dataset in tqdm(df_sorted.columns):
    exp2_pairs = []
    from_i = 0
    for idx, bin in enumerate(df_scorebins_thresh[dataset].index):
        to_i = df_scorebins_thresh[dataset][bin]
        random_idxs = np.random.choice(np.arange(from_i, to_i), size=to_i - from_i, replace=False)
        exp2_pairs += df_sorted[dataset][random_idxs].values.tolist()
        # Set next bin border
        from_i = df_scorebins_thresh[dataset][bin]
        if idx == 4:
            break
    exp2_sets[dataset] = set(exp2_pairs)


# Check nrof elements
numels = []
for elem in exp1_sets.values():
    numels.append(len(elem))
print(np.sum(numels))


numels = []
for elem in exp2_sets.values():
    numels.append(len(elem))
print(np.sum(numels))


# Read base pairs
with open("../temp/base_pairs.pkl", "rb") as f:
    base_pairs = pickle.load(f)


# Join all sets
final_sets = {}
for dataset, item in exp1_sets.items():
    item.update(exp2_sets[dataset])
    item.difference_update(set(base_pairs[dataset]))
    itemlist = list(item.copy())
    random.shuffle(itemlist)
    final_sets[dataset] = itemlist


# Write lists with scores and info for each user to file
NUM_USERS = 60
for dataset in final_sets.keys():
    with open(f"../survey_distribution/{dataset}.txt", "w") as f:
        batchsize = -(-len(final_sets[dataset]) // NUM_USERS)
        uid = 1
        for i in range(0, len(final_sets[dataset]), batchsize):
            for idx in final_sets[dataset][i : i + batchsize]:
                f.write(f"{uid}\t{idx}\n")
            uid += 1


# Find out how Many Questions per dataset per person needed
numels = []
for key, val in final_sets.items():
    numels.append(len(val))
    print(f"{key}:{len(val)}")
    print(f"{key} (per Person):{len(val)/60}")
print(f"Total:{np.sum(numels)/60}")
