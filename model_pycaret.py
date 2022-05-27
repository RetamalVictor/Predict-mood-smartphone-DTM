import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pycaret.regression import (
    setup,
    compare_models,
    pull,
    finalize_model,
    save_model,
    load_model,
    predict_model,
)

res_imp_feat = pd.read_csv("final_dataset.csv", engine="python")

new_user = res_imp_feat[res_imp_feat["id"] == "AS14.33"]
res_imp_feat = res_imp_feat[res_imp_feat["id"] != "AS14.33"]

# create empty dataframes for train_test and val
train_test = pd.DataFrame()
val = pd.DataFrame()
grouped = res_imp_feat.groupby("id")
for g in grouped.groups:
    group = grouped.get_group(g)
    group = group.sort_values(by=["id", "Date"])
    train_test_set, val_set = train_test_split(
        group, test_size=0.15, shuffle=False, random_state=42
    )
    train_test_set["id"] = g
    val_set["id"] = g
    train_test = pd.concat([train_test, train_test_set], ignore_index=True)
    val = pd.concat([val, val_set], ignore_index=True)

res_mod = train_test.sort_values(by=["id", "Date"])
all_ts = res_mod["id"].unique()

all_results = []
final_model = {}

for i in tqdm(all_ts):

    df_res = res_mod[res_mod["id"] == i]
    # initialize setup from pycaret.regression
    s = setup(
        df_res,
        target="mood",
        train_size=0.8,
        data_split_shuffle=False,
        fold_strategy="timeseries",
        fold=3,
        ignore_features=["id", "Date"],
        # log_experiment = True,
        experiment_name="DM",
        silent=True,
        verbose=False,
        session_id=123,
    )

    # compare all models and select best one based on MAE
    best_model = compare_models(sort="MAE", verbose=False)

    # capture the compare result grid and store best model in list
    p = pull().iloc[0:1]
    p["id"] = str(i)
    all_results.append(p)

    # finalize model i.e. fit on entire data including test set
    f = finalize_model(best_model)

    # attach final model to a dictionary
    final_model[i] = f

    # save transformation pipeline and model as pickle file
    save_model(f, model_name="trained_models" + str(i), verbose=False)

# Performance on Validation Set
all_score_df = []
for i in tqdm(val["id"].unique()):
    l = load_model("trained_models" + str(i), verbose=False)
    p = predict_model(l, data=val)
    p["id"] = i
    all_score_df.append(p)
concat_val = pd.concat(all_score_df, axis=0)
concat_val = concat_val[["id", "Date", "Label"]]
concat_val["Label"] = concat_val["Label"].astype(int)
val["mood"] = val["mood"].astype(int)
final_val_df = pd.merge(
    val, concat_val, how="left", left_on=["id", "Date"], right_on=["id", "Date"]
)
final_val_df = final_val_df.groupby("id").nth(0).reset_index()

# Global Mean Absolute Error
mean_absolute_error(final_val_df["mood"], final_val_df["Label"])

# View for all IDs
print(final_val_df.head(33))
