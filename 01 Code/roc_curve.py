#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:55:29 2020

@author: paulmora
"""

# %% Preliminaries

# Packages
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             plot_confusion_matrix,
                             recall_score,
                             precision_score,
                             auc, roc_curve)

# Paths
MAIN_PATH = r"/Users/paulmora/Documents/projects/roc_curve"
RAW_PATH = r"{}/00 Raw".format(MAIN_PATH)
CODE_PATH = r"{}/01 Code".format(MAIN_PATH)
DATA_PATH = r"{}/02 Data".format(MAIN_PATH)
OUTPUT_PATH = r"{}/03 Output".format(MAIN_PATH)

# Loading the data
total_labels = pd.read_csv(r"{}/train_labels.csv".format(RAW_PATH))
total_values = pd.read_csv(r"{}/train_values.csv".format(RAW_PATH))

# %% Mering data and dropping unwanted

total_values.info()
total_labels.loc[:, "damage_grade"].value_counts()

"""
We start by subsetting the data and only include damage grades which are
either 1 or 2.
"""

total_df = pd.concat([total_labels, total_values], axis=1)
not_3_bool = total_df.loc[:, "damage_grade"] != 3
subset_df = total_df.loc[not_3_bool, :]
subset_df.loc[:, "damage_grade"] = subset_df.loc[:, "damage_grade"] - 1
subset_df.rename(columns={"damage_grade": "high_damage"}, inplace=True)
subset_df.drop(columns=["building_id"], inplace=True)
subset_dummy_df = pd.get_dummies(subset_df)

plt.rcParams.update({"font.size": 20})
fig, axs = plt.subplots(figsize=(10, 10))
sns.countplot(subset_df.loc[:, "high_damage"], ax=axs)
axs.tick_params(axis="both", which="major")
axs.set_xlabel("High Grade")
axs.set_ylabel("Count")
path = (r"{}/int.png".format(OUTPUT_PATH))
fig.savefig(path, bbox_inches='tight')

# %% Model training

"""
We start by splitting the data into train and test and afterwards we try
out two different models.
"""

train, test = train_test_split(subset_dummy_df, test_size=0.3,
                               random_state=28, shuffle=False)
train.to_pickle("{}/train.pickle".format(DATA_PATH))
test.to_pickle("{}/test.pickle".format(DATA_PATH))

X_train = train.drop(columns="high_damage")
y_train = train.loc[:, "high_damage"]
X_test = test.drop(columns="high_damage")
y_test = test.loc[:, "high_damage"]

"""
After splitting the data, it is now time to train some models. For that we
use for once a normal logistic classifier model and as the other model
we use a GradientBoosting classifier model.
"""

logreg = LogisticRegression(max_iter=1000, random_state=28)
logreg.fit(X_train, y_train)

gbt = GradientBoostingClassifier(random_state=28)
gbt.fit(X_train, y_train)

sgdreg = SGDClassifier(fit_intercept=True, random_state=28)
sgdreg.fit(X_train, y_train)

# %% Classifier evaluation

"""
Now it is time to assess the performance of the two models, for that we
have quite different metrices.
"""

# Predictions
logreg_pred = logreg.predict(X_test)
gbt_pred = gbt.predict(X_test)
sgdreg_pred = sgdreg.predict(X_test)

# Confusion matrix
raw_conf_logreg = confusion_matrix(y_test, logreg_pred)
raw_conf_gbt = confusion_matrix(y_test, gbt_pred)
raw_conf_sgd = confusion_matrix(y_test, gbt_pred)

fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
axs = axs.ravel()
for i, (title, model) in enumerate(zip(["Logistic Model",
                                        "GradientBoosting Model",
                                        "Stochastic Gradient Descent"],
                                       [logreg, gbt, sgdreg])):
    plot_confusion_matrix(model, X_test, y_test, ax=axs[i])
    axs[i].set_title(title, fontsize=24)
    axs[i].set_ylabel("True Label")
    axs[i].set_xlabel("Predicted Label")
path = (r"{}/confusion_matrices.png".format(OUTPUT_PATH))
fig.savefig(path, bbox_inches='tight')

# Accuracy, Precision and Recall
eval_df = pd.DataFrame(columns=["Method", "Model", "Score"])

i = 0
for model, pred in zip(["Logistic Model",
                        "GradientBoosting Model",
                        "Stochastic Gradient Descent"],
                       [logreg_pred, gbt_pred, sgdreg_pred]):
    for meth, score in zip(["Accuracy", "Precision", "Recall"],
                           [accuracy_score, precision_score, recall_score]):

        eval_df.loc[i, "Method"] = meth
        eval_df.loc[i, "Model"] = model
        eval_df.loc[i, "Score"] = score(y_test, pred)
        i += 1

fig, axs = plt.subplots(figsize=(10, 10))
sns.barplot(x="Method", y="Score", hue="Model",
            data=eval_df, ax=axs)
axs.legend(loc="lower center")
path = (r"{}/acc_prec_rec.png".format(OUTPUT_PATH))
fig.savefig(path, bbox_inches='tight')

# TPR and FPR

"""
Now we take a look at how many false positive and true positive we got,
these can be taken from the confusion matrix
"""


def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i] == 1):
            TP += 1
        if (y_pred[i] == 1) and (y_true[i] != y_pred[i]):
            FP += 1
        if (y_true[i] == y_pred[i] == 0):
            TN += 1
        if (y_pred[i] == 0) and (y_true[i] != y_pred[i]):
            FN += 1

    return({"TP": TP, "FP": FP, "TN": TN, "FN": FN})


perf_measure(y_test, logreg_pred)
perf_measure(y_test, gbt_pred)

# %% ROC Curve

"""
We will now explore the ROC curve, we start by implementing the off-the-shelf
version from
"""

# ROC Curve
pred_proba_reg = logreg.predict_proba(X_test)
pred_proba_gbt = gbt.predict_proba(X_test)

fig, axs = plt.subplots(figsize=(10, 10))
for model, pred in zip(["Logistic Model",
                        "GradientBoosting Model"],
                       [pred_proba_reg,
                        pred_proba_gbt]):
    fpr, tpr, _ = roc_curve(y_test, pred[:, 1])
    auc_reg = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="{} AUC:{:.2f}".format(model, auc_reg))
plt.legend()
axs.set_xlabel("False Positive Rate")
axs.set_ylabel("True Positive Rate")
path = (r"{}/automatic_auc_roc_curve.png".format(OUTPUT_PATH))
fig.savefig(path, bbox_inches='tight')

"""
Now we build the ROC curve ourselves from scratch.
"""


def roc_manual(y_true, pred_prob):

    # Get the data ready
    pred_prob_series = pd.Series(pred_prob[:, 1])
    pred_prob_series.name = "pred"
    df = pd.DataFrame(pd.concat([y_true.reset_index(drop=True),
                                 pred_prob_series], axis=1))

    # Sorting probabilities in order to loop in an ascending manner
    sorted_df = df.sort_values(by="pred")

    # Calculate denominators
    true_num_pos = sum(y_true == 1)
    true_num_neg = sum(y_true == 0)

    # Create list container for results
    list_tpr, list_fpr = [], []

    for prob in tqdm(sorted_df.loc[:, "pred"]):

        # Create a boolean to mask only the values which qualify to be positive
        bool_classified_pos = sorted_df.loc[:, "pred"] > prob

        # Total number of positives and negative values
        tp = sum(sorted_df.loc[bool_classified_pos, "high_damage"] == 1)
        fp = sum(sorted_df.loc[bool_classified_pos, "high_damage"] == 0)

        # Calculate the TPR and FPR
        tpr = tp / true_num_pos
        fpr = fp / true_num_neg

        list_tpr.append(tpr)
        list_fpr.append(fpr)
    return list_tpr, list_fpr


gbt_list_tpr, gbt_list_fpr = roc_manual(y_test, pred_proba_gbt)
lgt_list_tpr, lgt_list_fpr = roc_manual(y_test, pred_proba_reg)

# Manual AUC and plotting
manual_auc_reg = abs(np.trapz(lgt_list_tpr, lgt_list_fpr))
manual_auc_gbt = abs(np.trapz(gbt_list_tpr, gbt_list_fpr))

fig, axs = plt.subplots(figsize=(10, 10))
plt.plot(lgt_list_fpr, lgt_list_tpr, color="orange",
         label="Logistic Regression AUC:{:.2f}".format(manual_auc_reg))
plt.plot(gbt_list_fpr, gbt_list_tpr, color="purple",
         label="GradientBoosting AUC:{:.2f}".format(manual_auc_gbt))
plt.legend()
axs.set_xlabel("False Positive Rate")
axs.set_ylabel("True Positive Rate")
path = (r"{}/manual_auc_roc_curve.png".format(OUTPUT_PATH))
fig.savefig(path, bbox_inches='tight')
