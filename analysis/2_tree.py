# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decision Tree

# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pingouin as pg

import matplotlib.pyplot as plt
import seaborn

import sklearn
import sklearn.impute
import sklearn.tree

import src.pandas

from njab.plotting.metrics import plot_split_prc

import njab.sklearn
from njab.sklearn import transform_DataFrame
from njab.sklearn.scoring import ConfusionMatrix

import config

# %% [markdown]
# # Set parameters

# %% tags=["parameters"]
# TARGET = 'hasLiverAdm90'
TARGET = 'liverDead090infl'
CLINIC = config.fname_pkl_prodoc_clinic
OLINK = config.fname_pkl_prodoc_olink
VAL_IDS: str = ''  #
use_val_split = True
FOLDER = ''

# %%
# VAL_IDS = '10129902,10146789,10146791,10146795,10146796,10146799,10146800,10146809,10146811,10146812,10146814,10146818,10146819,10146821,10146823,10146824,10146825,10146826,10146827,10146828,10146830,10146831,10146838,10146839,10146842,10146843,10146844,10146850,10146851'

# %% [markdown]
# set output folder

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %% [markdown]
# Load data

# %%
clinic = pd.read_pickle(CLINIC)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(OLINK)

# %% [markdown] tags=[]
# # Differences between groups defined by target

# %%
clinic

# %%
target_counts = clinic[TARGET].value_counts()

if target_counts.sum() < len(clinic):
    print(
        f"Target has missing values. Can only use {target_counts.sum()} of {len(clinic)} samples."
    )
    mask = clinic[TARGET].notna()
    clinic, olink = clinic.loc[mask], olink.loc[mask]

target_counts

# %%
pd.crosstab(clinic[TARGET], clinic["DecomensatedAtDiagnosis"])

# %%
y = clinic[TARGET].astype(int)  # NA is encoded as False for boolean type

# %% tags=[]
y

# %% [markdown]
# # Data Splits

# %%
olink_val, clinic_val = None, None
if use_val_split:
    if not VAL_IDS:
        logging.warning("Create train and test split.")
        _, VAL_IDS = sklearn.model_selection.train_test_split(
            clinic.index,
            test_size=0.2,
            random_state=123,
            stratify=clinic[TARGET])
        VAL_IDS = list(VAL_IDS)
    elif isinstance(VAL_IDS, str):
        VAL_IDS = VAL_IDS.split(",")
    else:
        raise ValueError("Provide IDs in csv format as str: 'ID1,ID2'")
VAL_IDS

# %%
if VAL_IDS:
    diff = pd.Index(VAL_IDS)
    VAL_IDS = clinic.index.intersection(VAL_IDS)
    if len(diff) < len(VAL_IDS):
        logging.warning("Some requested validation IDs are not in the data: "
                        ",".join(str(x) for x in diff.difference(VAL_IDS)))
    olink_val = olink.loc[VAL_IDS]
    olink = olink.drop(VAL_IDS)
    #
    clinic_val = clinic.loc[VAL_IDS]
    clinic = clinic.drop(VAL_IDS)
    use_val_split = True
    
    y_val = y.loc[VAL_IDS]
    y = y.drop(VAL_IDS)

# %%
pd.crosstab(clinic.DiagnosisPlace, clinic.dead)

# %% [markdown]
# FirstAdmission is also right-censored

# %%
time_from_diagnose_to_first_admission = clinic["DateFirstAdmission"].fillna(
    config.STUDY_ENDDATE) - clinic["DateDiagnose"]
time_from_diagnose_to_first_admission.describe()

# %% [markdown]
# Who dies without having a first Admission date?

# %%
dead_wo_adm = clinic["DateFirstAdmission"].isna() & clinic['dead']
idx_dead_wo_adm = dead_wo_adm.loc[dead_wo_adm].index
print('Dead without admission to hospital:',
      *dead_wo_adm.loc[dead_wo_adm].index)
clinic.loc[dead_wo_adm, ["DateFirstAdmission", "DateDiagnose"]]

# %% [markdown]
# ## All Features

# %%
clinic.dtypes.value_counts()

# %%
X = (olink.join(clinic[config.clinic_data.vars_cont]).join(
    clinic[config.clinic_data.comorbidities].astype('object').replace({
        'Yes': 1,
        'No': 0,
        'yes': 1,
        'no': 0
    })))

# %%
feat_w_missings = X.isna().sum()
feat_w_missings = feat_w_missings.loc[feat_w_missings > 0]
feat_w_missings

# %%
median_imputer = sklearn.impute.SimpleImputer(strategy='median')
X = transform_DataFrame(X, median_imputer.fit_transform)

# %% [markdown]
# ## DecisionTreeClassifier
#
# - Documentation for [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontree#sklearn.tree.DecisionTreeClassifier)
# - `gini`: weighted probab
# - `log_loss`: as in Logistic Regression (binary entropy)

# %%
clf = sklearn.tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_leaf=1,
    # min_samples_split=4,
    max_features=X.shape[-1])
clf = clf.fit(X, y)

# rerunning this shows differences in deeper nodes
fig, ax = plt.subplots()
nodes = sklearn.tree.plot_tree(clf,
                               feature_names=X.columns,
                               class_names=["False", "True", "none"],
                               filled=True,
                               ax=ax)
fig.suptitle(f"Decision tree for endpoint {TARGET}", fontsize='xx-large')
fig.tight_layout()
fig.savefig(FOLDER / '2_decision_tree.pdf')

# %%
pred_train = clf.predict(X)
ConfusionMatrix(y,  pred_train).as_dataframe

# %%
results_train = njab.sklearn.get_results_split(y_true=y, y_score=clf.predict_proba(X)[:,1])
fig, ax = plt.subplots(1, 1, figsize=None)
ax = plot_split_prc(results_train, 'Decision Tree', ax)

# %%
feat_used = clf.feature_importances_ > 0.0
feat_used = X.columns[feat_used]
feat_used

# %% [markdown]
# # Test split performance

# %%
if olink_val is not None and clinic_val is not None:
    X_val = (olink_val.join(clinic_val[config.clinic_data.vars_cont]).join(
        clinic_val[config.clinic_data.comorbidities].astype('object').replace({
            'Yes': 1,
            'No': 0,
            'yes': 1,
            'no': 0
        })))
    display(X_val)

# %%
if olink_val is not None and clinic_val is not None:
    feat_w_missings = X_val.isna().sum()
    feat_w_missings = feat_w_missings.loc[feat_w_missings > 0]
    display(feat_w_missings)

# %%
if olink_val is not None and clinic_val is not None:
    X_val = transform_DataFrame(X_val, median_imputer.transform)
    pred_val = clf.predict(X_val)
    pred_val = pd.Series(pred_val, index=X_val.index)
    score_val = pd.Series(clf.predict_proba(X_val)[:,1], index=X_val.index)
    display(ConfusionMatrix(y_val,  pred_val).as_dataframe)

# %%
if olink_val is not None and clinic_val is not None:
    out_val = pd.DataFrame({'true': y_val, 'pred': pred_val, 'score': score_val}).join(X_val[feat_used])
    display(out_val)

# %% [markdown]
# - [ ] olink data only
# - [ ] clinial data only
# - [ ] laboratory (biochemistry) only?
