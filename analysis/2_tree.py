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
from pathlib import Path

import numpy as np
import pandas as pd

import pingouin as pg

import matplotlib.pyplot as plt
import seaborn

import sklearn
import sklearn.impute
import sklearn.tree

import src.stats
from src.sklearn import transform_DataFrame
from src.plotting.metrics import plot_split_prc

import config

# %% [markdown]
# # Set parameters

# %% tags=["parameters"]
TARGET = 'dead090infl'
TARGET = 'hasLiverAdm090infl'
FOLDER = ''

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET

# %%
clinic = pd.read_pickle(config.fname_pkl_clinic)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(config.fname_pkl_olink)

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
# # Differences between groups defined by target

# %%
clinic

# %%
clinic[TARGET].value_counts()

# %%
pd.crosstab(clinic[TARGET], clinic["DecomensatedAtDiagnosis"])

# %%
y = clinic[TARGET].astype(bool)

# %%
y

# %% [markdown]
# ## All Features

# %%
clinic.dtypes.value_counts()

# %%
X = (olink
     .join(clinic[config.clinic_data.vars_cont])
     .join(clinic[config.clinic_data.comorbidities].astype('object').replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}))
    )

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
clf = sklearn.tree.DecisionTreeClassifier(criterion='log_loss',
                                          max_depth=3,
                                          min_samples_leaf=1,
                                          # min_samples_split=4,
                                          max_features=X.shape[-1]
                                         )
clf = clf.fit(X, y)

# rerunning this shows differences in deeper nodes
fig, ax = plt.subplots()
nodes = sklearn.tree.plot_tree(clf,
                               feature_names=X.columns,
                               class_names=["False", "True", "none"],
                               filled=True,
                               ax=ax)
fig.tight_layout()
fig.savefig(FOLDER / 'decision_tree.pdf')

# %%
results_train = src.sklearn.get_results_split(clf, X, y)
fig, ax = plt.subplots(1, 1, figsize=None)
ax = plot_split_prc(results_train, 'Decision Tree', ax)

# %% [markdown]
# - [ ] olink data only
# - [ ] clinial data only
# - [ ] laboratory (biochemistry) only?
