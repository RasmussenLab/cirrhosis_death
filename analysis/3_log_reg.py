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
# # Logistic regression model

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve

import src.stats
from src.sklearn import run_pca, StandardScaler
from src.sklearn.scoring import ConfusionMatrix

import config

# %% [markdown]
# # Set parameters

# %% tags=["parameters"]
TARGET = 'hasAdm180'
FOLDER = ''

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True)

# %%
clinic = pd.read_pickle(config.fname_pkl_clinic)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(config.fname_pkl_olink)

# %%
olink_scaled = StandardScaler().fit_transform(olink).fillna(0)

PCs, pca = run_pca(olink_scaled, n_components=None)
PCs.iloc[:10, :10]

# %% [markdown]
# # Initial Modeling

# %%
y_true = clinic[TARGET]
predictions = y_true.to_frame('true')
y_true.value_counts()

# %%
(y_true.value_counts() / len(y_true))

# %% [markdown]
# ## Baseline
# - `age`, `decompensated`, `child-pugh` 

# %%
X = [cols_clinic.Age, cols_clinic.ChildPugh]
X = clinic[X].copy()
X.loc[:,'decompensated'] = clinic.DecomensatedAtDiagnosis.cat.codes

weights= sklearn.utils.class_weight.compute_sample_weight('balanced', y_true)

log_reg = sklearn.linear_model.LogisticRegression()
log_reg = log_reg.fit(X=X, y=y_true, sample_weight=weights)

# %%
y_pred = log_reg.predict(X)
predictions['baseline weighted (LR)'] = y_pred
ConfusionMatrix(y_true, y_pred).as_dataframe

# %% [markdown]
# ## Logistic Regression

# %%
X = PCs.iloc[:,:5]

# %% [markdown]
# ### With weights

# %%
weights= sklearn.utils.class_weight.compute_sample_weight('balanced', y_true)

log_reg = sklearn.linear_model.LogisticRegression()
log_reg = log_reg.fit(X=X, y=y_true, sample_weight=weights)

# %%
scores = dict(ref_score=(y_true.value_counts() / len(clinic)).max(),
              model_score=log_reg.score(X, y_true, sample_weight=None))

scores

# %%
y_pred = log_reg.predict(X)
predictions['5 PCs weighted (LR)'] = y_pred

ConfusionMatrix(y_true, y_pred).as_dataframe

# %%
pivot = y_true.to_frame()
pivot['pred'] = y_pred
pivot = pivot.join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %%
pd.pivot_table(pivot, values='pred', index=TARGET, columns='dead', aggfunc='sum')

# %%
pd.pivot_table(pivot, values='dead', index=TARGET, columns='pred', aggfunc='sum')

# %%
pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed

# %% [markdown]
# ### Without weights, but adapting cutoff

# %%
log_reg = log_reg.fit(X=X, y=y_true, sample_weight=None)

y_prob = log_reg.predict_proba(X)[:,1]

# %%
fpr, tpr, cutoffs = roc_curve(y_true, y_prob)
roc = pd.DataFrame([fpr, tpr, cutoffs], index='fpr tpr cutoffs'.split())
ax = roc.T.plot('fpr', 'tpr')

# %%
precision, recall, cutoffs = precision_recall_curve(y_true, y_prob)
prc = pd.DataFrame([precision, recall, cutoffs], index='precision recall cutoffs'.split())
prc

# %%
ax = prc.T.plot('recall', 'precision', ylabel='precision')

# %%
prc.loc['f1_score'] = 2 * (prc.loc['precision'] * prc.loc['recall']) / (1/prc.loc['precision'] + 1/prc.loc['recall']) 
f1_max = prc[prc.loc['f1_score'].argmax()]
f1_max

# %%
y_pred = pd.Series((y_prob > f1_max.loc['cutoffs']), index=PCs.index).astype(int)

predictions['5 PCs (LR)'] = y_pred

ConfusionMatrix(y_true, y_pred).as_dataframe # this needs to be augmented with information if patient died by now (to see who is "wrongly classified)")

# %%
pivot = y_pred.to_frame('pred').join(y_true).join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %% [markdown]
# How many will die for those who have been predicted to die?

# %%
pd.pivot_table(pivot, values='pred', index=TARGET, columns='dead', aggfunc='sum')

# %%
pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed


# %% [markdown]
# ## Compare prediction errors between models

# %%
def get_mask_fp_tn(predictions:pd.DataFrame):
    N, M = predictions.shape
    row_sums = predictions.sum(axis=1)
    mask = (row_sums == 0) | (row_sums==M)
    return ~mask
mask_fp_tn = get_mask_fp_tn(predictions)
predictions.loc[mask_fp_tn].sort_values(by='true', ascending=False)

# %%
sel_clinic_cols = [cols_clinic.Age, cols_clinic.DiagnosisPlace, cols_clinic.HeartDiseaseTotal, cols_clinic.DaysToAdmFromInflSample, cols_clinic.DaysToDeathFromInfl, cols_clinic.DaysToDeathFromInfl, cols_clinic.DateInflSample, cols_clinic.DateBiochemistry_, cols_clinic.DateImmunoglobulins_, cols_clinic.DateInflSample]
predictions.loc[mask_fp_tn].loc[y_true.astype(bool)].sort_values(by='true', ascending=False).join(clinic[sel_clinic_cols])

# %%
mask_tp = predictions.sum(axis=1) == 4
predictions.loc[mask_tp].join(clinic[sel_clinic_cols])

# %% [markdown]
# ## Plot TP, TN, FP and FN on PCA plot

# %%
model_pred_cols = predictions.columns[1:5].to_list()
model_pred_cols

# %%
binary_labels = pd.DataFrame()

TRUE_COL = 'true'
for model_pred_col in model_pred_cols:
    binary_labels[model_pred_col] = predictions.apply(lambda x: src.sklearn.scoring.get_label_binary_classification(
        x[TRUE_COL], x[model_pred_col]),
                      axis=1)
binary_labels.sample(6)

# %%
colors = seaborn.color_palette(n_colors=4)
colors

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3,1, figsize=(10,20), sharex=True, sharey=True)
for model_pred_col, ax in zip(binary_labels.columns, axes.ravel()):
    ax = seaborn.scatterplot(x=PCs.iloc[:,0], y=PCs.iloc[:, 1], hue=binary_labels[model_pred_col], hue_order=['TN', 'TP', 'FN', 'FP'],
                             # palette=colors,
                             palette=[colors[0], colors[2], colors[1], colors[3]],
                             ax=ax)
    ax.set_title(model_pred_col)
