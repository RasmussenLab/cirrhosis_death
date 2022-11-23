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
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pingouin as pg
import matplotlib.pyplot as plt
import seaborn

import sklearn
import sklearn.impute
from sklearn.metrics import precision_recall_curve, roc_curve

import njab.sklearn
from njab.sklearn import StandardScaler
from njab.sklearn import pca as njab_pca
from njab.sklearn.scoring import ConfusionMatrix
from njab.sklearn.types import Splits
from njab.plotting.metrics import plot_split_auc, plot_split_prc

import src

import config

# %% [markdown]
# # Set parameters
#
# - [ ] one large dataset, composing of data should be done elsewhere
# - [ ] allow feature selection based on requested variables

# %% tags=["parameters"]
TARGET:str = 'liverDead180infl' # target column in CLINIC data
CLINIC:Path = config.fname_pkl_prodoc_clinic_num # clinic numeric pickled, can contain missing
feat_clinic:list = config.clinic_data.comorbidities + config.clinic_data.vars_cont # ToDo: make string
OLINK:Path = config.fname_pkl_prodoc_olink # olink numeric pickled, can contain missing
VAL_IDS: str = ''  #
use_val_split = True
FOLDER = 'reports/dev'

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %%
clinic = pd.read_pickle(CLINIC)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(OLINK)


# %% [markdown]
# ## Target
# %%
def value_counts_with_margins(y:pd.Series):
    ret = y.value_counts().to_frame('counts')
    ret.index.name = y.name
    ret['prop']  = y.value_counts(normalize=True)
    return ret

value_counts_with_margins(clinic[TARGET])

# %%
target_counts = clinic[TARGET].value_counts()

if target_counts.sum() < len(clinic):
    print(
        f"Target has missing values. Can only use {target_counts.sum()} of {len(clinic)} samples."
    )
    mask = clinic[TARGET].notna()
    clinic, olink = clinic.loc[mask], olink.loc[mask]

y = clinic[TARGET]

target_counts

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

# %% [markdown] tags=[]
# ## Combine dataset
#

# %%
predictors = feat_clinic + olink.columns.to_list()
model_name = 'combined LR'
X = clinic[feat_clinic].join(olink)
X

# %%
if VAL_IDS:
    diff = pd.Index(VAL_IDS)
    VAL_IDS = X.index.intersection(VAL_IDS)
    if len(diff) < len(VAL_IDS):
        logging.warning("Some requested validation IDs are not in the data: "
                        ",".join(str(x) for x in diff.difference(VAL_IDS)))
    X_val = X.loc[VAL_IDS]
    X = X.drop(VAL_IDS)

    use_val_split = True
    
    y_val = y.loc[VAL_IDS]
    y = y.drop(VAL_IDS)


# %% [markdown]
# ### Collect test predictions

# %%
predictions = y_val.to_frame('true')

# %% [markdown]
# ## Deal with missing values globally

# %%
feat_w_missings = X.isna().sum()
feat_w_missings = feat_w_missings.loc[feat_w_missings > 0]
feat_w_missings

# %%
median_imputer = sklearn.impute.SimpleImputer(strategy='median')

X = njab.sklearn.transform_DataFrame(X, median_imputer.fit_transform)
X_val = njab.sklearn.transform_DataFrame(X_val, median_imputer.transform)


# %%
X.isna().sum()

# %% [markdown]
# ## Principal Components
#
# - [ ]  base on selected data
# - binary features do not strictly need to be normalized

# %%
# on X
# scaler = StandardScaler()
# olink_scaled = scaler.fit_transform(olink).fillna(0)

# PCs, pca = njab_pca.run_pca(olink, n_components=None)
# njab_pca.plot_explained_variance(pca)
# PCs.iloc[:10, :10]

# %% [markdown]
# ## Baseline Model - Logistic Regression 
# - `age`, `decompensated`, `MELD-score`
# - use weigthing to counter class imbalances

# %%
# run nb with parameters
# name_model = 'baseline'
# cols_base_model = [cols_clinic.Age, cols_clinic.DecomensatedAtDiagnosis, cols_clinic.MELD_score] # MELD score -> death

# %%
weights= sklearn.utils.class_weight.compute_sample_weight('balanced', y)

# %% [markdown]
# ## Logistic Regression

# %%
# X_all = 
# X = X_all[include]

# X = olink
# X_val = olink_val
# model_name = 'olink LR'

# X = clinic[feat_clinic]
# X_val = clinic_val[feat_clinic]
# model_name = 'clinic LR'

# %%
# X = clinic[feat_clinic].join(olink)
# X_val = clinic_val[feat_clinic].join(olink_val)


# %%
cv_feat = njab.sklearn.find_n_best_features(
    X=X,
    y=y,
    model=sklearn.linear_model.LogisticRegression(),
    name=TARGET,
    groups=y,
    n_features_max=10,
    fit_params=dict(sample_weight=weights)
)
cv_feat = cv_feat.groupby('n_features').agg(['mean', 'std'])
cv_feat

# %%
mask = cv_feat.columns.levels[0].str[:4] == 'test'
scores_cols =  cv_feat.columns.levels[0][mask]
n_feat_best = cv_feat.loc[:, pd.IndexSlice[scores_cols, 'mean']].idxmax()
n_feat_best

# %%
splits = Splits(X_train=X, X_test=X_val, y_train=y, y_test=y_val)
results_model = njab.sklearn.run_model(
    splits,
    # n_feat_to_select=n_feat_best.loc['test_f1', 'mean'],
    n_feat_to_select=int(n_feat_best.mode()),
    fit_params=dict(sample_weight=weights)
)
results_model.name = model_name


# %%
def plot_auc(results, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1, **kwargs)
    ax = plot_split_auc(results.train,  f"{results.name} (train)", ax)
    ax = plot_split_auc(results.test, f"{results.name} (test)", ax)
    return ax

ax = plot_auc(results_model, figsize=(4,2))


# %%
def plot_prc(results, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1, **kwargs)
    ax = plot_split_prc(results.train,  f"{results.name} (train)", ax)
    ax = plot_split_prc(results.test, f"{results.name} (test)", ax)
    return ax

ax = plot_prc(results_model, figsize=(4,2))

# %%
# https://www.statsmodels.org/dev/discretemod.html
np.exp(results_model.model.coef_)

# %%
X[results_model.selected_features].describe()

# %%
X[results_model.selected_features].head()


# %%
def get_score(clf, X, pos=1):
    ret = clf.predict_proba(X)[:,pos]
    ret = pd.Series(ret, index=X.index)
    return ret

def get_pred(clf, X):
    ret = clf.predict(X)
    ret = pd.Series(ret, index=X.index)
    return ret

score = get_score(clf=results_model.model, X=X[results_model.selected_features], pos=1)
ax = score.hist(bins=20)

# %%
# score_val

N_BINS = 10
def get_target_count_per_bin(score, y, n_bins=N_BINS):
    pred_bins = pd.DataFrame({'score':pd.cut(score, bins=list(x/N_BINS for x in range(0,N_BINS+1))), 'y==1':y})
    pred_bins = pred_bins.groupby(by='score').sum().astype(int)
    return pred_bins

pred_bins = get_target_count_per_bin(score, y)    
pred_bins.plot(kind='bar', ylabel='count')
pred_bins

# %%
score_val = get_score(clf=results_model.model, X=X_val[results_model.selected_features], pos=1)
predictions['score'] = score_val
ax = score_val.hist(bins=20)
pred_bins_val = get_target_count_per_bin(score_val, y_val)    
pred_bins_val.plot(kind='bar', ylabel='count')
pred_bins_val

# %% [markdown]
# ## Performance evaluations

# %%
y_pred_val = get_pred(clf=results_model.model, X=X_val[results_model.selected_features])
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
ConfusionMatrix(y_val, y_pred_val).as_dataframe

# %%
predictions.sort_values('score', ascending=False)

# %% [markdown]
# ## Multiplicative decompositon

# %%
components = X[results_model.selected_features].multiply(results_model.model.coef_)
components['intercept'] = float(results_model.model.intercept_)
np.exp(components)

# # prediction is row entries multiplied (note: numerial instability of multiplications)
# import functools
# np.exp(components).apply(lambda s: functools.reduce(np.multiply, s), axis=1)

# %%
pivot = y.to_frame()
pivot['pred'] = results_model.model.predict(X[results_model.selected_features])
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
#
# - [ ] use when no weights are used

# %%
log_reg = sklearn.linear_model.LogisticRegression()
log_reg = log_reg.fit(X=X, y=y, sample_weight=None)

y_prob = log_reg.predict_proba(X)[:,1]

# %%
fpr, tpr, cutoffs = roc_curve(y, y_prob)
roc = pd.DataFrame([fpr, tpr, cutoffs], index='fpr tpr cutoffs'.split())
ax = roc.T.plot('fpr', 'tpr')

# %%
precision, recall, cutoffs = precision_recall_curve(y, y_prob)
prc = pd.DataFrame([precision, recall, cutoffs], index='precision recall cutoffs'.split())
prc

# %%
ax = prc.T.plot('recall', 'precision', ylabel='precision')

# %%
prc.loc['f1_score'] = 2 * (prc.loc['precision'] * prc.loc['recall']) / (1/prc.loc['precision'] + 1/prc.loc['recall'])
f1_max = prc[prc.loc['f1_score'].argmax()]
f1_max

# %%
y_pred = pd.Series((y_prob > f1_max.loc['cutoffs']), index=X.index).astype(int)

predictions['5 PCs (LR)'] = y_pred

ConfusionMatrix(y, y_pred).as_dataframe # this needs to be augmented with information if patient died by now (to see who is "wrongly classified)")

# %%
pivot = y_pred.to_frame('pred').join(y).join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %% [markdown]
# How many will die for those who have been predicted to die?

# %%
pd.pivot_table(pivot, values='pred', index=TARGET, columns='dead', aggfunc='sum')

# %%
pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed

# %% [markdown]
# ## Plot TP, TN, FP and FN on PCA plot

# %%
model_pred_cols = predictions.columns[1:5].to_list()
model_pred_cols

# %%
binary_labels = pd.DataFrame()

TRUE_COL = 'true'
for model_pred_col in model_pred_cols:
    binary_labels[model_pred_col] = predictions.apply(lambda x: njab.sklearn.scoring.get_label_binary_classification(
        x[TRUE_COL], x[model_pred_col]),
                      axis=1)
binary_labels.sample(6)

# %%
colors = seaborn.color_palette(n_colors=4)
colors

# %%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3,1, figsize=(10,20), sharex=True, sharey=True)
# for model_pred_col, ax in zip(binary_labels.columns, axes.ravel()):
#     ax = seaborn.scatterplot(x=PCs.iloc[:,0], y=PCs.iloc[:, 1], hue=binary_labels[model_pred_col], hue_order=['TN', 'TP', 'FN', 'FP'],
#                              # palette=colors,
#                              palette=[colors[0], colors[2], colors[1], colors[3]],
#                              ax=ax)
#     ax.set_title(model_pred_col)

# %%
