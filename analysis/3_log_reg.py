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
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pingouin as pg
import matplotlib.pyplot as plt
import seaborn
from heatmap import heatmap, corrplot

import sklearn
import sklearn.impute
from sklearn.metrics import precision_recall_curve, roc_curve

import njab.sklearn
from njab.sklearn import StandardScaler
from njab.sklearn import pca as njab_pca
from njab.sklearn.scoring import ConfusionMatrix
from njab.sklearn.types import Splits
from njab.plotting.metrics import plot_auc, plot_prc
from njab.sklearn.scoring import get_score, get_pred, get_target_count_per_bin

import src

import config

# %% [markdown]
# # Set parameters
#
# - [ ] one large dataset, composing of data should be done elsewhere
# - [ ] allow feature selection based on requested variables

# %% tags=["parameters"]
TARGET:str = 'dead180infl' # target column in CLINIC data
CLINIC:Path = config.fname_pkl_prodoc_clinic_num # clinic numeric pickled, can contain missing
# feat_clinic:list = config.clinic_data.comorbidities + config.clinic_data.vars_cont # ToDo: make string
OLINK:Path = config.fname_pkl_prodoc_olink # olink numeric pickled, can contain missing
# X_numeric: Path
feat_set_to_consider:str='OLINK_AND_CLINIC'
n_features_max:int = 15
VAL_IDS: str = ''  #
weights:bool = True
FOLDER = ''

# %%
weights = False

# %% [markdown]
# # Setup
# ## Load data

# %%
clinic = pd.read_pickle(CLINIC)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(OLINK)


# %% [markdown]
# ## Target
# %%
src.pandas.value_counts_with_margins(clinic[TARGET])

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
# # Test IDs

# %%
olink_val, clinic_val = None, None
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

# %% [markdown]
# ## Combine clinical and olink data
#

# %%
if feat_set_to_consider not in config.feat_sets:
    raise ValueError(f"Choose one of the available sets: {', '.join(config.feat_sets.keys())}")
feat_to_consider = config.feat_sets[feat_set_to_consider].split(',')
feat_to_consider

# %%
# predictors = feat_clinic + olink.columns.to_list()
model_name = feat_set_to_consider
X = clinic.join(olink)[feat_to_consider]
X

# %% [markdown]
# ## Data Splits -> train and test split

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
# ## Output folder


# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET / feat_set_to_consider
    FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %% [markdown]
# Outputs

# %%
# out
files_out = {}
files_out['3_log_reg.xlsx'] = FOLDER / '3_log_reg.xlsx'
writer = pd.ExcelWriter(files_out['3_log_reg.xlsx'])

# %% [markdown]
# ## Collect test predictions

# %%
predictions = y_val.to_frame('true')

# %% [markdown]
# ## Deal with missing values globally - impute

# %%
feat_w_missings = X.isna().sum()
feat_w_missings = feat_w_missings.loc[feat_w_missings > 0]
feat_w_missings

# %%
median_imputer = sklearn.impute.SimpleImputer(strategy='median')

X = njab.sklearn.transform_DataFrame(X, median_imputer.fit_transform)
X_val = njab.sklearn.transform_DataFrame(X_val, median_imputer.transform)


# %%
assert X.isna().sum().sum()  == 0

# %% [markdown]
# ## Principal Components
#
# - [ ]  base on selected data
# - binary features do not strictly need to be normalized

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

PCs, pca = njab_pca.run_pca(X_scaled, n_components=None)
files_out["var_explained_by_PCs.pdf"] = FOLDER / "var_explained_by_PCs.pdf"
ax = njab_pca.plot_explained_variance(pca)
ax.locator_params(axis='x', integer=True)
njab.plotting.savefig(ax.get_figure(), files_out["var_explained_by_PCs.pdf"])
X_scaled.shape

# %%
files_out['scatter_first_5PCs.pdf'] = FOLDER / 'scatter_first_5PCs.pdf'

fig, axes = plt.subplots(5, 2, figsize=(8.3, 11.7), layout='constrained')
PCs = PCs.join(y.astype('category'))
for(i,j), ax in zip(itertools.combinations(range(5), 2), axes.flatten()):
    PCs.plot.scatter(i, j, c=TARGET, cmap='Paired', ax=ax)
_ = PCs.pop(TARGET)
njab.plotting.savefig(fig, files_out['scatter_first_5PCs.pdf'])

# %% [markdown]
# ## UMAP

# %%
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_scaled)

# %%
files_out['umap.pdf'] = FOLDER / 'umap.pdf'

embedding = pd.DataFrame(embedding, index=X_scaled.index, columns=['UMAP 1', 'UMAP 2']).join(y.astype('category'))
ax = embedding.plot.scatter('UMAP 1', 'UMAP 2', c=TARGET, cmap='Paired')
njab.plotting.savefig(ax.get_figure(), files_out['umap.pdf'])

# %% [markdown]
# ## Baseline Model - Logistic Regression 
# - `age`, `decompensated`, `MELD-score`
# - use weigthing to counter class imbalances

# %%
# run nb with parameters
# name_model = 'baseline'
# cols_base_model = [cols_clinic.Age, cols_clinic.DecomensatedAtDiagnosis, cols_clinic.MELD_score] # MELD score -> death

# %%
if weights:
    weights= sklearn.utils.class_weight.compute_sample_weight('balanced', y)
    cutoff=0.5
else:
    cutoff=None
    weights=None

# %% [markdown]
# ## Logistic Regression
# Procedure:
# 1. Select best set of features from entire feature set selected using CV on train split
# 2. Retrain best model configuration using entire train split and evalute on test split

# %%
cv_feat = njab.sklearn.find_n_best_features(
    X=X,
    y=y,
    model=sklearn.linear_model.LogisticRegression(),
    name=TARGET,
    groups=y,
    n_features_max=n_features_max,
    fit_params=dict(sample_weight=weights)
)
cv_feat = cv_feat.groupby('n_features').agg(['mean', 'std'])
cv_feat.to_excel(writer, 'CV', float_format='%.3f')
cv_feat

# %%
mask = cv_feat.columns.levels[0].str[:4] == 'test'
scores_cols =  cv_feat.columns.levels[0][mask]
n_feat_best = cv_feat.loc[:, pd.IndexSlice[scores_cols, 'mean']].idxmax()
n_feat_best.name = 'best'
n_feat_best.to_excel(writer, 'n_feat_best')
n_feat_best

# %%
splits = Splits(X_train=X, X_test=X_val, y_train=y, y_test=y_val)
results_model = njab.sklearn.run_model(
    splits,
    # n_feat_to_select=n_feat_best.loc['test_f1', 'mean'],
    n_feat_to_select=n_feat_best.loc['test_roc_auc', 'mean'],
    # n_feat_to_select=int(n_feat_best.mode()),
    fit_params=dict(sample_weight=weights)
)
results_model.name = model_name


# %%
ax = plot_auc(results_model, figsize=(4,2))
files_out['ROAUC'] = FOLDER / 'plot_roauc.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['ROAUC'])

# %%
ax = plot_prc(results_model, figsize=(4,2))
files_out['PRAUC'] = FOLDER / 'plot_prauc.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['PRAUC'])

# %%
# https://www.statsmodels.org/dev/discretemod.html
np.exp(results_model.model.coef_)

# %%
des_selected_feat = X[results_model.selected_features].describe()
des_selected_feat.to_excel(writer, 'sel_feat', float_format='%.3f')
des_selected_feat

# %%
fig = plt.figure(figsize=(5,5,))
files_out['corr_plot_train.pdf'] = FOLDER / 'corr_plot_train.pdf' 
_ = corrplot(X[results_model.selected_features].join(y).corr(), size_scale=80);
fig.tight_layout()
njab.plotting.savefig(fig, files_out['corr_plot_train.pdf'])

# %% [markdown]
# Plot training data scores


# %%
N_BINS = 20
score = get_score(clf=results_model.model, X=X[results_model.selected_features], pos=1)
ax = score.hist(bins=N_BINS)
files_out['hist_score_train.pdf'] = FOLDER / 'hist_score_train.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_train.pdf'])

# %%
# score_val
pred_bins = get_target_count_per_bin(score, y, n_bins=N_BINS)    
ax = pred_bins.plot(kind='bar', ylabel='count')
files_out['hist_score_train_target.pdf'] = FOLDER / 'hist_score_train_target.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_train_target.pdf'])
# pred_bins

# %% [markdown]
# Test data scores

# %%
score_val = get_score(clf=results_model.model, X=X_val[results_model.selected_features], pos=1)
predictions['score'] = score_val
ax = score_val.hist(bins=N_BINS) # list(x/N_BINS for x in range(0,N_BINS)))
ax.set_ylabel('count')
ax.set_xlim(0,1)
files_out['hist_score_test.pdf'] = FOLDER / 'hist_score_test.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_test.pdf'])
pred_bins_val = get_target_count_per_bin(score_val, y_val, n_bins=N_BINS)    
ax = pred_bins_val.plot(kind='bar', ylabel='count')
ax.locator_params(axis='y', integer=True)
files_out['hist_score_test_target.pdf'] = FOLDER / 'hist_score_test_target.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_test_target.pdf'])
# pred_bins_val

# %% [markdown]
# ## Performance evaluations

# %%
prc = pd.DataFrame(results_model.train.prc, index='precision recall cutoffs'.split())
prc

# %%
prc.loc['f1_score'] = 2 * (prc.loc['precision'] * prc.loc['recall']) / (1/prc.loc['precision'] + 1/prc.loc['recall'])
f1_max = prc[prc.loc['f1_score'].argmax()]
f1_max

# %% [markdown]
# Cutoff set

# %%
cutoff = float(f1_max.loc['cutoffs'])
cutoff

# %%
y_pred_val = njab.sklearn.scoring.get_custom_pred(
    clf=results_model.model,
    X=X_val[results_model.selected_features],
    cutoff=cutoff)
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
_ = ConfusionMatrix(y_val, y_pred_val).as_dataframe
_.to_excel(writer, f"CM_test_cutoff_{cutoff:.3f}")
_

# %%
y_pred_val = njab.sklearn.scoring.get_custom_pred(
    clf=results_model.model,
    X=X_val[results_model.selected_features],
    cutoff=0.5)
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
_ = ConfusionMatrix(y_val, y_pred_val).as_dataframe
# _.to_excel(writer, "CM_test")
_

# %%
y_pred_val = get_pred(clf=results_model.model,
                      X=X_val[results_model.selected_features])
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
_ = ConfusionMatrix(y_val, y_pred_val).as_dataframe
_.to_excel(writer, "CM_test_cutoff_0.5")
_


# %% [markdown]
# ## Multiplicative decompositon

# %%
def get_lr_multiplicative_decomposition(results, X, score, y):
    components = X[results.selected_features].multiply(results.model.coef_)
    components['intercept'] = float(results.model.intercept_)
    components = np.exp(components)
    components['score'] = score
    components[TARGET]  = y
    components = components.sort_values('score', ascending=False)
    return components

components = get_lr_multiplicative_decomposition(results=results_model, X=X, score=score, y=y)
components.to_excel(writer, 'decomp_multiplicative_train')
components.to_excel(writer, 'decomp_multiplicative_train_view', float_format='%.3f')
components.head(10)

# %%
components_test = get_lr_multiplicative_decomposition(results=results_model, X=X_val, score=score_val, y=y_val)
components_test.to_excel(writer, 'decomp_multiplicative_test')
components_test.to_excel(writer, 'decomp_multiplicative_test_view', float_format='%.3f')
components_test.head(10)

# %%
pivot = y.to_frame()
pivot['pred'] = results_model.model.predict(X[results_model.selected_features])
pivot = pivot.join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %%
pivot_dead_by_pred_and_target = pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed
pivot_dead_by_pred_and_target.to_excel(writer, 'pivot_dead_by_pred_and_target')


# %%
writer.close()
# %% [markdown]
# ## Plot TP, TN, FP and FN on PCA plot

# %%
predictions['label'] = predictions.apply(lambda x: njab.sklearn.scoring.get_label_binary_classification(
        x['true'], x[model_name]),
                      axis=1)

mask = predictions[['true', model_name]].sum(axis=1).astype(bool)
predictions.loc[mask].sort_values('score', ascending=False)

# %%
X_val_scaled = scaler.transform(X_val)
PCs_val = pd.DataFrame(pca.transform(X_val_scaled), index=X_val.index, columns=PCs.columns)
PCs_val

# %%
pred_train = y.to_frame('true').join(get_pred(results_model.model, X[results_model.selected_features]).rename(model_name))
pred_train['label'] = pred_train.apply(lambda x: njab.sklearn.scoring.get_label_binary_classification(
        x['true'], x[model_name]),
                      axis=1)
pred_train.sample(5)

# %%
colors = seaborn.color_palette(n_colors=4)
colors

# %%
fig, axes = plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True)
for _pcs, ax, _title, _model_pred_label in zip([PCs, PCs_val],
                                                axes,
                                                ['train', 'test'],
                                                [pred_train['label'], predictions['label']]
    ):
    ax = seaborn.scatterplot(x=_pcs.iloc[:,0], y=_pcs.iloc[:, 1],
                             hue=_model_pred_label,
                             hue_order=['TN', 'TP', 'FN', 'FP'],
                             palette=[colors[0], colors[2], colors[1], colors[3]],
                             ax=ax)
    ax.set_title(_title)

files_out['pred_pca_labeled'] = FOLDER / 'pred_pca_labeled.pdf'
njab.plotting.savefig(fig, files_out['pred_pca_labeled'])

# %% [markdown]
# ## Outputs

# %%
files_out
