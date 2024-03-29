# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn
from heatmap import corrplot
import umap
from IPython.display import display

import sklearn
import sklearn.impute
from sklearn.metrics import make_scorer, log_loss
import statsmodels.api as sm

import njab.sklearn
from njab.sklearn import StandardScaler
from njab.sklearn import pca as njab_pca
from njab.sklearn.scoring import ConfusionMatrix
from njab.sklearn.types import Splits
from njab.plotting.metrics import plot_auc, plot_prc
from njab.sklearn.scoring import (get_score,
                                  get_pred,
                                  get_target_count_per_bin,
                                  get_lr_multiplicative_decomposition)
import config

logger = logging.getLogger('njab')
logger.setLevel(logging.INFO)

# adapt fond size:
# njab.plotting.set_font_sizes(7)

# %% [markdown]
# ## Set parameters
#
# - [ ] one large dataset, composing of data should be done elsewhere
# - [ ] allow feature selection based on requested variables

# %% tags=["parameters"]
# clinic numeric pickled, can contain missing
CLINIC: Path = config.fname_pkl_all_clinic_num
# olink numeric pickled, can contain missing
OLINK: Path = config.fname_pkl_all_olink
TARGET: str = 'dead180infl'  # target column in CLINIC data
feat_set_to_consider: str = 'OLINK_AND_CLINIC'
n_features_max: int = 15
VAL_IDS: str = ''  #
VAL_IDS_query: str = 'Cflow'
weights: bool = True
FOLDER = ''

# %%
# CLINIC:Path = config.fname_pkl_all_clinic_num
# OLINK:Path = config.fname_pkl_all_olink
# weights = False
# feat_set_to_consider = "SCORES_ONLY"
# FOLDER = "S:/SUND-CBMR-RegH-cohorts/ProDoc/reports_dev/prodoc/hasLiverAdm180/SCORES_ONLY"

# # Parameters
# TARGET = "hasLiverAdm180"
# CLINIC = "S:/SUND-CBMR-RegH-cohorts/ProDoc/data/processed/all_clinic_num.pkl"
# OLINK = "S:/SUND-CBMR-RegH-cohorts/ProDoc/data/processed/all_olink.pkl"
# feat_set_to_consider = "SCORES_ONLY"
# VAL_IDS_query = "Cflow"
# FOLDER = "S:/SUND-CBMR-RegH-cohorts/ProDoc/reports/prodoc/hasLiverAdm180/SCORES_ONLY"

# %% [markdown]
# # Setup
# ## Load data

# %%
clinic = pd.read_pickle(CLINIC)
cols_clinic = njab.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(OLINK)


# %%
olink.shape, clinic.shape

# %% [markdown]
# ## Target
# %%
njab.pandas.value_counts_with_margins(clinic[TARGET])

# %%
target_counts = clinic[TARGET].value_counts()

if target_counts.sum() < len(clinic):
    print(
        f"Target has missing values. Can only use {target_counts.sum()} of {len(clinic)} samples."
    )
    mask = clinic[TARGET].notna()
    clinic, olink = clinic.loc[mask], olink.loc[mask]

TARGET_LABEL = config.TARGET_LABELS[TARGET]

y = clinic[TARGET].rename(TARGET_LABEL)

target_counts

# %% [markdown]
# ## Test IDs

# %%
olink_val, clinic_val = None, None
if not VAL_IDS:
    if VAL_IDS_query:
        logging.warning(f"Querying index using: {VAL_IDS_query}")
        VAL_IDS = clinic.filter(like='Cflow', axis=0).index.to_list()
        logging.warning(f"Found {len(VAL_IDS)} Test-IDs")
    else:
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
    raise ValueError(
        f"Choose one of the available sets: {', '.join(config.feat_sets.keys())}"
    )
feat_to_consider = config.feat_sets[feat_set_to_consider].split(',')
feat_to_consider

# %%
# predictors = feat_clinic + olink.columns.to_list()
model_name = config.MODEL_NAMES[feat_set_to_consider]
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
    FOLDER = Path(
        config.folder_reports) / 'prodoc' / TARGET / feat_set_to_consider
    FOLDER.mkdir(exist_ok=True, parents=True)
else:
    FOLDER = Path(FOLDER)
FOLDER

# %% [markdown]
# ### Outputs

# %%
# out
files_out = {}
fname = FOLDER / '3_log_reg.xlsx'
files_out[fname.stem] = fname
writer = pd.ExcelWriter(fname)
fname

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
row_w_missing = X.isna().sum(axis=1).astype(bool)
col_w_missing = X.isna().sum(axis=0).astype(bool)
X.loc[row_w_missing, col_w_missing]

# %%
median_imputer = sklearn.impute.SimpleImputer(strategy='median')

X = njab.sklearn.transform_DataFrame(X, median_imputer.fit_transform)
X_val = njab.sklearn.transform_DataFrame(X_val, median_imputer.transform)


# %%
assert X.isna().sum().sum() == 0

# %%
X.shape, X_val.shape

# %% [markdown]
# # Principal Components
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
up_to = min(PCs.shape[-1], 5)
for (i, j), ax in zip(itertools.combinations(range(up_to), 2), axes.flatten()):
    PCs.plot.scatter(i, j, c=TARGET_LABEL, cmap='Paired', ax=ax)
_ = PCs.pop(TARGET_LABEL)
njab.plotting.savefig(fig, files_out['scatter_first_5PCs.pdf'])

# %% [markdown]
# # UMAP

# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_scaled)

# %%
files_out['umap.pdf'] = FOLDER / 'umap.pdf'

embedding = pd.DataFrame(embedding,
                         index=X_scaled.index,
                         columns=['UMAP 1',
                                  'UMAP 2']).join(y.astype('category'))
ax = embedding.plot.scatter('UMAP 1', 'UMAP 2', c=TARGET_LABEL, cmap='Paired')
njab.plotting.savefig(ax.get_figure(), files_out['umap.pdf'])

# %% [markdown]
# # Baseline Model - Logistic Regression
# - `age`, `decompensated`, `MELD-score`
# - use weigthing to counter class imbalances

# %%
# run nb with parameters
# name_model = 'baseline'
# cols_base_model = [cols_clinic.Age, cols_clinic.DecomensatedAtDiagnosis, cols_clinic.MELD_score] # MELD score -> death

# %%
if weights:
    weights = 'balanced'
    cutoff = 0.5
else:
    cutoff = None
    weights = None

# %% [markdown]
# # Logistic Regression
# Procedure:
# 1. Select best set of features from entire feature set selected using CV on train split
# 2. Retrain best model configuration using entire train split and evalute on test split

# %%
# # Scaled
splits = Splits(X_train=X_scaled,
                X_test=scaler.transform(X_val),
                y_train=y,
                y_test=y_val)

# splits = Splits(X_train=X,
#                 X_test=X_val,
#                 y_train=y, y_test=y_val)

model = sklearn.linear_model.LogisticRegression(
    penalty='l2', class_weight=weights)

# %%
scoring = [
    'precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc',
    'average_precision'
]
scoring = {k: k for k in scoring}
# do not average log loss for AIC and BIC calculations
scoring['log_loss'] = make_scorer(log_loss,
                                  greater_is_better=True,
                                  normalize=False)
cv_feat = njab.sklearn.find_n_best_features(
    X=splits.X_train,
    y=splits.y_train,
    model=model,
    name=TARGET_LABEL,
    groups=y,
    n_features_max=n_features_max,
    scoring=scoring,
    return_train_score=True,
    # fit_params=dict(sample_weight=weights)
)
cv_feat = cv_feat.groupby('n_features').agg(['mean', 'std'])
cv_feat

# %% [markdown]
# Add AIC and BIC for model selection

# %%
# AIC vs BIC on train and test data with bigger is better
IC_criteria = pd.DataFrame()
N_split = {
    'train': round(len(splits.X_train) * 0.8),
    'test': round(len(splits.X_train) * 0.2)
}

# IC_criteria[('test_log_loss', 'mean')] = cv_feat[
#                      ('test_log_loss', 'mean')]
# IC_criteria[('train_log_loss', 'mean')] = cv_feat[
#                      ('train_log_loss', 'mean')]
for _split in ('train', 'test'):

    IC_criteria[(f'{_split}_neg_AIC',
                 'mean')] = -(2 * cv_feat.index.to_series() -
                              2 * cv_feat[(f'{_split}_log_loss', 'mean')])
    IC_criteria[(
        f'{_split}_neg_BIC',
        'mean')] = -(cv_feat.index.to_series() * np.log(N_split[_split]) -
                     2 * cv_feat[(f'{_split}_log_loss', 'mean')])
IC_criteria.columns = pd.MultiIndex.from_tuples(IC_criteria.columns)
IC_criteria

# %%
cv_feat = cv_feat.join(IC_criteria)
cv_feat = cv_feat.filter(regex="train|test", axis=1).style.highlight_max(
    axis=0, subset=pd.IndexSlice[:, pd.IndexSlice[:, 'mean']])
cv_feat

# %%
cv_feat.to_excel(writer, 'CV', float_format='%.3f')
cv_feat = cv_feat.data

# %%
mask = cv_feat.columns.levels[0].str[:4] == 'test'
scores_cols = cv_feat.columns.levels[0][mask]
n_feat_best = cv_feat.loc[:, pd.IndexSlice[scores_cols, 'mean']].idxmax()
n_feat_best.name = 'best'
n_feat_best.to_excel(writer, 'n_feat_best')
n_feat_best

# %%
results_model = njab.sklearn.run_model(
    model=model,
    splits=splits,
    # n_feat_to_select=n_feat_best.loc['test_f1', 'mean'],
    n_feat_to_select=n_feat_best.loc['test_roc_auc', 'mean'],
    # n_feat_to_select=n_feat_best.loc['test_neg_AIC', 'mean'],
    # n_feat_to_select=int(n_feat_best.mode()),
    # fit_params=dict(sample_weight=weights)
)

results_model.name = model_name


# %% [markdown]
# ## ROC

# %%
ax = plot_auc(results_model,
              label_train=config.TRAIN_LABEL,
              label_test=config.TEST_LABEL,
              figsize=(4, 2))
files_out['ROAUC'] = FOLDER / 'plot_roauc.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['ROAUC'])

# %% [markdown]
# ## PRC

# %%
ax = plot_prc(results_model,
              label_train=config.TRAIN_LABEL,
              label_test=config.TEST_LABEL,
              figsize=(4, 2))
files_out['PRAUC'] = FOLDER / 'plot_prauc.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['PRAUC'])

# %% [markdown]
# ## Coefficients with/out std. errors

# %%
pd.DataFrame({
    'coef': results_model.model.coef_.flatten(),
    'name': results_model.model.feature_names_in_
})

# %%
results_model.model.intercept_

# %%
sm_logit = sm.Logit(endog=splits.y_train,
                    exog=sm.add_constant(
                        splits.X_train[results_model.selected_features]))
sm_logit = sm_logit.fit()
sm_logit.summary()

# %% [markdown]
# ## Selected Features

# %%
des_selected_feat = splits.X_train[results_model.selected_features].describe()
des_selected_feat.to_excel(writer, 'sel_feat', float_format='%.3f')
des_selected_feat

# %%
fig = plt.figure(figsize=(6, 6))
files_out['corr_plot_train.pdf'] = FOLDER / 'corr_plot_train.pdf'
heatmap_corr = X[results_model.selected_features].join(y).corr()
heatmap_corr.to_excel(writer, 'sel_feat_corr', float_format='%.3f')
_ = corrplot(heatmap_corr, size_scale=300)
njab.plotting.savefig(fig, files_out['corr_plot_train.pdf'])

# %% [markdown]
# ## Plot training data scores


# %%
N_BINS = 20
score = get_score(clf=results_model.model,
                  X=splits.X_train[results_model.selected_features],
                  pos=1)
ax = score.hist(bins=N_BINS)
files_out['hist_score_train.pdf'] = FOLDER / 'hist_score_train.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_train.pdf'])

# %%
# score_val
pred_bins = get_target_count_per_bin(score, y, n_bins=N_BINS)
ax = pred_bins.plot(kind='bar', ylabel='count')
files_out[
    'hist_score_train_target.pdf'] = FOLDER / 'hist_score_train_target.pdf'
njab.plotting.savefig(
    ax.get_figure(), files_out['hist_score_train_target.pdf'])
# pred_bins

# %% [markdown]
# ## Test data scores

# %%
score_val = get_score(clf=results_model.model,
                      X=splits.X_test[results_model.selected_features],
                      pos=1)
predictions['score'] = score_val
ax = score_val.hist(bins=N_BINS)  # list(x/N_BINS for x in range(0,N_BINS)))
ax.set_ylabel('count')
ax.set_xlim(0, 1)
files_out['hist_score_test.pdf'] = FOLDER / 'hist_score_test.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_test.pdf'])
pred_bins_val = get_target_count_per_bin(score_val, y_val, n_bins=N_BINS)
ax = pred_bins_val.plot(kind='bar', ylabel='count')
ax.locator_params(axis='y', integer=True)
files_out['hist_score_test_target.pdf'] = FOLDER / 'hist_score_test_target.pdf'
njab.plotting.savefig(ax.get_figure(), files_out['hist_score_test_target.pdf'])
# pred_bins_val

# %% [markdown]
# # KM plot

# %%
pred_train = get_pred(
    clf=results_model.model,
    X=splits.X_train[results_model.selected_features]).astype(bool)
ax, _, _ = njab.plotting.compare_km_curves(time=clinic.loc[pred_train.index,
                                                           "DaysToDeathFromInfl"],
                                           y=y[pred_train.index],
                                           pred=pred_train,
                                           xlabel='Days since inflammation sample',
                                           ylabel=f'rate {y.name}')

res = njab.plotting.km.log_rank_test(time=clinic.loc[pred_train.index,
                                                     "DaysToDeathFromInfl"],
                                     y=y[pred_train.index],
                                     mask=pred_train)
ax.set_title(
    f'KM curve for LR based on {model_name.lower()}\n (log-rank-test p={res.p_value:.3f})')
ax.legend([
    f"KP pred=0 (N={(~pred_train).sum()})", '95% CI (pred=0)',
    f"KP pred=1 (N={pred_train.sum()})", '95% CI (pred=1)'
])
fname = FOLDER / 'KM_plot_model_train.pdf'
files_out[fname.name] = fname
njab.plotting.savefig(ax.get_figure(), fname)

# %% [markdown]
# # Performance evaluations

# %%
prc = pd.DataFrame(results_model.train.prc,
                   index='precision recall cutoffs'.split())
prc

# %%
prc.loc['f1_score'] = 2 * (prc.loc['precision'] * prc.loc['recall']) / (
    1 / prc.loc['precision'] + 1 / prc.loc['recall'])
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
    X=splits.X_test[results_model.selected_features],
    cutoff=cutoff)
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
_ = ConfusionMatrix(y_val, y_pred_val).as_dataframe()
_.columns = pd.MultiIndex.from_tuples([
    (t[0] + f" - {cutoff:.3f}", t[1]) for t in _.columns
])
_.to_excel(writer, "CM_test_cutoff_adapted")
_

# %%
y_pred_val = get_pred(clf=results_model.model,
                      X=splits.X_test[results_model.selected_features])
predictions[model_name] = y_pred_val
predictions['dead'] = clinic['dead']
_ = ConfusionMatrix(y_val, y_pred_val).as_dataframe()
_.columns = pd.MultiIndex.from_tuples([
    (t[0] + f" - {0.5}", t[1]) for t in _.columns
])
_.to_excel(writer, "CM_test_cutoff_0.5")
_

# %%
# ! km plot
y_pred_val = y_pred_val.astype(bool)

ax, _, _ = njab.plotting.compare_km_curves(
    time=clinic.loc[y_pred_val.index, "DaysToDeathFromInfl"],
    y=y_val[y_pred_val.index],
    pred=y_pred_val,
    xlabel='Days since inflammation sample',
    ylabel=f'rate {y.name}')

res = njab.plotting.km.log_rank_test(time=clinic.loc[y_pred_val.index,
                                                     "DaysToDeathFromInfl"],
                                     y=y_val[y_pred_val.index],
                                     mask=y_pred_val)
ax.set_title(
    f'KM curve for LR based on {model_name.lower()}\n (log-rank-test p={res.p_value:.3f})')
ax.legend([
    f"KP pred=0 (N={(~y_pred_val).sum()})", '95% CI (pred=0)',
    f"KP pred=1 (N={y_pred_val.sum()})", '95% CI (pred=1)'
])
fname = FOLDER / 'KM_plot_model_val.pdf'
files_out[fname.name] = fname
njab.plotting.savefig(ax.get_figure(), fname)

# %% [markdown]
# # Multiplicative decompositon
# logarithmic transformation of the linear model
# $$ \ln\left(\frac{p}{1-p} \right) = \beta_0 + \beta_1 x_1 + \dots + \beta_M x_M $$
# Odds in terms of multiplicative compentents (which is dumped to excel)
# $$ \frac{p}{1-p} = \exp(\beta_0) \cdot \exp(\beta_1 x_1) * \dots * \exp(\beta_M x_M) = odds $$
# Going from odds to probability:
# $$ p = \frac{odds}{1+odds} $$

# %%
components = get_lr_multiplicative_decomposition(results=results_model,
                                                 X=splits.X_train,
                                                 prob=score,
                                                 y=y)
components.to_excel(writer, 'decomp_multiplicative_train')
components.to_excel(writer,
                    'decomp_multiplicative_train_view',
                    float_format='%.5f')
components.head(10)

# %%
components_test = get_lr_multiplicative_decomposition(results=results_model,
                                                      X=splits.X_test,
                                                      prob=score_val,
                                                      y=y_val)
components_test.to_excel(writer, 'decomp_multiplicative_test')
components_test.to_excel(writer,
                         'decomp_multiplicative_test_view',
                         float_format='%.5f')
components_test.head(10)

# %%
pivot = y.to_frame()
pivot['pred'] = results_model.model.predict(
    splits.X_train[results_model.selected_features])
pivot = pivot.join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %%
pivot_dead_by_pred_and_target = pivot.groupby(['pred', TARGET_LABEL
                                               ]).agg({'dead': ['count', 'sum']
                                                       })  # more detailed
pivot_dead_by_pred_and_target.to_excel(writer, 'pivot_dead_by_pred_and_target')


# %% [markdown]
# # Plot TP, TN, FP and FN on PCA plot
#
# ## UMAP
# %%
reducer = umap.UMAP(random_state=42)
# bug: how does UMAP works with only one feature?
# make sure to have two or more features?
M_sel = len(results_model.selected_features)
if M_sel > 1:
    embedding = reducer.fit_transform(
        X_scaled[results_model.selected_features])

    embedding = pd.DataFrame(embedding,
                             index=X_scaled.index,
                             columns=['UMAP dimension 1', 'UMAP dimension 2'
                                      ]).join(y.astype('category'))
    display(embedding.head(3))
else:
    embedding = None

# %%
predictions['DaysToDeathFromInfl'] = clinic['DaysToDeathFromInfl']
predictions['label'] = predictions.apply(
    lambda x: njab.sklearn.scoring.get_label_binary_classification(
        x['true'], x[model_name]),
    axis=1)
mask = predictions[['true', model_name]].sum(axis=1).astype(bool)
predictions.loc[mask].sort_values('score', ascending=False)

# %%
X_val_scaled = scaler.transform(X_val)
if embedding is not None:
    embedding_val = pd.DataFrame(reducer.transform(
        X_val_scaled[results_model.selected_features]),
        index=X_val_scaled.index,
        columns=['UMAP dimension 1', 'UMAP dimension 2'])
    embedding_val.sample(3)

# %%
pred_train = (
    y.to_frame('true')
    # .join(get_score(clf=results_model.model, X=splits.X_train[results_model.selected_features], pos=1))
    .join(score.rename('score')).join(
        get_pred(
            results_model.model,
            splits.X_train[results_model.selected_features]).rename(model_name))
)
pred_train['dead'] = clinic['dead']
pred_train['DaysToDeathFromInfl'] = clinic['DaysToDeathFromInfl']
pred_train['label'] = pred_train.apply(
    lambda x: njab.sklearn.scoring.get_label_binary_classification(
        x['true'], x[model_name]),
    axis=1)
pred_train.sample(5)

# %%
colors = seaborn.color_palette(n_colors=4)
colors

# %%
if embedding is not None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for _embedding, ax, _title, _model_pred_label in zip(
        [embedding, embedding_val], axes, [
            config.TRAIN_LABEL, config.TEST_LABEL],
            [pred_train['label'], predictions['label']]):
        ax = seaborn.scatterplot(
            x=_embedding.iloc[:, 0],
            y=_embedding.iloc[:, 1],
            hue=_model_pred_label,
            hue_order=['TN', 'TP', 'FN', 'FP'],
            palette=[colors[0], colors[2], colors[1], colors[3]],
            ax=ax)
        ax.set_title(_title)

    # files_out['pred_pca_labeled'] = FOLDER / 'pred_pca_labeled.pdf'
    # njab.plotting.savefig(fig, files_out['pred_pca_labeled'])

    files_out['umap_sel_feat.pdf'] = FOLDER / 'umap_sel_feat.pdf'
    njab.plotting.savefig(ax.get_figure(), files_out['umap_sel_feat.pdf'])

# %% [markdown]
# ### Interactive UMAP plot

# %%
if embedding is not None:
    embedding = embedding.join(X[results_model.selected_features])
    embedding_val = embedding_val.join(X_val[results_model.selected_features])
    embedding['label'], embedding_val['label'] = pred_train['label'], predictions['label']
    embedding['group'], embedding_val['group'] = config.TRAIN_LABEL, config.TEST_LABEL
    combined_embeddings = pd.concat([embedding, embedding_val])
    combined_embeddings.index.name = 'ID'

# %%
if embedding is not None:
    cols = combined_embeddings.columns

    TEMPLATE = 'none'
    defaults = dict(width=1600, height=700, template=TEMPLATE)

    fig = px.scatter(combined_embeddings.round(3).reset_index(),
                     x=cols[0],
                     y=cols[1],
                     color='label',
                     facet_col='group',
                     hover_data=['ID'] + results_model.selected_features,
                     **defaults)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    fname = FOLDER / 'umap_sel_feat.html'
    files_out[fname.name] = fname
    fig.write_html(fname)
    print(fname)

# %% [markdown]
# ## PCA

# %%
PCs_train, pca = njab_pca.run_pca(
    X_scaled[results_model.selected_features], n_components=None)
ax = njab_pca.plot_explained_variance(pca)
ax.locator_params(axis='x', integer=True)

fname = FOLDER / "feat_sel_PCA_var_explained_by_PCs.pdf"
files_out[fname.name] = fname
njab.plotting.savefig(ax.get_figure(), fname)

# %%
PCs_val = pca.transform(X_val_scaled[results_model.selected_features])
PCs_val = pd.DataFrame(PCs_val, index=X_val_scaled.index,
                       columns=PCs_train.columns)
PCs_val

# %%
if M_sel > 1:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for _embedding, ax, _title, _model_pred_label in zip(
        [PCs_train, PCs_val],
        axes,
        [config.TRAIN_LABEL, config.TEST_LABEL],
            [pred_train['label'], predictions['label']]):
        ax = seaborn.scatterplot(
            x=_embedding.iloc[:, 0],
            y=_embedding.iloc[:, 1],
            hue=_model_pred_label,
            hue_order=['TN', 'TP', 'FN', 'FP'],
            palette=[colors[0], colors[2], colors[1], colors[3]],
            ax=ax)
        ax.set_title(_title)

    fname = FOLDER / 'pca_sel_feat.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)


# %%
if M_sel > 1:
    max_rows = min(3, len(results_model.selected_features))
    fig, axes = plt.subplots(max_rows, 2,
                             figsize=(8.3, 11.7),
                             sharex=False, sharey=False,
                             layout='constrained')

    for axes_col, (_embedding, _title, _model_pred_label) in enumerate(zip(
        [PCs_train, PCs_val],
        [config.TRAIN_LABEL, config.TEST_LABEL],
            [pred_train['label'], predictions['label']])):
        _row = 0
        axes[_row, axes_col].set_title(_title)
        for (i, j) in itertools.combinations(range(max_rows), 2):
            ax = seaborn.scatterplot(
                x=_embedding.iloc[:, i],
                y=_embedding.iloc[:, j],
                hue=_model_pred_label,
                hue_order=['TN', 'TP', 'FN', 'FP'],
                palette=[colors[0], colors[2], colors[1], colors[3]],
                ax=axes[_row, axes_col])
            _row += 1

    fname = FOLDER / f'pca_sel_feat_up_to_{max_rows}.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)


# %% [markdown]
# ### Features
# - top 3 scaled n_features_max (scatter)
# - or unscalled single features (swarmplot)

# %%
if M_sel > 1:
    max_rows = min(3, len(results_model.selected_features))
    fig, axes = plt.subplots(max_rows, 2,
                             figsize=(6, 9),
                             sharex=False,
                             sharey=False,
                             layout='constrained')

    for axes_col, (_embedding, _title, _model_pred_label) in enumerate(zip(
        [X_scaled[results_model.selected_features],
         X_val_scaled[results_model.selected_features]],
        [config.TRAIN_LABEL, config.TEST_LABEL],
            [pred_train['label'], predictions['label']])):
        _row = 0
        axes[_row, axes_col].set_title(_title)
        for (i, j) in itertools.combinations(range(max_rows), 2):
            ax = seaborn.scatterplot(
                x=_embedding.iloc[:, i],
                y=_embedding.iloc[:, j],
                hue=_model_pred_label,
                s=12,
                hue_order=['TN', 'TP', 'FN', 'FP'],
                palette=[colors[0], colors[2], colors[1], colors[3]],
                ax=axes[_row, axes_col])
            # ! manipulate legend globally?
            # ax.legend(title='label', fontsize=7, title_fontsize=8)
            # ! scale each row and each column
            # X_scaled[results_model.selected_features][i]
            # X_val_scaled[results_model.selected_features][j]
            _row += 1

    fname = FOLDER / f'sel_feat_up_to_{max_rows}.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)
else:
    fig, axes = plt.subplots(1, 1,
                             figsize=(6, 2),
                             layout='constrained'
                             )
    single_feature = results_model.selected_features[0]
    data = pd.concat([
        X[single_feature].to_frame().join(
            pred_train['label']).assign(group=config.TRAIN_LABEL),
        X_val[single_feature].to_frame().join(
            predictions['label']).assign(group=config.TEST_LABEL)
    ])
    ax = seaborn.swarmplot(data=data,
                           x='group',
                           y=single_feature,
                           hue='label',
                           ax=axes
                           )
    fname = FOLDER / f'sel_feat_{single_feature}.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)


# %% [markdown]
# ## Annotation of Errors for manuel analysis
#
# -saved to excel table

# %%
X[results_model.selected_features].join(pred_train).to_excel(
    writer, sheet_name='pred_train_annotated', float_format="%.3f")
X_val[results_model.selected_features].join(predictions).to_excel(
    writer, sheet_name='pred_test_annotated', float_format="%.3f")


# %% [markdown]
# # Outputs

# %%
writer.close()

# %%
files_out

# %%
