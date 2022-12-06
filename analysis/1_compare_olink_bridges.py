# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Explorative Analysis

# %%
from functools import partial
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve

import src
import njab.plotting
from njab.sklearn import run_pca, StandardScaler
from njab.sklearn.scoring import ConfusionMatrix

import config
import njab

# %% [markdown]
# # Set parameters

# %%
TARGET = 'project'
FOLDER = Path(config.folder_reports) / 'bridges'
OLINK=config.fname_pkl_prodoc_olink

# %%
inputs = dict()
inputs['bridging_samples'] = config.data_processed / 'bridges.pkl'
olink_bridge = pd.read_pickle(inputs['bridging_samples'])
olink_bridge.sample(10)

# %%
olink_bridge = olink_bridge.reorder_levels(['Project', 'SampleID', 'Assay'])
olink_bridge.sample(2)

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %% [markdown]
# ## Compare subsets

# %%
olink_bridge = olink_bridge['NPX'].unstack()
olink_bridge

# %% [markdown]
# # Differences between two batches
#
# - create dummy to indicate

# %%
badge_tag = pd.Series(1, olink_bridge.index, name='batch')
badge_tag.loc['20202249'] = 0
badge_tag

# %%
happend = badge_tag.astype(bool)

# %% [markdown]
# ## Olink - uncontrolled

# %%
olink = olink_bridge
olink

# %%
assert olink.isna().sum().sum() == 0
# olink.loc[:, olink.isna().any()].describe()

# %%
ana_diff_olink = njab.stats.groups_comparision.diff_analysis(olink,
                                         happend,
                                         event_names=('2nd batch',
                                                      '1st batch')).sort_values(
                                                          ('ttest', 'p-val'))
ana_diff_olink.to_excel(FOLDER / "DA_batches.xlsx")

ana_diff_olink.head(20)


# %% [markdown]
# # PCA

# %% [markdown]
# ## Missing values handling


# %%
def info_missing(df):
    N, M = olink.shape
    msg = "{} missing features out of {} measurments, corresponding to {:.3f}%"
    msg = msg.format(df.isna().sum().sum(), N * M,
                     df.isna().sum().sum() / (N * M) * 100)
    print(msg)
    return msg


_ = info_missing(olink)

# %% [markdown]
# ## PCA on scaled data
#
# - missing values set to zero

# %%
olink_scaled = StandardScaler().fit_transform(olink).fillna(0)

PCs, pca = run_pca(olink_scaled, n_components=None)
PCs.iloc[:10, :10]

# %%
olink.columns[np.argmax(np.abs(
    pca.components_[:,
                    0]))]  # eigenvector first PCa, absolut arg max -> variable

# %%
exp_var_olink = pd.Series(
    pca.explained_variance_ratio_).to_frame('explained variance')
exp_var_olink["explained variance (cummulated)"] = exp_var_olink[
    'explained variance'].cumsum()
exp_var_olink.index.name = 'PC'
ax = exp_var_olink.plot()
fig = ax.get_figure()
njab.plotting.savefig(fig, name=FOLDER / '1_PCs_distribution')

# %%
ax = seaborn.scatterplot(x=PCs.iloc[:, 0],
                         y=PCs.iloc[:, 1],
                         hue=badge_tag)
fig = ax.get_figure()
njab.plotting.savefig(fig, name=FOLDER / '1_PC1_vs_PC2.pdf')
