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

# %% [markdown] tags=[]
# # Compare data splits
#
# - compare two data splits by a binary target variable

# %%
from pathlib import Path
import pandas as pd
import sweetviz

import src

import config

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
fname_pkl_train = config.fname_pkl_clinic
fname_pkl_val = config.fname_pkl_val_clinic

TARGET = 'liverDead090infl'
FOLDER = ''

name_report = 'train_val_comparison'

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True)
else:
    FOLDER = Path(FOLDER)

# %% [markdown]
# ## Read data

# %%
train_split = pd.read_pickle(fname_pkl_train)
cols_train = src.pandas.get_colums_accessor(train_split)

val_split = pd.read_pickle(fname_pkl_val)

# %% [markdown]
# retain entries with only non-missing targets

# %%
train_split = train_split.dropna(subset=[TARGET])
val_split = val_split.dropna(subset=[TARGET])

# %% [markdown]
# ## Create Report

# %%
sweetviz_report = sweetviz.compare([train_split, 'training data'],
                                   [val_split, 'validation data'],
                                   target_feat=TARGET,
                                   pairwise_analysis='off')
sweetviz_report.show_html(filepath=FOLDER / f'{name_report}.html')
