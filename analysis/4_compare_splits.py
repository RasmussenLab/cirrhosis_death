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
fname_pkl_clinic = config.fname_pkl_all_clinic_num
fname_pkl_olink = config.fname_pkl_all_olink

TARGET = 'hasLiverAdm180'
FOLDER = ''
feat_set_to_consider:str='OLINK_AND_CLINIC'
VAL_IDS=''
VAL_IDS_query = "Cflow"
name_report = 'train_val_comparison'

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True)
else:
    FOLDER = Path(FOLDER)
FOLDER

# %% [markdown]
# ## Read data

# %%
data = pd.read_pickle(fname_pkl_clinic).join(pd.read_pickle(fname_pkl_olink))
data

# cols = src.pandas.get_colums_accessor(clinic)

# %%
test_ids = src.find_val_ids(data, val_ids=VAL_IDS, val_ids_query=VAL_IDS_query)
# val_ids

# %% [markdown]
# retain entries with only non-missing targets

# %%
test_split = data.loc[test_ids]
train_split = data.drop(test_ids)
train_split.shape, test_split.shape

# %%
train_split = train_split.dropna(subset=[TARGET])
test_split = test_split.dropna(subset=[TARGET])
train_split.shape, test_split.shape

# %%
# def find_unique(df:pd.DataFrame) -> pd.Index:
#     drop_cols = df.describe(include='all').loc['unique'] == 1
#     drop_cols = df.columns[drop_cols]
#     return drop_cols

# drop_cols = find_unique(test_split)
# test_split[drop_cols].describe(include='all') if not test_split[drop_cols].empty else "None"

# %%
# drop_cols = find_unique(train_split)
# train_split[drop_cols].describe(include='all') if not train_split[drop_cols].empty else "None"

# %%
# test_split = test_split.drop(drop_cols, axis=1)
# train_split = train_split.drop(drop_cols, axis=1)

# %% [markdown]
# ## Create Report

# %%
sweetviz_report = sweetviz.compare([train_split, 'training data'],
                                   [test_split, 'test data'],
                                   target_feat=TARGET,
                                   pairwise_analysis='off')
sweetviz_report.show_html(filepath=FOLDER / f'{name_report}.html')
