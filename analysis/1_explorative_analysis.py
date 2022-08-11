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
import numpy as np
import pandas as pd

import pingouin as pg

import src.stats
from src.stats import means_between_groups

import config

# %% [markdown]
# ## Set parameters

# %%
TARGET = 'dead_90'

# %%
clinic = pd.read_pickle(config.fname_pkl_clinic)
olink = pd.read_pickle(config.fname_pkl_olink)

# %%
pd.crosstab(clinic.DiagnosisPlace, clinic.dead)

# %% [markdown]
# FirstAdmission is also right-censored

# %%
time_from_diagnose_to_first_admission = clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) - clinic["DateDiagnose"]
time_from_diagnose_to_first_admission.describe()

# %% [markdown]
# Who dies without having a first Admission date?

# %%
dead_wo_adm = clinic["DateFirstAdmission"].isna() & clinic['dead']
idx_dead_wo_adm = dead_wo_adm.loc[dead_wo_adm].index
print('Dead without admission to hospital:', *dead_wo_adm.loc[dead_wo_adm].index)
clinic.loc[dead_wo_adm, ["DateFirstAdmission", "DateDiagnose", "Admissions"]]

# %% [markdown]
# ## Differences between groups defined by target

# %%
clinic

# %%
happend = clinic[TARGET].astype(bool)

# %%
var = 'Age'
# import scipy.stats 
# scipy.stats.ttest_ind(clinic.loc[happend, var], clinic.loc[~happend, var], equal_var=False) # same results as pengoin
pg.ttest(clinic.loc[happend, var], clinic.loc[~happend, var])

# %%


group_diffs = means_between_groups(clinic, happend, event_names=('died', 'alive'))
group_diffs


# %%
def calc_stats(df:pd.DataFrame, boolean_array:pd.Series, vars:list[str]):
    ret = []
    for var in vars:
        _ = pg.ttest(df.loc[boolean_array, var], df.loc[~boolean_array, var])
        ret.append(_)
    ret = pd.concat(ret)
    ret = ret.set_index(group_diffs.index)
    ret.columns.name = 'ttest'
    ret.columns = pd.MultiIndex.from_product([['ttest'], ret.columns], names=('test', 'var'))
    return ret

ttests = calc_stats(clinic, happend, group_diffs.index)

ttests

# %%
group_diffs.join(ttests.loc[:, pd.IndexSlice[:,["alternative", "p-val", "cohen-d"]]])
