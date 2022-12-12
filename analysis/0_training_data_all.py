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
# # Combine cohorts: ProDoc and CircaFlow
#
# - ProDoc should be used for training
# - CircaFlow should be used for testing

# %%
import config
import pandas as pd

# %%
files_in = {'clinic': {p.stem: p for p in [config.fname_pkl_prodoc_clinic_num, config.fname_pkl_cirkaflow_clinic_num]}}
files_in['olink'] = {p.stem: p for p in [config.fname_pkl_prodoc_olink, config.fname_pkl_cirkaflow_olink]}
files_in

# %%
# not generic
clinic = pd.read_pickle(config.fname_pkl_prodoc_clinic_num)
clinic_cirkaflow = pd.read_pickle(config.fname_pkl_cirkaflow_clinic_num)

feat_in_both = clinic.columns.intersection(clinic_cirkaflow.columns)
feat_in_both

# %%
clinic = pd.concat([clinic[feat_in_both], clinic_cirkaflow[feat_in_both]])
clinic

# %%
olink = files_in['olink'].values()
olink = pd.concat(map(pd.read_pickle, olink))
olink

# %%
files_out = {'olink': config.fname_pkl_all_olink, 
            'clinic': config.fname_pkl_all_clinic_num}
files_out

# %%
clinic.to_pickle(files_out['clinic'])
olink.to_pickle(files_out['olink'])
