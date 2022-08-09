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

import config

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.min_rows = 50

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
