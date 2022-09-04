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
# # Outliers in clinic data
#
# - base outlier detection on interquartile range (IQR) for continuous features
#   - set fraction for deviation from IQR wrt. to 1st and 3rd quartile 
# - collect sample IDs with ouliers and respective values
# - join into new data view and dump to disk

# %%
import numpy as np
import pandas as pd

import src

import config

# %%
clinic = pd.read_pickle(config.fname_pkl_clinic).reset_index()
cols_clinic = src.pandas.get_colums_accessor(clinic)
clinic = clinic.set_index(cols_clinic.Study_ID)


# %% [markdown]
# ## Parameters

# %%
IQR_FACTOR = 1.5
EXCLUDED = 'TimeToDeath,TimeToAdmFromDiagnose,TimeToAdmFromSample,TimeToDeathFromDiagnose'

# %%
vars_cont_sel = [x for x in config.clinic_data.vars_cont if x not in EXCLUDED.split(',')]
cont_des = clinic[vars_cont_sel].describe()
cont_des

# %% [markdown]
# ## Find outliers
# Aim: Identify all dots in boxplot

# %%
ax = clinic[vars_cont_sel].boxplot(rot=90, whis=IQR_FACTOR)
fig = ax.get_figure()
fig.savefig(f"{config.folder_reports}/outlier_boxplot_iqr_factor_{IQR_FACTOR}.pdf")

# %%
cont_des.loc['iqr'] = cont_des.loc['75%'] - cont_des.loc['25%']
cont_des.loc['val_min'] = cont_des.loc['25%'] - IQR_FACTOR * cont_des.loc['iqr']
cont_des.loc['val_max'] = cont_des.loc['75%'] + IQR_FACTOR * cont_des.loc['iqr']
cont_des

# %%
cont_des.to_excel(f"{config.folder_reports}/clinic_cont_described.xlsx")

# %%
mask = (clinic[vars_cont_sel] < cont_des.loc['val_min']) | (clinic[vars_cont_sel] > cont_des.loc['val_max'])
msg = "Total number of outlier values: {}"
print(msg.format(mask.sum().sum()))

# %%
outliers = clinic[vars_cont_sel][mask].dropna(axis=0, how='all').dropna(axis=1, how='all')
outliers = outliers.style.format( na_rep='-', precision=2)
with pd.option_context('display.max_rows', len(outliers.data)):
    display(outliers)

# %%
outliers.to_excel(f'{config.folder_reports}/outliers.xlsx')

# %%
