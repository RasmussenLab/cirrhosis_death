# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Raw Data
#
# - join OLink and clinical data
# -

# +
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import src

import config


pd.options.display.max_columns = 100

# +
DATA_FOLDER = Path(config.data)
DATA_PROCESSED = Path(config.data_processed)
list(DATA_FOLDER.iterdir())

config.STUDY_ENDDATE
# -

DATA_CLINIC = DATA_FOLDER / '2022-08-08_clinical_data.xlsx'
DATA_OLINK = DATA_FOLDER / 'QC_OlinkProD_wide.tsv'

clinic = pd.read_excel(DATA_CLINIC)
clinic.SampleID = clinic.SampleID.str.replace(' ', '')
clinic = clinic.set_index('SampleID').sort_index()
clinic

olink = pd.read_table(DATA_OLINK)
olink = olink.set_index(olink.SampleID.str[4:]).sort_index()
olink

# ## Different overlaps

idx_overlap = olink.index.intersection(clinic.index)
idx_overlap

# in clinical data, but not in olink data
clinic.index.difference(olink.index)

# in olink data, but not in clinical data -> excluded samples
olink.index.difference(clinic.index)

# ## Dump feature names

# +
import yaml

with open('config/olink_features.yaml', 'w') as f:
    yaml.dump({k: '' for k in olink.columns.to_list()}, f, sort_keys=False)

with open('config/clinic_features.yaml', 'w') as f:
    yaml.dump({k: '' for k in clinic.columns.to_list()}, f, sort_keys=False)
    
olink.columns.to_series().to_excel('config/olink_features.xlsx')
clinic.columns.to_series().to_excel('config/clinic_features.xlsx')    
# -

# ## Subselect

# +
clinic = clinic.loc[idx_overlap]
olink = olink.loc[idx_overlap]

clinic['dead'] = (clinic['DateDeath'] - clinic['DateDiagnose']).notna()
clinic["DateDeath"] = clinic["DateDeath"].fillna(value=config.STUDY_ENDDATE)
# -

# ## Deaths over time
#
# - one plot with absolute time axis
# - one plot relative to diagnosis date


clinic.describe(datetime_is_numeric=True, include='all')

# +
din_a4 = (8.27 * 2, 11.69 * 2)
fig, ax = plt.subplots(figsize=din_a4)

src.plotting.plot_lifelines(clinic.sort_values('DateDiagnose'), ax=ax)
_ = plt.xticks(rotation=45)
ax.invert_yaxis()
# -

ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateDiagnose", y="dead", c='blue', rot=45)

ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateDiagnose",
                y='DateDeath',
                c="dead",
                rot=45,
                sharex=False)
min_date, max_date = clinic["DateDiagnose"].min(
), clinic["DateDiagnose"].max()
ax.plot([min_date, max_date], [min_date, max_date], 'k-', lw=2)
fig = ax.get_figure()

# ## Kaplan-Meier survival plot

# +
# from lifelines import KaplanMeierFitter
# kmf = KaplanMeierFitter()

# T = kp_plot['DateDeath']
# C = kp_plot['dead']

# kmf.fit(T, event_observed=C)
# -

# ## Dumped processed and selected data

# +
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)

clinic.to_pickle(config.fname_pkl_clinic)
olink.to_pickle(config.fname_pkl_olink)
