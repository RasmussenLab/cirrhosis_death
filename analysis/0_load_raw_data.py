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
# - create targets: 
#     
#   event | next 90 days | next 180 days
#   --- | --- | ---
#   death | `dead_90` | `dead_180`
#   admission to hospital | `adm_90`  | `adm_180`
#     
#   all cases within 90 days will be included into the 180 days

# +
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lifelines import KaplanMeierFitter

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

clinic.dead

# +
fig, axes = plt.subplots(2, sharex=True)
ax =  axes[0]
ax.set_yticks([])

ax = clinic.loc[clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateDiagnose", y="dead", c='blue', rot=45, ax=ax, ylabel='dead')
ax =  axes[1]
# ax.axes.yaxis.set_visible(False)
ax.set_yticks([])
ax = clinic.loc[~clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateDiagnose", y="dead", c='blue', rot=45, ax=ax, ylabel='alive')
_ = fig.suptitle("Diagnose date by survival status", fontsize=22)
# -

ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateDiagnose", y='DateDeath', c="dead", rot=45, sharex=False)
min_date, max_date = clinic["DateDiagnose"].min(), clinic["DateDiagnose"].max()
ax.plot([min_date, max_date], [min_date, max_date], 'k-', lw=2)
fig = ax.get_figure()

# ## Cleanup steps

# ### Clinic
#
# - [x] encode binary variables (yes, no) as `category`s 
#   > Be aware that this might cause unexpected behaviour!

# Fill derived variables with missing measurements

# fill missing Admissions with zero, and make it an integer
clinic["Admissions"] = clinic["Admissions"].fillna(0).astype(int)

# Encode binary variables

# binary variables
vars_binary = [
    'Sex',
    'EtiAlco',
    'EtiFat',
    'EtiHBV',
    'EtiHCV',
    'EtiPBC',
    'EtiAIH',
    'EtiMTX',
    'EtiOther',
    'EtiUnknown',
    'DecomensatedAtDiagnosis',
    'Ascites',
    'EsoBleeding',
    'HRS',
    'HE',
    'Icterus',
    'Heartdisease',
    'Hypertension',
    'HighCholesterol',
    'Cancer',
    'Depression',
    'Psychiatric',
    'Diabetes',
    'IschemicHeart',
    'HeartFailure',
    'Arrythmia',
    'OtherHeart',
    'TypeDiabetes',
    'InsulinDependent',
    # 'OtherComorb',
    'Statins'
]
clinic[vars_binary].head()
# clinic.columns.to_list()

clinic[vars_binary] = clinic[vars_binary].astype('category')

# ### Olink
#
# - [x] remove additional meta data
# - [x] highlight missing values
#

# 
olink.head()

# Remove additional metadata

olink = olink.loc[:,'IL8':]

# Which measurments have missing values
#
# - [ ] Imputation due to limit of detection (LOD) -> how to best impute

olink.loc[:, olink.isna().any()].describe() 

# ## Targets
#
# - death only has right censoring, no drop-out
# - admission has right censoring, and a few drop-outs who die before their first admission for the cirrhosis

# +
clinic["TimeToAdmFromDiagnose"] = (
    clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) -
    clinic["DateDiagnose"]).dt.days
clinic["TimeToDeathFromDiagnose"] = (
    clinic["DateDeath"].fillna(config.STUDY_ENDDATE) -
    clinic["DateDiagnose"]).dt.days

mask = clinic["TimeToDeathFromDiagnose"] < clinic["TimeToAdmFromDiagnose"]
cols_view = [
    "TimeToDeathFromDiagnose", "TimeToAdmFromDiagnose", "dead", "Admissions"
]
clinic[cols_view].loc[mask]
# -

# For these individuals, the diagnose time is censored as the persons died before.

clinic.loc[mask,
           "TimeToAdmFromDiagnose"] = clinic.loc[mask,
                                                 "TimeToDeathFromDiagnose"]
clinic.loc[mask, cols_view]

# ### Kaplan-Meier survival plot 

# +
kmf = KaplanMeierFitter()
kmf.fit(clinic["TimeToDeathFromDiagnose"], event_observed=clinic["dead"])

y_lim = (0, 1)
ax = kmf.plot(title='Kaplan Meier survival curve since diagnose',
              xlim=(0, None),
              ylim=(0, 1),
              xlabel='Time since diagnose',
              ylabel='survival rate',
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
# -

_ = sns.catplot(x="TimeToDeathFromDiagnose",
                y="dead",
                hue="DiagnosisPlace",
                data=clinic.astype({'dead': 'category'}),
                height=4,
                aspect=3)
ax = _.fig.get_axes()[0]
ylim = ax.get_ylim()
ax.vlines(90, *ylim)
ax.vlines(180, *ylim)

# ### KP plot admissions

# +
kmf = KaplanMeierFitter()
kmf.fit(clinic["TimeToAdmFromDiagnose"], event_observed=clinic['Admissions'])

y_lim = (0, 1)
ax = kmf.plot(title='Kaplan Meier curve for admissions',
              xlim=(0, None),
              ylim=(0, 1),
              xlabel='Time since diagnose',
              ylabel='non-admission rate',
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
# -

# ### Build targets

# +
targets = {}

for cutoff in [90, 180]:
    targets[f'dead_{cutoff}'] = (clinic["TimeToDeath"] <= cutoff).astype(int)
    targets[f'adm_{cutoff}'] = (clinic["TimeToAdmFromDiagnose"] <=
                                cutoff).astype(int)
targets = pd.DataFrame(targets)
targets = targets.sort_index(axis=1, ascending=False)
targets.head()

# +
from src.pandas import combine_value_counts

combine_value_counts(targets)

# +
ret = []
for var in targets.columns:
    _ = pd.crosstab(targets[var], clinic.DiagnosisPlace)
    _.index = [f'{var.replace("_", " <= ")} - {i}' for i in _.index]
    ret.append(_)
ret = pd.concat(ret)

tab_targets_by_diagnosisPlace = ret
tab_targets_by_diagnosisPlace
# -

# add to clinical targets

clinic = clinic.join(targets)

# ## Dumped processed and selected data

# +
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)

clinic.to_pickle(config.fname_pkl_clinic)
olink.to_pickle(config.fname_pkl_olink)
targets.to_pickle(config.fname_pkl_targets)
# -


