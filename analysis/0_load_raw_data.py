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
# # Raw Data
#
# - join OLink and clinical data
# - create targets: 
#     
# event | next 90 days | next 180 days |
# --- | --- | --- |
# death | `dead_90` | `dead_180` | 
# admission to hospital | `adm_90`  | `adm_180` |
#     
# all cases within 90 days will be included into the 180 days

# %%
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import seaborn as sns

from lifelines import KaplanMeierFitter

import src

import config

# %%
DATA_FOLDER = Path(config.data)
DATA_PROCESSED = Path(config.data_processed)
FOLDER_REPORTS = Path(config.folder_reports)
list(DATA_FOLDER.iterdir())

config.STUDY_ENDDATE

# %%
DATA_CLINIC = DATA_FOLDER / '2022-09-09_clinical_data.xlsx'
DATA_OLINK = DATA_FOLDER / 'QC_OlinkProD_wide.tsv'

# %%
clinic = pd.read_excel(DATA_CLINIC)
clinic.SampleID = clinic.SampleID.str.replace(' ', '')
cols_clinic = src.pandas.get_colums_accessor(clinic)
clinic = clinic.set_index('SampleID').sort_index()

# %%
# clinic
clinic.describe(datetime_is_numeric=True, include='all')

# %%
olink = pd.read_table(DATA_OLINK)
olink = olink.set_index(olink.SampleID.str[4:]).sort_index()
cols_olink = src.pandas.get_colums_accessor(olink)

# %%
# olink
olink.describe(datetime_is_numeric=True, include='all')

# %% [markdown]
# ## Dump feature names
#
# - maybe with data type information?

# %%
# import yaml

# with open('config/olink_features.yaml', 'w') as f:
#     yaml.dump({k: '' for k in olink.columns.to_list()}, f, sort_keys=False)

# with open('config/clinic_features.yaml', 'w') as f:
#     yaml.dump({k: '' for k in clinic.columns.to_list()}, f, sort_keys=False)

# olink.columns.to_series().to_excel('config/olink_features.xlsx')
# clinic.columns.to_series().to_excel('config/clinic_features.xlsx')

# %% [markdown]
# ## Deaths over time
#
# - one plot with absolute time axis
# - one plot relative to diagnosis date


# %%
clinic['dead'] = (clinic['DateDeath'] - clinic['DateInclusion']).notna()
clinic["DateDeath"] = clinic["DateDeath"].fillna(value=config.STUDY_ENDDATE)

# %%
din_a4 = (8.27 * 2, 11.69 * 2)
fig, ax = plt.subplots(figsize=din_a4)

src.plotting.plot_lifelines(clinic.sort_values('DateInclusion'), start_col='DateInclusion', ax=ax)
_ = plt.xticks(rotation=45)
ax.invert_yaxis()
fig.savefig(FOLDER_REPORTS/ 'lifelines.pdf')
fig

# %%
clinic.dead.value_counts()

# %%
fig, axes = plt.subplots(2, sharex=True)
ax =  axes[0]
ax.set_yticks([])

ax = clinic.loc[clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInclusion", y="dead", c='blue', rot=45, ax=ax, ylabel='dead')
ax =  axes[1]
# ax.axes.yaxis.set_visible(False)
ax.set_yticks([])
ax = clinic.loc[~clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInclusion", y="dead", c='blue', rot=45, ax=ax, ylabel='alive')
_ = fig.suptitle("Inclusion date by survival status", fontsize=22)
fig.savefig(FOLDER_REPORTS / 'death_vs_alive_diagonose_dates')
fig

# %%
ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateInclusion", y='DateDeath', c="dead", rot=45, sharex=False)
# ticks = ax.get_xticks()
# ax.set_xticklabels(ax.get_xticklabels(),  horizontalalignment='right')
# ax.set_xticks(ticks)
min_date, max_date = clinic["DateInclusion"].min(), clinic["DateInclusion"].max()
ax.plot([min_date, max_date],
        [min_date, max_date],
        'k-', lw=2)
_ = ax.annotate('date', [min_date, min_date + datetime.timedelta(days=20)], rotation=25)
offset, rot = 20 , 25
delta=90
_ = ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], rotation=25)
delta=180
ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], rotation=25)
fig = ax.get_figure()
fig.savefig(FOLDER_REPORTS / 'timing_deaths_over_time.pdf')
fig

# %% [markdown]
# ## Cleanup steps

# %% [markdown]
# ### Clinic
#
# - [x] encode binary variables (yes, no) as `category`s 
#   > Be aware that this might cause unexpected behaviour!

# %% [markdown]
# Fill derived variables with missing measurements

# %%
clinic.loc[:, clinic.columns.str.contains("Adm")].describe()

# %%
# fill missing Admissions with zero, and make it an integer
# clinic["Admissions"] = clinic["Admissions"].fillna(0).astype(int)
clinic["AmountLiverRelatedAdm"] = clinic["AmountLiverRelatedAdm"].fillna(0).astype(int)

# %% [markdown]
# Encode binary variables

# %%
# binary variables
vars_binary = config.clinic_data.vars_binary
clinic[vars_binary].head()
# clinic.columns.to_list()

# %%
clinic[vars_binary] = clinic[vars_binary].astype('category')

# %% [markdown]
# remaining non numeric variables

# %%
mask_cols_obj = clinic.dtypes == 'object'
clinic.loc[:,mask_cols_obj].describe()

# %%
clinic["HbA1c"] = clinic["HbA1c"].replace(to_replace="(NA)", value=np.nan).astype(pd.Int32Dtype())
clinic["LiverRelated1admFromInclu"] = clinic["LiverRelated1admFromInclu"].replace('x', 1).fillna(0).astype('category')
clinic["MaritalStatus"] = clinic["MaritalStatus"].astype('category')
clinic["HeartDiseaseTotal"] = clinic["HeartDiseaseTotal"].replace(0, 'no').astype('category')
clinic.loc[:,mask_cols_obj].describe(include='all')


# %%
def get_dummies_yes_no(s, prefix=None):
    return pd.get_dummies(s, prefix=prefix).replace({
        0: 'No',
        1: 'Yes'
    }).astype('category')
    
clinic = clinic.join(get_dummies_yes_no(clinic["DiagnosisPlace"]))
clinic = clinic.join(get_dummies_yes_no(clinic["MaritalStatus"], prefix='MaritalStatus'))
clinic = clinic.join(get_dummies_yes_no(clinic["CauseOfDeath"], prefix='CoD'))
clinic

# %% [markdown]
# - few have more than one etiology

# %%
etiology_mask_yes = clinic.loc[:, clinic.columns.str.contains("Eti")] == 'Yes'
etiology_mask_yes.sum(axis=1).value_counts()

# %%
etiology_mask_yes.drop('EtiAlco', axis=1).sum(axis=1).astype(bool)

# %%
clinic["EtiNonAlco"] = (clinic["EtiAlco"] == 'No') & (etiology_mask_yes.drop('EtiAlco', axis=1).sum(axis=1).astype(bool))
#clinic["EtiNonAlco"] = clinic["EtiNonAlco"].replace({False: 'No', True: 'Yes'}).astype('category')
clinic["EtiNonAlco"] = get_dummies_yes_no(clinic["EtiNonAlco"])[True]
clinic["EtiNonAlco"].value_counts()

# %% [markdown]
# ### Olink
#
# - [x] remove additional meta data
# - [x] highlight missing values
#

# %%
olink.head()

# %% [markdown]
# Remove additional metadata

# %%
olink = olink.loc[:,'IL8':]

# %% [markdown]
# Which measurments have missing values
#
# - [ ] Imputation due to limit of detection (LOD) -> how to best impute

# %%
olink.loc[:, olink.isna().any()].describe() 

# %% [markdown]
# ## Targets
#
# - death only has right censoring, no drop-out
# - admission has right censoring, and a few drop-outs who die before their first admission for the cirrhosis

# %%
clinic["TimeToAdmFromInclusion"] = (
    clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInclusion"]).dt.days
clinic["TimeToDeathFromInclusion"] = (
    clinic["DateDeath"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInclusion"]).dt.days

mask = clinic["TimeToDeathFromInclusion"] < clinic["TimeToAdmFromInclusion"]
cols_view = [
    "TimeToDeathFromInclusion", "TimeToAdmFromInclusion", "dead", cols_clinic.AmountLiverRelatedAdm, "Age"
]
clinic[cols_view].loc[mask]

# %% [markdown]
# For these individuals, the admission time is censored as the persons died before.

# %%
clinic.loc[mask,
           "TimeToAdmFromInclusion"] = clinic.loc[mask,
                                                 "TimeToDeathFromInclusion"]
clinic.loc[mask, cols_view]

# %%
clinic["TimeToAdmFromInflSample"] = (
    clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days
clinic["TimeToDeathFromInfl"] = (
    clinic["DateDeath"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days

cols_clinic = src.pandas.get_colums_accessor(clinic)

cols_view = [
    "TimeToDeathFromInclusion", cols_clinic.TimeToDeathFromInfl,
    "TimeToAdmFromInclusion", cols_clinic.TimeToAdmFromInflSample, "dead",
    "AmountLiverRelatedAdm", "Age"
]
mask = (clinic[cols_view] < 0).any(axis=1)
clinic[cols_view].loc[mask]

# %%
clinic[cols_view].describe()

# %%
clinic[cols_view].dtypes

# %% [markdown]
# ### Kaplan-Meier survival plot 

# %%
kmf = KaplanMeierFitter()
kmf.fit(clinic["TimeToDeathFromInclusion"], event_observed=clinic["dead"])

fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(#title='Kaplan Meier survival curve since inclusion',
              xlim=(0, None),
              ylim=y_lim,
              xlabel='Days since inclusion',
              ylabel='survival rate',
              ax=ax,
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
fig.savefig(FOLDER_REPORTS / 'km_plot_death.pdf')
fig

# %%
_ = sns.catplot(x="TimeToDeathFromInclusion",
                y="dead",
                hue="DiagnosisPlace",
                data=clinic.astype({'dead': 'category'}),
                height=4,
                aspect=3)
_.set_xlabels('Days from inclusion to death or until study end')
ax = _.fig.get_axes()[0]
ylim = ax.get_ylim()
ax.vlines(90, *ylim)
ax.vlines(180, *ylim)
fig = ax.get_figure()
fig.savefig(FOLDER_REPORTS / 'deaths_along_time.pdf')
fig

# %% [markdown]
# ### KP plot admissions

# %%
kmf = KaplanMeierFitter()
kmf.fit(clinic["TimeToAdmFromInclusion"], event_observed=clinic["AmountLiverRelatedAdm"])


fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(#title='Kaplan Meier curve for liver related admissions',
              xlim=(0, None),
              ylim=(0, 1),
              xlabel='Days since inclusion',
              ylabel='remaining with non-liver related admission',
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
fig = ax.get_figure()
fig.savefig(FOLDER_REPORTS / 'km_plot_admission.pdf')
fig

# %% [markdown]
# ### Build targets

# %%
mask = clinic.columns.str.contains("(90|180)")
clinic.loc[:,mask] = clinic.loc[:,mask].fillna(0)
clinic.loc[:,mask].describe()

# %%
mask = clinic.columns.str.contains("LiverAdm(90|180)")
clinic.loc[:,clinic.columns.str.contains("LiverAdm(90|180)")].describe() # four targets for liver related admissions

# %%
{k:k for k in clinic.columns[mask]}

# %%
targets = {}

for cutoff in [90, 180]:
    targets[f'dead{cutoff}incl'] = (clinic["TimeToDeathFromInclusion"] <=
                                    cutoff).astype(int)
    targets[f'dead{cutoff}infl'] = (clinic["TimeToDeathFromInfl"] <=
                                    cutoff).astype(int)
targets = pd.DataFrame(targets)
targets = targets.join((clinic.loc[:, mask] > 0).astype(int).rename(
    {
        'LiverAdm90infl': 'hasLiverAdm90infl',
        'LiverAdm180infl': 'hasLiverAdm180infl',
        'LiverAdm90incl': 'hasLiverAdm90incl',
        'LiverAdm180incl': 'hasLiverAdm180incl'
    },
    axis=1))
# targets = targets.sort_index(axis=1, ascending=False)
targets.describe()

# %%
from src.pandas import combine_value_counts
combine_value_counts(targets)

# %%
ret = []
for var in targets.columns:
    _ = pd.crosstab(targets[var], clinic.DiagnosisPlace)
    _.index = [f'{var.replace("_", " <= ", 1)} - {i}' for i in _.index]
    ret.append(_)
ret = pd.concat(ret)

tab_targets_by_diagnosisPlace = ret
tab_targets_by_diagnosisPlace

# %% [markdown]
# add to clinical targets

# %%
clinic = clinic.join(targets)

# %% [markdown]
# ## Different overlaps
#
# - save persons with clinical data as potential validation cohort separately
# - done after preprocessing of data
#

# %%
idx_overlap = olink.index.intersection(clinic.index)
idx_overlap

# %%
# in clinical data, but not in olink data
idx_validation = clinic.index.difference(olink.index)
idx_validation

# %%
# in olink data, but not in clinical data -> excluded samples
olink.index.difference(clinic.index)

# %% [markdown]
# ## Save validation cohort separately

# %%
clinic.loc[idx_validation].to_pickle(config.fname_pkl_val_clinic)

# %% [markdown]
# ## Dumped processed and selected data

# %%
clinic = clinic.loc[idx_overlap]
olink = olink.loc[idx_overlap]

# %%
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)

clinic.to_pickle(config.fname_pkl_clinic)
olink.to_pickle(config.fname_pkl_olink)
targets.to_pickle(config.fname_pkl_targets)
