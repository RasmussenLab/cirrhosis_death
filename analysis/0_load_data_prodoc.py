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
# - prepare OLink and clinical data
# - create views on data
# - create targets:
#
# event | next 90 days | next 180 days |
# --- | --- | --- |
# death | `dead90` | `dead180` |
# admission to hospital | `adm90`  | `adm180` |
#
# all cases within 90 days will be included into the 180 days, from `incl`usion and from `infl`ammation sample time.

# %%
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import seaborn as sns

from lifelines import KaplanMeierFitter

import src

import config

# %%
DATA_FOLDER = Path(config.data)
DATA_PROCESSED = Path(config.data_processed)
FOLDER_REPORTS = Path(config.folder_reports)
list(DATA_FOLDER.iterdir())

files_out=dict()

config.STUDY_ENDDATE

# %% tags=["parameters"]
DATA_CLINIC = DATA_FOLDER / 'DataSheet - fewer variables_2022-11-24.xlsx'
DATA_OLINK = DATA_FOLDER / 'QC_OlinkProD_wide.tsv'
DATA_OLINK_VAL = DATA_FOLDER / 'olink_prodoc_val.xlsx'

# %% [markdown]
# Load clinical data

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

# %%
olink_val = pd.read_excel(DATA_OLINK_VAL, index_col=0)
olink_val.index = olink_val.index.str[4:].str.replace(' ', '')
olink_val

# %% [markdown]
# ## Deaths over time
#
# - one plot with absolute time axis
# - one plot relative to diagnosis date


# %%
clinic['dead'] = (clinic['DateDeath'] - clinic['DateInflSample']).notna()
clinic["DateDeath"] = clinic["DateDeath"].fillna(value=config.STUDY_ENDDATE)

# %%
din_a4 = (8.27 * 2, 11.69 * 2)
fig, ax = plt.subplots(figsize=din_a4)

src.plotting.plot_lifelines(clinic.sort_values('DateInflSample'), start_col='DateInflSample', ax=ax)
_ = plt.xticks(rotation=45)
ax.invert_yaxis()
files_out['lifelines'] = FOLDER_REPORTS/ 'lifelines.pdf'
fig.savefig(files_out['lifelines'])

# %%
clinic.dead.value_counts()

# %%
fig, axes = plt.subplots(2, sharex=True)
ax =  axes[0]
ax.set_yticks([])

ax = clinic.loc[clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y="dead", c='blue', rot=45, ax=ax, ylabel='dead')
ax =  axes[1]
# ax.axes.yaxis.set_visible(False)
ax.set_yticks([])
ax = clinic.loc[~clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y="dead", c='blue', rot=45, ax=ax, ylabel='alive')
_ = fig.suptitle("Inclusion date by survival status", fontsize=22)
files_out['death_vs_alive_diagonose_dates'] = FOLDER_REPORTS / 'death_vs_alive_diagonose_dates'
fig.savefig(files_out['death_vs_alive_diagonose_dates'])

# %%
ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y='DateDeath', c="dead", rot=45, sharex=False)
# ticks = ax.get_xticks()
# ax.set_xticklabels(ax.get_xticklabels(),  horizontalalignment='right')
# ax.set_xticks(ticks)
min_date, max_date = clinic["DateInflSample"].min(), clinic["DateInflSample"].max()
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
delta=360
ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], rotation=25)
fig = ax.get_figure()
files_out['timing_deaths_over_time'] = FOLDER_REPORTS / 'timing_deaths_over_time.pdf'
fig.savefig(files_out['timing_deaths_over_time'])

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
clinic.loc[:, clinic.columns.str.contains("Adm")].sum()

# %% [markdown]
# Encode binary variables

# %%
# binary variables
vars_binary = config.clinic_data.vars_binary
clinic[vars_binary].head()

# %%
clinic[vars_binary] = clinic[vars_binary].astype('category')

# %% [markdown]
# remaining non numeric variables

# %%
mask_cols_obj = clinic.dtypes == 'object'
clinic.loc[:,mask_cols_obj].describe()

# %%
clinic["HbA1c"] = clinic["HbA1c"].replace(to_replace="(NA)", value=np.nan).astype(pd.Int32Dtype())
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
clinic["EtiNonAlco"] = (clinic["EtiAlco"] == 'No') & (etiology_mask_yes.drop('EtiAlco', axis=1).sum(axis=1).astype(bool))
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
# ## Timespans
#
# - death only has right censoring, no drop-out
# - admission has right censoring, and a few drop-outs who die before their first admission for the cirrhosis

# %% [markdown]
# For these individuals, the admission time is censored as the persons died before.

# %%
clinic["DaysToAdmFromInflSample"] = (
    clinic["DateAdm"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days
clinic["DaysToDeathFromInfl"] = (
    clinic["DateDeath"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days

cols_clinic = src.pandas.get_colums_accessor(clinic)

cols_view = [
    cols_clinic.DaysToDeathFromInfl,
    cols_clinic.DateAdm,
    # cols_clinic.DateFirstAdmission,
    cols_clinic.DaysToAdmFromInflSample,
    "dead",
    "Adm90", "Adm180",
    "Age"
]

mask = (clinic.dead == True) & (clinic.Adm180.isna())
view = clinic.loc[mask, cols_view].sort_values(cols_view)
files_out['died_before_admission'] = FOLDER_REPORTS / 'died_before_adm.xlsx'
view.to_excel(files_out['died_before_admission'])
view

# %% [markdown]
# ## Kaplan-Meier survival plot

# %%
kmf = KaplanMeierFitter()
kmf.fit(clinic["DaysToDeathFromInfl"], event_observed=clinic["dead"])

fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(  #title='Kaplan Meier survival curve since inclusion',
    xlim=(0, None),
    ylim=y_lim,
    xlabel='Days since inflammation sample',
    ylabel='survival rate',
    ax=ax,
    legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
files_out['km_plot_death'] = FOLDER_REPORTS / 'km_plot_death.pdf'
fig.savefig(files_out['km_plot_death'])

# %%
_ = sns.catplot(x="DaysToDeathFromInfl",
                y="dead",
                hue="DiagnosisPlace",
                data=clinic.astype({'dead': 'category'}),
                height=4,
                aspect=3)
_.set_xlabels('Days from inflammation sample to death or until study end')
ax = _.fig.get_axes()[0]
ylim = ax.get_ylim()
ax.vlines(90, *ylim)
ax.vlines(180, *ylim)
fig = ax.get_figure()
files_out['deaths_along_time'] = FOLDER_REPORTS / 'deaths_along_time.pdf'
fig.savefig(files_out['deaths_along_time'])

# %% [markdown]
# ## KP plot admissions

# %%
kmf = KaplanMeierFitter()
kmf.fit(clinic["DaysToDeathFromInfl"], event_observed=clinic["LiverAdm180"].fillna(0))


fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(#title='Kaplan Meier curve for liver related admissions',
              xlim=(0, None),
              ylim=(0, 1),
              xlabel='Days since inflammation sample',
              ylabel='remaining with non-liver related admission',
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
fig = ax.get_figure()
files_out['km_plot_admission'] = FOLDER_REPORTS / 'km_plot_admission.pdf'
fig.savefig(files_out['km_plot_admission'])

# %% [markdown]
# ## Targets

# %%
mask = clinic.columns.str.contains("(90|180)")
clinic.loc[:,mask] = clinic.loc[:,mask].fillna(0)
clinic.loc[:,mask].describe()

# %%
mask = clinic.columns.str.contains("Adm(90|180)")
clinic.loc[:,mask].describe() # four targets for liver related admissions

# %%
target_name = {k:f'has{k}' for k in clinic.columns[mask]}
target_name

# %%
targets = {}

for cutoff in [90, 180]:
    targets[f'dead{cutoff:03}infl'] = (clinic["DaysToDeathFromInfl"] <=
                                       cutoff).astype(int)
    targets[f"liverDead{cutoff:03}infl"] = (
        clinic.loc[clinic["CauseOfDeath"] != 'NonLiver',
                   "DaysToDeathFromInfl"] <= cutoff).astype(int)

targets = pd.DataFrame(targets)
targets = targets.join(
    (clinic.loc[:, mask] > 0).astype(int).rename(columns=target_name))
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
# ## Censoring

# %% [markdown]
# FirstAdmission is also right-censored

# %%
time_from_inclusion_to_first_admission = clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) - clinic["DateInflSample"]
time_from_inclusion_to_first_admission.describe()

# %% [markdown]
# Who dies without having a first Admission date?

# %%
dead_wo_adm = clinic["DateFirstAdmission"].isna() & clinic['dead']
idx_dead_wo_adm = dead_wo_adm.loc[dead_wo_adm].index
print('Dead without admission to hospital:', *dead_wo_adm.loc[dead_wo_adm].index)
clinic.loc[dead_wo_adm, ["DateFirstAdmission", "DateInflSample", cols_clinic.LiverAdm180]]

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
idx_validation = idx_validation.intersection(olink_val.index)
idx_validation

# %%
# in olink data, but not in clinical data -> excluded samples
olink.index.difference(clinic.index)

# %% [markdown]
# ## Save training cohort separately

# %%
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
files_out[config.fname_pkl_clinic.stem] = config.fname_pkl_clinic
files_out[config.fname_pkl_olink.stem] = config.fname_pkl_olink
clinic.loc[idx_overlap].to_pickle(config.fname_pkl_clinic)
olink.loc[idx_overlap].to_pickle(config.fname_pkl_olink)

# %% [markdown]
# ## Save validation cohort separately

# %%
files_out[config.fname_pkl_val_clinic.stem] = config.fname_pkl_val_clinic
clinic.loc[idx_validation].to_pickle(config.fname_pkl_val_clinic)

# %%
# potentially save to yaml to be used elsewhere (or json)
','.join(idx_validation)

# %% [markdown]
# ## Dump combined data for comparision


# %%
idx_valid_proDoc = [*idx_overlap, *idx_validation]
clinic = clinic.loc[idx_valid_proDoc]
clinic[config.COMPARE_PRODOC] = clinic.index.isin(idx_validation).astype('float')
files_out[config.fname_pkl_prodoc_clinic.stem] = config.fname_pkl_prodoc_clinic
clinic.to_pickle(config.fname_pkl_prodoc_clinic)
olink = pd.concat(
    [olink.loc[idx_overlap], olink_val.loc[idx_validation]])
files_out[config.fname_pkl_prodoc_olink.stem] = config.fname_pkl_prodoc_olink
olink.to_pickle(config.fname_pkl_prodoc_olink)


# %%
files_out[config.fname_pkl_targets.stem] = config.fname_pkl_targets
targets.to_pickle(config.fname_pkl_targets)
clinic[targets.columns].describe()

# %% [markdown]
# ## Dumped combined clinical data as numeric data for ML applications

# %%
cols_cat = clinic.dtypes == 'category'
cols_cat = clinic.columns[cols_cat]
clinic[cols_cat]

# %%
encode = {**{k:1 for k in ['Yes', 'yes', 'Male']},
          **{k:0 for k in ['No', 'no', 'Female']}}
encode

# %%
clinic[cols_cat] = clinic[cols_cat].astype('object').replace(encode)
clinic[cols_cat]

# %%
clinic[cols_cat].describe()

# %% [markdown]
# The martial status was made into three dummy variables before (see above):
# `MaritalStatus_Divorced, MaritalStatus_Married, MaritalStatus_Relationship, MaritalStatus_Separated, MaritalStatus_Unmarried, MaritalStatus_Widow/widower`

# %%
mask = clinic.dtypes == 'object'
clinic.loc[:, mask]

# %%
mask = clinic.dtypes == 'datetime64[ns]'
clinic.loc[:,mask]

# %%
numeric_cols = (clinic.apply(pd.api.types.is_numeric_dtype))
numeric_cols = clinic.columns[numeric_cols]
clinic[numeric_cols]

# %%
clinic[numeric_cols].dtypes.value_counts()

# %%
to_drop = ["Study ID"]
clinic[numeric_cols].drop(to_drop, axis=1)

# %%
files_out[config.fname_pkl_prodoc_clinic_num.stem] = config.fname_pkl_prodoc_clinic_num
clinic[numeric_cols].drop(to_drop, axis=1).to_pickle(config.fname_pkl_prodoc_clinic_num)

# %% [markdown]
# ## Files saved by notebook

# %%
src.io.print_files(files_out)
