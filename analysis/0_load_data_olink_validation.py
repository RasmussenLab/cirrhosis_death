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
# # Olink validation data
#
# both for 
#
# - `ProDoc` (N=29) samples
# - `CircaFlow` samples

# %%
from collections import namedtuple
from pathlib import Path

import pandas as pd

import config

# %% [markdown]
# Set default paths and collection

# %%
DATA_FOLDER = Path(config.data)

inputs = {}
outputs = {}

# %% [markdown]
# Define Measurment

# %%
Measurement = namedtuple('Measurment', 'idx measure')
measure_olink = Measurement(['SampleID', 'Assay'], 'NPX')
measure_olink

# %% [markdown]
# Load Olink validation data

# %%
inputs['olink'] = DATA_FOLDER / "Validation Results" / "ProDoc_Olink_bridged_QC.tsv"
olink = pd.read_table(inputs['olink'])
olink = olink.set_index(measure_olink.idx)
olink

# %% [markdown]
# Contains duplicated for bridging samples

# %%
duplicated = olink[measure_olink.measure].index.duplicated(keep=False)
olink.loc[duplicated].sort_index(level=-1).head(20)

# %% [markdown]
# Metadata for Olink features
#
# - `UniProt` ID of `OlinkID`
# - limit of detection (`LOD`)

# %%
inputs['metadata'] = DATA_FOLDER / "Validation Results" / "metadata.tsv"
metadata = pd.read_table(inputs["metadata"])
metadata

# %% [markdown]
# Sample name to ID mapping  - find subcohorts

# %%
inputs['id_map'] = DATA_FOLDER / "Validation Results" / "id.xlsx"
id_map = pd.read_excel(inputs["id_map"], index_col='SampleID')
id_map

# %%
print(id_map["CBMRID"].str[:4].value_counts().to_string())


# %%
def _select_idx(query: str,
                expected: int,
                id_map: pd.DataFrame = id_map,
                id_col: str = 'CBMRID'):
    idx = id_map.loc[id_map[id_col].str.contains(query)]
    idx = idx[id_col].to_list()
    assert len(
        idx
    ) == expected, f"Excepcted {expected} Prodoc validation samples, not {len(idx)}"
    return idx


# %%
idx_prodoc = _select_idx(query='ProD', expected=29)
# idx_prodoc

# %%
idx_circaflow = _select_idx(query='Cflow', expected=101)
# idx_circaflow

# %%
olink

# %%
olink_prodoc_val = olink.loc[idx_prodoc, measure_olink.measure].unstack()
olink_prodoc_val.describe()

# %%
stem = 'olink_prodoc_val'
outputs[f'{stem}'] = DATA_FOLDER / f'{stem}.pkl'
olink_prodoc_val.to_pickle(outputs[f'{stem}'])
outputs[f'{stem}'] = DATA_FOLDER / f'{stem}.xlsx'
olink_prodoc_val.to_excel(outputs[f'{stem}'])

# %%
olink_cflow = olink.loc[idx_circaflow, measure_olink.measure].unstack()
olink_cflow.describe()

# %%
stem = 'olink_cflow'
fname = DATA_FOLDER / f'{stem}.pkl'
olink_cflow.to_pickle(fname)
outputs[stem] = DATA_FOLDER / f'{stem}.xlsx'
olink_cflow.to_excel(outputs[stem])

# %% [markdown]
# Log all input and selected output files 

# %%
inputs

# %%
outputs
