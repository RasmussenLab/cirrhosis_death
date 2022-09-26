import config.pandas

from pathlib import Path

STUDY_ENDDATE: str = '2022-09-09'

base_folder = 'S:/SUND-CBMR-RegH-cohorts/ProDoc'
data: str = base_folder
data_processed: str = f'{base_folder}/data/processed'
folder_reports = f'{base_folder}/reports'

fname_pkl_clinic = Path(data_processed) / 'clinic.pkl'
fname_pkl_val_clinic = Path(data_processed) / 'val_clinic.pkl'
fname_pkl_olink = Path(data_processed) / 'olink.pkl'
fname_pkl_targets = Path(data_processed) / 'targets.pkl'

from . import clinic_data

TARGETS = [
    'dead90incl', 'dead90infl', 'dead180incl', 'dead180infl',
    'hasLiverAdm90infl', 'hasLiverAdm180infl', 'hasLiverAdm90incl',
    'hasLiverAdm180incl'
]
