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
    # dead within 90 days or 180 days
    # incl -> from inclusion date
    # infl -> from inflammation sample date
    'dead090incl',
    'dead090infl',
    'dead180incl',
    'dead180infl',
    ##### different admission variables
    # -> TotalAdm: number of total admissions
    'hasTotalAdm180infl',
    # InfecAdm: Infection related admission (with link to liver disease)
    'hasInfecAdm180infl',
    'hasTotalAdm090infl',
    'hasInfecAdm090infl',
    'hasTotalAdm180incl',
    'hasInfecAdm180incl',
    'hasTotalAdm090incl',
    'hasInfecAdm090incl',
    # LiverAdm: Liver related Admission
    'hasLiverAdm090infl',
    # has InfecAdm or LiverAdm
    'hasLiverInfecAdm090infl',
    'hasLiverAdm180infl',
    'hasLiverInfecAdm180infl',
    'hasLiverAdm090incl',
    'hasLiverInfecAdm090incl',
    'hasLiverAdm180incl',
    'hasLiverInfecAdm180incl'
]
