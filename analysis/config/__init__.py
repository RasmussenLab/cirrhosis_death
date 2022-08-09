from pathlib import Path

STUDY_ENDDATE:str = '2022-08-08'

data:str = 'S:/SUND-CBMR-RegH-cohorts/ProDoc'
data_processed:str = 'S:/SUND-CBMR-RegH-cohorts/ProDoc/data/processed'


fname_pkl_clinic = Path(data_processed) / 'clinic.pkl'
fname_pkl_olink = Path(data_processed) /  'olink.pkl'