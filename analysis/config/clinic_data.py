vars_binary = [
    # Demographics
    'Sex',
    # Etiology of disease
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
    # clinical variables
    'Ascites',
    'EsoBleeding',
    'HRS',
    'HE',
    'Icterus',
    'SBP',
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
    # Medication
    'InsulinDependent',
    'Statins',
    'NonselectBetaBlock'
]

vars_binary_created = [
    # from DiagnosisPlace
    'Admission',
    'AdmissionForOtherDisease',
    'Outpatient'
]

vars_cont = [
    # Demographics
    'Age',
    # laboratory markers
    'IgM',
    'IgG',
    'IgA',
    'Hgb',
    'Leucocytes',
    'Platelets',
    'Bilirubin',
    'Albumin',
    'CRP',
    'pp',
    'INR',
    'ALAT',
    # Intervals
    'TimeToDeath',
    'TimeToAdmFromDiagnose',
    'TimeToAdmFromSample',
    'TimeToDeathFromDiagnose',
    # clinical variables
    'Admissions',
    # derived scores
    'MELD-score',
    'MELD-Na',
    'ChildPugh'
]
