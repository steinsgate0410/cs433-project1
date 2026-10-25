"""BRFSS preprocessing for EPFL CS-433 Project 1.

The original V6 codebook-aware encoder is preserved to reproduce
AIcrowd submission #337816. Fit preprocessing on training partitions only.
"""
import numpy as np

SURVEY_METADATA = frozenset({
    'Id', '_STATE', 'STATERES', 'CSTATE', 'MSCODE',
    'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR',
    'DISPCODE', 'QSTVER', 'QSTLANG',
    'SEQNO', '_PSU',
    'CTELENUM', 'PVTRESD1', 'COLGHOUS', 'CELLFON3', 'LADULT',
    'NUMADULT', 'NUMMEN', 'NUMWOMEN', 'CTELNUM1', 'CELLFON2',
    'CADULT', 'PVTRESD2', 'CCLGHOUS', 'LANDLINE', 'HHADULT',
    'NUMHHOL2', 'NUMPHON2', 'CPDEMO1',
    '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE',
    '_CHISPNC', '_CRACE1', '_CPRACE', '_CLLCPWT',
    '_DUALUSE', '_DUALCOR', '_LLCPWT',
    'RCSGENDR', 'RCSRLTN2',
})


def _eligible(names):
    """Enforce the exact same global exclusion policy in every feature pool."""
    return [name for name in names if name not in SURVEY_METADATA]

BASE_CAT = [
    'SEX','GENHLTH','_RFHYPE5','_RFCHOL','DIABETE3','_SMOKER3',
    '_TOTINDA','_INCOMG','_EDUCAG','CVDSTRK3','CHCKIDNY','CHCCOPD1',
    'MARITAL','_RACEGR3','HLTHPLN1','_PA150R2','ADDEPEV2'
]
BASE_NUM = ['_AGE80','_BMI5','PHYSHLTH','MENTHLTH','POORHLTH']
EXTRA_CAT = [
    'QLACTLM2','DIFFWALK','DIFFDRES','DIFFALON','DECIDE','USEEQUIP','BLIND',
    'PERSDOC2','MEDCOST','CHECKUP1','BLOODCHO','_CHOLCHK',
    'DRNKANY5','_RFBING5','_RFDRHV5','_PACAT1','_PAINDX1',
    'BPHIGH4','TOLDHI2','SMOKE100','SMOKDAY2','EXERANY2','_BMI5CAT','_AGEG5YR'
]
EXTRA_NUM = ['_FRUTSUM','_VEGESUM']
# Broader candidate pool: only fields with an explicit codebook rule.
# The global survey-metadata exclusion below also applies to this pool.
# Optional health modules are deliberately retained.
BROAD_CAT = [
    '_STATE','FMONTH','QSTVER','MSCODE','QSTLANG',
    'BPMEDS','CHOLCHK','ASTHMA3','ASTHNOW','CHCSCNCR','CHCOCNCR','HAVARTH3',
    'PREDIAB1','INSULIN','DIABEYE','DIABEDU','RENTHOM1','EMPLOY1','EDUCA',
    'INCOME2','INTERNET','PREGNANT','VETERAN3','USENOW3','STOPSMK2','LASTSMK2',
    'FLUSHOT6','PNEUVAC3','SEATBELT','HIVTST6','EXRACT11','EXRACT21',
    '_RACE','_RFBMI5','_RFHLTH','_HCVU651','_CHLDCNT','_DRDXAR1',
    '_LTASTH1','_CASTHM1','_ASTHMS1','_PA300R2','_PA30021','_PASTRNG',
    '_PAREC1','_PASTAE1','_LMTACT1','_LMTWRK1','_LMTSCL1',
    '_RFSEAT2','_RFSEAT3','_FLSHOT6','_PNEUMO2','_AIDTST3',
    'PDIABTST','LMTJOIN3','ARTHDIS2','ARTHSOCL','EYEEXAM',
    'SEATBELT','CIMEMLOS','EMTSUPRT','LSATISFY',
    'CAREGIV1','CRGVREL1','CRGVHRS1','WTCHSALT',
    'VIDFCLT2','VIREDIF3','VIPRFVS2','VIEYEXM2',
    'SHINGLE2','TETANUS','_FRTLT1','_VEGLT1',
    '_FRTRESP','_VEGRESP','_BMI5CAT','_AGE65YR',
]
BROAD_CAT = list(dict.fromkeys(n for n in BROAD_CAT if n not in BASE_CAT + EXTRA_CAT))
BROAD_NUM = [
    'CHILDREN','DIABAGE2','JOINPAIN','HHADULT','NUMADULT',
    'ALCDAY5','AVEDRNK2','DRNK3GE5','MAXDRNKS',
    'FRUITJU1','FRUIT1','FVBEANS','FVGREEN','FVORANG','VEGETAB1',
    'EXEROFT1','EXEROFT2','EXERHMM1','EXERHMM2','STRENGTH',
    'DOCTDIAB','CHKHEMO3','FEETCHK',
    'PADUR1_','PADUR2_','PA1MIN_',
    'ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1',
    'ADFAIL','ADTHINK','ADMOVE',
]
# Apply the blacklist once, before input loading, encoding, CV or inference.
BASE_CAT, BASE_NUM = _eligible(BASE_CAT), _eligible(BASE_NUM)
EXTRA_CAT, EXTRA_NUM = _eligible(EXTRA_CAT), _eligible(EXTRA_NUM)
BROAD_CAT, BROAD_NUM = _eligible(BROAD_CAT), _eligible(BROAD_NUM)
ALL_NAMES = list(dict.fromkeys(BASE_CAT + BASE_NUM + EXTRA_CAT + EXTRA_NUM
                               + BROAD_CAT + BROAD_NUM))
LEAKAGE = {'_MICHD','_MICHDCV','CVDINFR4','CVDCRHD4','HAREHAB1'}
assert not SURVEY_METADATA.intersection(ALL_NAMES)
assert not LEAKAGE.intersection(ALL_NAMES)

# Allowed *observed* survey codes from the supplied BRFSS-2015 quick-reference.
# Every value outside the given domain (including 7/9/77/99 when they represent
# nonresponse) becomes NaN. The meaning of 7 and 9 varies by question.
YES_NO = {'HLTHPLN1','CVDSTRK3','CHCKIDNY','CHCCOPD1','ADDEPEV2',
          'QLACTLM2','DIFFWALK','DIFFDRES','DIFFALON','DECIDE',
          'USEEQUIP','BLIND','MEDCOST','BLOODCHO','TOLDHI2',
          'SMOKE100','EXERANY2','DRNKANY5'}
DOMAIN = {name: (1,2) for name in YES_NO}
DOMAIN.update({
    'SEX': (1,2), 'GENHLTH': (1,2,3,4,5),
    '_RFHYPE5': (1,2), '_RFCHOL': (1,2),
    'DIABETE3': (1,2,3,4), '_SMOKER3': (1,2,3,4),
    '_TOTINDA': (1,2), '_INCOMG': (1,2,3,4,5),
    '_EDUCAG': (1,2,3,4), 'MARITAL': (1,2,3,4,5,6),
    '_RACEGR3': (1,2,3,4,5), '_PA150R2': (1,2,3),
    'PERSDOC2': (1,2,3), 'CHECKUP1': (1,2,3,4,8),
    '_CHOLCHK': (1,2,3), '_RFBING5': (1,2), '_RFDRHV5': (1,2),
    '_PACAT1': (1,2,3,4), '_PAINDX1': (1,2),
    'BPHIGH4': (1,2,3,4), 'SMOKDAY2': (1,2,3),
    '_BMI5CAT': (1,2,3,4), '_AGEG5YR': tuple(range(1,14)),
})
assert set(BASE_CAT + EXTRA_CAT) == set(DOMAIN), 'Missing domain for categorical feature'
# Broad categorical domains. In particular, 7/8/9 can be VALID answers
# in some columns, hence each column has its own set of allowed codes.
_BROAD_YN = {'BPMEDS','ASTHMA3','ASTHNOW','CHCSCNCR','CHCOCNCR','HAVARTH3',
             'DIABEYE','DIABEDU','INTERNET','PREGNANT','VETERAN3',
             'FLUSHOT6','PNEUVAC3','HIVTST6','PDIABTST','LMTJOIN3',
             'ARTHDIS2','CIMEMLOS','WTCHSALT','SHINGLE2'}
DOMAIN.update({n: (1,2) for n in _BROAD_YN})
DOMAIN.update({
    'CHOLCHK':(1,2,3,4),
    'PREDIAB1':(1,2,3), 'INSULIN':(1,2),
    'RENTHOM1':(1,2,3), 'EMPLOY1':tuple(range(1,9)),
    'EDUCA':tuple(range(1,7)), 'INCOME2':tuple(range(1,9)),
    'USENOW3':(1,2,3), 'STOPSMK2':(1,2),
    'LASTSMK2':tuple(range(1,9)),
    'SEATBELT':(1,2,3,4,5,8), 'EXRACT11':tuple(range(1,77))+(98,),
    'EXRACT21':(0,)+tuple(range(1,77))+(98,),
    '_RACE':tuple(range(1,9)), '_RFBMI5':(1,2),
    '_RFHLTH':(1,2), '_HCVU651':(1,2),
    '_CHLDCNT':tuple(range(1,7)), '_DRDXAR1':(1,2),
    '_LTASTH1':(1,2),'_CASTHM1':(1,2),
    '_ASTHMS1':(1,2,3), '_PA300R2':(1,2,3),
    '_PA30021':(1,2), '_PASTRNG':(1,2),
    '_PAREC1':(1,2,3,4), '_PASTAE1':(1,2),
    '_LMTACT1':(1,2,3), '_LMTWRK1':(1,2,3),
    '_LMTSCL1':(1,2,3,4), '_RFSEAT2':(1,2),
    '_RFSEAT3':(1,2), '_FLSHOT6':(1,2),
    '_PNEUMO2':(1,2), '_AIDTST3':(1,2),
    'ARTHSOCL':(1,2,3), 'EYEEXAM':(1,2,3,4,8),
    'EMTSUPRT':(1,2,3,4,5), 'LSATISFY':(1,2,3,4),
    'CAREGIV1':(1,2,8), 'CRGVREL1':tuple(range(1,16)),
    'CRGVHRS1':(1,2,3,4), 'VIDFCLT2':tuple(range(1,7))+(8,),
    'VIREDIF3':tuple(range(1,7)), 'VIPRFVS2':(1,2,3,4,5),
    'VIEYEXM2':(1,2,3,4,5,8), 'TETANUS':(1,2,3,4),
    '_FRTLT1':(1,2), '_VEGLT1':(1,2),
    '_FRTRESP':(0,1), '_VEGRESP':(0,1),
    '_AGE65YR':(1,2),
})
assert set(BROAD_CAT).issubset(DOMAIN), sorted(set(BROAD_CAT)-set(DOMAIN))


# NOTE ON UNITS: the EPFL CSV *already* contains floating-point BMI and fruit/
# vegetable totals (_BMI5=26.63, _FRUTSUM=1.43), while the CDC fixed-width source
# often stores these multiplied by 100. DO NOT divide this competition CSV again.


def clean_column(name, X, index=None):
    """Apply *fixed*, label-free BRFSS codebook rules to one raw EPFL column.

    - NaN and out-of-domain survey answers become missing.
    - A validated logical skip is represented as 0, a separate category
      meaning 'not asked', NOT an affirmative or negative answer.
    - Derived numeric variables are already in decoded units in EPFL CSV.
    """
    if index is None:
        index = {n:i for i,n in enumerate(ALL_NAMES)}
    v = np.asarray(X[:,index[name]], dtype=np.float32).copy()
    if name in DOMAIN:
        if name == 'EXRACT21':
            v[v == 88] = 0  # no second exercise, separate valid category
        v[~np.isin(v, DOMAIN[name])] = np.nan
        if name == 'SMOKDAY2' and 'SMOKE100' in index:
            # Asked only if smoked >=100 cigarettes; a definite 'no' = skipped.
            parent = X[:,index['SMOKE100']]
            v[np.isnan(v) & (parent == 2)] = 0.0
        elif name == 'TOLDHI2' and 'BLOODCHO' in index:
            # Not examined = not asked about elevated result (0 means N/A).
            parent = X[:,index['BLOODCHO']]
            v[np.isnan(v) & (parent == 2)] = 0.0
        elif name in ('BPMEDS',) and 'BPHIGH4' in index:
            v[np.isnan(v) & (X[:,index['BPHIGH4']] != 1)
              & np.isfinite(X[:,index['BPHIGH4']])] = 0
        elif name == 'ASTHNOW' and 'ASTHMA3' in index:
            v[np.isnan(v) & (X[:,index['ASTHMA3']] == 2)] = 0
        elif name == 'STOPSMK2' and 'SMOKDAY2' in index:
            v[np.isnan(v) & (X[:,index['SMOKDAY2']] == 3)] = 0
        return v
    if name == '_AGE80':
        v[(v < 18) | (v > 80)] = np.nan  # 77 is a VALID age
    elif name == '_BMI5':
        # A wide plausibility gate, not an official CDC exclusion criterion.
        v[(v < 10) | (v > 100)] = np.nan
    elif name in ('PHYSHLTH', 'MENTHLTH', 'POORHLTH'):
        v[v == 88] = 0.0  # none: zero days, not missing
        v[(v < 0) | (v > 30)] = np.nan  # includes 77/99
        if name == 'POORHLTH':
            physically_zero = X[:,index['PHYSHLTH']] == 88
            mentally_zero = X[:,index['MENTHLTH']] == 88
            v[np.isnan(v) & physically_zero & mentally_zero] = 0.0
    elif name in ('_FRUTSUM', '_VEGESUM'):
        # Predecoded daily counts (e.g. 1.43); no arbitrary cap at 25.
        v[v < 0] = np.nan
    elif name in ('CHILDREN',):
        v[v == 88] = 0
        v[(v < 0) | (v > 87)] = np.nan
    elif name in ('DIABAGE2',):
        v[(v < 1) | (v > 97)] = np.nan
    elif name in ('JOINPAIN',):
        v[(v < 0) | (v > 10)] = np.nan
    elif name == 'ALCDAY5':
        # 1xx: drinking days per week; 2xx: days per 30 days; 888: none.
        raw = v.copy(); v[:] = np.nan
        a = (raw >= 101) & (raw <= 107)
        b = (raw >= 201) & (raw <= 230)
        v[a] = raw[a]-100
        v[b] = (raw[b]-200)*7/30
        v[raw == 888] = 0
    elif name in ('FRUITJU1','FRUIT1','FVBEANS','FVGREEN','FVORANG','VEGETAB1'):
        raw = v.copy();v[:] = np.nan
        for unit, divisor in ((1,1),(2,7),(3,30)):
            valid=(raw >= unit*100+1) & (raw <= unit*100+99)
            v[valid]=(raw[valid]-unit*100)/divisor
        v[(raw == 555) | (raw == 300)] = 0
    elif name in ('EXEROFT1','EXEROFT2','STRENGTH'):
        raw = v.copy();v[:] = np.nan
        for unit, divisor in ((1,1),(2,4.345)):
            valid=(raw >= unit*100+1) & (raw <= unit*100+99)
            v[valid]=(raw[valid]-unit*100)/divisor  # counts per week
        if name == 'STRENGTH': v[raw == 888] = 0
        elif 'EXERANY2' in index:
            v[np.isnan(v) & (X[:,index['EXERANY2']] == 2)] = 0
    elif name in ('EXERHMM1','EXERHMM2'):
        raw=v.copy();v[:]=np.nan
        valid=(raw>=1)&(raw<=959)&((np.nan_to_num(raw, nan=0).astype(np.int32)%100)<60)
        v[valid]=(raw[valid]//100)*60+(raw[valid]%100)
        if 'EXERANY2' in index:
            v[np.isnan(v) & (X[:,index['EXERANY2']] == 2)] = 0
    elif name in ('AVEDRNK2','DRNK3GE5','MAXDRNKS'):
        if name == 'DRNK3GE5': v[v==88]=0
        v[(v<0)|(v>76)] = np.nan
        if 'ALCDAY5' in index:
            v[np.isnan(v) & (X[:,index['ALCDAY5']] == 888)] = 0
    elif name in ('DOCTDIAB','CHKHEMO3','FEETCHK'):
        v[v == 88] = 0
        # 98 means unaware for CHKHEMO3, not a count.
        v[(v<0)|(v>76)] = np.nan
    elif name in ('PADUR1_','PADUR2_'):
        v[(v<0)|(v>599)] = np.nan
    elif name in ('PA1MIN_',):
        v[(v<0)|(v>=99999)] = np.nan
    elif name in ('ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1',
                  'ADFAIL','ADTHINK','ADMOVE'):
        v[v == 88] = 0
        v[(v<0)|(v>14)] = np.nan
    else:
        raise KeyError(f'No BRFSS numeric cleaning rule for {name}')
    return v


def preprocessing_audit(X, feature_set='full'):
    """Row-level audit of input missing, invalid-code cleaning and logical skips.

    Return list of dictionaries for easy CSV writing or notebook inspection.
    No label information is used; these counts are descriptive only.
    """
    cats = BASE_CAT + (EXTRA_CAT if feature_set != 'base' else []) + (BROAD_CAT if feature_set == 'broad' else [])
    nums = BASE_NUM + (EXTRA_NUM if feature_set != 'base' else []) + (BROAD_NUM if feature_set == 'broad' else [])
    ix = {n:i for i,n in enumerate(ALL_NAMES)}
    summary = []
    for name in cats + nums:
        raw = X[:,ix[name]]
        clean = clean_column(name,X,ix)
        raw_blank = int(np.sum(~np.isfinite(raw)))
        final_missing = int(np.sum(~np.isfinite(clean)))
        invalid = int(np.sum(np.isfinite(raw) & ~np.isfinite(clean)))
        skip_filled = int(np.sum(~np.isfinite(raw) & np.isfinite(clean)))
        summary.append(dict(feature=name, raw_blank=raw_blank,
                            invalid_code_or_value=invalid,
                            logical_skip_to_zero=skip_filled,
                            missing_after_cleaning=final_missing,
                            missing_rate=final_missing/len(X)))
    return summary


class BRFSSEncoder:
    """Categorical one-hot + continuous imputation + standardization, train-fit only."""
    def __init__(self, feature_set='full', min_category=20, max_missing=0.97):
        if feature_set not in ('base', 'full', 'broad'):
            raise ValueError("feature_set must be 'base', 'full', or 'broad'")
        self.cats = BASE_CAT + (EXTRA_CAT if feature_set != 'base' else []) + (BROAD_CAT if feature_set == 'broad' else [])
        self.nums = BASE_NUM + (EXTRA_NUM if feature_set != 'base' else []) + (BROAD_NUM if feature_set == 'broad' else [])
        self.min_category = min_category
        self.feature_set = feature_set
        self.max_missing = max_missing
        self.index = {name: i for i, name in enumerate(ALL_NAMES)}

    def fit(self, X):
        if self.feature_set == 'broad':
            # Unsupervised filtering, fitted separately on each training fold.
            self.cats = [n for n in self.cats if
                np.isfinite(clean_column(n,X,self.index)).mean() >= 1-self.max_missing]
            self.nums = [n for n in self.nums if
                np.isfinite(clean_column(n,X,self.index)).mean() >= 1-self.max_missing]
        self.raw_features_ = self.cats + self.nums
        self.levels = {}
        self.impute = {}
        for name in self.cats:
            col = clean_column(name, X, self.index)
            levels, count = np.unique(col[np.isfinite(col)], return_counts=True)
            self.levels[name] = levels[count >= self.min_category]
        for name in self.nums:
            col = clean_column(name, X, self.index)
            good = col[np.isfinite(col)]
            self.impute[name] = float(np.median(good)) if good.size else 0.0
        encoded, names = self._encode(X)
        self.names_ = names
        self.mean_ = encoded.mean(axis=0, dtype=np.float64)
        self.scale_ = np.maximum(encoded.std(axis=0, dtype=np.float64), 0.05)
        # Avoid fitting permanent all-constant dummies (but keep raw feature list).
        self.keep_ = encoded.std(axis=0) > 1e-7
        self.names_ = [n for n,keep in zip(self.names_,self.keep_) if keep]
        self.mean_ = self.mean_[self.keep_]
        self.scale_ = self.scale_[self.keep_]
        return self

    def _encode(self, X):
        blocks, names = [], []
        for name in self.cats:
            x = clean_column(name, X, self.index)
            levels = self.levels[name]
            for code in levels[1:]:  # omit reference category
                blocks.append((x == code).astype(np.float32))
                names.append(f'{name}=={code:g}')
            blocks.append((~np.isfinite(x)).astype(np.float32))
            names.append(name + ':missing')
            blocks.append((np.isfinite(x) & ~np.isin(x, levels)).astype(np.float32))
            names.append(name + ':rare_or_unseen')
        for name in self.nums:
            x = clean_column(name, X, self.index)
            missing = ~np.isfinite(x)
            filled = np.where(missing, self.impute[name], x)
            blocks.extend((filled.astype(np.float32), missing.astype(np.float32)))
            names.extend((name + ':value', name + ':missing'))
        return np.column_stack(blocks), names

    def transform(self, X):
        values, names = self._encode(X)
        values = values[:,self.keep_]
        return ((values - self.mean_) / self.scale_).astype(np.float32)


# Lightweight teammate utilities; not used by the final submission model.
def subsetting_data(dataset, featurelist):
    """Return a subset of the dataset containing only the requested features.

    Args:
        dataset (dict): Mapping from feature names to arrays.
        featurelist (list[str]): Feature names to keep.

    Returns:
        dict: A dictionary containing only the requested features that exist in the dataset.
    """
    return {feature: dataset[feature] for feature in featurelist if feature in dataset}


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the given split ratio using a single permutation.

    Parameters:
    - x: numpy array of shape (N, D) or (N,)
    - y: numpy array of shape (N,)
    - ratio: float, fraction of data for training (e.g., 0.8)
    - seed: int, random seed for reproducibility

    Returns:
    - x_tr, y_tr: training split
    - x_te, y_te: test split
    """
    np.random.seed(seed)
    num_rows = len(y)

    # Generate a single random permutation of indices
    indices = np.random.permutation(num_rows)

    # Determine the split index
    split_idx = int(np.floor(ratio * num_rows))

    # Partition indices into training and testing sets
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Select data using the permuted index arrays
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_te = x[test_idx]
    y_te = y[test_idx]

    return x_tr, y_tr, x_te, y_te


def shape(dataset):
    num_cols = len(dataset)
    num_rows = dataset[list(dataset.keys())[0]].shape[0] if num_cols > 0 else 0
    return (num_rows, num_cols)


def compute_covariance_matrix(tx):
    """
    Computes the (D, D) covariance matrix for an (N, D) feature matrix.
    """
    # 1. Center the features by subtracting the mean of each column
    mean_vector = np.mean(tx, axis=0)
    tx_centered = tx - mean_vector
    
    # 2. Compute the sample covariance matrix: (tx_centered.T @ tx_centered) / (N - 1)
    N = tx.shape[0]
    covariance_matrix = (tx_centered.T @ tx_centered) / (N - 1)
    
    return covariance_matrix


def compute_correlation_matrix(tx):
    """
    Computes the (D, D) Pearson correlation matrix from the covariance matrix.
    """
    cov_matrix = compute_covariance_matrix(tx)
    
    # Extract the standard deviations (square root of the diagonal variances)
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Handle any constant features (variance = 0) to avoid division by zero
    std_devs[std_devs == 0] = 1e-9
    
    # Correlation = Cov(X, Y) / (Std(X) * Std(Y))
    # Using outer product to create a (D, D) denominator matrix
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    return correlation_matrix

