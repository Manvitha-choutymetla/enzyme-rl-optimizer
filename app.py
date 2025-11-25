import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

from catboost import CatBoostRegressor, Pool
import joblib

# ---------- RDKit imports (for substrate features) ----------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError as e:
    raise ImportError(
        "RDKit is required for this pipeline.\n"
        "Install via: conda install -c rdkit rdkit"
    ) from e

RANDOM_STATE = 42

# ============================================================
# 1. HELPER FUNCTIONS: parsing numeric / pH / temperature / EC / SMILES
# ============================================================

def parse_numeric(value):
    """
    Extract a single float from messy strings like:
      '37', '37°C', '37 +/- 2', '30-40', '<0.5', '~10'
    Returns np.nan if we cannot parse.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()

    # Handle explicit ranges: "30-40", "30–40", "30 to 40"
    range_match = re.search(r'(\d+(\.\d+)?)\s*[-–to]+\s*(\d+(\.\d+)?)', s)
    if range_match:
        a = float(range_match.group(1))
        b = float(range_match.group(3))
        return (a + b) / 2.0

    # Handle +/- : "37 ± 2"
    plusminus_match = re.search(r'(\d+(\.\d+)?)\s*[±\+\/-]\s*(\d+(\.\d+)?)', s)
    if plusminus_match:
        # we only care about central value
        return float(plusminus_match.group(1))

    # Generic first float
    num_match = re.search(r'(-?\d+(\.\d+)?)', s)
    if num_match:
        return float(num_match.group(1))

    return np.nan


def parse_temperature(value):
    """
    Parse temperature in °C with some heuristics:
    - 'RT', 'room temperature' -> 25.0
    - Ranges & +/- handled via parse_numeric
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    if 'rt' in s or 'room' in s:
        return 25.0

    temp = parse_numeric(s)
    if temp is np.nan:
        return np.nan

    # Basic sanity clamp
    if temp < -5 or temp > 150:
        return np.nan
    return temp


def parse_ph(value):
    """
    Parse pH value; handle things like '7.0', '6.5-8.0', '~7', etc.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    ph = parse_numeric(s)
    if ph is np.nan:
        return np.nan

    # Clamp to plausible pH range
    if ph < 0 or ph > 14:
        return np.nan
    return ph


def split_ec_number(ec_str):
    """
    Split EC number 'a.b.c.d' into 4 integer parts.
    Missing / malformed -> zeros.
    """
    if pd.isna(ec_str):
        return 0, 0, 0, 0

    parts = str(ec_str).split(".")
    parts += ["0"] * (4 - len(parts))
    out = []
    for p in parts[:4]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out)


def smiles_to_chem_features(smiles):
    """
    Convert SMILES to a small set of RDKit-derived features.
    If parsing fails, return NaNs.
    Features:
      - mol_wt
      - h_acceptors
      - h_donors
      - rot_bonds
      - tpsa
      - ring_count
      - frac_csp3
    """
    if pd.isna(smiles):
        return [np.nan] * 7
    s = str(smiles).strip()
    if not s:
        return [np.nan] * 7

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return [np.nan] * 7

    mol_wt = Descriptors.MolWt(mol)
    h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    h_donors = rdMolDescriptors.CalcNumHBD(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    frac_csp3 = Descriptors.FractionCSP3(mol)

    return [mol_wt, h_acceptors, h_donors, rot_bonds, tpsa, ring_count, frac_csp3]


CHEM_FEATURE_NAMES = [
    "mol_wt",
    "h_acceptors",
    "h_donors",
    "rot_bonds",
    "tpsa",
    "ring_count",
    "frac_csp3",
]

# ============================================================
# 2. LOAD DATASET (SKiD kcat_dataset)
# ============================================================

def load_kcat_dataset(xlsx_path: Path):
    excel_file = pd.ExcelFile(xlsx_path)
    print("Available sheets:")
    print(excel_file.sheet_names)

    # Try detecting kcat sheet
    kcat_sheet = None
    for sheet in excel_file.sheet_names:
        if any(keyword in sheet.lower() for keyword in ['kcat', 'turnover', 'cat']):
            kcat_sheet = sheet
            break

    if kcat_sheet is None:
        raise ValueError("Could not auto-detect kcat sheet. Set it manually.")

    print(f"\nLoading kcat data from sheet: '{kcat_sheet}'")
    df = pd.read_excel(xlsx_path, sheet_name=kcat_sheet)
    print(f"Raw dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    required_cols = [
        'kcat_value (1/s)',
        'pH',
        'Temperature',
        'EC_number',
        'UniProt_ID',
        'Substrate_SMILES',
        'Mutant',
        'Organism_name'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ============================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# ============================================================

def prepare_features(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # 3.1 Filter to wild-type (no mutation)
    df = df[df['Mutant'] == 'no'].copy()

    # 3.2 Parse kcat, pH, Temperature
    df['kcat_value_parsed'] = df['kcat_value (1/s)'].apply(parse_numeric)
    df['pH_parsed'] = df['pH'].apply(parse_ph)
    df['Temperature_parsed'] = df['Temperature'].apply(parse_temperature)

    # Drop rows where any of these are missing
    df = df.dropna(subset=['kcat_value_parsed', 'pH_parsed', 'Temperature_parsed'])

    # Remove non-positive kcat
    df = df[df['kcat_value_parsed'] > 0]

    # 3.3 Log-transform kcat
    df['log_kcat'] = np.log10(df['kcat_value_parsed'])

    # 3.4 EC hierarchy
    ec_parts = df['EC_number'].apply(split_ec_number)
    df[['ec1', 'ec2', 'ec3', 'ec4']] = pd.DataFrame(ec_parts.tolist(), index=df.index)

    # 3.5 RDKit features from Substrate_SMILES
    chem_features = df['Substrate_SMILES'].apply(smiles_to_chem_features)
    chem_df = pd.DataFrame(chem_features.tolist(), columns=CHEM_FEATURE_NAMES, index=df.index)
    df = pd.concat([df, chem_df], axis=1)

    # Drop rows where all chem features are NaN
    df = df.dropna(subset=CHEM_FEATURE_NAMES, how='all')

    # Final sanity drop
    df = df.dropna(subset=['log_kcat', 'pH_parsed', 'Temperature_parsed'])

    print(f"\nAfter cleaning: {len(df)} rows")
    print("EC_number distribution (top 10):")
    print(df['EC_number'].value_counts().head(10))

    # 3.6 Build feature matrix
    numeric_features = [
        'pH_parsed',
        'Temperature_parsed',
        'ec1', 'ec2', 'ec3', 'ec4',
    ] + CHEM_FEATURE_NAMES

    categorical_features = [
        'EC_number',
        'Organism_name',
        'UniProt_ID',
        # you can add 'Substrate' here if you want
    ]

    # Only keep columns that exist
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    feature_cols = numeric_features + categorical_features

    X = df[feature_cols].copy()
    y = df['log_kcat'].copy()

    print(f"\nFinal feature matrix shape: {X.shape}")
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    return X, y, feature_cols, categorical_features


# ============================================================
# 4. TRAIN CATBOOST MODEL
# ============================================================

def train_catboost_model(X, y, feature_cols, cat_feature_names):
    # Identify indices of categorical features
    cat_indices = [X.columns.get_loc(c) for c in cat_feature_names]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)

    model = CatBoostRegressor(
        loss_function='RMSE',
        depth=8,
        learning_rate=0.05,
        n_estimators=1200,
        random_seed=RANDOM_STATE,
        eval_metric='RMSE',
        verbose=200
    )

    model.fit(train_pool, eval_set=test_pool)

    # Predictions
    y_pred_train = model.predict(train_pool)
    y_pred_test = model.predict(test_pool)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n--- Global kcat model performance (CatBoost) ---")
    print(f"Train R² : {train_r2:.3f} | RMSE: {train_rmse:.3f}")
    print(f"Test  R² : {test_r2:.3f} | RMSE: {test_rmse:.3f}")

    # Manual 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = []

    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        m = CatBoostRegressor(
            loss_function='RMSE',
            depth=8,
            learning_rate=0.05,
            n_estimators=800,
            random_seed=RANDOM_STATE + fold_idx,
            verbose=False
        )
        m.fit(tr_pool)

        y_val_pred = m.predict(val_pool)
        r2 = r2_score(y_val, y_val_pred)
        cv_r2_scores.append(r2)
        fold_idx += 1

    cv_mean = np.mean(cv_r2_scores)
    cv_std = np.std(cv_r2_scores)
    print(f"\n5-fold CV R²: {cv_mean:.3f} ± {cv_std:.3f}")

    return model, {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "cv_r2_mean": float(cv_mean),
        "cv_r2_std": float(cv_std),
    }


# ============================================================
# 5. SAVE MODEL + METADATA
# ============================================================

def save_model_and_metadata(model, feature_cols, cat_feature_names, metrics):
    # CatBoost native model
    model.save_model("kcat_catboost_model.cbm")

    meta = {
        "feature_cols": feature_cols,
        "categorical_feature_names": cat_feature_names,
        "metrics": metrics,
    }
    with open("kcat_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved model to kcat_catboost_model.cbm")
    print("Saved metadata to kcat_model_metadata.json")


# ============================================================
# 6. EXAMPLE PREDICTION FUNCTION FOR RL OR DEMO
# ============================================================

def build_predict_function(feature_cols, cat_feature_names):
    """
    Returns a convenience function:
      predict_kcat(ph, temp, ec_number, organism, uniprot_id, smiles)
    It rebuilds a one-row DataFrame in the same feature order and calls the model.
    """

    # Load model + metadata inside closure
    model = CatBoostRegressor()
    model.load_model("kcat_catboost_model.cbm")

    with open("kcat_model_metadata.json", "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    cat_feature_names = meta["categorical_feature_names"]

    def predict_kcat(ph, temp, ec_number, organism_name, uniprot_id, smiles):
        # Parse inputs
        ph_val = parse_ph(ph)
        temp_val = parse_temperature(temp)
        ec1, ec2, ec3, ec4 = split_ec_number(ec_number)
        chem_vals = smiles_to_chem_features(smiles)

        if ph_val is np.nan or temp_val is np.nan:
            raise ValueError("Invalid pH or Temperature input.")

        # Start with empty dict
        row = {col: np.nan for col in feature_cols}

        # Fill what we know
        if 'pH_parsed' in row:
            row['pH_parsed'] = ph_val
        if 'Temperature_parsed' in row:
            row['Temperature_parsed'] = temp_val

        if 'ec1' in row: row['ec1'] = ec1
        if 'ec2' in row: row['ec2'] = ec2
        if 'ec3' in row: row['ec3'] = ec3
        if 'ec4' in row: row['ec4'] = ec4

        for name, val in zip(CHEM_FEATURE_NAMES, chem_vals):
            if name in row:
                row[name] = val

        if 'EC_number' in row:
            row['EC_number'] = ec_number
        if 'Organism_name' in row:
            row['Organism_name'] = organism_name
        if 'UniProt_ID' in row:
            row['UniProt_ID'] = uniprot_id

        X_row = pd.DataFrame([row], columns=feature_cols)

        cat_indices = [X_row.columns.get_loc(c) for c in cat_feature_names if c in X_row.columns]
        pool = Pool(X_row, cat_features=cat_indices)

        log_kcat_pred = model.predict(pool)[0]
        kcat_pred = 10 ** log_kcat_pred  # back-transform

        return float(kcat_pred)

    return predict_kcat


# ============================================================
# 7. MAIN
# ============================================================

def main():
    file_path = Path("./Main_dataset_v1.xlsx")  # adjust if needed

    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found at: {file_path}")

    df_raw = load_kcat_dataset(file_path)
    X, y, feature_cols, cat_feature_names = prepare_features(df_raw)
    model, metrics = train_catboost_model(X, y, feature_cols, cat_feature_names)
    save_model_and_metadata(model, feature_cols, cat_feature_names, metrics)

    # Build example predictor and test it
    predict_kcat = build_predict_function(feature_cols, cat_feature_names)
    example_kcat = predict_kcat(
        ph=1.0,
        temp=98.0,
        ec_number="1.1.1.1",
        organism_name="Escherichia coli",
        uniprot_id="P00330",
        smiles="CCO"  # ethanol
    )
    print(f"\nExample prediction: kcat ≈ {example_kcat:.3f} s⁻1")

if __name__ == "__main__":
    main()
