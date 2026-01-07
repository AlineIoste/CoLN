from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def oversample_positive_class(
    X: np.ndarray,
    y: np.ndarray,
    pos_target: float = 0.35,
    seed: int = 42,
):
    """Simple random oversampling of the positive class to reach pos_target prevalence."""
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y

    n_pos, n_neg = len(pos_idx), len(neg_idx)
    p_new = int(np.ceil(pos_target * n_neg / (1 - pos_target)))
    if p_new <= n_pos:
        return X, y

    need = p_new - n_pos
    extra = rng.choice(pos_idx, size=need, replace=True)
    X_new = np.concatenate([X, X[extra]], axis=0)
    y_new = np.concatenate([y, y[extra]], axis=0)
    idx = rng.permutation(len(y_new))
    return X_new[idx], y_new[idx]

def prepare_clients_from_dataframe(
    df: pd.DataFrame,
    target_col: str,
    hospital_col: str,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Prepare the `clients` dict expected by `run_federated_coln`.

    - One-hot encodes categorical columns (global)
    - Splits each hospital into train/test
    - Fits a *global* StandardScaler using concatenated train sets
    - Returns clients[cid] = {X_tr, y_tr, X_te, y_te, n_train, n_test}

    The returned arrays are float32 for X and int for y.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found.")
    if hospital_col not in df.columns:
        raise ValueError(f"hospital_col '{hospital_col}' not found.")

    df = df.copy()
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # One-hot encode everything except hospital_col
    X_no_h = X.drop(columns=[hospital_col])
    X_ohe = pd.get_dummies(X_no_h, drop_first=False)
    hospitals = df[hospital_col].astype(str).values

    clients = {}
    for hosp in sorted(pd.unique(hospitals)):
        idx = np.where(hospitals == hosp)[0]
        if len(idx) < 5:
            # too small to be meaningful; still include but warn via counts
            pass
        X_h = X_ohe.iloc[idx].to_numpy()
        y_h = y.iloc[idx].to_numpy()

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_h, y_h,
            test_size=test_size,
            random_state=random_state,
            stratify=y_h if len(np.unique(y_h)) > 1 else None,
        )
        clients[hosp] = {
            "X_tr_raw": X_tr_raw,
            "X_te_raw": X_te_raw,
            "y_tr": y_tr.astype(int),
            "y_te": y_te.astype(int),
        }

    # Global scaler fit on concatenated train sets
    X_concat_tr = np.vstack([c["X_tr_raw"] for c in clients.values()])
    scaler = StandardScaler().fit(X_concat_tr)

    for c in clients.values():
        c["X_tr"] = scaler.transform(c["X_tr_raw"]).astype("float32")
        c["X_te"] = scaler.transform(c["X_te_raw"]).astype("float32")
        c["n_train"] = int(len(c["y_tr"]))
        c["n_test"] = int(len(c["y_te"]))

    return clients, scaler, X_ohe.columns.tolist()
