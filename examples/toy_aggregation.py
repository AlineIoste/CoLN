import numpy as np
import pandas as pd

from coln.data import prepare_clients_from_dataframe
from coln.trainer import run_federated_coln

def make_synthetic_df(n=2000, n_hosp=4, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "source_hospital": rng.integers(0, n_hosp, size=n).astype(str),
        "age": rng.normal(60, 15, size=n),
        "creatinine": rng.lognormal(mean=0.0, sigma=0.5, size=n),
        "sex": rng.choice(["M","F"], size=n),
    })
    # logistic signal + hospital shift
    h_shift = df["source_hospital"].astype(int).to_numpy() * 0.15
    logit = 0.02*(df["age"].to_numpy()-60) + 0.6*(df["creatinine"].to_numpy()-1.0) + h_shift
    p = 1/(1+np.exp(-logit))
    df["icu_admission"] = (rng.random(n) < p).astype(int)
    return df

if __name__ == "__main__":
    df = make_synthetic_df()
    clients, scaler, feature_cols = prepare_clients_from_dataframe(
        df,
        target_col="icu_admission",
        hospital_col="source_hospital",
        test_size=0.2,
        random_state=42,
    )
    model, hist, res = run_federated_coln(
        clients,
        rounds=5,
        epochs=3,
        batch_size=128,
        use_oversample=True,
        pos_target=0.35,
        agg_equal_weights=False,
        threshold_fixed=0.5,
        verbose_rounds=True,
    )
    print("\nFinal metrics:", res)
    print("\nHistory head:")
    print(hist.head())
