from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .models import build_mlp
from .data import oversample_positive_class
from .aggregation import combined_learning_coln

def schedule_c(round_idx: int, total_rounds: int) -> float:
    # starts smooth and increases (avoids very sharp alpha early on)
    return 0.5 + 1.0 * (round_idx - 1) / max(total_rounds - 1, 1)

def schedule_lambda_beta(round_idx: int, total_rounds: int) -> float:
    # warm-up (rounds 1-2): 0; then ramps and saturates
    if round_idx <= 2:
        return 0.0
    return min(0.10, 0.03 * (round_idx - 2))

def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens + 1e-12)
    auc  = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "Accuracy": acc,
        "F1-score": f1,
        "Precision": prec,
        "Sensitivity (Recall)": sens,
        "Specificity": spec,
        "AUC": auc,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }

def run_federated_coln(
    clients: dict,
    rounds: int = 15,
    epochs: int = 10,
    batch_size: int = 256,
    use_oversample: bool = True,
    pos_target: float = 0.35,
    agg_equal_weights: bool = True,
    threshold_fixed: float = 0.5,
    verbose_rounds: bool = True,
    lr: float = 3e-4,
):
    """Federated training loop for CoLN(+).

    Expected client dict format (per client):
        X_tr, y_tr, X_te, y_te

    Returns
    -------
    global_model : tf.keras.Model
    df_history : pd.DataFrame
        Round-by-round metrics on the *global test set*.
    res_final : dict
        Final metrics dict.
    """
    # Build global test set
    X_test_global = np.vstack([c["X_te"] for c in clients.values()])
    y_test_global = np.concatenate([c["y_te"] for c in clients.values()])
    input_dim = next(iter(clients.values()))["X_tr"].shape[1]

    global_model = build_mlp(input_dim, lr=lr)
    global_weights = global_model.get_weights()

    history_rounds = []

    for rnd in range(1, rounds + 1):
        local_weights_list = []
        local_rh = []

        for _, cdata in clients.items():
            X_tr = cdata["X_tr"]
            y_tr = cdata["y_tr"]

            if use_oversample:
                X_fit, y_fit = oversample_positive_class(X_tr, y_tr, pos_target=pos_target, seed=42 + rnd)
            else:
                X_fit, y_fit = X_tr, y_tr

            local_model = build_mlp(input_dim, lr=lr)
            local_model.set_weights(global_weights)

            local_model.fit(
                X_fit, y_fit,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[
                    EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_lr=1e-6),
                ],
            )

            w = 1.0 if agg_equal_weights else float(len(y_fit))
            local_rh.append(w)
            local_weights_list.append(local_model.get_weights())

        # Normalize r_h
        s = float(sum(local_rh))
        if s == 0.0:
            raise RuntimeError("Sum of local weights is zero; check aggregation.")
        local_rh = [w / s for w in local_rh]

        # Aggregate
        global_weights = combined_learning_coln(
            models_weights=local_weights_list,
            r_h_list=local_rh,
            c=schedule_c(rnd, rounds),
            lambda_beta=schedule_lambda_beta(rnd, rounds),
            use_layer_c=True,
            gamma=1.5,
        )
        global_model.set_weights(global_weights)

        # Evaluate
        y_proba_round = global_model.predict(X_test_global, verbose=0).ravel()
        res_round = _compute_metrics(y_test_global, y_proba_round, threshold_fixed)
        res_round["Round"] = rnd
        res_round["Threshold"] = threshold_fixed
        history_rounds.append(res_round)

        if verbose_rounds:
            print(
                f"[Round {rnd:02d}/{rounds}] "
                f"Acc={res_round['Accuracy']:.4f} | "
                f"F1={res_round['F1-score']:.4f} | "
                f"Prec={res_round['Precision']:.4f} | "
                f"Sens={res_round['Sensitivity (Recall)']:.4f} | "
                f"Spec={res_round['Specificity']:.4f} | "
                f"AUC={res_round['AUC']:.4f} | "
                f"Thr={threshold_fixed:.2f}"
            )

    df_history = pd.DataFrame(history_rounds)
    y_proba_final = global_model.predict(X_test_global, verbose=0).ravel()
    res_final = _compute_metrics(y_test_global, y_proba_final, threshold_fixed)
    res_final["Threshold"] = threshold_fixed
    return global_model, df_history, res_final
