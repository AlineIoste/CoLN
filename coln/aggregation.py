import numpy as np

def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)  # stability
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / s if s != 0 else np.ones_like(ex) / max(len(ex), 1)

def combined_learning_coln(
    models_weights,
    r_h_list,
    c: float,
    lambda_beta: float,
    use_layer_c: bool,
    gamma: float = 1.0,
):
    """CoLN aggregation (as described in the paper).

    Parameters
    ----------
    models_weights : list[list[np.ndarray]]
        List of clients; each client is a list of layer weight arrays as returned by
        `tf.keras.Model.get_weights()`.
        Shape: [H][L].
    r_h_list : list[float]
        Client reliability/importance scores (one per client). Must have length H.
        Common choices: normalized local sample counts, local AUROC, etc.
    c : float
        Sharpness parameter for the softmax weighting.
    lambda_beta : float
        Strength of the non-convex term (beta). Use 0.0 to disable.
    use_layer_c : bool
        If True, adjusts c per layer using LayerDistance.
    gamma : float
        Exponent applied to the softmax weights (paper notation).

    Returns
    -------
    list[np.ndarray]
        Aggregated weights (list of arrays), compatible with `model.set_weights()`.

    Notes
    -----
    The beta term is O(m_l * H^2) per layer (m_l = number of parameters in the layer),
    which can be expensive for large layers.
    """
    if not models_weights:
        raise ValueError("models_weights is empty.")
    H = len(models_weights)
    if H != len(r_h_list):
        raise ValueError(f"len(r_h_list)={len(r_h_list)} must equal number of clients H={H}.")
    L = len(models_weights[0])
    if any(len(mw) != L for mw in models_weights):
        raise ValueError("All clients must have the same number of layers (same get_weights() structure).")

    r_h_arr = np.asarray(r_h_list, dtype=np.float64)

    # ---- α^h (global) ----
    alpha_logits = c * r_h_arr
    alpha_h_base = _softmax_stable(alpha_logits) ** float(gamma)
    alpha_h_base = alpha_h_base / np.sum(alpha_h_base)

    combined_weights = []

    for layer_idx in range(L):
        layer_weights = [mw[layer_idx] for mw in models_weights]
        layer_shape = layer_weights[0].shape
        if any(w.shape != layer_shape for w in layer_weights):
            raise ValueError(f"Layer {layer_idx} shape mismatch across clients.")
        m_l = int(np.prod(layer_shape)) or 1
        flat = np.asarray([w.reshape(-1) for w in layer_weights], dtype=np.float64)

        # ---- LayerDistance ----
        if H >= 2:
            pair_sum = 0.0
            for i in range(H):
                for j in range(i + 1, H):
                    diff = flat[i] - flat[j]
                    pair_sum += np.dot(diff, diff)
            layer_dist = np.sqrt(pair_sum) / m_l
        else:
            layer_dist = 0.0

        # ---- Layer-adjusted α (optional) ----
        if use_layer_c and layer_dist > 1e-12:
            local_c = c / (1.0 + layer_dist)
            alpha = _softmax_stable(local_c * r_h_arr) ** float(gamma)
            alpha_h = alpha / np.sum(alpha)
        else:
            alpha_h = alpha_h_base

        # ---- Convex aggregation ----
        combined_flat = np.zeros(m_l, dtype=np.float64)
        for h in range(H):
            combined_flat += alpha_h[h] * flat[h]

        # ---- Non-convex beta term ----
        if H >= 2 and lambda_beta and lambda_beta > 0:
            for i in range(m_l):
                wd_sum = 0.0
                for j in range(H):
                    for k in range(j + 1, H):
                        wd_sum += (flat[j, i] * r_h_arr[j] - flat[k, i] * r_h_arr[k]) ** 2
                weight_dist = np.sqrt(wd_sum)
                beta = weight_dist if weight_dist < layer_dist else 0.0
                combined_flat[i] += float(lambda_beta) * beta

        combined_layer = combined_flat.reshape(layer_shape).astype(layer_weights[0].dtype, copy=False)
        combined_weights.append(combined_layer)

    return combined_weights
