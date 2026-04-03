"""
HUSK: Hedonic Unfolding with Shrinkage Kernel

Probabilistic unfolding model for hedonic data (liking, purchase intent,
acceptability, and other consumer metrics).

Uses the Ennis-Johnson (1993) Thurstone-Shepard kernel:

    E[L_ij] = kappa^(-d/2) * exp(-D^2_ij / kappa)
    kappa = 1 + 2(tau^2_i + sigma^2_j)

with L2 shrinkage on log-variance parameters and adaptive response transforms.

Reference:
    Ennis, D. M., & Johnson, N. L. (1993). Thurstone-Shepard similarity
    models as special cases of moment generating functions. Journal of
    Mathematical Psychology, 37(1), 104-110.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional


def fit_husk(
    ratings: np.ndarray,
    n_dims: int = 3,
    sigma_sq: float = 1.0,
    optimizer: str = 'bfgs',
    response_transform: str = 'per_consumer_bias',
    shrinkage: float = 0.005,
    learning_rate: float = 0.06,
    seed: int = 42,
    num_runs: int = 1,
    n_outer_iter: Optional[int] = None,
    bfgs_maxiter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit the HUSK model to a hedonic ratings matrix.

    Parameters
    ----------
    ratings : np.ndarray
        I x J matrix of consumer-product ratings. NaN for missing.
        Ratings should be in (0, 1] range.
    n_dims : int
        Number of latent dimensions (2 or 3). Default 3.
    sigma_sq : float
        Initial product variance. Default 1.0 (prevents kernel collapse in 3D).
    optimizer : str
        'bfgs' (fast, recommended for < 150 consumers) or
        'adam' (accurate, recommended for 150+ consumers).
    response_transform : str
        'none', 'per_consumer_bias', or 'per_consumer_affine'.
    shrinkage : float
        L2 shrinkage on log-variance parameters. Default 0.005.
    learning_rate : float
        Adam learning rate. Ignored for BFGS. Default 0.06.
    seed : int
        Random seed for reproducibility.
    num_runs : int
        Multi-start runs (best of N). Default 1 for BFGS.
    n_outer_iter : int, optional
        Adam iterations. None = auto (60 for 2D, 80 for 3D).
    bfgs_maxiter : int, optional
        BFGS max iterations. None = 100.

    Returns
    -------
    dict with keys:
        product_positions : np.ndarray (J, d)
        consumer_positions : np.ndarray (I, d)
        tau_sq : np.ndarray (I,) per-consumer variance
        sigma_sq : np.ndarray (J,) per-product variance
        predicted_ratings : np.ndarray (I, J)
        rating_corr : float
        rating_rmse : float
        response_transform : str
        transform_alpha : np.ndarray or None
        transform_beta : np.ndarray or None
    """
    rng = np.random.default_rng(seed)
    I, J = ratings.shape
    obs_mask = ~np.isnan(ratings)
    n_obs = obs_mask.sum()

    if n_obs < 10:
        raise ValueError(f"Too few observed ratings ({n_obs}). Need at least 10.")

    # Initialize parameters
    consumer_pos = rng.normal(0, 0.5, (I, n_dims))
    product_pos = rng.normal(0, 0.5, (J, n_dims))
    log_tau_sq = np.full(I, np.log(0.3))
    log_sigma_sq = np.full(J, np.log(sigma_sq))

    def _pack(cp, pp, lt, ls):
        return np.concatenate([cp.ravel(), pp.ravel(), lt, ls])

    def _unpack(x):
        idx = 0
        cp = x[idx:idx + I * n_dims].reshape(I, n_dims); idx += I * n_dims
        pp = x[idx:idx + J * n_dims].reshape(J, n_dims); idx += J * n_dims
        lt = x[idx:idx + I]; idx += I
        ls = x[idx:idx + J]
        return cp, pp, lt, ls

    def _predict(cp, pp, lt, ls):
        tau_sq = np.exp(lt)
        sig_sq = np.exp(ls)
        dist_sq = np.sum((cp[:, np.newaxis, :] - pp[np.newaxis, :, :]) ** 2, axis=2)
        kappa = 1.0 + 2.0 * (tau_sq[:, np.newaxis] + sig_sq[np.newaxis, :])
        pred = (kappa ** (-n_dims / 2.0)) * np.exp(-dist_sq / kappa)
        return pred

    def _loss(x):
        cp, pp, lt, ls = _unpack(x)
        pred = _predict(cp, pp, lt, ls)
        residuals = (pred - ratings)[obs_mask]
        mse = np.mean(residuals ** 2)
        # L2 shrinkage on log-variances toward their means
        shrink_tau = shrinkage * np.mean((lt - np.mean(lt)) ** 2) if shrinkage > 0 else 0
        shrink_sig = shrinkage * np.mean((ls - np.mean(ls)) ** 2) if shrinkage > 0 else 0
        return mse + shrink_tau + shrink_sig

    if optimizer == 'bfgs':
        x0 = _pack(consumer_pos, product_pos, log_tau_sq, log_sigma_sq)
        maxiter = bfgs_maxiter or 100
        result = minimize(_loss, x0, method='L-BFGS-B', options={'maxiter': maxiter})
        cp, pp, lt, ls = _unpack(result.x)
    elif optimizer == 'adam':
        iters = n_outer_iter or (60 if n_dims == 2 else 80)
        x = _pack(consumer_pos, product_pos, log_tau_sq, log_sigma_sq)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, iters + 1):
            grad = np.zeros_like(x)
            fx = _loss(x)
            h = 1e-5
            for i in range(len(x)):
                x[i] += h
                grad[i] = (_loss(x) - fx) / h
                x[i] -= h
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        cp, pp, lt, ls = _unpack(x)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Use 'bfgs' or 'adam'.")

    # Compute predictions
    pred = _predict(cp, pp, lt, ls)

    # Apply response transform
    transform_alpha = None
    transform_beta = None

    if response_transform == 'per_consumer_bias':
        transform_beta = np.zeros(I)
        for i in range(I):
            mask_i = obs_mask[i]
            if mask_i.sum() > 0:
                p = pred[i, mask_i]
                r = ratings[i, mask_i]
                transform_beta[i] = np.dot(p, r) / (np.dot(p, p) + 1e-10)
            else:
                transform_beta[i] = 1.0
        pred = transform_beta[:, np.newaxis] * pred

    elif response_transform == 'per_consumer_affine':
        transform_alpha = np.zeros(I)
        transform_beta = np.zeros(I)
        for i in range(I):
            mask_i = obs_mask[i]
            if mask_i.sum() > 1:
                p = pred[i, mask_i]
                r = ratings[i, mask_i]
                A = np.column_stack([np.ones(mask_i.sum()), p])
                params = np.linalg.lstsq(A, r, rcond=None)[0]
                transform_alpha[i] = params[0]
                transform_beta[i] = params[1]
            else:
                transform_alpha[i] = 0.0
                transform_beta[i] = 1.0
        pred = transform_alpha[:, np.newaxis] + transform_beta[:, np.newaxis] * pred

    # Metrics on observed ratings
    obs_pred = pred[obs_mask]
    obs_true = ratings[obs_mask]
    corr = float(np.corrcoef(obs_pred, obs_true)[0, 1]) if len(obs_true) > 5 else 0.0
    rmse = float(np.sqrt(np.mean((obs_pred - obs_true) ** 2)))

    return {
        'product_positions': pp,
        'consumer_positions': cp,
        'tau_sq': np.exp(lt),
        'sigma_sq': np.exp(ls),
        'predicted_ratings': pred,
        'rating_corr': corr,
        'rating_rmse': rmse,
        'response_transform': response_transform,
        'transform_alpha': transform_alpha,
        'transform_beta': transform_beta,
    }
