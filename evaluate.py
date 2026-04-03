"""
evaluate.py — Cross-validation harness and data generation for HUSK.

Provides dataset generation from both probabilistic and Ennis
data-generating processes, k-fold cross-validation at the rating
level, and structured result formatting.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from husk import fit_husk


# ============================================================================
# Data Generation
# ============================================================================

def generate_dataset_probabilistic(
    n_consumers: int, n_products: int, n_segments: int, n_dims: int,
    noise: float, missing_rate: float, seed: int
) -> np.ndarray:
    """Generate data from the Probabilistic model (Model A).
    Returns ratings in (0, 1] with NaN for missing.
    """
    rng = np.random.default_rng(seed)
    segment_centers = rng.normal(0, 1.5, (n_segments, n_dims))
    product_pos = rng.normal(0, 1.5, (n_products, n_dims))
    assignments = rng.integers(0, n_segments, n_consumers)
    consumer_pos = segment_centers[assignments] + rng.normal(0, 0.5, (n_consumers, n_dims))

    tau_sq = rng.uniform(0.1, 0.8, n_consumers)
    sigma_sq = rng.uniform(0.1, 0.5, n_products)

    dist_sq = np.sum((consumer_pos[:, np.newaxis, :] - product_pos[np.newaxis, :, :]) ** 2, axis=2)
    omega_sq = tau_sq[:, np.newaxis] + sigma_sq[np.newaxis, :]
    kappa = 1.0 + 2.0 * omega_sq
    base = (kappa ** (-n_dims / 2.0)) * np.exp(-dist_sq / kappa)

    ratings = base + rng.normal(0, noise, (n_consumers, n_products))
    ratings = np.clip(ratings, 0.01, 0.99)

    mask = rng.random((n_consumers, n_products)) > missing_rate
    ratings[~mask] = np.nan
    return ratings


def generate_dataset_ennis(
    n_consumers: int, n_products: int, n_segments: int, n_dims: int,
    noise: float, missing_rate: float, seed: int
) -> np.ndarray:
    """Generate data from the Ennis model (Model B).
    Returns ratings in (0, 1] with NaN for missing.
    """
    rng = np.random.default_rng(seed)
    segment_centers = rng.normal(0, 1.5, (n_segments, n_dims))
    product_pos = rng.normal(0, 1.5, (n_products, n_dims))
    assignments = rng.integers(0, n_segments, n_consumers)
    consumer_pos = segment_centers[assignments] + rng.normal(0, 0.5, (n_consumers, n_dims))

    biases = rng.normal(0, 0.5, n_consumers)
    sigmas = rng.uniform(0.3, 1.5, n_products)

    dist_sq = np.sum((consumer_pos[:, np.newaxis, :] - product_pos[np.newaxis, :, :]) ** 2, axis=2)
    sigma_sq = sigmas ** 2
    kernel = np.exp(-dist_sq / (2 * sigma_sq[np.newaxis, :] + 1e-8))
    logit = biases[:, np.newaxis] + 2 * kernel - 1
    base = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    ratings = base + rng.normal(0, noise, (n_consumers, n_products))
    ratings = np.clip(ratings, 0.01, 0.99)

    mask = rng.random((n_consumers, n_products)) > missing_rate
    ratings[~mask] = np.nan
    return ratings


# ============================================================================
# Cross-Validation
# ============================================================================

def create_cv_folds(ratings: np.ndarray, k: int = 5, seed: int = 42) -> List[np.ndarray]:
    """Create k-fold CV splits over observed ratings.
    Returns list of k arrays, each containing (i, j) indices of held-out entries.
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(ratings)
    observed = np.argwhere(mask)
    perm = rng.permutation(len(observed))
    fold_size = len(observed) // k

    folds = []
    for f in range(k):
        start = f * fold_size
        end = start + fold_size if f < k - 1 else len(observed)
        folds.append(observed[perm[start:end]])
    return folds


def evaluate_fold(
    ratings: np.ndarray, held_out_indices: np.ndarray,
    optimizer: str, response_transform: str,
    n_dims: int = 2, seed: int = 42, num_runs: int = 1,
    learning_rate: float = 0.06, sigma_sq: float = 0.3,
    n_outer_iter: int = None, bfgs_maxiter: int = None,
    gn_max_feval: int = None, newton_maxiter: int = None,
    shrinkage: float = 0.005, tau_lr_multiplier: float = 0.5,
    use_prefit: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Fit model on training data, evaluate on held-out ratings.
    Accepts the FULL hyperparameter set from LSAConfig.
    """
    # Create training ratings (mask held-out)
    train = ratings.copy()
    for i, j in held_out_indices:
        train[i, j] = np.nan

    # Check minimum ratings per entity
    train_mask = ~np.isnan(train)
    if np.any(train_mask.sum(axis=1) < 3) or np.any(train_mask.sum(axis=0) < 3):
        return {'skip': True, 'reason': 'sparse fold'}

    # Fit via HUSK
    start = time.time()
    try:
        result = fit_husk(
            ratings=train,
            n_dims=n_dims,
            sigma_sq=sigma_sq,
            optimizer=optimizer,
            response_transform=response_transform,
            seed=seed,
            num_runs=num_runs,
            learning_rate=learning_rate,
            n_outer_iter=n_outer_iter,
            bfgs_maxiter=bfgs_maxiter,
            shrinkage=shrinkage,
        )
    except Exception as e:
        return {'skip': True, 'reason': str(e)[:100]}
    elapsed = time.time() - start

    pred = result['predicted_ratings']
    obs_mask = ~np.isnan(ratings)

    # In-sample metrics (training data only)
    train_obs = ratings[train_mask & obs_mask]
    train_pred = pred[train_mask & obs_mask]
    in_corr = float(np.nan_to_num(np.corrcoef(train_obs, train_pred)[0, 1])) if len(train_obs) > 5 else 0.0
    in_rmse = float(np.sqrt(np.mean((train_obs - train_pred) ** 2))) if len(train_obs) > 0 else 99.0

    # Held-out metrics
    ho_obs = np.array([ratings[i, j] for i, j in held_out_indices])
    ho_pred = np.array([pred[i, j] for i, j in held_out_indices])
    out_corr = float(np.nan_to_num(np.corrcoef(ho_obs, ho_pred)[0, 1])) if len(ho_obs) > 5 else 0.0
    out_rmse = float(np.sqrt(np.mean((ho_obs - ho_pred) ** 2))) if len(ho_obs) > 0 else 99.0

    return {
        'skip': False,
        'in_corr': in_corr, 'in_rmse': in_rmse,
        'out_corr': out_corr, 'out_rmse': out_rmse,
        'elapsed': elapsed,
    }


def evaluate_cv(
    ratings: np.ndarray, optimizer: str, response_transform: str,
    n_dims: int = 2, k: int = 5, seed: int = 42, num_runs: int = 1,
    learning_rate: float = 0.06, sigma_sq: float = 0.3,
    n_outer_iter: int = None, bfgs_maxiter: int = None,
    gn_max_feval: int = None, newton_maxiter: int = None,
    shrinkage: float = 0.005, tau_lr_multiplier: float = 0.5,
    use_prefit: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Run full k-fold cross-validation with ALL hyperparameters.
    Returns mean/std metrics.
    """
    folds = create_cv_folds(ratings, k=k, seed=seed)

    in_corrs, out_corrs, in_rmses, out_rmses, times = [], [], [], [], []
    skipped_folds = []

    for fold_idx, held_out in enumerate(folds):
        result = evaluate_fold(
            ratings, held_out, optimizer, response_transform,
            n_dims=n_dims, seed=seed + fold_idx, num_runs=num_runs,
            learning_rate=learning_rate, sigma_sq=sigma_sq,
            n_outer_iter=n_outer_iter, bfgs_maxiter=bfgs_maxiter,
            gn_max_feval=gn_max_feval, newton_maxiter=newton_maxiter,
            shrinkage=shrinkage, tau_lr_multiplier=tau_lr_multiplier,
            use_prefit=use_prefit,
        )
        if result['skip']:
            skipped_folds.append((fold_idx, result.get('reason', 'unknown')))
            continue
        in_corrs.append(result['in_corr'])
        out_corrs.append(result['out_corr'])
        in_rmses.append(result['in_rmse'])
        out_rmses.append(result['out_rmse'])
        times.append(result['elapsed'])

    if not out_corrs:
        return {'success': False, 'reason': 'all folds skipped'}

    # Fail if any fold was skipped — partial results are not comparable to full k-fold
    if skipped_folds:
        reasons = '; '.join(f"fold {f}: {r}" for f, r in skipped_folds)
        return {'success': False, 'reason': f'{len(skipped_folds)}/{k} folds skipped ({reasons})'}

    return {
        'success': True,
        'folds_completed': len(out_corrs),
        'in_corr_mean': float(np.mean(in_corrs)),
        'in_corr_std': float(np.std(in_corrs)),
        'out_corr_mean': float(np.mean(out_corrs)),
        'out_corr_std': float(np.std(out_corrs)),
        'in_rmse_mean': float(np.mean(in_rmses)),
        'out_rmse_mean': float(np.mean(out_rmses)),
        'elapsed_mean': float(np.mean(times)),
        'elapsed_total': float(np.sum(times)),
        'overfit_gap': float(np.mean(in_corrs) - np.mean(out_corrs)),
    }


# ============================================================================
# Standard Datasets
# ============================================================================

STANDARD_DATASETS = [
    # 2D data (original)
    {'name': 'small_low_noise',     'nc': 30,  'np': 6,  'ns': 2, 'noise': 0.03, 'miss': 0.00, 'gen_dims': 2},
    {'name': 'small_high_noise',    'nc': 30,  'np': 6,  'ns': 2, 'noise': 0.10, 'miss': 0.10, 'gen_dims': 2},
    {'name': 'medium_low_noise',    'nc': 50,  'np': 8,  'ns': 3, 'noise': 0.05, 'miss': 0.00, 'gen_dims': 2},
    {'name': 'medium_moderate',     'nc': 50,  'np': 10, 'ns': 3, 'noise': 0.05, 'miss': 0.05, 'gen_dims': 2},
    {'name': 'medium_high_noise',   'nc': 50,  'np': 8,  'ns': 3, 'noise': 0.10, 'miss': 0.10, 'gen_dims': 2},
    {'name': 'large_low_noise',     'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.05, 'miss': 0.05, 'gen_dims': 2},
    {'name': 'large_moderate',      'nc': 100, 'np': 8,  'ns': 2, 'noise': 0.03, 'miss': 0.00, 'gen_dims': 2},
    {'name': 'large_high_noise',    'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.10, 'miss': 0.10, 'gen_dims': 2},
    {'name': 'xlarge_low_noise',    'nc': 150, 'np': 15, 'ns': 4, 'noise': 0.03, 'miss': 0.05, 'gen_dims': 2},
    {'name': 'xlarge_high_noise',   'nc': 150, 'np': 15, 'ns': 4, 'noise': 0.10, 'miss': 0.15, 'gen_dims': 2},

    # 3D data — true 3D structure the model needs 3 dims to capture
    {'name': 'medium_3d',           'nc': 50,  'np': 8,  'ns': 3, 'noise': 0.05, 'miss': 0.00, 'gen_dims': 3},
    {'name': 'large_3d',            'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.05, 'miss': 0.05, 'gen_dims': 3},
    {'name': 'large_3d_noisy',      'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.10, 'miss': 0.10, 'gen_dims': 3},
    {'name': 'xlarge_3d',           'nc': 150, 'np': 15, 'ns': 4, 'noise': 0.03, 'miss': 0.05, 'gen_dims': 3},

    # 4D data — model can only fit 2D or 3D, so there's always lost signal
    {'name': 'medium_4d',           'nc': 50,  'np': 8,  'ns': 3, 'noise': 0.05, 'miss': 0.00, 'gen_dims': 4},
    {'name': 'large_4d',            'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.05, 'miss': 0.05, 'gen_dims': 4},
    {'name': 'large_4d_noisy',      'nc': 80,  'np': 12, 'ns': 4, 'noise': 0.10, 'miss': 0.10, 'gen_dims': 4},
    {'name': 'xlarge_4d',           'nc': 150, 'np': 15, 'ns': 4, 'noise': 0.03, 'miss': 0.05, 'gen_dims': 4},
]


def get_dataset(name: str, dgp: str = 'probabilistic', seed: int = 42) -> np.ndarray:
    """Get a standard dataset by name and DGP.
    gen_dims controls the dimensionality of the data-generating process.
    This is SEPARATE from n_dims used for fitting.
    """
    for ds in STANDARD_DATASETS:
        if ds['name'] == name:
            gen = generate_dataset_probabilistic if dgp == 'probabilistic' else generate_dataset_ennis
            gen_dims = ds.get('gen_dims', 2)
            return gen(ds['nc'], ds['np'], ds['ns'], gen_dims, ds['noise'], ds['miss'], seed)
    raise ValueError(f"Unknown dataset: {name}. Available: {[d['name'] for d in STANDARD_DATASETS]}")


# ============================================================================
# The full hyperparameter space (for reference by the agent)
# ============================================================================

HYPERPARAMETER_SPACE = {
    # Model choice (HUSK supports bfgs and adam only; newton/gauss_newton/ennis are in lsa.py)
    'optimizer': ['bfgs', 'adam'],
    'response_transform': ['none', 'per_consumer_bias', 'per_consumer_affine'],

    # Shared hyperparameters
    'n_dims': [2, 3],
    'sigma_sq': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],       # initial product variance
    'num_runs': [1, 3, 5, 10],                            # multi-start restarts
    'use_prefit': [False, True],                          # super-consumer init

    # Adam-specific
    'learning_rate': [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
    'tau_lr_multiplier': [0.25, 0.5, 0.75, 1.0],         # variance LR relative to position LR
    'n_outer_iter': [20, 40, 60, 80, 100, 120, 150],     # Adam iterations
    'shrinkage': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],  # L2 regularization on variances

    # BFGS-specific
    'bfgs_maxiter': [50, 100, 200, 500, 1000],

    # Gauss-Newton-specific
    'gn_max_feval': [50, 100, 250, 500, 1000],

    # Newton-specific
    'newton_maxiter': [20, 40, 80, 150, 300],

    # Data conditions
    'dataset': [d['name'] for d in STANDARD_DATASETS],
    'dgp': ['probabilistic', 'ennis'],
}


# ============================================================================
# Reporting
# ============================================================================

TSV_V2_COLUMNS = [
    'run_id', 'run_group', 'optimizer', 'transform', 'dataset', 'dgp',
    'n_dims', 'sigma_sq', 'num_runs', 'learning_rate', 'n_outer_iter',
    'bfgs_maxiter', 'gn_max_feval', 'newton_maxiter',
    'shrinkage', 'tau_lr_multiplier', 'use_prefit',
    'k_folds', 'cv_seed', 'folds_completed',
    'run_outcome', 'out_corr', 'out_corr_std', 'in_corr', 'overfit_gap',
    'out_rmse', 'elapsed_mean',
    # Dataset metadata for reproducibility (added v2.1)
    'ds_n_consumers', 'ds_n_products', 'ds_n_segments', 'ds_noise',
    'ds_missing_rate', 'ds_gen_dims',
]

# Lookup table: dataset name → generation parameters
_DS_PARAMS = {d['name']: d for d in STANDARD_DATASETS}


def _get_dataset_params(dataset_name: str) -> Dict[str, Any]:
    """Resolve dataset name to generation parameters. Returns empty dict for non-standard names."""
    ds = _DS_PARAMS.get(dataset_name)
    if ds:
        return {
            'ds_n_consumers': ds['nc'], 'ds_n_products': ds['np'],
            'ds_n_segments': ds['ns'], 'ds_noise': ds['noise'],
            'ds_missing_rate': ds['miss'], 'ds_gen_dims': ds.get('gen_dims', 2),
        }
    return {}

TSV_V2_HEADER = '\t'.join(TSV_V2_COLUMNS)

import uuid as _uuid

def make_run_id(run_group: str) -> str:
    """Generate a globally unique run ID that survives process restarts."""
    short_uuid = _uuid.uuid4().hex[:8]
    return f"{run_group}_{short_uuid}"


def format_result_v2(config: Dict, result: Dict, run_group: str = 'default') -> str:
    """Format a single experiment as a structured TSV row with all metadata."""
    run_id = make_run_id(run_group)

    ds_params = _get_dataset_params(config.get('dataset', ''))

    if not result.get('success', False):
        vals = {col: '' for col in TSV_V2_COLUMNS}
        vals['run_id'] = run_id
        vals['run_group'] = run_group
        vals['optimizer'] = config.get('optimizer', '')
        vals['transform'] = config.get('response_transform', config.get('transform', ''))
        vals['dataset'] = config.get('dataset', '')
        vals['dgp'] = config.get('dgp', '')
        vals['run_outcome'] = 'skip'
        vals['out_corr'] = '0.0000'
        vals.update(ds_params)
        return '\t'.join(str(vals.get(c, '')) for c in TSV_V2_COLUMNS)

    oc = result['out_corr_mean']
    ic = result['in_corr_mean']
    gap = result['overfit_gap']
    ormse = result['out_rmse_mean']
    t = result['elapsed_mean']
    oc_std = result.get('out_corr_std', 0.0)

    vals = {
        'run_id': run_id,
        'run_group': run_group,
        'optimizer': config.get('optimizer', ''),
        'transform': config.get('response_transform', config.get('transform', '')),
        'dataset': config.get('dataset', ''),
        'dgp': config.get('dgp', ''),
        'n_dims': config.get('n_dims', 2),
        'sigma_sq': config.get('sigma_sq', 0.3),
        'num_runs': config.get('num_runs', 1),
        'learning_rate': config.get('learning_rate', 0.06),
        'n_outer_iter': config.get('n_outer_iter', ''),
        'bfgs_maxiter': config.get('bfgs_maxiter', ''),
        'gn_max_feval': config.get('gn_max_feval', ''),
        'newton_maxiter': config.get('newton_maxiter', ''),
        'shrinkage': config.get('shrinkage', 0.005),
        'tau_lr_multiplier': config.get('tau_lr_multiplier', 0.5),
        'use_prefit': config.get('use_prefit', False),
        'k_folds': config.get('k_folds', config.get('k', 5)),
        'cv_seed': config.get('seed', 42),
        'folds_completed': result.get('folds_completed', ''),
        'run_outcome': 'success',
        'out_corr': f"{oc:.4f}",
        'out_corr_std': f"{oc_std:.4f}",
        'in_corr': f"{ic:.4f}",
        'overfit_gap': f"{gap:.4f}",
        'out_rmse': f"{ormse:.4f}",
        'elapsed_mean': f"{t:.3f}",
    }
    vals.update(ds_params)

    return '\t'.join(str(vals.get(c, '')) for c in TSV_V2_COLUMNS)


# Keep old format for backward compatibility
TSV_HEADER = "tag\tout_corr\tin_corr\toverfit_gap\tout_rmse\ttime_s\tstatus\tdescription"

def format_result_tsv(tag: str, result: Dict, description: str = '') -> str:
    """Legacy format — use format_result_v2 for new runs."""
    if not result.get('success', False):
        return f"{tag}\t0.0000\t0.0000\t0.0000\t0.0000\t0.00\tskip\t{result.get('reason','unknown')}"
    oc = result['out_corr_mean']
    ic = result['in_corr_mean']
    gap = result['overfit_gap']
    ormse = result['out_rmse_mean']
    t = result['elapsed_mean']
    return f"{tag}\t{oc:.4f}\t{ic:.4f}\t{gap:.4f}\t{ormse:.4f}\t{t:.2f}\tkeep\t{description}"


def print_result_block(tag: str, config: Dict, result: Dict, best_so_far: float = 0.0,
                        hypothesis: str = '', next_step: str = ''):
    """Print formatted result block for terminal output."""
    print("---")
    if result.get('success', False):
        oc = result['out_corr_mean']
        ic = result['in_corr_mean']
        gap = result['overfit_gap']
        t = result['elapsed_mean']
        status = 'keep' if oc > best_so_far else 'discard'
        improved = oc > best_so_far

        print(f"RESULT: {status}{'  *** NEW BEST ***' if improved else ''}")
        print(f"config: {tag}")
        print(f"out_corr: {oc:.4f}")
        print(f"in_corr: {ic:.4f}")
        print(f"overfit_gap: {gap:.4f}")
        print(f"out_rmse: {result['out_rmse_mean']:.4f}")
        print(f"time_s: {t:.2f}")
        print(f"folds: {result['folds_completed']}")
        print(f"best_so_far: {max(oc, best_so_far):.4f}")

        # Print non-default hyperparameters
        defaults = {'n_dims': 2, 'sigma_sq': 0.3, 'learning_rate': 0.06,
                    'tau_lr_multiplier': 0.5, 'shrinkage': 0.005, 'num_runs': 1,
                    'use_prefit': False, 'n_outer_iter': None, 'bfgs_maxiter': None,
                    'gn_max_feval': None, 'newton_maxiter': None}
        non_default = {k: v for k, v in config.items()
                       if k in defaults and v != defaults[k]
                       and k not in ('optimizer', 'transform', 'dataset', 'dgp', 'seed', 'k_folds')}
        if non_default:
            print(f"non_default_params: {non_default}")
    else:
        print(f"RESULT: skip")
        print(f"config: {tag}")
        print(f"reason: {result.get('reason', 'unknown')}")

    if hypothesis:
        print(f"hypothesis: {hypothesis}")
    if next_step:
        print(f"next: {next_step}")
    print("---")


if __name__ == "__main__":
    # Quick test
    print("Testing prepare.py...")
    ratings = generate_dataset_probabilistic(50, 8, 3, 2, 0.05, 0.0, 42)
    print(f"Dataset: {ratings.shape}, obs: {(~np.isnan(ratings)).sum()}")

    result = evaluate_cv(ratings, 'bfgs', 'none', n_dims=2, k=5, seed=42, num_runs=1)
    print(f"CV result: out_corr={result['out_corr_mean']:.4f}")

    # Test with non-default hyperparameters
    result2 = evaluate_cv(ratings, 'adam', 'per_consumer_bias', n_dims=2, k=5, seed=42,
                          num_runs=1, learning_rate=0.04, shrinkage=0.01, sigma_sq=0.5)
    print(f"Adam+pcb (lr=0.04, shrink=0.01, sig=0.5): out_corr={result2['out_corr_mean']:.4f}")
    print("OK")
