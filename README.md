# HUSK: Hedonic Unfolding with Shrinkage Kernel

Open-source implementation of the HUSK framework for probabilistic unfolding of hedonic data.

HUSK applies the Ennis-Johnson (1993) Thurstone-Shepard kernel with three innovations:
- **L2 shrinkage** on log-variance parameters to prevent degeneracy
- **Dimensionality stability** via sigma_sq=1.0 initialization for 3D analyses
- **Adaptive response transforms** (per-consumer bias or per-consumer affine)

## Quick Start

```python
from husk import fit_husk
import numpy as np

# Your ratings matrix: consumers x products, NaN for missing
ratings = np.array([
    [0.8, 0.3, np.nan, 0.6],
    [0.2, 0.9, 0.7, np.nan],
    [0.5, 0.4, 0.6, 0.8],
])

result = fit_husk(ratings, n_dims=3, sigma_sq=1.0, response_transform='per_consumer_bias')

print(result['product_positions'])    # J x 3 array
print(result['consumer_positions'])   # I x 3 array
print(result['predicted_ratings'])    # I x J array
print(result['rating_corr'])         # Pearson correlation
```

## Recommended Configurations

| Consumers | Config | Time |
|-----------|--------|------|
| < 150 | `fit_husk(ratings, optimizer='bfgs', response_transform='per_consumer_bias')` | 0.02s |
| >= 150 | `fit_husk(ratings, optimizer='adam', response_transform='per_consumer_affine')` | 0.6s |

Both use `n_dims=3`, `sigma_sq=1.0`, `shrinkage=0.005` by default.

## Cross-Validation

```python
from husk.evaluate import evaluate_cv

result = evaluate_cv(ratings, optimizer='bfgs', response_transform='per_consumer_bias',
                     n_dims=3, k=5, seed=42)
print(f"Held-out correlation: {result['out_corr_mean']:.4f}")
```

## Autoresearch Results

The `results/` directory contains the full corpus of 2,237 cross-validated experiments
described in the paper. See `results/results_v2.tsv` (33-column structured TSV).

## Citation

```bibtex
@article{ennis2026husk,
  title={HUSK: Hedonic Unfolding with Shrinkage Kernel},
  author={Ennis, John M. and Ennis, Daniel M.},
  journal={Journal of Mathematical Psychology},
  year={2026}
}
```

## References

- Ennis, D. M., & Johnson, N. L. (1993). Thurstone-Shepard similarity models as special cases of moment generating functions. *Journal of Mathematical Psychology*, 37(1), 104-110.
- Karpathy, A. (2026). autoresearch. GitHub. https://github.com/karpathy/autoresearch

## License

MIT
