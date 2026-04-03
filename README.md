# HUSK: Hedonic Unfolding with Shrinkage Kernel

Open-source implementation of the HUSK framework for probabilistic unfolding of hedonic data (liking, purchase intent, acceptability).

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

The `results/` directory contains 1,790 cross-validated experiments reproducible with `fit_husk` (BFGS and Adam optimizers with none, per_consumer_bias, and per_consumer_affine transforms). The paper describes a broader evaluation of 2,237 experiments that also includes LSA (Ennis) and other optimizers for comparison. See `results/results_v2.tsv` (33-column structured TSV).

## Theoretical Background

The HUSK kernel is the expected value of Shepard's (1987) generalization gradient under Thurstonian noise:

```
E[L_ij] = kappa^(-d/2) * exp(-D^2_ij / kappa)
kappa = 1 + 2(tau^2_i + sigma^2_j)
```

where D is the distance between consumer i and product j in d-dimensional latent space, tau^2 is per-consumer variance, and sigma^2 is per-product variance. This formula is the moment generating function of a noncentral chi-squared distribution evaluated at the appropriate point (Ennis & Johnson, 1993).

HUSK builds directly on Landscape Segmentation Analysis (Ennis, 1993), which solved the degeneracy problem that had made multidimensional unfolding impractical for applied work. LSA introduced three ideas that proved critical: sigmoid-bounded predictions, per-consumer biases, and multi-start optimization. These innovations made preference mapping viable for the food and consumer products industries, where LSA has been the standard tool for three decades (Ennis & Ennis, 2013).

HUSK takes the theoretical kernel that underlies LSA (Ennis & Johnson, 1993), derived from Coombs's (1964) ideal point theory, Thurstone's (1927) discriminal process, and Shepard's (1987) universal law of generalization, and adds three modern techniques: L2 shrinkage on log-variance parameters to prevent degeneracy, a dimensionality stability correction for 3D analyses, and adaptive response transforms that can approximate LSA's built-in bias structure when the data calls for it.

The relationship between HUSK and LSA is one of generalization. LSA's structural commitments (sigmoid bounding, built-in biases) are well-chosen defaults validated by decades of applied success. HUSK makes those commitments configurable, confirming their effectiveness while extending the model to conditions where different choices prove optimal. For details, see Ennis and Ennis (in preparation), "HUSK: An open-source model for hedonic unfolding with shrinkage kernel."

## Citation

```bibtex
@article{ennis2026husk,
  title={HUSK: An Open-Source Model for Hedonic Unfolding with Shrinkage Kernel},
  author={Ennis, John M. and Ennis, Daniel M.},
  note={In preparation},
  year={2026}
}
```

## References

Andrich, D. (1988). The application of an unfolding model of the PIRT type to the measurement of attitude. *Applied Psychological Measurement*, 12(1), 33-51.

Bechtel, G. G. (1968). Folded and unfolded scaling from preferential paired comparisons. *Journal of Mathematical Psychology*, 5(2), 333-357.

Brockhoff, P. B., & Skovgaard, I. M. (1994). Modelling individual differences between assessors in sensory evaluations. *Food Quality and Preference*, 5(3), 215-224.

Busing, F. M. T. A., Groenen, P. J., & Heiser, W. J. (2005). Avoiding degeneracy in multidimensional unfolding by penalizing on the coefficient of variation. *Psychometrika*, 70(1), 71-98.

Busing, F. M. T. A., Heiser, W. J., & Cleaver, G. (2010). Restricted unfolding: Preference analysis with optimal transformations of preferences and attributes. *Food Quality and Preference*, 21(1), 82-92.

Coombs, C. H. (1964). *A Theory of Data*. Wiley.

De Soete, G., Carroll, J. D., & DeSarbo, W. S. (1986). The wandering ideal point model: A probabilistic multidimensional unfolding model for paired comparisons data. *Journal of Mathematical Psychology*, 30(1), 28-41.

DeSarbo, W. S., & Rao, V. R. (1984). GENFOLD2: A set of models and algorithms for the general unfolding analysis of preference/dominance data. *Journal of Classification*, 1(1), 147-186.

DeSarbo, W. S., Green, P. E., & Carroll, J. D. (1986). An alternating least-squares procedure for estimating missing preference data in product-concept testing. *Decision Sciences*, 17(2), 163-185.

Engelhard, G. (2023). Functional approaches for modeling unfolding data. *Educational and Psychological Measurement*, 83(6), 1139-1159.

Ennis, D. M. (1993). The mapping of sensory and liking data. *Food Quality and Preference*, 4(3), 149-159.

Ennis, D. M., & Ennis, J. M. (2013). Mapping hedonic data: A process perspective. *Journal of Sensory Studies*, 28(5), 324-334.

Ennis, D. M., & Johnson, N. L. (1993). Thurstone-Shepard similarity models as special cases of moment generating functions. *Journal of Mathematical Psychology*, 37(1), 104-110.

Ennis, D. M., & Johnson, N. L. (1994). A general model for preferential and triadic choice in terms of central F distribution functions. *Psychometrika*, 59, 91-96.

Ennis, D. M., & Mullen, K. (1986). A multivariate model for discrimination methods. *Journal of Mathematical Psychology*, 30, 206-219.

Ennis, D. M., Palen, J. J., & Mullen, K. (1988). A multidimensional stochastic theory of similarity. *Journal of Mathematical Psychology*, 32, 449-465.

Karpathy, A. (2026). autoresearch [Computer software]. GitHub. https://github.com/karpathy/autoresearch

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

Luo, G. (2001). A class of probabilistic unfolding models for polytomous responses. *Journal of Mathematical Psychology*, 45(2), 224-248.

France, S. L., Vaghefi, M. S., & Batchelder, W. H. (2018). FlexCCT: A methodological framework and software for ratings analysis and wisdom of the crowd applications. *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 48(12), 2321-2335.

Gower, J. C., & Dijksterhuis, G. B. (2004). *Procrustes Problems*. Oxford University Press.

MacKay, D. B. (2001). Probabilistic unfolding models for sensory data. *Food Quality and Preference*, 12(5-7), 427-436.

MacKay, D. B., & Zinnes, J. L. (1995). Probabilistic multidimensional unfolding: An anisotropic model for preference ratio judgments. *Journal of Mathematical Psychology*, 39(1), 99-111.

Meullenet, J. F., Xiong, R., & Findlay, C. J. (2008). *Multivariate and Probabilistic Analyses of Sensory Science Problems*. Wiley-Blackwell.

Park, J., DeSarbo, W. S., & Liechty, J. (2008). A hierarchical Bayesian multidimensional scaling methodology for accommodating both structural and preference heterogeneity. *Psychometrika*, 73, 451-472.

Park, J., DeSarbo, W. S., & Rajagopal, P. (2012). A new heterogeneous multidimensional unfolding procedure. *Psychometrika*, 77(2), 263-287.

Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000). A general item response theory model for unfolding unidimensional polytomous responses. *Applied Psychological Measurement*, 24(1), 3-32.

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000). A general item response theory model for unfolding unidimensional polytomous responses. *Applied Psychological Measurement*, 24(1), 3-32.

Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*, 237(4820), 1317-1323.

Thurstone, L. L. (1927). A law of comparative judgment. *Psychological Review*, 34(4), 273-286.

Wold, S. (1978). Cross-validatory estimation of the number of components in factor and principal components models. *Technometrics*, 20(4), 397-405.

Zinnes, J. L., & MacKay, D. B. (1983). Probabilistic multidimensional scaling: Complete and incomplete data. *Psychometrika*, 48(1), 27-48.

## License

MIT
