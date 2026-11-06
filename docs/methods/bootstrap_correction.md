# Bootstrap Correction for Ranking Bias

Core module: `src/llm_wc/wc/core.py`

## Problem

When models are ranked by empirical benchmark scores, the top-ranked model's score is systematically inflated. This is the **winner's curse**: the model that happened to get lucky on this particular sample of tasks appears better than it truly is. The bias is strongest at the extremes of the ranking.

## API

```python
from llm_wc.wc.core import bootstrap_correction, BootstrapResult

result: BootstrapResult = bootstrap_correction(
    obs=obs,                        # (N, J) array: N tasks, J models
    n_bootstraps=5000,              # number of bootstrap resamples
    boot_size=None,                 # resample size m (default: N or auto)
    alpha=0.05,                     # FWER level for simultaneous CIs
    compute_partial_order=True,     # set False to skip CI/tier computation
    eps_se=1e-12,                   # floor for SEs to avoid divide-by-zero
    max_t_chunk=512,                # chunk size for max-t computation
    cluster_labels=None,            # (N,) int array — cluster per task
    bootstrap_method=None,          # "iid" | "moon" | "cluster" | "wild_cluster" | "stratified" | "stratified_moon" | None (auto)
    wild_weight_dist="webb",        # "rademacher" | "mammen" | "webb"
)
```

### Input

- **`obs`**: `np.ndarray` of shape `(N, J)` where `N` is the number of tasks and `J` is the number of models. Each entry is the outcome of model `j` on task `i` (e.g., 0/1 for binary benchmarks, or a continuous score).
- **`n_bootstraps`**: Number of bootstrap resamples. 5000-10000 is typical.
- **`boot_size`**: Resample size `m`. For `"iid"`, defaults to `N`. For `"cluster"`, defaults to `C` (or `⌊C^0.95⌋` when `C ≥ 50`). For `"moon"`, must be set explicitly to a value `< N`. Ignored for `"wild_cluster"` and `"stratified"`.
- **`alpha`**: Family-wise error rate for simultaneous pairwise CIs. Default 0.05 gives 95% coverage.
- **`compute_partial_order`**: If `False`, skips the CI/dominance/tier computation and returns only the basic correction. Useful when you only need corrected scores.
- **`eps_se`**: Floor for pairwise gap standard errors to avoid divide-by-zero. Default `1e-12`.
- **`max_t_chunk`**: Chunk size over bootstrap replicates when computing the max-t critical value. Controls peak memory usage. Default `512`.
- **`cluster_labels`**: Optional `np.ndarray` of shape `(N,)` assigning each task to a cluster (e.g., repository index for SWE-Bench). When provided, enables cluster-aware bootstrap methods. `None` for i.i.d. (default).
- **`bootstrap_method`**: Bootstrap strategy. `None` (default) auto-selects based on number of clusters `C`: `C ≥ 50` → `"cluster"` with `m = ⌊C^0.95⌋`; `20 ≤ C < 50` → `"cluster"` with `m = C`; `C < 20` → `"wild_cluster"`. When `cluster_labels` is `None`, defaults to `"iid"`. Additional methods:
  - `"moon"`: m-out-of-n bootstrap — resamples `m < N` tasks i.i.d. and rescales the bias estimate by `√(m/N)`. Requires an explicit `boot_size`. Cannot be used with `cluster_labels`.
  - `"stratified"`: Stratified task bootstrap — resamples tasks within each cluster independently (each cluster contributes exactly `n_g` tasks per replicate). Requires `cluster_labels`.
  - `"stratified_moon"`: Stratified m-out-of-n bootstrap — resamples `m_g = max(1, round(n_g * m/N))` tasks within each cluster (total `m ≈ boot_size`). Rescales bias by `√(m/N)`. Requires `cluster_labels` and an explicit `boot_size` with `C ≤ m < N`.
- **`wild_weight_dist`**: Weight distribution for wild cluster bootstrap. One of `"rademacher"`, `"mammen"`, or `"webb"` (default, recommended when `C` is small).

### Output: `BootstrapResult`

#### Basic correction fields

| Field | Shape | Description |
|---|---|---|
| `n_bootstraps` | scalar | Number of bootstrap resamples used |
| `sample_size` | scalar | N (number of tasks) |
| `n_arms` | scalar | J (number of models) |
| `emp_rank` | `(J,)` | Permutation that sorts models by descending empirical score |
| `emp_rank_est` | `(J,)` | Empirical scores in rank order (naive estimates) |
| `boot_wc` | `(J,)` | Estimated winner's curse at each rank position |
| `corr_rank_est` | `(J,)` | Corrected scores: `emp_rank_est - boot_wc` |
| `rank_prob_matrix` | `(J, J)` | `P[j, k]` = probability that model `j` lands at rank position `k` |
| `bootstrap_method` | `str` | Method used: `"iid"`, `"moon"`, `"cluster"`, `"wild_cluster"`, `"stratified"`, or `"stratified_moon"` |
| `boot_m` | `int \| None` | Resample size `m` for m-out-of-n bootstrap (None unless method is `"moon"` or `"stratified_moon"`) |
| `n_clusters` | `int \| None` | Number of clusters `C` (None for i.i.d.) |
| `boot_cluster_size` | `int \| None` | Number of clusters resampled per bootstrap (cluster method only) |
| `wild_weight_dist` | `str \| None` | Weight distribution used (wild cluster method only) |

#### Simultaneous inference fields (when `compute_partial_order=True`)

| Field | Shape | Description |
|---|---|---|
| `alpha` | scalar | FWER level used |
| `max_t_crit` | scalar | Max-t critical value `c_{1-alpha}` |
| `gap_hat` | `(J, J)` | Pairwise gap estimates: `gap_hat[j,k] = mean_j - mean_k` |
| `gap_se` | `(J, J)` | Standard errors for pairwise gaps |
| `gap_ci_lower` | `(J, J)` | Lower bound of simultaneous CI for each gap |
| `gap_ci_upper` | `(J, J)` | Upper bound of simultaneous CI for each gap |
| `dominance` | `(J, J)` bool | `dominance[j,k] = True` if j significantly beats k (CI lower > 0) |
| `rank_lower` | `(J,)` int | Lower bound on true rank for each model |
| `rank_upper` | `(J,)` int | Upper bound on true rank for each model |
| `tiers` | `List[List[int]]` | Ordered tiers of model indices (peeled from dominance DAG) |
| `tier_id` | `(J,)` int | Tier index for each model |

## Algorithm

### Step 1: Winner's curse correction

1. Compute empirical means `emp_est[j]` for each model.
2. Rank models by descending `emp_est` to get `emp_rank`.
3. Draw `B` bootstrap resamples of size `m` (with replacement).
4. For each bootstrap sample, compute means and re-rank.
5. The winner's curse at rank position `k` is:

   ```
   boot_wc[k] = mean_b( boot_est[boot_rank_b[k]] - emp_est[boot_rank_b[k]] )
   ```

   This measures how much the score at position `k` is inflated by the act of ranking.

6. Corrected estimates: `corr_rank_est = emp_rank_est - boot_wc`.

### Step 2: Simultaneous pairwise CIs (max-t method)

1. For each pair `(j, k)`, compute the gap `gap_hat[j,k]` and its standard error `gap_se[j,k]` from the sample covariance.
2. For each bootstrap sample, compute studentized pairwise gap statistics and take the maximum absolute value across all pairs.
3. The `(1-alpha)` quantile of these maxima gives the critical value `max_t_crit`.
4. Simultaneous CIs: `gap_hat[j,k] +/- max_t_crit * gap_se[j,k]`.

This controls the family-wise error rate across all `J*(J-1)/2` pairwise comparisons.

### Step 3: Dominance graph and tiers

1. Build a directed **dominance graph**: edge `j -> k` if the CI for `gap(j,k)` is strictly above 0 (j significantly beats k).
2. **Rank bounds**: `rank_lower[j] = 1 + in_degree(j)`, `rank_upper[j] = J - out_degree(j)`.
3. **Tiers**: Iteratively peel nodes with zero in-degree from the dominance graph. Each layer forms a tier of statistically indistinguishable models. Within each tier, models are sorted by descending empirical score.

Cycle handling uses Tarjan's SCC algorithm to condense the graph before peeling.

## Example

```python
import numpy as np
from llm_wc.wc.core import bootstrap_correction

# Binary outcomes: 500 tasks, 40 models
np.random.seed(42)
obs = (np.random.rand(500, 40) > 0.5).astype(float)

result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    boot_size=int(500 ** 0.95),
    alpha=0.05,
)

# Corrected scores at each rank position
print(result.corr_rank_est)

# Number of tiers
print(len(result.tiers))

# Rank bounds for the top model
top = result.emp_rank[0]
print(f"Model {top}: rank in [{result.rank_lower[top]}, {result.rank_upper[top]}]")
```

## m-out-of-n bootstrap (`"moon"`)

Use `bootstrap_method="moon"` with an explicit `boot_size` (`m < N`) to run an m-out-of-n bootstrap. This resamples `m` tasks i.i.d. and rescales the winner's curse bias by `√(m/N)` (selection bias ∝ `σ/√n`). The max-t computation accounts for the scaling factor `√(N/m)` so that CIs remain calibrated. Cannot be used with `cluster_labels`.

```python
result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    boot_size=int(500 ** 0.75),  # m must be < N
    bootstrap_method="moon",
    alpha=0.05,
)
print(result.boot_m)  # the m value used
```

## Cluster bootstrap

When tasks are grouped into clusters (e.g., repositories in SWE-Bench), resampling individual tasks destroys within-cluster correlation. The cluster bootstrap resamples entire clusters instead.

```python
import numpy as np
from llm_wc.wc.core import bootstrap_correction

np.random.seed(42)
obs = (np.random.rand(500, 10) > 0.5).astype(float)
labels = np.repeat(np.arange(25), 20)  # 25 clusters of 20 tasks each

# Auto-selects method based on C=25 (cluster bootstrap with m=C)
result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    cluster_labels=labels,
)
print(result.bootstrap_method)   # "cluster"
print(result.n_clusters)         # 25
print(result.corr_rank_est)
```

### When to use which method

| Setting | Method | Rationale |
|---|---|---|
| No clusters | `"iid"` (default) | Standard task-level bootstrap |
| No clusters, weak dependence | `"moon"` | m-out-of-n bootstrap; resamples `m < N` tasks and rescales bias by `√(m/N)` |
| C ≥ 50 clusters | `"cluster"` with `m = ⌊C^0.95⌋` | Sufficient clusters; subsampling smooths ties |
| 20 ≤ C < 50 | `"cluster"` with `m = C` | Moderate C; subsampling may remove too many clusters |
| C < 20 | `"wild_cluster"` | Too few clusters for resampling; uses perturbation instead |
| Clusters, preserve composition | `"stratified"` | Resamples tasks within each cluster; each cluster contributes exactly `n_g` tasks |
| Clusters, subsample within each | `"stratified_moon"` | m-out-of-n within each cluster; preserves composition with subsampling; rescales bias by `√(m/N)` |

## Stratified bootstrap (`"stratified"`)

When tasks are grouped into clusters and you want to preserve the exact cluster composition of the benchmark, the stratified bootstrap resamples tasks *within* each cluster independently. For each cluster `g` with `n_g` tasks, it draws `n_g` tasks with replacement from that cluster. This ensures every replicate has the same number of tasks per cluster as the original sample.

```python
result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    cluster_labels=labels,
    bootstrap_method="stratified",
)
print(result.bootstrap_method)  # "stratified"
```

## Stratified m-out-of-n bootstrap (`"stratified_moon"`)

Combines the stratified and m-out-of-n approaches. For each cluster `g` with `n_g` tasks, resamples `m_g = max(1, round(n_g * m/N))` tasks with replacement (instead of all `n_g`). The total resampled tasks `m = sum(m_g)` may differ slightly from the requested `boot_size` due to rounding; `boot_m` stores the actual value used.

The winner's curse bias is rescaled by `√(m/N)` and the max-t scale is `√(N/m)`, matching the moon bootstrap. Requires `cluster_labels` and an explicit `boot_size` with `C ≤ m < N`.

```python
result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    cluster_labels=labels,
    bootstrap_method="stratified_moon",
    boot_size=int(500 ** 0.75),
)
print(result.boot_m)           # actual m used (may differ slightly from boot_size)
print(result.bootstrap_method)  # "stratified_moon"
```

## Wild cluster bootstrap

When the number of clusters is small (< 20), the wild cluster bootstrap perturbs cluster-level residuals with random weights instead of resampling. This preserves the original data structure and within-cluster correlation.

```python
result = bootstrap_correction(
    obs=obs,
    n_bootstraps=5000,
    cluster_labels=labels,
    bootstrap_method="wild_cluster",
    wild_weight_dist="webb",  # recommended for small C
)
```

The `wild_weight_dist` parameter controls the weight distribution:
- `"rademacher"`: ±1 with equal probability
- `"mammen"`: two-point distribution matching first 3 moments of N(0,1)
- `"webb"`: six-point distribution (default, recommended when C is very small)
