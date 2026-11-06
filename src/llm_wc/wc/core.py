from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    # existing
    n_bootstraps: int
    sample_size: int
    n_arms: int
    emp_rank: np.ndarray  # (J,)
    emp_rank_est: np.ndarray  # (J,)
    boot_wc: np.ndarray  # (J,)
    corr_rank_est: np.ndarray  # (J,)
    rank_prob_matrix: Optional[np.ndarray] = None  # (J, J)

    # bootstrap method metadata
    bootstrap_method: str = "iid"
    boot_m: Optional[int] = None  # m for m-out-of-n bootstrap (None unless method="moon" or "stratified_moon")
    n_clusters: Optional[int] = None
    boot_cluster_size: Optional[int] = None
    wild_weight_dist: Optional[str] = None

    # simultaneous inference / partial order / tiers
    alpha: float = 0.05
    max_t_crit: Optional[float] = None  # c_{1-alpha}
    gap_hat: Optional[np.ndarray] = None  # (J, J), hatΔ_{jk} = hatθ_j - hatθ_k
    gap_se: Optional[np.ndarray] = None  # (J, J), se(hatΔ_{jk}) based on N
    gap_ci_lower: Optional[np.ndarray] = None  # (J, J)
    gap_ci_upper: Optional[np.ndarray] = None  # (J, J)
    dominance: Optional[np.ndarray] = (
        None  # (J, J) bool, edge j->k if CI_{jk} strictly > 0
    )
    rank_lower: Optional[np.ndarray] = None  # (J,) int, lower bound on true rank
    rank_upper: Optional[np.ndarray] = None  # (J,) int, upper bound on true rank
    tiers: Optional[List[List[int]]] = (
        None  # ordered tiers (list of lists of arm indices)
    )
    tier_id: Optional[np.ndarray] = None  # (J,) int, tier index for each arm


def _tarjan_scc(adj: np.ndarray) -> List[List[int]]:
    """
    Tarjan's algorithm for SCCs on a directed graph.
    adj: (n,n) boolean adjacency matrix, adj[u,v]=True means u->v.
    Returns list of SCCs, each SCC is list of node indices in [0,n).
    """
    n = adj.shape[0]
    index = 0
    stack: List[int] = []
    onstack = np.zeros(n, dtype=bool)
    indices = -np.ones(n, dtype=int)
    lowlink = np.zeros(n, dtype=int)
    sccs: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        # iterate neighbors
        for w in np.flatnonzero(adj[v]):
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif onstack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        # root of SCC
        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    return sccs


def _compute_tiers_from_dominance(
    dominance: np.ndarray, sort_key: Optional[np.ndarray] = None
) -> Tuple[List[List[int]], np.ndarray]:
    """
    dominance: (J,J) bool adjacency, j->k means j significantly beats k.
    sort_key: (J,) optional; used to sort nodes within each tier (descending).
    Returns (tiers, tier_id).
    """
    J = dominance.shape[0]
    remaining = set(range(J))
    tiers: List[List[int]] = []
    tier_id = -np.ones(J, dtype=int)

    while remaining:
        nodes = np.array(sorted(remaining), dtype=int)
        sub = dominance[np.ix_(nodes, nodes)]
        indeg = sub.sum(axis=0)

        # nodes with no incoming edges from remaining subgraph
        tier_nodes = nodes[indeg == 0]

        if tier_nodes.size == 0:
            # handle rare cycles by SCC condensation and peel SCCs with zero indegree
            sccs = _tarjan_scc(sub)
            scc_id = -np.ones(len(nodes), dtype=int)
            for c, comp in enumerate(sccs):
                for u in comp:
                    scc_id[u] = c

            C = len(sccs)
            indeg_scc = np.zeros(C, dtype=int)
            for u in range(len(nodes)):
                for v in np.flatnonzero(sub[u]):
                    if scc_id[u] != scc_id[v]:
                        indeg_scc[scc_id[v]] += 1

            zero_scc = np.flatnonzero(indeg_scc == 0)
            pick_sub_idx = np.array(
                [u for u in range(len(nodes)) if scc_id[u] in set(zero_scc)], dtype=int
            )
            tier_nodes = nodes[pick_sub_idx]

        tier_list = tier_nodes.tolist()

        if sort_key is not None:
            # sort within tier by descending sort_key (e.g., empirical estimate)
            tier_list.sort(key=lambda j: float(sort_key[j]), reverse=True)

        t = len(tiers)
        for j in tier_list:
            tier_id[j] = t

        tiers.append(tier_list)
        remaining.difference_update(tier_list)

    return tiers, tier_id


def _precompute_cluster_data(
    obs: np.ndarray, cluster_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Precompute cluster sums and sizes.

    Returns:
        cluster_sums: (C, J) sum of obs within each cluster
        cluster_sizes: (C,) number of tasks in each cluster
        C: number of clusters
    """
    unique_labels = np.unique(cluster_labels)
    C = len(unique_labels)
    N, J = obs.shape
    cluster_sums = np.zeros((C, J), dtype=float)
    cluster_sizes = np.zeros(C, dtype=int)
    # Map labels to contiguous indices
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped = np.array([label_map[l] for l in cluster_labels], dtype=int)
    np.add.at(cluster_sums, mapped, obs)
    np.add.at(cluster_sizes, mapped, 1)
    return cluster_sums, cluster_sizes, C


def _iid_bootstrap_estimates(
    obs: np.ndarray, n_bootstraps: int, boot_size: int
) -> np.ndarray:
    """Standard i.i.d. task bootstrap. Returns (B, J) bootstrap estimates."""
    sample_size = obs.shape[0]
    boot_indices = np.random.randint(0, sample_size, size=(n_bootstraps, boot_size))
    boot_samples = obs[boot_indices]
    return boot_samples.mean(axis=1)


def _cluster_bootstrap_estimates(
    obs: np.ndarray,
    cluster_labels: np.ndarray,
    n_bootstraps: int,
    boot_size: int,
) -> np.ndarray:
    """Cluster bootstrap: resample entire clusters. Returns (B, J)."""
    cluster_sums, cluster_sizes, C = _precompute_cluster_data(obs, cluster_labels)
    # boot_size here is the number of clusters to resample (m)
    boot_cluster_idx = np.random.randint(0, C, size=(n_bootstraps, boot_size))
    # Process in chunks to avoid (B, m, J) memory blowup
    J = obs.shape[1]
    boot_est = np.empty((n_bootstraps, J), dtype=float)
    chunk = 256
    for start in range(0, n_bootstraps, chunk):
        end = min(n_bootstraps, start + chunk)
        idx = boot_cluster_idx[start:end]  # (chunk, m)
        totals = cluster_sums[idx].sum(axis=1)  # (chunk, J)
        n_tasks = cluster_sizes[idx].sum(axis=1, keepdims=True)  # (chunk, 1)
        boot_est[start:end] = totals / n_tasks
    return boot_est


def _stratified_bootstrap_estimates(
    obs: np.ndarray,
    cluster_labels: np.ndarray,
    n_bootstraps: int,
) -> np.ndarray:
    """Stratified task bootstrap: resample tasks within each cluster.

    For each cluster g with n_g tasks, independently resample n_g tasks
    with replacement from that cluster. Concatenate across clusters and
    compute per-arm means. This preserves the cluster composition of the
    benchmark (each cluster contributes exactly n_g tasks every replicate).

    Returns (B, J) bootstrap estimates.
    """
    N, J = obs.shape
    unique_labels = np.unique(cluster_labels)
    # Precompute per-cluster task indices
    cluster_indices = [np.flatnonzero(cluster_labels == lab) for lab in unique_labels]

    boot_est = np.empty((n_bootstraps, J), dtype=float)
    chunk = 256
    for start in range(0, n_bootstraps, chunk):
        end = min(n_bootstraps, start + chunk)
        bsz = end - start
        # Build resampled index arrays: for each cluster, resample its tasks
        all_indices = []
        for idx_g in cluster_indices:
            n_g = len(idx_g)
            # (bsz, n_g) random indices into this cluster's tasks
            resampled = idx_g[np.random.randint(0, n_g, size=(bsz, n_g))]
            all_indices.append(resampled)
        # Concatenate across clusters: (bsz, N)
        boot_idx = np.concatenate(all_indices, axis=1)
        # Gather and compute means
        boot_est[start:end] = obs[boot_idx].mean(axis=1)
    return boot_est


def _stratified_moon_bootstrap_estimates(
    obs: np.ndarray,
    cluster_labels: np.ndarray,
    n_bootstraps: int,
    boot_size: int,
) -> Tuple[np.ndarray, int]:
    """Stratified m-out-of-n bootstrap: subsample tasks within each cluster.

    For each cluster g with n_g tasks, resample m_g = max(1, round(n_g * m/N))
    tasks with replacement. The total resampled size actual_m = sum(m_g) may
    differ slightly from *boot_size* due to rounding.

    Returns (boot_est, actual_m) where boot_est has shape (B, J).
    """
    N, J = obs.shape
    unique_labels = np.unique(cluster_labels)
    cluster_indices = [np.flatnonzero(cluster_labels == lab) for lab in unique_labels]

    # Proportional allocation
    ratio = boot_size / N
    cluster_m = [max(1, round(len(idx_g) * ratio)) for idx_g in cluster_indices]
    actual_m = sum(cluster_m)

    boot_est = np.empty((n_bootstraps, J), dtype=float)
    chunk = 256
    for start in range(0, n_bootstraps, chunk):
        end = min(n_bootstraps, start + chunk)
        bsz = end - start
        all_indices = []
        for idx_g, m_g in zip(cluster_indices, cluster_m):
            n_g = len(idx_g)
            resampled = idx_g[np.random.randint(0, n_g, size=(bsz, m_g))]
            all_indices.append(resampled)
        boot_idx = np.concatenate(all_indices, axis=1)  # (bsz, actual_m)
        boot_est[start:end] = obs[boot_idx].mean(axis=1)
    return boot_est, actual_m


def _wild_cluster_bootstrap_estimates(
    obs: np.ndarray,
    cluster_labels: np.ndarray,
    n_bootstraps: int,
    weight_dist: str,
    emp_est: np.ndarray,
) -> np.ndarray:
    """Wild cluster bootstrap: perturb cluster residuals. Returns (B, J)."""
    cluster_sums, cluster_sizes, C = _precompute_cluster_data(obs, cluster_labels)
    N = obs.shape[0]
    # Cluster residuals: eta[c, j] = cluster_sums[c, j] - n_c * emp_est[j]
    eta = cluster_sums - cluster_sizes[:, None] * emp_est[None, :]  # (C, J)

    # Generate weight matrix W of shape (B, C)
    if weight_dist == "rademacher":
        W = np.where(np.random.random((n_bootstraps, C)) < 0.5, -1.0, 1.0)
    elif weight_dist == "mammen":
        sqrt5 = np.sqrt(5.0)
        p = (sqrt5 + 1.0) / (2.0 * sqrt5)
        vals = np.where(
            np.random.random((n_bootstraps, C)) < p,
            -(sqrt5 - 1.0) / 2.0,
            (sqrt5 + 1.0) / 2.0,
        )
        W = vals
    elif weight_dist == "webb":
        points = np.array([
            -np.sqrt(3.0 / 2.0), -np.sqrt(2.0 / 2.0), -np.sqrt(1.0 / 2.0),
            np.sqrt(1.0 / 2.0), np.sqrt(2.0 / 2.0), np.sqrt(3.0 / 2.0),
        ])
        idx = np.random.randint(0, 6, size=(n_bootstraps, C))
        W = points[idx]
    else:
        raise ValueError(
            f"Unknown wild_weight_dist '{weight_dist}'. "
            "Choose from 'rademacher', 'mammen', 'webb'."
        )

    # boot_est = emp_est + (W @ eta) / N
    boot_est = emp_est[None, :] + (W @ eta) / float(N)  # (B, J)
    return boot_est


def _compute_gap_se(
    obs: np.ndarray,
    cluster_labels: Optional[np.ndarray],
    eps_se: float,
) -> np.ndarray:
    """Compute pairwise gap standard errors.

    When cluster_labels is None, uses the i.i.d. formula.
    When clustered, uses the cluster-robust sandwich formula.
    Returns (J, J) SE matrix with NaN on diagonal.
    """
    N, J = obs.shape
    if N < 2:
        raise ValueError("Need sample_size >= 2 to estimate pairwise SEs.")

    if cluster_labels is None:
        cov = np.cov(obs, rowvar=False, ddof=1)
        var = np.diag(cov)
        var_diff = var[:, None] + var[None, :] - 2.0 * cov
        var_diff = np.maximum(var_diff, 0.0)
        gap_se = np.sqrt(var_diff / float(N))
    else:
        # Cluster-robust SE: Var(gap_{jk}) = (1/N^2) * sum_c eta_{cj}*eta_{ck} style
        cluster_sums, cluster_sizes, C = _precompute_cluster_data(obs, cluster_labels)
        emp_est = obs.mean(axis=0)
        eta = cluster_sums - cluster_sizes[:, None] * emp_est[None, :]  # (C, J)
        # Cluster-robust covariance: (1/N^2) * eta.T @ eta  gives Cov(hat_tau)
        cov_robust = (eta.T @ eta) / float(N * N)  # (J, J)
        var = np.diag(cov_robust)
        var_diff = var[:, None] + var[None, :] - 2.0 * cov_robust
        var_diff = np.maximum(var_diff, 0.0)
        gap_se = np.sqrt(var_diff)

    gap_se = np.where(gap_se < eps_se, eps_se, gap_se)
    np.fill_diagonal(gap_se, np.nan)
    return gap_se


def bootstrap_correction(
    *,
    obs: np.ndarray,
    n_bootstraps: int,
    boot_size: int = None,
    alpha: float = 0.05,
    compute_partial_order: bool = True,
    eps_se: float = 1e-12,
    max_t_chunk: int = 512,
    cluster_labels: Optional[np.ndarray] = None,
    bootstrap_method: Optional[str] = None,
    wild_weight_dist: str = "webb",
) -> BootstrapResult:
    """
    Bootstrap correction of ranking value estimates + partial order / rank bounds / tiers.

    Args:
        obs: array, shape (sample_size, n_arms). Each row is a task; each col is a model/arm.
        n_bootstraps: number of bootstrap samples.
        boot_size: bootstrap resample size m (default N for iid, or auto for cluster).
        alpha: FWER level for simultaneous pairwise-gap inference (1-alpha coverage).
        compute_partial_order: if False, skips CI/dominance/tier computation.
        eps_se: floor for SEs to avoid divide-by-zero.
        max_t_chunk: chunk size over bootstraps to control memory for max-t computation.
        cluster_labels: (N,) int array assigning each task to a cluster. None for i.i.d.
        bootstrap_method: "iid" | "moon" | "cluster" | "wild_cluster" | "stratified" | None (auto-select).
            "moon" is the m-out-of-n bootstrap: resamples m < N tasks i.i.d. and rescales
            the bias estimate by √(m/N). Requires an explicit boot_size (m).
        wild_weight_dist: weight distribution for wild cluster bootstrap.
            One of "rademacher", "mammen", "webb" (default).

    Returns:
        BootstrapResult with:
            - n_bootstraps, sample_size, n_arms
            - emp_rank: empirical ranking permutation
            - emp_rank_est: naive ranking value estimates
            - boot_wc: position-specific winner's curse estimate
            - corr_rank_est: corrected ranking estimates
            - rank_prob_matrix: P(arm j at rank k)
            - bootstrap_method, n_clusters, boot_cluster_size, wild_weight_dist: metadata
            - max_t_crit: max-t critical value c_{1-alpha}
            - gap_hat, gap_se, gap_ci_lower, gap_ci_upper: pairwise gap CIs
            - dominance: (j->k if CI lower > 0)
            - rank_lower, rank_upper: rank bounds
            - tiers, tier_id: ordered tiers
    """
    if obs.ndim != 2:
        raise ValueError(
            f"`obs` must have shape (sample_size, n_arms), got {obs.shape}"
        )
    if n_bootstraps <= 0:
        raise ValueError(f"`n_bootstraps` must be > 0, got {n_bootstraps}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0,1), got {alpha}")

    sample_size, n_arms = obs.shape

    # --- Cluster validation and method auto-selection ---
    n_clusters_out: Optional[int] = None
    boot_cluster_size_out: Optional[int] = None
    wild_weight_dist_out: Optional[str] = None

    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels)
        if cluster_labels.shape != (sample_size,):
            raise ValueError(
                f"`cluster_labels` must have shape ({sample_size},), "
                f"got {cluster_labels.shape}"
            )
        C = len(np.unique(cluster_labels))
        if C < 2:
            raise ValueError(
                f"Need at least 2 clusters, got {C}."
            )
        n_clusters_out = C

        if bootstrap_method is None:
            # Auto-select
            if C >= 50:
                bootstrap_method = "cluster"
                if boot_size is None:
                    boot_size = int(C ** 0.95)
            elif C >= 20:
                bootstrap_method = "cluster"
                if boot_size is None:
                    boot_size = C
            else:
                bootstrap_method = "wild_cluster"
        elif bootstrap_method in ("iid", "moon"):
            raise ValueError(
                f"Cannot use bootstrap_method='{bootstrap_method}' with cluster_labels. "
                "Pass cluster_labels=None for i.i.d. bootstrap."
            )
    else:
        if bootstrap_method is not None and bootstrap_method not in ("iid", "moon"):
            raise ValueError(
                f"bootstrap_method='{bootstrap_method}' requires cluster_labels."
            )
        if bootstrap_method is None:
            bootstrap_method = "iid"

    if bootstrap_method not in ("iid", "cluster", "wild_cluster", "stratified", "moon", "stratified_moon"):
        raise ValueError(
            f"Unknown bootstrap_method '{bootstrap_method}'. "
            "Choose from 'iid', 'cluster', 'wild_cluster', 'stratified', 'moon', 'stratified_moon'."
        )

    # --- Set boot_size defaults ---
    boot_m_out: Optional[int] = None
    if bootstrap_method == "iid":
        if boot_size is None:
            boot_size = sample_size
        if boot_size <= 0:
            raise ValueError(f"`boot_size` must be > 0, got {boot_size}")
    elif bootstrap_method == "moon":
        if boot_size is None:
            raise ValueError(
                "m-out-of-n bootstrap requires an explicit `boot_size` (m). "
                "Pass e.g. boot_size=int(N**0.75)."
            )
        if boot_size <= 0:
            raise ValueError(f"`boot_size` must be > 0, got {boot_size}")
        if boot_size >= sample_size:
            raise ValueError(
                f"m-out-of-n bootstrap requires boot_size (m={boot_size}) "
                f"< sample_size (N={sample_size})."
            )
        boot_m_out = boot_size
    elif bootstrap_method == "stratified_moon":
        if boot_size is None:
            raise ValueError(
                "stratified_moon bootstrap requires an explicit `boot_size` (m). "
                "Pass e.g. boot_size=int(N**0.75)."
            )
        if boot_size <= 0:
            raise ValueError(f"`boot_size` must be > 0, got {boot_size}")
        if boot_size < n_clusters_out:
            raise ValueError(
                f"stratified_moon requires boot_size (m={boot_size}) >= "
                f"number of clusters (C={n_clusters_out}) so each cluster gets >= 1 task."
            )
        if boot_size >= sample_size:
            raise ValueError(
                f"stratified_moon requires boot_size (m={boot_size}) "
                f"< sample_size (N={sample_size})."
            )
        # boot_m_out set after allocation (actual_m may differ from boot_size)
    elif bootstrap_method == "cluster":
        # boot_size is number of clusters to resample
        if boot_size is None:
            boot_size = n_clusters_out
        if boot_size <= 0:
            raise ValueError(f"`boot_size` must be > 0, got {boot_size}")
        boot_cluster_size_out = boot_size

    if bootstrap_method == "wild_cluster":
        wild_weight_dist_out = wild_weight_dist

    # empirical estimates (J,)
    emp_est = obs.mean(axis=0)

    # empirical ranking permutation (J,)
    emp_rank = np.argsort(-emp_est)

    # naive ranking value estimates (J,)
    emp_rank_est = emp_est[emp_rank]

    # --- Bootstrap estimation (B, J) ---
    if bootstrap_method in ("iid", "moon"):
        boot_est = _iid_bootstrap_estimates(obs, n_bootstraps, boot_size)
    elif bootstrap_method == "cluster":
        boot_est = _cluster_bootstrap_estimates(
            obs, cluster_labels, n_bootstraps, boot_size
        )
    elif bootstrap_method == "wild_cluster":
        boot_est = _wild_cluster_bootstrap_estimates(
            obs, cluster_labels, n_bootstraps, wild_weight_dist, emp_est
        )
    elif bootstrap_method == "stratified":
        boot_est = _stratified_bootstrap_estimates(
            obs, cluster_labels, n_bootstraps
        )
    elif bootstrap_method == "stratified_moon":
        boot_est, actual_m = _stratified_moon_bootstrap_estimates(
            obs, cluster_labels, n_bootstraps, boot_size
        )
        boot_m_out = actual_m
        boot_size = actual_m  # downstream rescaling uses actual m

    # bootstrap rankings (B,J)
    boot_rank = np.argsort(-boot_est, axis=1)

    # winner's curse estimate (position-specific), (J,)
    boot_wc = np.mean(
        np.take_along_axis(boot_est, boot_rank, axis=1) - emp_est[boot_rank],
        axis=0,
    )

    # Scale WC bias from m-observation noise level to N-observation noise level.
    # Selection bias ∝ σ/√n, so bias_N = bias_m × √(m/N).
    if bootstrap_method in ("moon", "stratified_moon"):
        boot_wc = boot_wc * np.sqrt(float(boot_size) / float(sample_size))

    # corrected ranking estimates (J,)
    # Note: the correction is defined per rank POSITION, not per model. Models
    # that happen to share an empirical score in the original data therefore
    # receive different corrections based on the arbitrary tie-break order, and
    # that is the honest position-level statement of the winner's curse.
    corr_rank_est = emp_rank_est - boot_wc

    # rank probability matrix P(j,k) = Pr(arm j at rank k), (J,J)
    rank_prob_matrix = np.zeros((n_arms, n_arms), dtype=float)
    rank_pos = np.broadcast_to(np.arange(n_arms), boot_rank.shape)  # (B,J) positions
    np.add.at(rank_prob_matrix, (boot_rank.ravel(), rank_pos.ravel()), 1.0)
    rank_prob_matrix /= float(n_bootstraps)

    # Metadata dict for BootstrapResult
    _meta = dict(
        bootstrap_method=bootstrap_method,
        boot_m=boot_m_out,
        n_clusters=n_clusters_out,
        boot_cluster_size=boot_cluster_size_out,
        wild_weight_dist=wild_weight_dist_out,
    )

    # If user only wants the original outputs, return early
    if not compute_partial_order:
        return BootstrapResult(
            n_bootstraps=n_bootstraps,
            sample_size=sample_size,
            n_arms=n_arms,
            boot_wc=boot_wc,
            emp_rank=emp_rank,
            emp_rank_est=emp_rank_est,
            corr_rank_est=corr_rank_est,
            rank_prob_matrix=rank_prob_matrix,
            alpha=alpha,
            **_meta,
        )

    # -----------------------------
    # Simultaneous CIs for all pairwise gaps (max-t)
    # -----------------------------
    # gap_hat[j,k] = emp_est[j] - emp_est[k]
    gap_hat = emp_est[:, None] - emp_est[None, :]

    # SE for gaps (i.i.d. or cluster-robust)
    gap_se = _compute_gap_se(obs, cluster_labels, eps_se)
    np.fill_diagonal(gap_hat, 0.0)

    # Scale factor for max-t: accounts for bootstrap variance vs population variance
    if bootstrap_method in ("iid", "moon", "stratified_moon"):
        scale = np.sqrt(float(sample_size) / float(boot_size))
    elif bootstrap_method == "cluster":
        scale = np.sqrt(float(n_clusters_out) / float(boot_size))
    elif bootstrap_method == "wild_cluster":
        scale = 1.0
    else:  # stratified
        scale = 1.0

    # Precompute upper-triangular pair indices
    p, q = np.triu_indices(n_arms, k=1)
    gap_hat_pairs = gap_hat[p, q]
    gap_se_pairs = gap_se[p, q]  # se_N

    # Compute max-t per bootstrap in chunks to control memory
    T = np.empty(n_bootstraps, dtype=float)
    for start in range(0, n_bootstraps, max_t_chunk):
        end = min(n_bootstraps, start + max_t_chunk)
        be = boot_est[start:end]  # (chunk,J)
        # gaps on pairs: (chunk, P)
        gap_b_pairs = be[:, p] - be[:, q]
        # studentized using se_m ≈ se_N * sqrt(N/m)
        t_pairs = (gap_b_pairs - gap_hat_pairs[None, :]) / (
            gap_se_pairs[None, :] * scale
        )
        T[start:end] = np.max(np.abs(t_pairs), axis=1)

    max_t_crit = float(np.quantile(T, 1.0 - alpha, method="higher"))

    # Simultaneous CI for each gap Δ_{jk}: gap_hat ± c * se_N
    gap_ci_lower = gap_hat - max_t_crit * gap_se
    gap_ci_upper = gap_hat + max_t_crit * gap_se
    np.fill_diagonal(gap_ci_lower, 0.0)
    np.fill_diagonal(gap_ci_upper, 0.0)

    # Enforce antisymmetry numerically (optional but nice)
    # CI_{kj} should be -CI_{jk} swapped
    # (This keeps things consistent even with tiny numerical noise.)
    gap_ci_lower = 0.5 * (gap_ci_lower - gap_ci_upper.T)
    gap_ci_upper = 0.5 * (gap_ci_upper - gap_ci_lower.T)

    # -----------------------------
    # Dominance partial order, rank bounds, tiers
    # -----------------------------
    # dominance[j,k] True if CI_{jk} strictly above 0 => j significantly beats k
    dominance = gap_ci_lower > 0.0
    np.fill_diagonal(dominance, False)

    in_deg = dominance.sum(axis=0).astype(int)  # models that beat j
    out_deg = dominance.sum(axis=1).astype(int)  # models j beats

    rank_lower = 1 + in_deg
    rank_upper = n_arms - out_deg

    # tiers (peel layers of zero in-degree in dominance graph)
    tiers, tier_id = _compute_tiers_from_dominance(dominance, sort_key=emp_est)

    return BootstrapResult(
        n_bootstraps=n_bootstraps,
        sample_size=sample_size,
        n_arms=n_arms,
        boot_wc=boot_wc,
        emp_rank=emp_rank,
        emp_rank_est=emp_rank_est,
        corr_rank_est=corr_rank_est,
        rank_prob_matrix=rank_prob_matrix,
        alpha=alpha,
        max_t_crit=max_t_crit,
        gap_hat=gap_hat,
        gap_se=gap_se,
        gap_ci_lower=gap_ci_lower,
        gap_ci_upper=gap_ci_upper,
        dominance=dominance,
        rank_lower=rank_lower,
        rank_upper=rank_upper,
        tiers=tiers,
        tier_id=tier_id,
        **_meta,
    )
