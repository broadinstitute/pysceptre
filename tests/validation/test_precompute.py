import numpy as np

from pysceptre.precompute.pieces import compute_precomputation_pieces


def _lower_right(D: np.ndarray, trt_idxs_0based: np.ndarray) -> float:
    """The only way D ever gets consumed by the test statistic: sum_k (sum_{i in trt} D[k,i])^2."""
    inner = D[:, trt_idxs_0based].sum(axis=1)
    return float(np.sum(inner**2))


def test_mu_w_a_match_r_exactly(ground_truth):
    X = np.array(ground_truth["X"])
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        np.testing.assert_allclose(pieces.mu, gene["mu"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(pieces.w, gene["w"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(pieces.a, gene["a"], rtol=1e-6, atol=1e-8)


def test_D_matrix_gram_invariant_matches_r(ground_truth):
    """D itself may differ from R's in row order/sign (eigh vs eigen convention),
    but D'D (and therefore every downstream use of D in the test statistic) must match."""
    X = np.array(ground_truth["X"])
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        D_r = np.array(gene["D"])
        gram_python = pieces.D.T @ pieces.D
        gram_r = D_r.T @ D_r
        np.testing.assert_allclose(gram_python, gram_r, rtol=1e-5, atol=1e-6)


def test_D_matrix_statistic_invariant_matches_r_for_random_treatment_sets(ground_truth):
    """The specific reduction used by the test statistic should match R's D for
    arbitrary treated-cell subsets, which is the only invariant that actually matters."""
    rng = np.random.default_rng(0)
    X = np.array(ground_truth["X"])
    n = X.shape[0]
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        D_r = np.array(gene["D"])
        for _ in range(5):
            trt_idxs = rng.choice(n, size=rng.integers(5, 60), replace=False)
            lr_python = _lower_right(pieces.D, trt_idxs)
            lr_r = _lower_right(D_r, trt_idxs)
            np.testing.assert_allclose(lr_python, lr_r, rtol=1e-5, atol=1e-8)
