import numpy as np

from analog_geometry.metrics import linear_cka, linear_cka_stimcentered, rsa


def test_precomputed_gram_gives_identical_result():
    from analog_geometry.metrics import _centered_gram, stimcenter
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(100, 300)), rng.normal(size=(100, 200))
    assert linear_cka(X, Y) == linear_cka(X, Y, gram_x=_centered_gram(X))
    assert linear_cka_stimcentered(X, Y) == linear_cka_stimcentered(X, Y, gram_x=_centered_gram(stimcenter(X)))


def test_stimcentered_cka_removes_per_stimulus_offset():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 50))
    Y = X + 5.0 * rng.normal(size=(300, 1))  # offset constant across units, varying across stimuli
    assert linear_cka(X, Y) < 0.5
    assert abs(linear_cka_stimcentered(X, Y) - 1) < 1e-10
    assert abs(rsa(X, Y) - 1) < 1e-10  # correlation distance is offset-invariant as well


def test_cka_identity_and_invariances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))
    Q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    assert abs(linear_cka(X, X) - 1) < 1e-10
    assert abs(linear_cka(X, 3.0 * X @ Q + 5.0) - 1) < 1e-10  # rotation, scale, shift


def test_cka_independent_is_low():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(500, 20)), rng.normal(size=(500, 20))
    assert linear_cka(X, Y) < 0.15


def test_rsa_identity_and_monotone_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 40))
    assert abs(rsa(X, X) - 1) < 1e-10
    small = rsa(X, X + 0.1 * rng.normal(size=X.shape))
    large = rsa(X, X + 2.0 * rng.normal(size=X.shape))
    assert 1 > small > large


def test_cka_wide_matches_feature_space_formula():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(50, 7)), rng.normal(size=(50, 9))
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    ref = np.linalg.norm(Yc.T @ Xc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))
    assert abs(linear_cka(X, Y) - ref) < 1e-10
