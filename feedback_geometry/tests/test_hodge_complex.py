"""Unit tests for feedback_geometry/src/hodge_complex.py (SGB-045 Stage 0).

Run with either:
    ./venv/bin/python3 feedback_geometry/tests/test_hodge_complex.py
    ./venv/bin/python3 -m pytest feedback_geometry/tests/test_hodge_complex.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hodge_complex import (  # noqa: E402
    build_clique_complex,
    canonicalize_flow,
    hodge_decompose_2d,
)

TOL = 1e-9


def _random_graph(rng, n, p):
    return [(i, j) for i, j in combinations(range(n), 2) if rng.random() < p]


# 3-cycle 0 -> 1 -> 2 -> 0 ("1 beats 0", "2 beats 1", "0 beats 2").
# Canonical edges (0,1),(1,2),(0,2); positive flow = second index preferred.
CYCLE_EDGES = [(0, 1), (1, 2), (0, 2)]
CYCLE_FLOW = np.array([1.0, 1.0, -1.0])


def test_pure_gradient_is_all_gradient():
    rng = np.random.default_rng(0)
    n = 12
    edges = _random_graph(rng, n, 0.4)
    cx = build_clique_complex(n, edges)
    assert cx.n_triangles > 0, "test graph must contain triangles"
    phi = rng.normal(size=n)
    f = cx.d0 @ phi
    w = rng.uniform(0.5, 3.0, size=len(edges))
    for weights in (None, w):
        r = hodge_decompose_2d(cx, f, weights)
        assert abs(r.fractions["gradient"] - 1.0) < 1e-9, r.fractions
        assert np.max(np.abs(r.curl)) < 1e-9
        assert np.max(np.abs(r.harmonic)) < 1e-9
        assert np.max(np.abs(r.triangle_curl)) < 1e-9
        # potential recovered up to a constant
        assert np.max(np.abs((r.potential - r.potential.mean()) - (phi - phi.mean()))) < 1e-8


def test_filled_three_cycle_is_all_curl():
    cx = build_clique_complex(3, CYCLE_EDGES)
    assert cx.n_triangles == 1
    r = hodge_decompose_2d(cx, CYCLE_FLOW)
    assert abs(r.fractions["curl"] - 1.0) < TOL, r.fractions
    assert r.fractions["gradient"] < TOL and r.fractions["harmonic"] < TOL
    assert abs(r.triangle_curl[0] - 3.0) < TOL
    assert r.betti1 == 0
    # With unequal weights the W-divergence is no longer zero, so part of the flow can be
    # gradient; but beta_1 = 0, so there is never a harmonic part.
    r2 = hodge_decompose_2d(cx, CYCLE_FLOW, [1.0, 2.0, 5.0])
    assert r2.fractions["harmonic"] < 1e-9, r2.fractions
    assert r2.fractions["curl"] > 0.5, r2.fractions


def test_unfilled_three_cycle_is_all_harmonic():
    cx = build_clique_complex(3, CYCLE_EDGES, fill_triangles=False)
    assert cx.n_triangles == 0
    r = hodge_decompose_2d(cx, CYCLE_FLOW)
    assert abs(r.fractions["harmonic"] - 1.0) < TOL, r.fractions
    assert r.betti1 == 1


def test_four_cycle_hole_is_all_harmonic_and_chord_fills_it():
    # square 0->1->2->3->0: the clique complex has no triangle, so the loop is a hole.
    edges, f = canonicalize_flow([(0, 1), (1, 2), (2, 3), (3, 0)], [1.0, 1.0, 1.0, 1.0])
    cx = build_clique_complex(4, edges)
    assert cx.n_triangles == 0
    r = hodge_decompose_2d(cx, f)
    assert abs(r.fractions["harmonic"] - 1.0) < TOL, r.fractions
    assert r.betti1 == 1
    # adding the chord (0,2) with any value fills two triangles -> harmonic vanishes
    for chord in (-1.0, 0.0, 0.7):
        edges2 = edges + [(0, 2)]
        f2 = np.append(f, chord)
        cx2 = build_clique_complex(4, edges2)
        assert cx2.n_triangles == 2
        r2 = hodge_decompose_2d(cx2, f2)
        assert r2.betti1 == 0
        assert r2.fractions["harmonic"] < 1e-9, (chord, r2.fractions)


def test_three_parts_are_w_orthogonal_and_sum_to_flow():
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = int(rng.integers(6, 25))
        edges = _random_graph(rng, n, float(rng.uniform(0.15, 0.6)))
        if not edges:
            continue
        cx = build_clique_complex(n, edges)
        f = rng.normal(size=len(edges))
        w = rng.uniform(0.2, 5.0, size=len(edges))
        r = hodge_decompose_2d(cx, f, w)
        assert np.max(np.abs(r.gradient + r.curl + r.harmonic - f)) < 1e-9
        ip = lambda a, b: float(np.sum(w * a * b))  # noqa: E731
        scale = ip(f, f)
        assert abs(ip(r.gradient, r.curl)) < 1e-8 * scale
        assert abs(ip(r.gradient, r.harmonic)) < 1e-8 * scale
        assert abs(ip(r.curl, r.harmonic)) < 1e-8 * scale
        assert abs(sum(r.fractions.values()) - 1.0) < 1e-9
        # harmonic is closed (d1 h = 0) and W-co-closed (d0^T W h = 0)
        if cx.n_triangles:
            assert np.max(np.abs(cx.d1 @ r.harmonic)) < 1e-8
        assert np.max(np.abs(cx.d0.T @ (w * r.harmonic))) < 1e-8
        # beta_1 = dim of harmonic space; check it via the combinatorial Laplacian kernel
        L1 = cx.d0 @ cx.d0.T + (cx.d1.T @ cx.d1 if cx.n_triangles else 0)
        ker = int(np.sum(np.linalg.eigvalsh(L1) < 1e-8))
        assert ker == r.betti1, (ker, r.betti1)


def test_d1_d0_is_zero():
    rng = np.random.default_rng(2)
    for _ in range(10):
        n = int(rng.integers(4, 30))
        edges = _random_graph(rng, n, 0.5)
        cx = build_clique_complex(n, edges)
        if cx.n_triangles:
            assert np.max(np.abs(cx.d1 @ cx.d0)) == 0.0


def test_no_triangles_reproduces_old_residual():
    """With fill_triangles=False the harmonic part equals the old code's residual f - d0 phi."""
    rng = np.random.default_rng(3)
    n = 15
    edges = _random_graph(rng, n, 0.4)
    f = rng.normal(size=len(edges))
    cx0 = build_clique_complex(n, edges, fill_triangles=False)
    r0 = hodge_decompose_2d(cx0, f)
    assert np.max(np.abs(r0.curl)) == 0.0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hodge_utils import hodge_decompose as old  # noqa: E402

    old_r = old(f, edges, n)
    # old code's B is (E x n) with -1 at source, +1 at target: same orientation as d0
    assert np.max(np.abs(old_r.harmonic - r0.harmonic)) < 1e-6


def test_canonicalize_flips_sign():
    e, f = canonicalize_flow([(2, 0), (0, 1)], [0.5, -1.0])
    assert e == [(0, 2), (0, 1)]
    assert list(f) == [-0.5, -1.0]


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
