import numpy as np

from riemannian_score_sde.noise_floor import (
    comparison_metrics,
    heat_oracle_residual_rows,
    heat_comparison_rows,
    noise_floor_rows,
    residual_energy,
    uniform_time_edges,
)


def test_residual_energy_matches_raw_and_sigma_weighted_definitions():
    target = np.array([[1.0, 0.0], [0.0, 2.0]])
    prediction = np.array([[0.0, 0.0], [0.0, 1.0]])
    sigma_squared = np.array([1.0, 4.0])

    raw = residual_energy(target, prediction)
    weighted = residual_energy(target, prediction, sigma_squared)

    np.testing.assert_allclose(raw["numerator"], 1.0)
    np.testing.assert_allclose(raw["denominator"], 2.5)
    np.testing.assert_allclose(raw["ratio"], 1.0 / 2.5, rtol=1e-7)
    np.testing.assert_allclose(weighted["numerator"], 2.5)
    np.testing.assert_allclose(weighted["denominator"], 8.5)
    np.testing.assert_allclose(weighted["ratio"], 2.5 / 8.5, rtol=1e-7)


def test_comparison_metrics_are_exact_for_identical_vectors():
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    metrics = comparison_metrics(vectors, vectors)
    np.testing.assert_allclose(metrics["rmse"], 0.0)
    np.testing.assert_allclose(metrics["relative_rmse"], 0.0)
    np.testing.assert_allclose(metrics["cosine_similarity"], 1.0)


def test_time_rows_include_both_conditioning_sets_and_last_edge():
    times = np.array([0.1, 0.3, 0.5, 0.9])
    target = np.ones((4, 3))
    marginal = np.zeros((4, 3))
    transition = 0.5 * np.ones((4, 3))
    heat = transition.copy()
    sigma_squared = np.array([0.1, 0.2, 0.3, 0.4])
    edges = uniform_time_edges(0.1, 0.9, 2)

    noise_rows = noise_floor_rows(
        times, target, marginal, transition, sigma_squared, edges
    )
    assert [row["count"] for row in noise_rows] == [2, 2]
    assert noise_rows[-1]["time_upper"] == 0.9
    assert noise_rows[0]["marginal_ratio"] > noise_rows[0]["transition_ratio"]

    heat_rows = heat_comparison_rows(
        times, marginal, transition, heat, sigma_squared, edges
    )
    assert len(heat_rows) == 6  # two regressors: overall plus two bins
    transition_rows = [
        row for row in heat_rows if row["regressor"] == "transition_x0_xt_t"
    ]
    assert all(row["rmse"] == 0.0 for row in transition_rows)


def test_heat_oracle_residual_rows_include_overall_and_time_bins():
    times = np.array([0.1, 0.3, 0.5, 0.9])
    target = np.ones((4, 3))
    heat = 0.5 * np.ones((4, 3))
    sigma_squared = np.array([0.1, 0.2, 0.3, 0.4])
    edges = uniform_time_edges(0.1, 0.9, 2)

    rows = heat_oracle_residual_rows(
        times,
        target,
        heat,
        sigma_squared,
        edges,
    )

    assert [row["scope"] for row in rows] == ["overall", "time_bin", "time_bin"]
    assert rows[0]["count"] == 4
    assert rows[1]["count"] == 2
    assert rows[2]["time_upper"] == 0.9
    assert rows[0]["heat_oracle_ratio"] > 0.0
    assert rows[0]["heat_oracle_sigma_weighted_ratio"] > 0.0
