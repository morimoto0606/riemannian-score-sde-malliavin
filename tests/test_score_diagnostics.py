import numpy as np

from riemannian_score_sde.score_diagnostics import (
    score_error_rows,
    uniform_time_edges,
)


def test_uniform_time_edges_cover_requested_ten_bins():
    np.testing.assert_allclose(uniform_time_edges(10), np.linspace(0.0, 1.0, 11))


def test_score_error_rows_report_unweighted_overall_and_bin_metrics():
    times = np.array([0.001, 0.05, 0.15, 1.0])
    teacher_energy = np.array([1.0, 3.0, 2.0, 4.0])
    model_error = np.array([4.0, 0.0, 2.0, 2.0])
    cosine = np.array([1.0, 0.5, 0.0, -1.0])

    rows = score_error_rows(
        times,
        teacher_energy,
        model_error,
        cosine,
        time_bins=10,
        metadata={"metric_time_weighting": False},
    )

    assert len(rows) == 11
    overall = rows[0]
    assert overall["scope"] == "overall"
    assert overall["count"] == 4
    np.testing.assert_allclose(overall["teacher_energy"], 2.5)
    np.testing.assert_allclose(overall["model_error"], 2.0)
    np.testing.assert_allclose(overall["relative_error"], 0.8)
    np.testing.assert_allclose(overall["RMSE"], np.sqrt(2.0))
    np.testing.assert_allclose(overall["cosine_similarity"], 0.125)
    assert overall["metric_time_weighting"] is False

    first_bin = rows[1]
    assert first_bin["count"] == 2
    np.testing.assert_allclose(first_bin["teacher_energy"], 2.0)
    np.testing.assert_allclose(first_bin["model_error"], 2.0)
    np.testing.assert_allclose(first_bin["relative_error"], 1.0)
    np.testing.assert_allclose(first_bin["RMSE"], np.sqrt(2.0))
    np.testing.assert_allclose(first_bin["cosine_similarity"], 0.75)

    assert rows[2]["count"] == 1
    assert rows[-1]["count"] == 1


def test_empty_time_bins_are_preserved_in_csv_shape():
    rows = score_error_rows(
        [0.5],
        [2.0],
        [1.0],
        [0.25],
        time_bins=10,
    )

    assert len(rows) == 11
    assert sum(row["count"] for row in rows[1:]) == 1
    empty_rows = [row for row in rows[1:] if row["count"] == 0]
    assert len(empty_rows) == 9
    assert all(np.isnan(row["model_error"]) for row in empty_rows)
