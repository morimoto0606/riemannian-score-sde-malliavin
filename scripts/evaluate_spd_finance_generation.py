#!/usr/bin/env python3
"""AIRM sample metrics and finance diagnostics; never trains a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generation_metrics import geodesic_rbf_mmd, nearest_neighbor_distances, distribution_summary


def financial_features(x):
    eigenvalues = np.linalg.eigvalsh(x)
    variance = np.diagonal(x, axis1=-2, axis2=-1)
    correlation = x / np.sqrt(variance[:, :, None] * variance[:, None, :])
    return {"covariance": x, "variance": variance, "correlation": correlation,
            "eigenvalues": eigenvalues,
            "log_eigenvalues": np.log(eigenvalues), "largest_eigenvalue": eigenvalues[:, -1],
            "trace": np.trace(x, axis1=-2, axis2=-1), "determinant": np.linalg.det(x),
            "logdet": np.linalg.slogdet(x)[1], "condition_number": eigenvalues[:, -1] / eigenvalues[:, 0]}


def frechet_mean(train, space, max_iterations=64, tolerance=1e-7):
    """Train-only Karcher iteration with monotone backtracking (AIRM)."""
    mean = np.mean(train, axis=0)
    cost = lambda p: float(np.mean(np.asarray(space.metric.squared_dist(train, p))))
    current_cost = cost(mean)
    converged = False
    for iteration in range(max_iterations):
        tangent = np.mean(np.asarray(space.log(train, mean)), axis=0)
        norm = float(space.metric.norm(tangent, mean))
        if norm < tolerance:
            converged = True
            break
        accepted = False
        for step in 0.5 ** np.arange(16):
            candidate = np.asarray(space.exp(step * tangent, mean))
            candidate_cost = cost(candidate)
            if np.isfinite(candidate_cost) and candidate_cost < current_cost:
                mean, current_cost, accepted = candidate, candidate_cost, True
                break
        if not accepted:
            break
    return mean, {"converged": converged, "iterations": iteration + 1,
                  "gradient_norm": norm, "mean_squared_distance": current_cost}


def plot_comparison(output, real_features, generated_features, tickers, real_pc, generated_pc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def hist(ax, real, generated, title):
        edges = np.histogram_bin_edges(np.concatenate([real, generated]), bins=35)
        ax.hist(real, bins=edges, density=True, histtype="step", label="reference", linewidth=1.5)
        ax.hist(generated, bins=edges, density=True, histtype="step", label="generated", linewidth=1.5)
        ax.set_title(title)

    for key, title, filename in [("log_eigenvalues", "log eigenvalue", "log_eigenvalues.png"),
                                  ("variance", "variance", "asset_variances.png")]:
        fig, axes = plt.subplots(1, len(tickers), figsize=(18, 3.5))
        for i, ax in enumerate(axes):
            label = tickers[i] if key == "variance" else str(i + 1)
            hist(ax, real_features[key][:, i], generated_features[key][:, i], title + " " + label)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, key in zip(axes.flat, ("largest_eigenvalue", "trace", "determinant", "logdet", "condition_number")):
        hist(ax, real_features[key], generated_features[key], key)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / "matrix_statistics.png", dpi=160)
    plt.close(fig)
    pairs = list(zip(*np.triu_indices(len(tickers), 1)))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (i, j) in zip(axes.flat, pairs):
        hist(ax, real_features["correlation"][:, i, j], generated_features["correlation"][:, i, j],
             f"{tickers[i]} / {tickers[j]}")
        ax.set_xlim(-1, 1)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / "pairwise_correlations.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(real_pc[:, 0], real_pc[:, 1], s=10, alpha=0.4, label="reference")
    ax.scatter(generated_pc[:, 0], generated_pc[:, 1], s=10, alpha=0.4, label="generated")
    ax.set(xlabel="PC1", ylabel="PC2", title="AIRM tangent PCA (mean and basis fitted on train only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "tangent_pca.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None, help="Override saved dataset path (e.g. after moving servers)")
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.sigma <= 0 or not np.isfinite(args.sigma) or args.chunk_size < 1 or args.max_samples < 2:
        raise ValueError("Positive sigma/chunk-size and max-samples>=2 required")
    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["JAX_ENABLE_X64"] = "True"
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from riemannian_score_sde.spd import AffineSPD, to_frame
    from riemannian_score_sde.spd_generation import spd_summary

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    # Saved work_dir uses a Hydra runtime resolver; restore only this location.
    # User may instead point --dataset at the server-local immutable snapshot.
    if args.dataset is None:
        if "${hydra:" in str(OmegaConf.to_container(cfg, resolve=False).get("work_dir", "")):
            hydra_cfg = OmegaConf.load(run_dir / ".hydra/hydra.yaml")
            cfg.work_dir = hydra_cfg.hydra.runtime.cwd
        dataset_path = Path(str(cfg.dataset.data_path)).expanduser()
    else:
        dataset_path = args.dataset.expanduser()
    samples_path = args.samples_path or run_dir / "generated_samples.npy"
    output = args.output_dir or run_dir / "spd_evaluation"
    with np.load(dataset_path, allow_pickle=False) as data:
        cov = data["covariances"]
        train = cov[data["train_indices"]]
        real = cov[data[f"{args.split}_indices"]]
        tickers = data["tickers"].tolist()
    generated = np.load(samples_path, allow_pickle=False)
    provenance_path = samples_path.with_suffix(".metadata.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance and provenance.get("training_data_sha256") != hashlib.sha256(
            np.ascontiguousarray(train, dtype=np.float64).tobytes()).hexdigest():
        raise ValueError("Evaluation dataset's train split differs from the generation terminal data")
    if generated.shape[1:] != (5, 5) or real.shape[1:] != (5, 5):
        raise ValueError("This financial experiment requires (N,5,5) samples")
    real_spd, generated_spd = spd_summary(real), spd_summary(generated)
    rng = np.random.default_rng(args.seed)
    selected = lambda x: x[rng.choice(len(x), min(args.max_samples, len(x)), replace=False)]
    metric_real, metric_generated = selected(real), selected(generated)
    space = AffineSPD(5)
    pair_jit = jax.jit(lambda a, b: jax.vmap(lambda x: jax.vmap(lambda y: space.metric.dist(x, y))(b))(a))

    def pairwise(a, b):
        # Both axes chunked, so no unbounded intermediate n*n*5*5 tensor.
        blocks = []
        for start in range(0, len(a), args.chunk_size):
            blocks.append(np.concatenate([np.asarray(pair_jit(jnp.asarray(a[start:start + args.chunk_size]),
                                                               jnp.asarray(b[j:j + args.chunk_size])))
                                          for j in range(0, len(b), args.chunk_size)], axis=1))
        result = np.concatenate(blocks, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite AIRM distance")
        return result

    mmd = geodesic_rbf_mmd(metric_real, metric_generated, args.sigma, args.chunk_size, pairwise)
    g_to_r = nearest_neighbor_distances(metric_generated, metric_real, args.chunk_size, pairwise)
    r_to_g = nearest_neighbor_distances(metric_real, metric_generated, args.chunk_size, pairwise)
    gram_points = np.concatenate([metric_real[:32], metric_generated[:32]])
    gram = np.exp(-pairwise(gram_points, gram_points) ** 2 / (2 * args.sigma**2))
    features = {"reference": financial_features(real), "generated": financial_features(generated)}
    summaries = {}
    for group, values in features.items():
        summaries[group] = {k: distribution_summary(values[k]) for k in
                            ("largest_eigenvalue", "trace", "determinant", "logdet", "condition_number")}
        summaries[group]["asset_variances"] = {ticker: distribution_summary(values["variance"][:, i]) for i, ticker in enumerate(tickers)}
        summaries[group]["eigenvalues"] = [distribution_summary(values["eigenvalues"][:, i])
                                            for i in range(5)]
        summaries[group]["mean_covariance"] = values["covariance"].mean(axis=0).tolist()
        summaries[group]["mean_correlation"] = values["correlation"].mean(axis=0).tolist()
        summaries[group]["pairwise_correlations"] = {
            f"{tickers[i]}/{tickers[j]}": distribution_summary(values["correlation"][:, i, j])
            for i, j in zip(*np.triu_indices(5, 1))}
    output.mkdir(parents=True, exist_ok=True)
    if cfg.loss._target_.endswith("get_ism_loss_fn"):
        teacher = "ism"
    elif cfg.teacher._target_.endswith("SPDMalliavinTeacher"):
        teacher = "malliavin_" + cfg.teacher.divergence_mode
    elif cfg.teacher._target_.endswith("VaradhanTeacher"):
        teacher = "varadhan"
    else:
        raise ValueError("Unknown SPD objective in saved config")
    report = {"experiment": str(cfg.experiment), "teacher": teacher, "training_seed": int(cfg.seed),
              "generation_provenance": provenance,
              "time_weighting": bool(cfg.loss.get("time_weighting", False)),
              "time_weight_lambda": float(cfg.loss.get("time_weight_lambda", 0.)),
              "number_generated_samples": len(generated), "number_reference_samples": len(real),
              "metric_generated_samples": len(metric_generated), "metric_reference_samples": len(metric_real),
              "reference_split": args.split, "distance": "AIRM", "sigma": args.sigma,
              "rbf_mmd2_unbiased": float(mmd), "generated_to_reference_nn": distribution_summary(g_to_r),
              "reference_to_generated_nn": distribution_summary(r_to_g),
              "kernel_gram_probe_min_eigenvalue": float(np.linalg.eigvalsh((gram + gram.T) / 2).min()),
              "kernel_note": "AIRM geodesic Gaussian is not guaranteed PSD for all bandwidths; this is a kernel discrepancy, not a guaranteed RKHS metric. The finite Gram probe cannot prove global PSD.",
              "reference_spd": real_spd, "generated_spd": generated_spd, "finance": summaries,
              "evaluation_seed": args.seed, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest()}
    if not args.no_plots:
        mean, mean_info = frechet_mean(train, space)
        coordinate_fn = jax.jit(lambda x: to_frame(space.log(x, jnp.asarray(mean)), jnp.asarray(mean)))
        train_coordinates = np.asarray(coordinate_fn(jnp.asarray(train)))
        center = train_coordinates.mean(axis=0)
        _, singular_values, basis = np.linalg.svd(train_coordinates - center, full_matrices=False)
        real_pc = (np.asarray(coordinate_fn(jnp.asarray(real))) - center) @ basis[:2].T
        gen_pc = (np.asarray(coordinate_fn(jnp.asarray(generated))) - center) @ basis[:2].T
        np.savez_compressed(output / "tangent_pca.npz", reference_point=mean, center=center,
                            components=basis[:2], singular_values=singular_values,
                            reference=real_pc, generated=gen_pc)
        report["pca_reference_mean"] = mean_info
        plot_comparison(output, features["reference"], features["generated"], tickers, real_pc, gen_pc)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved SPD diagnostics in {output}")


if __name__ == "__main__":
    main()
