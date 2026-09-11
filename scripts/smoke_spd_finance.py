#!/usr/bin/env python3
"""Server-only: three one-update runs, checkpoint restores and 8 SPD samples each.

Uses production model/forward defaults unless --tiny is explicitly requested.
No existing output directories or checkpoint files are overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Maximum seconds per subprocess, including compilation")
    parser.add_argument("--tiny", action="store_true",
                        help="Use batch 2, forward N=1, hidden=[16,16]; not production-shape validation")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.output_root:
        output = args.output_root.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "results").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="spd_finance_smoke_", dir=ROOT / "results"))
    env = dict(os.environ, GEOMSTATS_BACKEND="jax", JAX_ENABLE_X64="true")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "geomstats"), str(ROOT), env.get("PYTHONPATH", "")])

    def run(arguments, log):
        with log.open("w") as handle:
            subprocess.run([sys.executable] + arguments, cwd=ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)

    def hashes(checkpoint):
        return {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                for name in ("tree.pkl", "arrays.npy")}

    results = []
    for method in ("varadhan", "ism", "malliavin_hutchinson"):
        experiment = "spd_finance_" + method
        train_dir, gen_dir = output / experiment, output / (experiment + "_generation")
        common = ["experiment=" + experiment, "logger=csv", "seed=0", "steps=1"]
        if args.tiny:
            common += ["batch_size=2", "eval_batch_size=2", "flow.N=1",
                       "architecture.hidden_shapes=[16,16]"]
        if method == "malliavin_hutchinson":
            common += ["loss.time_weighting=true", "loss.time_weight_lambda=5.0"]
        print("Checking", experiment, "logs:", output, flush=True)
        run(["main.py", "--info", "defaults-tree"] + common, output / (method + "_tree.log"))
        run(["main.py", "--info", "defaults"] + common, output / (method + "_defaults.log"))
        run(["main.py", "--cfg", "job"] + common, output / (method + "_config.log"))
        run(["main.py"] + common + ["mode=train", "generation.enabled=false",
            "ckpt_dir=" + str(train_dir / "ckpt"), "hydra.run.dir=" + str(train_dir)],
            output / (method + "_train.log"))
        before = hashes(train_dir / "ckpt")
        samples = gen_dir / "generated_samples.npy"
        run(["main.py"] + common + ["mode=test", "generation.enabled=true",
            "generation.count=8", "generation.batch_size=8", "generation.steps=4",
            "ckpt_dir=" + str(train_dir / "ckpt"), "hydra.run.dir=" + str(gen_dir),
            "generated_samples_path=" + str(samples)], output / (method + "_generation.log"))
        if hashes(train_dir / "ckpt") != before:
            raise RuntimeError("Generation modified checkpoint: " + experiment)
        report = json.loads(samples.with_suffix(".metadata.json").read_text())
        if report["checkpoint_step"] != 1 or report["spd"]["count"] != 8 or report["spd"]["non_spd_count"] != 0:
            raise RuntimeError("Unexpected checkpoint/sample validation: " + experiment)
        run(["scripts/evaluate_spd_finance_generation.py", "--run-dir", str(gen_dir),
             "--max-samples", "8", "--chunk-size", "8", "--no-plots"],
            output / (method + "_evaluation.log"))
        results.append({"experiment": experiment, "tiny": args.tiny, "checkpoint_step": 1,
                        "checkpoint_unchanged": True, "spd": report["spd"]})
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS:", output / "summary.json")


if __name__ == "__main__":
    main()
