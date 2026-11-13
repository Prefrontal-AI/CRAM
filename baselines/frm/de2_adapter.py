"""Run the pinned author FRM recipe and record exact Sudoku solves.

A copy of the Table 2 adapter frm_reproduction.py (the campaign's frozen file, SHA-256 e4e9ff14...) for the
data-efficiency sweep, built into each run root by baselines/frm/de2_setup.py. The authors' code is
github.com/helblazer811/Flow-Reasoning-Models at 6149a01. Run through frm/de2.sbatch. Changes, and only these:
- training rows: the root's cache/sudoku_extreme/train.npz holds the stored rows (k unique puzzles, each the original
  and its 99 stored shuffle_sudoku copies; experiments/runs/trm/data/sudoku-smalldata-v99), read whole (data.train_max_samples
  null) with FRM's on-the-fly symmetry augmentation off (data.aug_symmetry_prob 0); data/train-s<seed>.npz is that file.
- the in-training and final "train" monitor evaluates data/train-monitor-s<seed>.npz (the first min(k, 1,000)
  originals) instead of every training row; it does not enter checkpoint or sampler selection.
- finish scores the selected checkpoint and sampler on data/test10k.npz (full-all rows 10,000-19,999, which hold FRM's
  1,000 test puzzles) at batch 1,024, as Table 2's unseen scoring (frm/score_unseen.py), and writes runs/s<seed>/test10k.json
  and test10k.solved.npy (full-all order); FRM's separate 1,000-puzzle test evaluation is dropped (a subset of test10k).

The upstream package owns the model, objective, optimizer, data augmentation and
samplers. This adapter owns frozen inputs, paired evaluation, selection and records.
Tensor axes: b = puzzles, l = 81 cells, v = 10 token classes.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import random
import runpy
import signal
import statistics
import sys
import time


@dataclass
class Config:
    root: Path  # Frozen campaign directory with plan.json and upstream source.
    action: str = "train"  # prepare, train, finish, audit, or summarize.
    seed: int = 0  # Training, subset and augmentation seed.
    stage: str = "A"  # A for self-conditioning; B for fixed-point forcing.
    max_steps: int = 100_000  # Total updates in this stage, including resumed updates.
    smoke: bool = False  # Separate short qualification run; never contributes to reproduction results.
    checkpoint: Path | None = None  # Existing checkpoint for the audit action.


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return data["solutions"].copy(), data["clue_mask"].copy()


def prepare(root, plan):
    import numpy as np
    from flow_map_reasoning.datasets.sudoku_extreme import SudokuExtremeDataset

    out = root / "data"
    out.mkdir(exist_ok=True)
    records = {}
    for seed in plan["seeds"]:
        ds = SudokuExtremeDataset(root=root / "cache", split="train", download=False, max_samples=1000, subsample_seed=seed, balanced_by_rating=True)
        path = out / f"train-s{seed}.npz"
        np.savez_compressed(path, solutions=ds.solutions, clue_mask=ds.clue_masks, ratings=ds.ratings)
        records[path.name] = {"sha256": digest(path), "count": len(ds), "subsample_seed": seed}
    for split in ("dev", "test"):
        original = Path(plan["historical_data"]) / ("validation" if split == "dev" else "test") / "sudoku_extreme/test.npz"
        ds = SudokuExtremeDataset(root=original.parent.parent, split="test", download=False)
        path = out / f"{split}.npz"
        np.savez_compressed(path, solutions=ds.solutions, clue_mask=ds.clue_masks, ratings=ds.ratings)
        records[path.name] = {"sha256": digest(path), "count": len(ds), "origin": str(original), "origin_sha256": digest(original)}
    identities = {}
    for path in out.glob("*.npz"):
        solutions, clues = arrays(path)
        identities[path.stem] = {row.tobytes() for row in (solutions * clues).astype("uint8")}
        assert len(identities[path.stem]) == len(solutions), f"Duplicate puzzles: {path}"
    assert not identities["dev"] & identities["test"]
    for seed in plan["seeds"]:
        assert not identities[f"train-s{seed}"] & (identities["dev"] | identities["test"])
    save(out / "manifest.json", {"files": records, "exact_puzzle_disjointness": True, "scope": "Exact puzzle identity; not a symmetry-equivalence audit."})


def profile(model):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[0]), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        z = torch.zeros((1, 81, 10), device="cuda")
        model(z, torch.full((1,), 0.5, device="cuda"), self_cond=z)
    model.train(training)
    return {"params": sum(p.numel() for p in model.parameters()), "flops_per_forward": int(counter.get_total_flops()), "convention": "Forward operator FLOPs, multiply-add=2; excludes uncounted elementwise operations and overhead."}


def evaluate(model, lut, root, out, *, seed, stage, step, split, iterations, kind="held_t", smoke=False, check_hidden=False, cost=None, batch_size=128):
    import numpy as np
    import torch
    from flow_map_reasoning import SamplerConfig, run_sampler

    path = root / "data" / (f"train-monitor-s{seed}.npz" if split == "train" else f"{split}.npz")  # train monitor = first min(k, 1,000) originals
    solutions, clues = arrays(path)
    if smoke:
        solutions, clues = solutions[:8], clues[:8]
    sampler = SamplerConfig(kind=kind, self_cond=True, steps=iterations, held_iters=iterations, t_hold=0.4, renoise=1.0, batch_size=batch_size)
    cost = cost or profile(model)
    calls = [0]

    def count(_module, args):
        calls[0] += len(args[0])

    handle = model.register_forward_pre_hook(count)
    training = model.training
    model.eval()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = run_sampler(model, solutions, clues, sampler, lut_a2g=lut, vocab_size=10, device="cuda")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            measured_calls = calls[0]
            if check_hidden:
                changed = np.where(clues.astype(bool), solutions, solutions % 9 + 1)
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)
                counterfactual = run_sampler(model, changed, clues, sampler, lut_a2g=lut, vocab_size=10, device="cuda")
                assert np.array_equal(predictions, counterfactual), "Hidden targets affected predictions"
    finally:
        handle.remove()
        model.train(training)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    assert predictions.shape == solutions.shape
    assert np.array_equal(predictions[clues.astype(bool)], solutions[clues.astype(bool)])
    assert measured_calls == len(solutions) * (iterations + 1), (measured_calls, iterations)
    solved = np.all(predictions == solutions, axis=1)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", predictions=predictions, solved=solved)
    result = dict(seed=seed, stage=stage, step=step, split=split, solved=int(solved.sum()), count=len(solved), accuracy=float(solved.mean()), nfe=measured_calls / len(solved), seconds=seconds, sampler=asdict(sampler), smoke=smoke, data_sha256=digest(path), predictions=str(out / "predictions.npz"), **cost)
    result["flops_per_puzzle"] = result["nfe"] * cost["flops_per_forward"]
    save(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("stage", "step", "split", "solved", "count", "nfe", "seconds", "smoke")}), flush=True)
    return result


def train(c, root, plan):
    import torch
    import wandb
    import flow_map_reasoning as fmr
    import flow_map_reasoning.trainer as trainer

    lane = root / ("qualification" if c.smoke else "runs") / f"s{c.seed}"
    out = lane / f"stage{c.stage}"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "upstream"
    assert Path(fmr.__file__).resolve().is_relative_to(source)
    requested_stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: requested_stop.__setitem__(0, True))
    original_save = trainer._save_checkpoint

    def atomic_checkpoint(path, step, *args, **kwargs):
        path = Path(path)
        temporary = path.with_suffix(".pending")
        original_save(temporary, step, *args, **kwargs)
        temporary.replace(path)
        if requested_stop[0] and path.name.startswith("step_"):
            save(out / "interrupted.json", {"step": step, "checkpoint": str(path)})
            raise SystemExit(75)

    trainer._save_checkpoint = atomic_checkpoint
    original_init = wandb.init
    active_log = [None]

    def init(*args, **kwargs):
        kwargs["config"] = dict(kwargs.get("config", {}), provenance=plan, resolved_recipe=resolved)
        kwargs["settings"] = wandb.Settings(init_timeout=30)
        run = original_init(*args, **kwargs)
        active_log[0] = run.log
        # wandb.init replaces module-level log; install our local journal afterward.
        wandb.log = log
        save(out / "wandb.json", {"id": run.id, "url": run.url, "mode": run.settings.mode})
        return run

    def log(values, *args, **kwargs):
        values = dict(values)
        if values.get("train/steps_per_sec", 0) > 0:
            values["train/update_seconds"] = 1 / values["train/steps_per_sec"]
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": kwargs.get("step"), "time": time.time(), **values}) + "\n")
        active_log[0](values, *args, **kwargs)

    wandb.init, wandb.log = init, log
    original_train = fmr.train

    def recorded_train(model, train_loader, cfg, **kwargs):
        cfg.val_every = c.max_steps if c.smoke else 5000
        cfg.save_best_metric = "DevSet/accuracy"
        dataset = train_loader.dataset
        while not hasattr(dataset, "solutions"):
            dataset = dataset.dataset
        import numpy as np
        solutions, clues = arrays(root / "data" / f"train-s{c.seed}.npz")
        assert np.array_equal(dataset.solutions, solutions) and np.array_equal(dataset.clue_masks, clues)
        cost = profile(model)
        assert cost["params"] == 7_027_978
        save(out / "model-cost.json", cost)
        lut = fmr.TorchLUT(fmr.build_luts(K=10)[0], device="cuda")

        def paired(core, step):
            iterations = 2 if c.smoke else (128 if c.stage == "A" else 1024)
            kind = "euler" if c.stage == "A" else "held_t"
            results = [evaluate(core, lut, root, out / "monitor" / f"{step:08d}" / split, seed=c.seed, stage=c.stage, step=step, split=split, iterations=iterations, kind=kind, smoke=c.smoke, cost=cost) for split in ("train", "dev")]
            return {f"{label}/{key}": result[key] for label, result in zip(("TrainSet", "DevSet"), results) for key in ("solved", "count", "accuracy", "nfe", "flops_per_puzzle")}

        kwargs.update(val_fn=paired, checkpoint_provenance=plan)
        return original_train(model, train_loader, cfg, **kwargs)

    fmr.train = recorded_train
    entry = source / "experiments/sudoku_extreme_train/2_train.py"
    sys.argv = [str(entry), "--config", str(entry.parent / f"recipes/sudoku_7m_stage{c.stage}.yaml"), "--seed", str(c.seed), "--checkpoint_dir", str(out), "--wandb_run_name", f"frm-sudoku-reproduction/{'smoke-' if c.smoke else ''}{c.stage}-s{c.seed}-{root.name}"]
    checkpoints = sorted(out.glob("step_*.pt"))
    if checkpoints:
        sys.argv += ["--resume", str(checkpoints[-1])]
    elif c.stage == "B":
        parent = lane / "stageA/best.pt"
        assert parent.exists(), "Stage A must finish and select its weights first"
        sys.argv += ["--init_from", str(parent)]
        save(out / "parent.json", {"checkpoint": str(parent), "sha256": digest(parent), "selection": "Highest Stage-A DevSet pass@1 at Euler 128; earliest tie. EMA weights, fresh optimizer and EMA."})
    overrides = {
        "data.extreme_root": str(root / "cache"), "data.train_subsample_seed": c.seed, "data.aug_symmetry_seed": c.seed,
        "train.max_steps": c.max_steps, "train.save_every": (c.max_steps if c.smoke else 1000), "train.log_every": (16 if c.smoke else 100),
        "train.no_wandb": False, "train.wandb_project": "frm-reproduction", "val.extreme_val": False,
        "data.train_max_samples": "null", "data.aug_symmetry_prob": 0.0,  # every stored row, no on-the-fly augmentation
    }
    for key, value in overrides.items():
        sys.argv += ["--set", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
    namespace = runpy.run_path(str(entry))
    _, built = namespace["build_config"](namespace["parse_args"]())
    resolved = {key: asdict(value) for key, value in built.items()}
    resolved["adapter"] = {"paired_evaluation": True, "stage_a_parent": "best EMA", "save_every": overrides["train.save_every"]}
    save(out / "resolved.json", resolved)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "job": os.environ.get("SLURM_JOB_ID"), "source": str(source)}), flush=True)
    started = time.perf_counter()
    namespace["main"]()
    wandb.finish()
    save(out / "completed.json", {"step": c.max_steps, "seconds": time.perf_counter() - started, "checkpoint": str(out / "final.pt"), "sha256": digest(out / "final.pt")})


def load_model(root, checkpoint):
    sys.path.insert(0, str(root / "upstream/experiments/sudoku_extreme_train"))
    from evaluate import load_checkpoint_model
    return load_checkpoint_model(checkpoint, "cuda", use_ema=True)


def finish(c, root, plan):
    import torch
    import wandb

    lane = root / ("qualification" if c.smoke else "runs") / f"s{c.seed}"
    candidates = []
    steps = [c.max_steps] if c.smoke else [25_000, 50_000, 75_000, 100_000]
    budgets = [2] if c.smoke else plan["evaluation_iterations"]
    run = wandb.init(project="frm-reproduction", name=f"frm-sudoku-reproduction/{'smoke-' if c.smoke else ''}evaluation-s{c.seed}-{root.name}", config=plan)
    for step in steps:
        checkpoint = lane / "stageB" / f"step_{step:08d}.pt"
        model, meta = load_model(root, checkpoint)
        cost = profile(model)
        for iterations in budgets:
            out = lane / "frontier" / f"step{step}-held{iterations}"
            result_path = out / "dev/result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                assert result["checkpoint_sha256"] == digest(checkpoint)
            else:
                result = evaluate(model, meta["lut_a2g"], root, out / "dev", seed=c.seed, stage="B", step=step, split="dev", iterations=iterations, smoke=c.smoke, check_hidden=c.smoke, cost=cost)
                result.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
                save(result_path, result)
            candidates.append(result)
        del model
        torch.cuda.empty_cache()
    selected = min(candidates, key=lambda item: (-item["solved"], item["flops_per_puzzle"], item["step"]))
    save(lane / "selection.json", {"selected": selected, "candidates": candidates, "rule": "Most exact DevSet solves, then fewer executed FLOPs, then earlier update. TestSet unopened by selection."})
    model, meta = load_model(root, selected["checkpoint"])
    results = {"dev": selected}
    for split in ("train", "test10k"):  # test10k replaces FRM's 1,000-puzzle test
        done = lane / "selected" / split / "result.json"
        if done.exists() and json.loads(done.read_text()).get("checkpoint_sha256") == selected["checkpoint_sha256"]:
            results[split] = json.loads(done.read_text())
            continue
        result = evaluate(model, meta["lut_a2g"], root, lane / "selected" / split, seed=c.seed, stage="B", step=selected["step"], split=split, iterations=selected["sampler"]["held_iters"], smoke=c.smoke, batch_size=1024 if split == "test10k" else 128)
        result.update(checkpoint=selected["checkpoint"], checkpoint_sha256=selected["checkpoint_sha256"])
        save(done, result)
        results[split] = result
    save(lane / "result.json", results)
    import numpy as np
    t = results["test10k"]
    with np.load(t["predictions"]) as p:
        solved = p["solved"].astype(bool)
    assert len(solved) == t["count"] == (8 if c.smoke else 10000)
    np.save(lane / "test10k.solved.npy", solved)
    save(lane / "test10k.json", {"method": "FRM", "seed": c.seed, "n": int(len(solved)), "solved": int(solved.sum()), "rate": float(solved.mean()), "rows": "full-all 10000-19999", "checkpoint": t["checkpoint"], "checkpoint_sha256": t["checkpoint_sha256"], "step": t["step"], "sampler": t["sampler"], "batch_size": 1024, "nfe": t["nfe"], "flops_per_puzzle": t["flops_per_puzzle"], "seconds": t["seconds"], "flags": str(lane / "test10k.solved.npy"), "smoke": c.smoke})
    run.log({f"{split}/{key}": result[key] for split, result in results.items() for key in ("solved", "count", "accuracy", "nfe", "flops_per_puzzle")})
    run.finish()


def audit(c, root):
    import torch
    assert c.checkpoint is not None
    model, meta = load_model(root, c.checkpoint)
    step = torch.load(c.checkpoint, map_location="cpu", weights_only=False)["step"]
    results = {}
    for split in ("train", "dev"):
        results[split] = evaluate(model, meta["lut_a2g"], root, root / "historical-audit" / split, seed=0, stage="B", step=step, split=split, iterations=4096, check_hidden=False)
    save(root / "historical-audit/result.json", {"checkpoint": str(c.checkpoint), "sha256": digest(c.checkpoint), **results})


def summarize(root, plan):
    lanes = [json.loads((root / f"runs/s{seed}/result.json").read_text()) for seed in plan["seeds"]]
    scores = [lane["test"]["accuracy"] * 100 for lane in lanes]
    summary = {"lanes": lanes, "test_mean_percent": statistics.mean(scores), "test_sample_sd_percent": statistics.stdev(scores), "test_peak_percent": max(scores), "paper_peak_percent": 99.5, "paper_mean_percent": 99.2, "paper_sd_percent": 0.3, "peak_target_met": max(scores) >= 99.5, "same_author_test_indices": False}
    save(root / "summary.json", summary)
    lines = ["# Fresh FRM Sudoku reproduction", "", "| Seed | Stage B update | TrainSet | DevSet | TestSet | Calls/puzzle | FLOPs/puzzle |", "|---|---:|---:|---:|---:|---:|---:|"]
    for seed, lane in zip(plan["seeds"], lanes):
        counts = [f"{lane[split]['solved']}/{lane[split]['count']}" for split in ("train", "dev", "test")]
        lines.append(f"| {seed} | {lane['test']['step']} | {' | '.join(counts)} | {lane['test']['nfe']:g} | {lane['test']['flops_per_puzzle']:.6g} |")
    lines += ["", f"Test mean {summary['test_mean_percent']:.2f}%, sample SD {summary['test_sample_sd_percent']:.2f}%, peak {max(scores):.2f}%. Paper: mean 99.2% ± 0.3%, peak 99.5%.", "", "The exact author test indices and original trained weights are unavailable. This is a recipe reproduction on a fixed local 1,000-puzzle test subset, not bitwise reproduction or verification of the 44× baseline comparison."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


def main(c):
    root = c.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    assert digest(__file__) == plan["harness_sha256"], "Frozen adapter changed"
    for relative, expected in json.loads((root / "source-files.json").read_text()).items():
        assert digest(root / "upstream" / relative) == expected, f"Upstream source changed: {relative}"
    if c.action == "summarize":
        return summarize(root, plan)
    import torch
    assert torch.cuda.is_available(), "Run inside the requested Slurm GPU allocation"
    if c.action == "prepare":
        prepare(root, plan)
    elif c.action == "train":
        train(c, root, plan)
    elif c.action == "finish":
        finish(c, root, plan)
    elif c.action == "audit":
        audit(c, root)
    else:
        raise ValueError(c.action)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
