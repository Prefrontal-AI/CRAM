"""Run the pinned author FRM Maze-Unique recipe and record exact maze solves.

The upstream package owns the model, objective, optimizer and samplers. This adapter
owns frozen inputs, paired monitoring, DevSet-only selection, the sealed TestSet and
records. Tensor axes: b = puzzles, l = 900 cells, v = 6 cell states.

Protocols: dev900 trains on 900 official training mazes and selects on the other 100
(split of data/maze.py, seed 0). paper1000 trains on all 1,000 and evaluates one
operating point declared before training, on the TestSet only.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import ast
import hashlib
import json
import os
import random
import runpy
import statistics
import sys
import time

ARMS = {"det": (0.0, 16), "sde": (0.5, 16), "sde32": (0.5, 32)}  # sfr_noise_scale, sfr_fixed_depth
SAMPLERS = ("euler_selfcond", "sde_selfcond", "held_t")
PROTOCOLS = ("dev900", "paper1000")
PAPER_POINT = {"arm": "det", "step": 100_000, "sampler": "sde_selfcond", "nfe": 257, "sampler_seed": 0}
EQR_REVISION = "bcb7d45c157c06a054b715933557d266f5bd9d7f"


@dataclass
class Config:
    root: Path  # Frozen campaign directory with plan.json and upstream source.
    action: str = "train"  # prepare, train, frontier, finish, or summarize.
    seed: int = 0  # Training seed; the data split is fixed separately.
    stage: str = "A"  # A: self-conditioning from scratch. B: FPF from the Stage-A parent.
    arm: str = "det"  # Stage-B rollout arm: det, sde or sde32.
    protocol: str = "dev900"  # dev900 (select on 100 held-out training mazes) or paper1000.
    smoke: bool = False  # Qualification run in a separate tree; never a result.
    max_steps: int | None = None  # Total updates in this stage; defaults to the plan.
    evaluation_step: int | None = None  # Frontier: Stage-B update, or 0 for the Stage-A parent.


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def upstream_mapping(maze_unique_py):
    """Read upstream's EqR->FMR token table without importing its package."""
    for node in ast.walk(ast.parse(Path(maze_unique_py).read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_EQR_TO_FMR" for t in node.targets):
            return [ast.literal_eval(v) for v in node.value.args[0].elts]
    raise ValueError("_EQR_TO_FMR not found")


def convert(inputs, labels, mapping):
    """EqR ids -> upstream (solutions, clue_mask), with an exact round trip."""
    import numpy as np
    mapping = np.asarray(mapping, dtype=np.uint8)
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    assert sorted(mapping.tolist()) == list(range(6))
    assert np.array_equal(inputs, np.where(labels == 5, 2, labels)), "Inputs must equal labels with the path hidden"
    solutions = mapping[labels]
    clue_mask = (inputs != 2).astype(np.uint8)
    inverse = np.argsort(mapping)
    assert np.array_equal(inverse[solutions], labels)
    assert np.array_equal(np.where(clue_mask.astype(bool), inverse[solutions], 2), inputs)
    assert set(np.unique(solutions[clue_mask == 0]).tolist()) <= {2, 3}
    return solutions, clue_mask


def protocol_indices(protocol, split_seed=0, validation_size=100):
    """Official-train row indices per split, from data/maze.py's split."""
    import numpy as np
    from data.maze import split_indices
    train, dev = split_indices(1000, validation_size, split_seed)
    if protocol == "dev900":
        return {"train": train, "dev": dev}
    assert protocol == "paper1000"
    full = np.sort(np.concatenate([train, dev]))
    assert np.array_equal(full, np.arange(1000))
    return {"train": full}


def lane_dir(root, protocol, seed, smoke):
    return Path(root) / ("qualification" if smoke else "runs") / protocol / f"s{seed}"


def load_split(root, protocol, split, lane, smoke=False):
    """Frozen arrays. The TestSet opens only after this lane's selection is frozen."""
    import numpy as np
    if split == "test":
        assert (Path(lane) / "selection.json").exists(), "TestSet is sealed until selection.json exists"
    path = Path(root) / "data" / protocol / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        solutions, clues = data["solutions"].astype(np.int64), data["clue_mask"].astype(np.int64)
    if smoke:
        solutions, clues = solutions[:8], clues[:8]
    return solutions, clues, path


def prepare(root, plan):
    import numpy as np
    sys.path.insert(0, plan["reasoning_dir"])
    from CONFIG import DataConfig
    from data.maze import _read, audit_dataset, dataset_directory

    config = DataConfig(cache_dir=Path(plan["reasoning_dir"]) / "outputs/data", revision=EQR_REVISION)
    directory = dataset_directory(config)
    official = {split: _read(directory, split) for split in ("train", "test")}
    audit = audit_dataset(config)
    mapping = upstream_mapping(root / "upstream/flow_map_reasoning/datasets/maze_unique.py")
    files, identities = {}, {}
    for protocol in PROTOCOLS:
        splits = {**protocol_indices(protocol), "test": np.arange(1000)}
        for split, indices in splits.items():
            source = "test" if split == "test" else "train"
            inputs, labels = (array[indices] for array in official[source])
            solutions, clue_mask = convert(inputs, labels, mapping)
            ids = np.asarray([f"{source}/{i}" for i in indices])
            path = root / "data" / protocol / f"{split}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, solutions=solutions, clue_mask=clue_mask, ids=ids, inputs_eqr=inputs, labels_eqr=labels)
            files[f"data/{protocol}/{split}.npz"] = {"sha256": digest(path), "count": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
            identities[protocol, split] = {row.tobytes() for row in inputs}
            assert len(identities[protocol, split]) == len(ids)
            if split == "train":
                # Upstream reads this file itself; same layout as download_maze_unique writes.
                cache = root / "cache" / protocol / "maze_unique/train.npz"
                cache.parent.mkdir(parents=True, exist_ok=True)
                path_lengths = (labels == 5).sum(axis=1) + 1
                assert 100 <= path_lengths.min() and path_lengths.max() <= 140
                np.savez_compressed(cache, solutions=solutions, clue_mask=clue_mask, path_lengths=path_lengths, source_revision=np.asarray(EQR_REVISION))
                files[f"cache/{protocol}/maze_unique/train.npz"] = {"sha256": digest(cache), "count": len(ids)}
        assert set(audit["splits"]["test"]) == {f"test/{i}" for i in splits["test"]}
    dev_ids = {f"train/{i}" for i in protocol_indices("dev900")["dev"]}
    train_ids = {f"train/{i}" for i in protocol_indices("dev900")["train"]}
    assert dev_ids == set(audit["splits"]["validation"]) and train_ids == set(audit["splits"]["train"])
    assert not identities["dev900", "train"] & identities["dev900", "dev"]
    for protocol in PROTOCOLS:
        assert not identities[protocol, "train"] & identities[protocol, "test"]
    assert identities["paper1000", "train"] == identities["dev900", "train"] | identities["dev900", "dev"]
    save(root / "data/manifest.json", {
        "files": files, "eqr_revision": EQR_REVISION, "eqr_to_fmr": mapping, "clue_rule": "inputs != 2 (walls, start, goal)",
        "split": "data/maze.py split_indices(1000, 100, seed 0)", "data_maze_sha256": digest(Path(plan["reasoning_dir"]) / "data/maze.py"),
        "official_sha256": audit["sha256"], "audit": audit["audit"], "exact_puzzle_disjointness": True,
    })


def profile(model, length=900, vocab=6):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[0]), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        z = torch.zeros((1, length, vocab), device="cuda")
        model(z, torch.full((1,), 0.5, device="cuda"), self_cond=z)
    model.train(training)
    return {"params": sum(p.numel() for p in model.parameters()), "flops_per_forward": int(counter.get_total_flops()), "convention": "FlopCounterMode, math SDPA, multiply-add=2; excludes uncounted elementwise operations."}


def upstream_eval(root):
    """Upstream maze metrics, loaded by path (the module is not a package)."""
    if "frm_maze_upstream_eval" not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location("frm_maze_upstream_eval", root / "upstream/experiments/maze_30x30/evaluate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["frm_maze_upstream_eval"]


def evaluate(model, lut, root, out, solutions, clues, data_path, *, sampler, nfe, sampler_seed=0, check_hidden=False, cost=None, meta=None):
    """Score one sampler operating point; hidden targets never influence predictions."""
    import numpy as np
    import torch
    from flow_map_reasoning import build_sampler_config, run_sampler

    nfe = int(nfe)
    recipe = root / "upstream/experiments/sudoku_extreme_train/samplers" / f"{sampler}.yaml"
    config, _ = build_sampler_config([str(recipe)], [f"sampler.num_samples={len(solutions)}", "sampler.batch_size=250"])
    if config.kind == "held_t":
        config.held_iters = nfe - 1
    else:
        config.steps = nfe - 1
    cost = cost or profile(model)
    calls = [0]
    handle = model.register_forward_pre_hook(lambda _module, args: calls.__setitem__(0, calls[0] + len(args[0])))
    training = model.training
    model.eval()
    numpy_state, python_state = np.random.get_state(), random.getstate()

    def seeded_run(targets):
        torch.manual_seed(sampler_seed)
        np.random.seed(sampler_seed)
        random.seed(sampler_seed)
        return run_sampler(model, targets, clues, config, lut_a2g=lut, vocab_size=6, device="cuda")

    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = seeded_run(solutions)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            measured_calls = calls[0]
            if check_hidden:
                changed = np.where(clues.astype(bool), solutions, 5 - solutions)  # swap empty/path on free cells
                assert np.array_equal(predictions, seeded_run(changed)), "Hidden targets affected predictions"
    finally:
        handle.remove()
        model.train(training)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    clue = clues.astype(bool)
    assert predictions.shape == solutions.shape
    assert np.array_equal(predictions[clue], solutions[clue]), "Clues changed"
    assert measured_calls == len(solutions) * nfe, (measured_calls, nfe)
    solved = np.all(predictions == solutions, axis=1)
    author = upstream_eval(root).maze_metrics(predictions, solutions, clues)
    assert author["exact_gold"] == float(solved.mean())
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", predictions=predictions.astype(np.uint8), solved=solved)
    result = dict(sampler=sampler, nfe=measured_calls // len(solved), sampler_seed=sampler_seed, solved=int(solved.sum()), count=len(solved), accuracy=float(solved.mean()),
                  seconds=seconds, hidden_target_check=check_hidden, sampler_config=asdict(config), data=str(data_path), data_sha256=digest(data_path),
                  predictions=str(out / "predictions.npz"), author_metrics=author, **cost, **(meta or {}))
    result["flops_per_puzzle"] = result["nfe"] * cost["flops_per_forward"]
    save(out / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("stage", "arm", "step", "split", "sampler", "nfe", "sampler_seed", "solved", "count", "seconds")}), flush=True)
    return result


def monitor_specs(protocol):
    specs = [("train", "euler_selfcond", 65)]
    if protocol == "dev900":
        specs += [("dev", "euler_selfcond", 65), ("dev", "sde_selfcond", 129)]
    return specs


def load_model(root, checkpoint, use_ema=True):
    sys.path.insert(0, str(root / "upstream/experiments/sudoku_extreme_train"))
    from evaluate import load_checkpoint_model
    return load_checkpoint_model(str(checkpoint), "cuda", use_ema=use_ema)


def write_parent(c, root, plan, lane):
    """Freeze the Stage-A EMA weights that every Stage-B arm of this lane starts from."""
    import torch
    stage = lane / "stageA"
    if c.protocol == "dev900":
        rows = [json.loads(p.read_text()) for p in stage.glob("monitor/*/dev-sde_selfcond129-seed0/result.json")]
        assert rows, "No Stage-A DevSet monitor results"
        chosen = min(rows, key=lambda r: (-r["solved"], r["step"]))
        checkpoint = stage / f"step_{chosen['step']:08d}.pt"
        rule = "Earliest Stage-A checkpoint with the most DevSet solves at sde_selfcond NFE 129 (sampler seed 0). EMA weights."
    else:
        chosen, checkpoint = None, stage / "final.pt"
        rule = "Pre-declared: final Stage-A EMA weights (no DevSet exists in paper1000)."
    model, _ = load_model(root, checkpoint)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = {"step": source["step"], "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": source["config"], "provenance": source["provenance"]}
    temporary = stage / "parent.pending"
    torch.save(payload, temporary)
    temporary.replace(stage / "parent.pt")
    save(stage / "parent.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "step": source["step"], "parent": str(stage / "parent.pt"), "parent_sha256": digest(stage / "parent.pt"), "rule": rule, "monitor": chosen})


def train(c, root, plan):
    import numpy as np
    import torch
    import wandb
    import flow_map_reasoning as fmr
    import flow_map_reasoning.trainer as trainer

    assert c.stage in ("A", "B") and c.arm in ARMS and c.protocol in PROTOCOLS
    assert c.protocol == "dev900" or c.stage == "A" or c.arm == PAPER_POINT["arm"]
    lane = lane_dir(root, c.protocol, c.seed, c.smoke)
    out = lane / ("stageA" if c.stage == "A" else f"{c.arm}/stageB")
    schedule = plan["smoke" if c.smoke else "production"]
    max_steps = c.max_steps or plan["stage_steps"][c.stage]
    if (out / "completed.json").exists() and json.loads((out / "completed.json").read_text())["step"] == max_steps:
        print(f"Already complete: {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    source = root / "upstream"
    assert Path(fmr.__file__).resolve().is_relative_to(source.resolve())
    keep_every, save_every = schedule["keep_every"], schedule["save_every"]
    original_save = trainer._save_checkpoint

    def atomic_checkpoint(path, step, *args, **kwargs):
        path = Path(path)
        temporary = path.with_suffix(".pending")
        original_save(temporary, step, *args, **kwargs)
        temporary.replace(path)
        if path.name.startswith("step_"):
            for old in path.parent.glob("step_*.pt"):
                if old != path and int(old.stem[5:]) % keep_every:
                    old.unlink()

    trainer._save_checkpoint = atomic_checkpoint
    original_init = wandb.init
    active_log = [None]
    arm_label = "sc" if c.stage == "A" else c.arm

    def init(*args, **kwargs):
        kwargs["config"] = dict(kwargs.get("config", {}), provenance=plan, resolved_recipe=resolved, local_output=str(out))
        kwargs["group"] = f"{plan['attempt']}-{c.protocol}{'-smoke' if c.smoke else ''}"
        kwargs["settings"] = wandb.Settings(init_timeout=60)
        run = original_init(*args, **kwargs)
        active_log[0] = run.log
        wandb.log = log  # wandb.init replaces module-level log; install the local journal afterward.
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
        cfg.val_every = max_steps if c.smoke else schedule[f"monitor_every_{c.stage}"]
        cfg.save_best_metric = "DevSet/sde_selfcond129/solved" if c.protocol == "dev900" else ""
        dataset = train_loader.dataset
        while not hasattr(dataset, "solutions"):
            dataset = dataset.dataset
        solutions, clues, _ = load_split(root, c.protocol, "train", lane)
        assert np.array_equal(dataset.solutions, solutions) and np.array_equal(dataset.clue_masks, clues)
        cost = profile(model)
        save(out / "model-cost.json", cost)
        lut = fmr.TorchLUT(fmr.build_luts(K=6)[0], device="cuda")
        data = {split: load_split(root, c.protocol, split, lane) for split in {s for s, _, _ in monitor_specs(c.protocol)}}

        def paired(core, step):
            metrics = {}
            for split, sampler, nfe in monitor_specs(c.protocol):
                result = evaluate(core, lut, root, out / "monitor" / f"{step:08d}" / f"{split}-{sampler}{nfe}-seed0", *data[split], sampler=sampler, nfe=nfe, cost=cost,
                                  meta=dict(stage=c.stage, arm=arm_label, seed=c.seed, step=step, split=split, protocol=c.protocol, smoke=c.smoke))
                label = {"train": "TrainSet", "dev": "DevSet"}[split]
                metrics.update({f"{label}/{sampler}{nfe}/{key}": result[key] for key in ("solved", "count", "accuracy", "nfe", "flops_per_puzzle", "seconds")})
            return metrics

        kwargs.update(val_fn=paired)
        return original_train(model, train_loader, cfg, **kwargs)

    fmr.train = recorded_train
    entry = source / "experiments/maze_30x30/2_train.py"
    recipes = entry.parent / "recipes"
    run_name = f"frm-maze/{arm_label}-{c.stage}-s{c.seed}-{c.protocol}-{plan['attempt']}{'-smoke' if c.smoke else ''}-j{os.environ.get('SLURM_JOB_ID', 'local')}"
    sys.argv = [str(entry), "--config", str(recipes / f"maze_8m_stage{c.stage}.yaml"), "--config", str(recipes / "datasets/maze_unique.yaml"),
                "--seed", str(c.seed), "--checkpoint_dir", str(out), "--wandb_run_name", run_name]
    checkpoints = sorted(out.glob("step_*.pt"))
    if checkpoints:
        sys.argv += ["--resume", str(checkpoints[-1])]
    elif c.stage == "B":
        parent = lane / "stageA/parent.pt"
        assert parent.exists(), "Stage A must finish and freeze its parent first"
        sys.argv += ["--init_from", str(parent)]
        save(out / "parent.json", {**json.loads((lane / "stageA/parent.json").read_text()), "loaded_sha256": digest(parent)})
    overrides = {
        "data.dataset_root": str(root / "cache" / c.protocol), "train.max_steps": max_steps, "train.save_every": save_every,
        "train.log_every": 16 if c.smoke else 100, "train.no_wandb": False, "train.wandb_project": "frm-reproduction", "val.maze_val": False,
    }
    if c.stage == "B":
        overrides["loss.sfr_noise_scale"], overrides["loss.sfr_fixed_depth"] = ARMS[c.arm]
    for key, value in overrides.items():
        sys.argv += ["--set", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
    namespace = runpy.run_path(str(entry))
    _, built = namespace["build_config"](namespace["parse_args"]())
    resolved = {key: asdict(value) for key, value in built.items()}
    expected = plan["recipe"]["common"] | plan["recipe"][c.stage] | ({"loss.sfr_noise_scale": ARMS[c.arm][0], "loss.sfr_fixed_depth": ARMS[c.arm][1]} if c.stage == "B" else {})
    for key, value in expected.items():
        section, field = key.split(".")
        assert resolved[section][field] == value, (key, resolved[section][field], value)
    resolved["adapter"] = {"protocol": c.protocol, "arm": arm_label, "save_every": save_every, "keep_every": keep_every, "max_steps": max_steps}
    save(out / "resolved.json", resolved)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "job": os.environ.get("SLURM_JOB_ID"), "argv": sys.argv}), flush=True)
    started = time.perf_counter()
    namespace["main"]()
    wandb.finish()
    if c.stage == "A":
        write_parent(c, root, plan, lane)
    save(out / "completed.json", {"step": max_steps, "seconds": time.perf_counter() - started, "checkpoint": str(out / "final.pt"), "sha256": digest(out / "final.pt")})


def grid(plan, smoke):
    g = plan["smoke" if smoke else "production"]
    return [(s, n, k) for s in SAMPLERS for n in g["frontier_nfe"] for k in g["sampler_seeds"]]


def checkpoint_for(lane, arm, step):
    return lane / "stageA/parent.pt" if step == 0 else lane / arm / "stageB" / f"step_{step:08d}.pt"


def frontier(c, root, plan):
    """DevSet grid for one checkpoint (step 0 = the frozen Stage-A parent)."""
    import torch
    import wandb
    assert c.protocol == "dev900" and c.evaluation_step is not None
    steps = plan["smoke"]["evaluation_steps"] if c.smoke else plan["production"]["evaluation_steps"]
    assert c.evaluation_step == 0 or c.evaluation_step in steps
    lane = lane_dir(root, c.protocol, c.seed, c.smoke)
    arm = "parent" if c.evaluation_step == 0 else c.arm
    checkpoint = checkpoint_for(lane, c.arm, c.evaluation_step)
    directory = lane / "frontier" / arm / f"step{c.evaluation_step}"
    solutions, clues, path = load_split(root, c.protocol, "dev", lane, c.smoke)
    model = None
    results = []
    for sampler, nfe, seed in grid(plan, c.smoke):
        result_path = directory / f"{sampler}-nfe{nfe}-seed{seed}/result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result["checkpoint_sha256"] == digest(checkpoint) and result["data_sha256"] == digest(path)
        else:
            if model is None:
                model, meta = load_model(root, checkpoint)
                cost = profile(model)
            result = evaluate(model, meta["lut_a2g"], root, result_path.parent, solutions, clues, path, sampler=sampler, nfe=nfe, sampler_seed=seed, check_hidden=c.smoke, cost=cost,
                              meta=dict(stage="A" if arm == "parent" else "B", arm=arm, seed=c.seed, step=c.evaluation_step, split="dev", protocol=c.protocol, smoke=c.smoke,
                                        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint)))
        results.append(result)
    save(directory / "complete.json", results)
    run = wandb.init(project="frm-reproduction", group=f"{plan['attempt']}-{c.protocol}{'-smoke' if c.smoke else ''}", config=plan,
                     name=f"frm-maze/{arm}-frontier-s{c.seed}-u{c.evaluation_step}-{plan['attempt']}{'-smoke' if c.smoke else ''}")
    for r in results:
        run.log({f"dev/{r['sampler']}/nfe{r['nfe']:g}/seed{r['sampler_seed']}/{k}": r[k] for k in ("solved", "count", "flops_per_puzzle", "seconds")})
    run.finish()
    del model
    torch.cuda.empty_cache()


def pool(results):
    """Sum each (arm, step, sampler, NFE) over sampler seeds."""
    groups = {}
    for r in results:
        groups.setdefault((r["arm"], r["step"], r["sampler"], r["nfe"]), []).append(r)
    rows = []
    for (arm, step, sampler, nfe), members in groups.items():
        members.sort(key=lambda r: r["sampler_seed"])
        rows.append(dict(arm=arm, step=step, sampler=sampler, nfe=nfe, solved=sum(r["solved"] for r in members), count=sum(r["count"] for r in members),
                         per_seed_solved=[r["solved"] for r in members], sampler_seeds=[r["sampler_seed"] for r in members], flops_per_puzzle=members[0]["flops_per_puzzle"],
                         checkpoint=members[0]["checkpoint"], checkpoint_sha256=members[0]["checkpoint_sha256"]))
    return rows


def select(rows):
    """Most pooled DevSet solves, then fewer FLOPs per puzzle, then earlier update (then fixed arm/sampler order)."""
    arms, samplers = list(ARMS), list(SAMPLERS)
    return min(rows, key=lambda r: (-r["solved"], r["flops_per_puzzle"], r["step"], arms.index(r["arm"]), samplers.index(r["sampler"])))


def score(root, plan, lane, c, point, split, sampler_seed, tag):
    model, meta = load_model(root, point["checkpoint"])
    solutions, clues, path = load_split(root, c.protocol, split, lane, c.smoke)
    result = evaluate(model, meta["lut_a2g"], root, lane / "scored" / tag / f"{split}-seed{sampler_seed}", solutions, clues, path, sampler=point["sampler"], nfe=point["nfe"], sampler_seed=sampler_seed,
                      meta=dict(arm=point["arm"], step=point["step"], split=split, seed=c.seed, protocol=c.protocol, smoke=c.smoke, checkpoint=str(point["checkpoint"]), checkpoint_sha256=digest(point["checkpoint"])))
    return result


def finish(c, root, plan):
    import wandb
    lane = lane_dir(root, c.protocol, c.seed, c.smoke)
    parent = {"arm": "parent", "step": 0, "checkpoint": str(lane / "stageA/parent.pt")}
    if c.protocol == "dev900":
        steps = plan["smoke" if c.smoke else "production"]["evaluation_steps"]
        arms = plan["smoke"]["arms"] if c.smoke else list(ARMS)
        expected = len(grid(plan, c.smoke))
        results = []
        for arm in arms:
            for step in steps:
                complete = json.loads((lane / "frontier" / arm / f"step{step}/complete.json").read_text())
                assert len(complete) == expected
                results += complete
        rows = pool(results)
        chosen = select(rows)
        per_arm = {arm: select([r for r in rows if r["arm"] == arm]) for arm in arms}
        parent_rows = pool(json.loads((lane / "frontier/parent/step0/complete.json").read_text()))
        save(lane / "selection.json", {"selected": chosen, "per_arm": per_arm, "candidates": rows, "parent_dev": parent_rows,
                                       "rule": "Most DevSet solves pooled over sampler seeds, then fewer FLOPs/puzzle, then earlier update, then arm det<sde<sde32, then sampler order. TestSet unopened by selection."})
    else:
        stageB = lane / PAPER_POINT["arm"] / "stageB"
        step = plan["smoke"]["paper_step"] if c.smoke else PAPER_POINT["step"]
        chosen = {**PAPER_POINT, "step": step, "checkpoint": str(stageB / f"step_{step:08d}.pt")}
        per_arm = {}
        save(lane / "selection.json", {"selected": chosen, "rule": "Pre-declared before training: final Stage-A EMA parent, det arm, final Stage-B update, sde_selfcond NFE 257, sampler seed 0. TestSet only; no selection."})
    report = {"selected": chosen, "protocol": c.protocol}
    report["test"] = score(root, plan, lane, c, chosen, "test", 0, "selected")
    report["train"] = score(root, plan, lane, c, chosen, "train", 0, "selected")
    if c.protocol == "dev900":
        report["dev"] = chosen
        report["test_other_sampler_seeds"] = [score(root, plan, lane, c, chosen, "test", k, "selected") for k in (1, 2)]
        report["per_arm"] = {arm: {"dev": point, "test": score(root, plan, lane, c, point, "test", 0, f"arm-{arm}"), "train": score(root, plan, lane, c, point, "train", 0, f"arm-{arm}")} for arm, point in per_arm.items()}
    matched = {**parent, "sampler": chosen["sampler"], "nfe": chosen["nfe"]}
    report["parent_test_matched_sampler"] = score(root, plan, lane, c, matched, "test", 0, "parent")
    save(lane / "result.json", report)
    run = wandb.init(project="frm-reproduction", group=f"{plan['attempt']}-{c.protocol}{'-smoke' if c.smoke else ''}", config=plan,
                     name=f"frm-maze/{chosen['arm']}-finish-s{c.seed}-{c.protocol}-{plan['attempt']}{'-smoke' if c.smoke else ''}")
    run.log({f"{split}/{k}": report[split][k] for split in ("train", "test") for k in ("solved", "count", "accuracy", "nfe", "flops_per_puzzle")}
            | {"parent/test/solved": report["parent_test_matched_sampler"]["solved"]})
    run.finish()


def summarize(root, plan):
    groups = {}
    for protocol in PROTOCOLS:
        lanes = {seed: json.loads((lane_dir(root, protocol, seed, False) / "result.json").read_text()) for seed in plan["seeds"]}
        scores = [lane["test"]["accuracy"] * 100 for lane in lanes.values()]
        group = {"lanes": lanes, "test_mean_percent": statistics.mean(scores), "test_sample_sd_percent": statistics.stdev(scores), "test_peak_percent": max(scores),
                 "stage_b_beats_parent_all_seeds": all(l["test"]["solved"] > l["parent_test_matched_sampler"]["solved"] for l in lanes.values())}
        groups[protocol] = group
    dev = groups["dev900"]
    mean = dev["test_mean_percent"]
    arm_tests = {arm: [dev["lanes"][s]["per_arm"][arm]["test"]["solved"] for s in plan["seeds"]] for arm in ARMS}
    summary = {
        "groups": groups, "paper_mean_percent": 98.0, "paper_sd_percent": 1.8, "paper_peak_percent": 99.9,
        "FRM-1": "supports" if mean >= 97.0 else ("refutes" if mean < 96.2 else "inconclusive"),
        "FRM-2": {"sde_beats_det_all_seeds": all(s > d for s, d in zip(arm_tests["sde"], arm_tests["det"])),
                  "sde32_beats_det_all_seeds": all(s > d for s, d in zip(arm_tests["sde32"], arm_tests["det"])), "per_arm_test_solved": arm_tests},
        "FRM-3": {p: groups[p]["stage_b_beats_parent_all_seeds"] for p in PROTOCOLS},
    }
    save(root / "summary.json", summary)
    lines = ["# FRM Maze-Unique results", "", "| Protocol | Seed | Arm | Stage B update | Sampler | NFE | TrainSet | DevSet (pooled) | TestSet | Parent TestSet | FLOPs/puzzle |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for protocol, group in groups.items():
        for seed, lane in group["lanes"].items():
            s, dev_cell = lane["selected"], (f"{lane['dev']['solved']}/{lane['dev']['count']}" if "dev" in lane else "none")
            lines.append(f"| {protocol} | {seed} | {s['arm']} | {s['step']} | {s['sampler']} | {s['nfe']:g} | {lane['train']['solved']}/{lane['train']['count']} | {dev_cell} | "
                         f"{lane['test']['solved']}/{lane['test']['count']} | {lane['parent_test_matched_sampler']['solved']}/1000 | {lane['test']['flops_per_puzzle']:.4g} |")
    lines += ["", f"dev900 test mean {mean:.2f}% (SD {dev['test_sample_sd_percent']:.2f}); paper1000 test mean {groups['paper1000']['test_mean_percent']:.2f}%. Paper: 98.0 ± 1.8%, peak 99.9%.",
              f"FRM-1 {summary['FRM-1']}; FRM-2 {summary['FRM-2']}; FRM-3 {summary['FRM-3']}."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, default=str)[:4000], flush=True)


def qualify(root, plan):
    """Check the qualification tree and turn its timings into a production estimate."""
    import torch
    report = {}
    q = root / "qualification"
    for name, directory, steps in [("A-dev900", q / "dev900/s0/stageA", 256), ("A-paper1000", q / "paper1000/s0/stageA", 64)] + \
            [(f"B-{arm}", q / f"dev900/s0/{arm}/stageB", 64) for arm in ARMS] + [("B-det-paper1000", q / "paper1000/s0/det/stageB", 32)]:
        final = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
        assert final["step"] == steps and max(float(s["step"]) for s in final["optimizer"]["state"].values()) == steps
        assert all(torch.isfinite(v).all() for v in final["model"].values() if v.is_floating_point())
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        losses = [r["train/loss"] for r in rows if "train/loss" in r]
        timings = [r["train/update_seconds"] for r in rows if "train/update_seconds" in r and r["step"] > 16]
        assert losses and all(map(lambda x: x == x and abs(x) < 1e6, losses)) and timings
        monitors = [json.loads(p.read_text()) for p in directory.glob("monitor/*/*/result.json")]
        report[name] = {"updates": final["step"], "median_update_seconds": statistics.median(timings), "peak_allocated_gb": max(r.get("train/vram_alloc_gb", 0) for r in rows),
                        "last_loss": losses[-1], "monitor_seconds_per_point": sum(m["seconds"] for m in monitors) / max(1, len({m["step"] for m in monitors})),
                        "monitor": {f"{m['split']}-{m['sampler']}{m['nfe']:g}@{m['step']}": f"{m['solved']}/{m['count']}" for m in monitors}}
        del final
    stageA = q / "dev900/s0/stageA"
    kept = sorted(int(p.stem[5:]) for p in stageA.glob("step_*.pt"))
    assert kept == [64, 128, 192, 256], kept
    assert torch.load(stageA / "final.pt", map_location="cpu", weights_only=False)["config"]["resume"].endswith("step_00000128.pt")
    assert json.loads((stageA / "parent.json").read_text())["step"] in (128, 256)
    frontier = [json.loads(p.read_text()) for p in (q / "dev900/s0/frontier").glob("*/*/*/result.json")]
    assert frontier and all(r["hidden_target_check"] and r["count"] == 8 for r in frontier)
    results = {p: json.loads((q / p / "s0/result.json").read_text()) for p in PROTOCOLS}
    cost = results["dev900"]["test"]
    report["model"] = {"params": cost["params"], "flops_per_forward": cost["flops_per_forward"]}
    report["selected"] = {p: {k: r["selected"][k] for k in ("arm", "step", "sampler", "nfe")} | {"train": f"{r['train']['solved']}/{r['train']['count']}", "test": f"{r['test']['solved']}/{r['test']['count']}"} for p, r in results.items()}
    # Frontier throughput: puzzle-forwards per second over the smoke grid.
    report["eval_puzzle_forwards_per_second"] = sum(r["nfe"] * r["count"] for r in frontier) / sum(r["seconds"] for r in frontier)
    monitor_A = report["A-dev900"]["monitor_seconds_per_point"]
    monitor_B = report["B-det"]["monitor_seconds_per_point"]
    p = plan["production"]
    hours = {"A": (plan["stage_steps"]["A"] * report["A-dev900"]["median_update_seconds"] + plan["stage_steps"]["A"] / p["monitor_every_A"] * monitor_A) / 3600}
    for arm in ARMS:
        hours[f"B-{arm}"] = (plan["stage_steps"]["B"] * report[f"B-{arm}"]["median_update_seconds"] + plan["stage_steps"]["B"] / p["monitor_every_B"] * monitor_B) / 3600
    forwards = 100 * sum(n for _, n, _ in grid(plan, False))
    hours["frontier_per_checkpoint"] = forwards / report["eval_puzzle_forwards_per_second"] / 3600
    report["estimated_hours"] = hours
    report["passed"] = True
    save(root / "qualification-report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main(c):
    root = c.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    assert digest(__file__) == plan["harness_sha256"], "Frozen adapter changed"
    for relative, expected in json.loads((root / "source-files.json").read_text()).items():
        assert digest(root / "upstream" / relative) == expected, f"Upstream source changed: {relative}"
    if c.action != "prepare" and (root / "data/manifest.json").exists():
        for name, record in json.loads((root / "data/manifest.json").read_text())["files"].items():
            assert digest(root / name) == record["sha256"], f"Frozen data changed: {name}"
    if c.action == "prepare":
        return prepare(root, plan)
    if c.action == "summarize":
        return qualify(root, plan) if c.smoke else summarize(root, plan)
    import torch
    assert torch.cuda.is_available(), "Run inside the requested Slurm GPU allocation"
    {"train": train, "frontier": frontier, "finish": finish}[c.action](c, root, plan)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
