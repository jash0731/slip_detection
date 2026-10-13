import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import ct_lif_v3.core as core
from ct_lif_v4.core import features, load_model
from ct_lif_study.run import load_data, windows, verify_splits, detailed
from ct_lif_study.reference.ct_lif import sigmoid


def file_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob("*") if p.is_file()
    }


def trial_weights(y, objects, groups):
    weights = np.zeros(len(y))
    for obj in np.unique(objects):
        for label in np.unique(y[objects == obj]):
            cell = (objects == obj) & (y == label)
            trials = np.unique(groups[cell])
            for group in trials:
                ids = cell & (groups == group)
                weights[ids] = 1 / (len(trials) * ids.sum())
    return weights / weights.mean()


def modified_trainer():
    source = inspect.getsource(core.fit)
    replacements = [
        ("def fit(", "def fit_modified("),
        ("rate, seed):", "rate, seed, train_weights, penalty):"),
        ("weights = balanced_weights(y, objects)", "weights = train_weights.copy()"),
        ("            norm = np.sqrt(",
         "            for k in ['W', 'w']:\n"
         "                grads[k] += penalty * model.params[k]\n"
         "            norm = np.sqrt("),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Training source changed; review the experiment adapter")
        source = source.replace(old, new)
    namespace = dict(core.__dict__)
    exec(compile(source, "<in-memory training adapter>", "exec"), namespace)
    return namespace["fit_modified"]


def selection_key(entry):
    point = entry["point"]
    return (-point["balanced_accuracy"], point["false_alarm_rate"],
            entry["parameters"], entry["name"])


def main():
    before = file_hashes()
    train = modified_trainer()
    data, metadata = load_data()
    results = ROOT / "ct_lif_v4/results"
    saved = json.loads((results / "all_choices_before_test.json").read_text())["folds"]
    frozen = []
    fits = 0
    for fold in metadata["folds"]:
        held, seed, splits = verify_splits(data, fold)
        old = next(f for f in saved if f["held"] == held and f["seed"] == seed)
        choice = next(c for c in old["selected"] if c["variant"] == "ct_only")
        baseline, scale, _ = load_model(results / (choice["name"] + ".npz"))
        hidden = len(baseline.params["b"])
        tr, va = windows(data, splits["train"]), windows(data, splits["validation"])
        objects = np.array([r["object"] for r in tr[2]])
        val_objects = np.array([r["object"] for r in va[2]])
        groups = np.array([r["group"] for r in tr[2]])
        x = features(tr[0], scale, "ct_only", False)
        xv = features(va[0], scale, "ct_only", False)
        normal = core.balanced_weights(tr[1], objects)
        balanced_trials = trial_weights(tr[1], objects, groups)
        for obj in np.unique(objects):
            for label in np.unique(tr[1][objects == obj]):
                ids = (objects == obj) & (tr[1] == label)
                assert np.isclose(balanced_trials[ids].sum(), normal[ids].sum())
        ref = core.validation_metrics(
            va[1], sigmoid(baseline.forward(core.compress(xv))),
            val_objects, choice["threshold"])
        base = dict(name="baseline", model=baseline, point=ref,
                    parameters=choice["parameters"])
        joint, balanced, families = [base], [base], {}
        regimes = [
            ("trial_balanced", balanced_trials, 0.),
            ("regularized", normal, .001),
            ("trial_regularized", balanced_trials, .001),
        ]
        for family, weights, penalty in regimes:
            candidates = []
            for pooling in ["uniform", "recent"]:
                for rate in [.005, .01]:
                    model = core.ResidualLIF(12, hidden, False, pooling, seed)
                    train(model, x, tr[1], objects, xv, va[1], val_objects,
                          rate, seed, weights, penalty)
                    points = core.operating_points(
                        va[1], sigmoid(model.forward(core.compress(xv))), val_objects, ref)
                    entry = dict(
                        name=f"{family}_{pooling}_h{hidden}_lr{rate}", model=model,
                        parameters=sum(v.size for v in model.params.values()))
                    assert entry["parameters"] == base["parameters"]
                    candidate = dict(entry, point=points["balanced"])
                    candidates.append(candidate)
                    balanced.append(candidate)
                    if points["joint"]["validation_joint_improvement"]:
                        joint.append(dict(entry, point=points["joint"]))
                    fits += 1
            families[family] = min(candidates, key=selection_key)
        selected = dict(baseline=base, joint=min(joint, key=selection_key),
                        balanced=min(balanced, key=selection_key), **families)
        frozen.append((held, seed, splits, scale, selected))
        print("TRAIN", fits, "/72", held, seed,
              {k: v["name"] for k, v in selected.items()}, flush=True)
    print("ALL CHOICES FROZEN BEFORE TEST INFERENCE", flush=True)
    rows = {}
    for held, seed, splits, scale, selected in frozen:
        raw, y, _ = windows(data, splits["test"])
        xt = features(raw, scale, "ct_only")
        for method, entry in selected.items():
            p = sigmoid(entry["model"].forward(xt))
            row = detailed(y, p, entry["point"]["threshold"])
            row.update(held=held, seed=seed, parameters=entry["parameters"],
                       name=entry["name"])
            rows.setdefault(method, []).append(row)
    for method, entries in rows.items():
        summary = {
            k: float(np.mean([r[k] for r in entries]))
            for k in ["balanced_accuracy", "slip_recall", "false_alarm_rate", "parameters"]
        }
        summary["accuracy"] = (
            sum(r["tp"] + r["tn"] for r in entries) /
            sum(r["tp"] + r["tn"] + r["fp"] + r["fn"] for r in entries))
        summary["input_MACs"] = float(np.mean([
            (r["parameters"] - 1) / 14 * 12 * 32 for r in entries]))
        print("RESULT", method, json.dumps(summary), flush=True)
    print("PRIMARY_FOLDS", json.dumps(rows["joint"]), flush=True)
    after = file_hashes()
    print("FILES_UNCHANGED", before == after, "COUNT", len(after), flush=True)
    assert before == after


if __name__ == "__main__":
    main()
