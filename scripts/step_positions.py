"""Compare boundary, interior and pooled states on saved StepCluster responses."""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

from pipeline import load_lm, marker_positions, read_json, sha, write_json
from token_alignment import original_token_offsets


SPLITS = ("train", "validation", "test")
VIEWS = ("boundary", "body25", "body50", "body75", "body_mean", "nearby_control")
HEADER = re.compile(r"(?im)^[ \t]*(?:\*\*)?Step[ \t]+(\d+)[ \t]*:(?:\*\*)?")


def selections(text, offsets, prompt_length, min_body):
    end = text.find("####")
    positions, numbers = marker_positions(text, offsets, prompt_length)
    headers = list(HEADER.finditer(text[:end]))
    if not numbers or [int(m[1]) for m in headers] != numbers:
        raise ValueError("Step headers disagree with Experiment A")
    bodies = []
    for i, header in enumerate(headers):
        stop = headers[i + 1].start() if i + 1 < len(headers) else end
        bodies.append([
            prompt_length + t for t, (a, b) in enumerate(offsets)
            if header.end() <= a < b <= stop and text[a:b].strip()
        ])
    result = []
    for i in range(1, len(numbers)):
        body = bodies[i]
        boundary = positions[i] - 1
        # A nearby control from the previous step's prose, safely away from
        # the boundary and from that step's own header and first few tokens.
        controls = [p for p in bodies[i - 1][2:] if boundary - p >= 4]
        if len(body) < min_body or not controls:
            continue
        control = controls[-1]
        samples = [[boundary], *[[body[round(q * (len(body) - 1))]] for q in (.25, .5, .75)],
                   body, [control]]
        result.append((numbers[i], samples))
    return result


def source_split(root, split):
    cell = root / split
    gen = read_json(cell / "generation.json")
    audit = read_json(cell / "extraction.json")
    if not gen["complete"] or gen != audit["source_generation"] or audit["state"] != "author-hidden":
        raise ValueError(f"{cell}: expected completed Experiment A author-hidden extraction")
    files = {int(p.stem.rsplit("_", 1)[-1]): p for p in (cell / "merged").glob("gsm8k_*.json")}
    ids = audit["included"]
    if len(files) != gen["n"] or len(ids) != len(set(ids)) or not set(ids) <= files.keys():
        raise ValueError(f"{cell}: incomplete or inconsistent source rows")
    return gen, [(i, files[i]) for i in sorted(ids)]


def cpu_states(model, ids, positions, layers):
    import torch

    capture = {}
    last_layer = len(model.model.layers) - 1
    indices = torch.tensor(positions, device=model.device)

    def hook(layer):
        def save(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            values = hidden[0, indices]
            if layer == last_layer:
                values = model.model.norm(values)
            capture[layer] = values.float().cpu().numpy()
        return save

    handles = [model.model.layers[layer].register_forward_hook(hook(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            input_ids = torch.tensor([ids[:positions[-1] + 1]], device=model.device)
            model.model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return np.stack([capture[layer] for layer in layers])


def extract(args):
    from tqdm import tqdm

    sources = {split: source_split(args.source, split) for split in SPLITS}
    model_path = Path(args.model or sources["train"][0]["config"]["model"]).resolve()
    model_hash = sha(model_path / "config.json")
    model_configs = [manifest["config"] for manifest, _ in sources.values()]
    for config in model_configs:
        if (model_hash != config["model_config"]
                or model_path.name != Path(config["model"]).name):
            raise ValueError("Use the same model snapshot as Experiment A")
        for key in ("adapter", "prompt", "stop_mode", "max_new_tokens"):
            if config.get(key) != model_configs[0].get(key):
                raise ValueError(f"Inconsistent source generation setting: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    for split, (manifest, rows) in sources.items():
        output = args.output / f"{split}.npz"
        audit_file = args.output / f"{split}-audit.json"
        settings = {
            "source": str(args.source.resolve()), "generation": manifest, "model": str(model_path),
            "layers": args.layers, "min_body": args.min_body, "limit": args.limit,
            "sources": [[qid, sha(path)] for qid, path in rows],
            "code": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("pipeline.py"),
                                             Path(__file__).with_name("token_alignment.py"))},
        }
        if output.exists() and audit_file.exists():
            audit = read_json(audit_file)
            if audit["settings"] != settings or audit["npz_sha256"] != sha(output):
                raise ValueError(f"{output}: provenance mismatch. Use a new output directory.")
            print(f"Reusing {output}", flush=True)
            continue
        if model is None:
            model, tokenizer = load_lm(str(model_path))
            if max(args.layers) >= len(model.model.layers):
                raise ValueError("Layer index outside model")
        vectors, metadata, empty = [], [], []
        selected_rows = rows[:args.limit] if args.limit else rows
        for qid, path in tqdm(selected_rows, desc=f"position extraction/{split}"):
            row = read_json(path)
            ids, prompt = row["full_seq_ids"], row["input_ids"]
            prefix = len(prompt)
            if not prefix or ids[:prefix] != prompt or row["truncated"]:
                raise ValueError(f"Unexpected source trajectory: {path}")
            text, offsets, _ = original_token_offsets(tokenizer, ids[prefix:])
            if text != row["decoded_with_specials"]:
                raise ValueError(f"Saved token decoding differs: {path}")
            samples = selections(text, offsets, prefix, args.min_body)
            if not samples:
                empty.append(qid)
                continue
            positions = sorted({p for _, groups in samples for group in groups for p in group})
            index = {p: i for i, p in enumerate(positions)}
            states = cpu_states(model, ids, positions, args.layers)
            question = " ".join(row["question"].split())
            for step, groups in samples:
                pooled = [states[:, [index[p] for p in group]].mean(axis=1) for group in groups]
                vectors.append(np.stack(pooled).astype(np.float16))
                metadata.append({
                    "question": question, "question_id": qid, "step": step,
                    "positions": groups, "token_ids": [ids[g[0]] if len(g) == 1 else -1 for g in groups],
                    "prompt_tokens": prefix, "generated_tokens": len(offsets),
                })
        if not metadata:
            raise ValueError(f"{split}: no eligible steps")
        temp = output.with_suffix(".tmp.npz")
        np.savez(temp, features=np.stack(vectors), layers=args.layers, views=VIEWS,
                 questions=[r["question"] for r in metadata], steps=[r["step"] for r in metadata],
                 token_ids=[r["token_ids"] for r in metadata],
                 token_positions=[[np.mean(g) - r["prompt_tokens"] + 1 for g in r["positions"]]
                                  for r in metadata],
                 progress=[[((np.mean(g) - r["prompt_tokens"] + 1) / r["generated_tokens"])
                            for g in r["positions"]] for r in metadata])
        temp.replace(output)
        write_json(audit_file, {"settings": settings, "npz_sha256": sha(output),
                               "questions_sampled": len(selected_rows), "questions_without_rows": empty,
                               "samples": metadata})
        print(f"{split}: {len(metadata)} matched steps from {len(selected_rows) - len(empty)} questions", flush=True)


def analyze(args):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score

    data = {}
    rows, confusions = [], []
    for split in SPLITS:
        path = args.output / f"{split}.npz"
        audit = read_json(args.output / f"{split}-audit.json")
        if sha(path) != audit["npz_sha256"]:
            raise ValueError(f"Integrity check failed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            data[split] = {key: archive[key] for key in archive.files}
        if tuple(data[split]["views"]) != VIEWS or not np.array_equal(data[split]["layers"], data["train"]["layers"]):
            raise ValueError("Inconsistent feature schema")
        for earlier in data:
            if earlier != split and set(data[earlier]["questions"]) & set(data[split]["questions"]):
                raise ValueError(f"Question overlap: {earlier}/{split}")
    labels = np.arange(2, args.max_step + 1)
    for split, d in data.items():
        keep = np.isin(d["steps"], labels)
        data[split] = {k: v[keep] if k not in ("layers", "views") else v for k, v in d.items()}
        if set(data[split]["steps"]) != set(labels):
            raise ValueError(f"{split}: missing stage class, lower --max-step or use more questions")
    train, val, test = (data[s] for s in SPLITS)

    def fit_predict(task, layer, fit_view, eval_views, x_train, x_val, x_tests, y_train, y_val, y_tests, classes):
        best = None
        for c in (.01, .1, 1.):
            model = LogisticRegression(C=c, max_iter=1000).fit(x_train, y_train)
            score = f1_score(y_val, model.predict(x_val), labels=classes, average="macro", zero_division=0)
            if best is None or score > best[0]:
                best = score, c, model
        score, c, model = best
        for view, x, y in zip(eval_views, x_tests, y_tests):
            prediction = model.predict(x)
            row = dict(task=task, layer=layer, train_view=fit_view, test_view=view, C=c,
                       validation_macro_f1=score, n=len(y), accuracy=accuracy_score(y, prediction),
                       macro_f1=f1_score(y, prediction, labels=classes, average="macro", zero_division=0))
            if task == "boundary_detection":
                row["boundary_ap"] = average_precision_score(y, model.predict_proba(x)[:, 1])
            rows.append(row)
            confusions.append({**row, "labels": classes.tolist(),
                               "matrix": confusion_matrix(y, prediction, labels=classes).tolist()})

    def hidden(d, view, layer):
        return d["features"][:, view, layer].astype(np.float32)

    def surface(d, view):
        return [{"progress": float(x), "progress_sq": float(x * x),
                 "log_position": float(np.log1p(p)), "token_id": str(t)}
                for x, p, t in zip(d["progress"][:, view], d["token_positions"][:, view], d["token_ids"][:, view])]

    y_train, y_val, y_test = (d["steps"] for d in (train, val, test))
    for view, name in enumerate(VIEWS[:-1]):
        encoder = DictVectorizer().fit(surface(train, view))
        fit_predict("stage", "surface", name, [name], encoder.transform(surface(train, view)),
                    encoder.transform(surface(val, view)), [encoder.transform(surface(test, view))],
                    y_train, y_val, [y_test], labels)
    control = VIEWS.index("nearby_control")
    binary = [np.repeat([1, 0], len(d["steps"])) for d in (train, val, test)]
    pair_surface = [surface(d, 0) + surface(d, control) for d in (train, val, test)]
    encoder = DictVectorizer().fit(pair_surface[0])
    fit_predict("boundary_detection", "surface", "boundary_vs_control", ["boundary_vs_control"],
                encoder.transform(pair_surface[0]), encoder.transform(pair_surface[1]),
                [encoder.transform(pair_surface[2])], *binary[:2], binary[2:], np.array([0, 1]))
    for layer_index, layer in enumerate(train["layers"]):
        for view, name in enumerate(VIEWS[:-1]):
            checks = list(range(len(VIEWS)-1)) if view == 0 else [view]
            fit_predict("stage", int(layer), name, [VIEWS[i] for i in checks],
                        hidden(train, view, layer_index), hidden(val, view, layer_index),
                        [hidden(test, i, layer_index) for i in checks], y_train, y_val, [y_test]*len(checks), labels)
        pairs = [np.concatenate([hidden(d, 0, layer_index), hidden(d, control, layer_index)])
                 for d in (train, val, test)]
        fit_predict("boundary_detection", int(layer), "boundary_vs_control", ["boundary_vs_control"],
                    *pairs[:2], pairs[2:], *binary[:2], binary[2:], np.array([0, 1]))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = args.output / f"probes-step2-{args.max_step}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(path.with_suffix(".json"), {
        "counts": {s: {int(k): int(sum(d["steps"] == k)) for k in labels} for s, d in data.items()},
        "source_npz": {s: sha(args.output / f"{s}.npz") for s in SPLITS},
        "results": confusions,
    })
    print(f"POSITION PROBES COMPLETE: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "analyze", "all"))
    parser.add_argument("--source", type=Path, help="Experiment A GSM8K directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--layers", type=lambda x: [int(i) for i in x.split(",")], default=[0, 9, 18, 27])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-body", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 0 or args.min_body < 5 or args.max_step < 3 or min(args.layers) < 0 or len(set(args.layers)) != len(args.layers):
        parser.error("Invalid sample or layer settings")
    if args.phase in ("extract", "all"):
        if args.source is None:
            parser.error("--source is required for extraction")
        source, output = args.source.resolve(), args.output.resolve()
        if source == output or output in source.parents or any(source/s in output.parents or source/s == output for s in SPLITS):
            parser.error("Use an independent output directory, outside the existing split directories")
        extract(args)
    if args.phase in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
