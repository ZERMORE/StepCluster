"""Server adapter for the official reasoning-trajectory implementation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def prepare(args):
    """Prefer the existing datasets cache, or explicitly supplied local data."""
    import random
    from datasets import load_dataset
    target = args.root / "datasets" / args.task
    target.mkdir(parents=True, exist_ok=True)
    config = {"task":args.task,"revision":args.dataset_revision,"local_source":args.data_jsonl,"seed":args.seed}
    if (target / "manifest.json").exists():
        existing = read_json(target / "manifest.json")
        if any(existing.get(k) != v for k,v in config.items()):
            raise ValueError("Dataset preparation settings differ; use a new root")
        for split, info in existing["files"].items():
            if sha(target / (split + ".jsonl")) != info["sha256"]:
                raise ValueError("Prepared dataset changed")
        print("Using prepared dataset", target)
        return
    splits = ["train", "test"] if args.task == "gsm8k" else ["test"]
    data = {}
    for split in splits:
        if args.data_jsonl:
            source = Path(args.data_jsonl.format(split=split))
            raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            dataset_id = "openai/gsm8k" if args.task == "gsm8k" else "HuggingFaceH4/MATH-500"
            extra = {"name": "main"} if args.task == "gsm8k" else {}
            raw = list(load_dataset(dataset_id, split=split, revision=args.dataset_revision, **extra))
        converted = []
        for i, row in enumerate(raw):
            question = row.get("question", row.get("problem"))
            answer = row.get("answer", row.get("solution", row.get("gold_answer")))
            if not question or answer is None:
                raise ValueError(f"Missing question or reference at {split}:{i}")
            converted.append({"id": i, "question": question, "answer": answer,
                              "source_id": row.get("id", row.get("unique_id", f"{split}:{i}"))})
        if split == "train":
            random.Random(args.seed).shuffle(converted)
            cut = max(1, int(len(converted) * .15))
            data["validation"], data["train"] = converted[:cut], converted[cut:]
        else:
            data[split] = converted
    if "train" in data:
        normalized = lambda s: " ".join(s.split())
        test_questions = {normalized(r["question"]) for r in data["test"]}
        for split in ("train", "validation"):
            data[split] = [r for r in data[split] if normalized(r["question"]) not in test_questions]
        assert not {r["id"] for r in data["train"]} & {r["id"] for r in data["validation"]}
    files = {}
    for split, rows in data.items():
        path = target / (split + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        files[split] = {"n": len(rows), "sha256": sha(path)}
    write_json(target / "manifest.json", {"task": args.task, "revision": args.dataset_revision,
               "local_source": args.data_jsonl, "seed": args.seed, "files": files})
    print(json.dumps(files, indent=2))


def marker_positions(text, offsets, prompt_length):
    """Map marker starts to actual generated tokens, never retokenize the response."""
    end = text.find("####")
    if end < 0:
        return [], []
    matches = list(re.finditer(r"(?im)^\s*(?:\*\*)?(?P<marker>Step)\s+(\d+)\s*:", text[:end]))
    positions, steps = [], []
    for match in matches:
        start = match.start("marker")
        token = next((i for i, (a, b) in enumerate(offsets) if a <= start < b), None)
        if token is None:
            raise ValueError("Step marker has no matching token offset")
        positions.append(prompt_length + token)
        steps.append(int(match.group(2)))
    token = next((i for i, (a, b) in enumerate(offsets) if a <= end < b), None)
    if token is None:
        raise ValueError("Answer marker has no matching token offset")
    if steps != list(range(1, len(steps) + 1)):
        return [], []
    return positions + [prompt_length + token], steps


def load_lm(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not Path(model_path).is_dir():
        raise ValueError("MODEL_PATH must be an already downloaded checkpoint directory")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa").to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tokenizer


def final_answer_line_complete(text):
    """Stop only after a nonempty final-answer line ends, never at its first digit."""
    marker = text.find("####")
    if marker < 0:
        return False
    tail = text[marker + 4:]
    for line in tail.splitlines(keepends=True):
        payload = line.strip().strip("#").strip()
        if payload:
            return line.endswith(("\n", "\r"))
    return False


class FinalAnswerLineStop:
    """Single-response stopping criterion; uses generated text, never the reference."""
    def __init__(self, tokenizer, prefix):
        self.tokenizer, self.prefix = tokenizer, prefix
        self.finished = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        text = self.tokenizer.decode(input_ids[0, self.prefix:].tolist(),
                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.finished = final_answer_line_complete(text)
        return torch.tensor([self.finished], device=input_ids.device, dtype=torch.bool)


def generate(args):
    import torch
    from tqdm import tqdm
    utils = module("author_utils", ROOT / "author/src/utils.py")
    records = [json.loads(line) for line in (args.root / "datasets" / args.task / (args.split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    records = records[:args.limit] if args.limit else records
    cell = args.root / args.name / args.task / args.split
    config = {"model": str(Path(args.model).resolve()), "model_config": sha(Path(args.model)/"config.json"),
              "dataset": sha(args.root / "datasets" / args.task / (args.split + ".jsonl")),
              "limit": args.limit, "seed": args.seed, "max_new_tokens": args.max_tokens,
              "prompt": args.prompt, "stop_mode": args.stop_mode, "adapter": sha(__file__)}
    manifest = cell / "generation.json"
    if manifest.exists() and read_json(manifest)["config"] != config:
        raise ValueError("Generation configuration changed; use another --name")
    write_json(manifest, {"config": config, "complete": False})
    model, tokenizer = load_lm(args.model)
    prompt_mode = args.prompt
    if prompt_mode == "auto":
        prompt_mode = "chat" if getattr(tokenizer, "chat_template", None) else "plain"
    # Greedy decoding should not inherit sampling-only settings from Qwen's checkpoint.
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    print(f"Generation: prompt={prompt_mode}, stop={args.stop_mode}, greedy=True", flush=True)
    torch.manual_seed(args.seed)
    for row in tqdm(records, desc=f"{args.task}/{args.split}"):
        output = cell / "merged" / f"gsm8k_{row['id']}.json"
        if output.exists():
            read_json(output)
            continue
        prompt = utils.format_prompt(row["question"], template="cot")
        if prompt_mode == "chat":
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=prompt_mode == "plain").to(model.device)
        prefix = inputs.input_ids.shape[1]
        stopper = FinalAnswerLineStop(tokenizer, prefix)
        stop_options = {"stopping_criteria": [stopper]} if args.stop_mode == "final-line" else {}
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=tokenizer.eos_token_id, **stop_options)[0]
        generated = ids[prefix:].tolist()
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        ended_eos = bool(generated) and generated[-1] in eos_ids
        truncated = len(generated) >= args.max_tokens and not ended_eos and not stopper.finished
        stop_reason = "final_answer_line" if stopper.finished else "eos" if ended_eos else "max_tokens" if truncated else "other"
        text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = utils.extract_answer(text, task="gsm8k" if args.task == "gsm8k" else "math-500")
        gold = utils.extract_answer(str(row["answer"]), task="gsm8k") if args.task == "gsm8k" else str(row["answer"])
        if args.task == "gsm8k":
            # The official numeric parser recognizes the ####-formatted reference.
            correct = bool(utils.evaluate_answer(parsed, gold))
        else:
            if "\\boxed" in gold:
                gold = utils.extract_answer(gold, task="math-500")
            correct = bool(utils.evaluate_math_answer(parsed or "", gold or ""))
        write_json(output, {**row, "produced_text": text, "decoded_with_specials": raw,
                   "input_ids": inputs.input_ids[0].tolist(), "full_seq_ids": ids.tolist(),
                   "dp1_idx": prefix - 1, "produced_answer": parsed, "gold_answer": gold,
                   "is_correct": correct, "truncated": truncated, "stop_reason": stop_reason,
                   "generated_tokens": len(generated), "prompt_mode": prompt_mode})
    write_json(manifest, {"config": config, "complete": True, "n": len(records)})


def extract(args):
    """Boundary-only capture. Export official NPZ schema, without full-vocabulary logits."""
    import numpy as np
    import torch
    from tqdm import tqdm
    cell = args.root / args.name / args.task / args.split
    output = cell / "activations.npz"
    if output.exists():
        audit = read_json(cell / "extraction.json")
        if audit["state"] != args.state or audit["source_generation"] != read_json(cell/"generation.json") or audit["npz_sha256"] != sha(output) or audit["source_generation"]["config"]["model"] != str(Path(args.model).resolve()):
            raise ValueError("Extraction provenance differs; choose a new name")
        print("Using existing extraction", output)
        return
    model, tokenizer = load_lm(args.model)
    if read_json(cell / "generation.json")["config"]["model"] != str(Path(args.model).resolve()):
        raise ValueError("Extract with the exact checkpoint used for generation")
    n = len(model.model.layers)
    per_layer = {key: [[] for _ in range(n)] for key in
          ("step_activations", "hash_activations", "step_numbers", "question_ids_step", "question_ids_hash", "is_correct_step", "is_correct_hash")}
    excluded, included = [], []
    files = sorted((cell / "merged").glob("gsm8k_*.json"))
    if not files:
        raise ValueError("Run generation first")
    for path in tqdm(files, desc="Boundary extraction"):
        row = read_json(path)
        prefix = len(row["input_ids"])
        ids = row["full_seq_ids"]
        generated = ids[prefix:]
        text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded.input_ids != generated:
            raise ValueError(f"Tokenizer roundtrip differs for {path}; cannot silently realign markers")
        positions, steps = marker_positions(text, encoded.offset_mapping, prefix)
        if not steps or row["truncated"]:
            excluded.append({"id":row["id"],"reason":"truncated" if row["truncated"] else "missing_or_nonsequential_markers"})
            continue
        capture = {}
        handles = []
        targets = torch.tensor([p-1 for p in positions], device=model.device)
        def hook(layer):
            def save(_module, _inputs, outputs):
                state = outputs[0] if isinstance(outputs, tuple) else outputs
                selected = state[0, targets]
                if layer == n-1 and args.state == "author-hidden":
                    selected = model.model.norm(selected)
                capture[layer] = selected.float().cpu().numpy()
            return save
        try:
            for layer, block in enumerate(model.model.layers):
                handles.append(block.register_forward_hook(hook(layer)))
            # Run only the backbone. The language head would materialize sequence x vocabulary logits.
            with torch.inference_mode():
                model.model(input_ids=torch.tensor([ids[:max(positions)]], device=model.device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for layer in range(n):
            vectors = capture[layer]
            per_layer["step_activations"][layer].extend(vectors[:-1])
            per_layer["hash_activations"][layer].append(vectors[-1])
            per_layer["step_numbers"][layer].extend(steps)
            per_layer["question_ids_step"][layer].extend([row["id"]]*len(steps))
            per_layer["question_ids_hash"][layer].append(row["id"])
            per_layer["is_correct_step"][layer].extend([row["is_correct"]]*len(steps))
            per_layer["is_correct_hash"][layer].append(row["is_correct"])
        included.append(row["id"])
    if not included:
        raise ValueError("No valid trajectories; inspect generated format before continuing")
    payload = {}
    for key, layers in per_layer.items():
        dtype = np.float32 if "activations" in key else np.bool_ if "is_correct" in key else np.int64
        payload[key] = np.array([np.asarray(x, dtype=dtype) for x in layers])
    # Use the author's trimmed-mean implementation unchanged.
    collector = module("official_collector", ROOT / "author/scripts/steering/collect_steering_vectors.py")
    vectors = []
    for layer in range(n):
        answer = dict(zip(payload["question_ids_hash"][layer], payload["hash_activations"][layer]))
        differences = [answer[q]-h for q,h in zip(payload["question_ids_step"][layer],payload["step_activations"][layer])]
        vectors.append(collector.trimmed_mean_by_norm(differences))
    payload.update(steering_vectors=np.asarray(vectors), num_layers=n, hidden_dim=model.config.hidden_size,
                   task=args.task, stats=json.dumps({"successful_questions":len(included)}))
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(output)
    write_json(cell / "extraction.json", {"state":args.state,"layers":n,"included":included,"excluded":excluded,
                "source_generation":read_json(cell/"generation.json"),"npz_sha256":sha(output)})
    print("Extracted",len(included),"excluded",len(excluded),output)


def author(args):
    # Keep downloaded source pristine. All compatibility patches are in a
    # content-addressed runtime copy on the server data disk.
    files = sorted((ROOT / "author").rglob("*.py"))
    fingerprint = hashlib.sha256((sha(__file__) + "".join(sha(p) for p in files)).encode()).hexdigest()[:16]
    runtime = args.root / "runtime" / fingerprint
    if not (runtime / "ready.json").exists():
        shutil.copytree(ROOT / "author", runtime, dirs_exist_ok=True)
        target = runtime / "scripts/steering/traj_based/compute_ideal_trajectory_thresholds.py"
        text = target.read_text(encoding="utf-8")
        before = "    # ALWAYS run full grid search for best thresholds on training set"
        after = """    # StepCluster: fit prototypes on training, choose thresholds on validation.
    train_divergences = compute_all_divergences(test_trajs, test_qids, test_correct, model)
    # Grid search below now receives only held-out validation trajectories."""
        if text.count(before) != 1:
            raise ValueError("Upstream threshold implementation changed")
        target.write_text(text.replace(before, after), encoding="utf-8")
        write_json(runtime / "ready.json", {"source":str(ROOT/"author"), "patches":["threshold selection on supplied validation NPZ"]})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
    cwd = args.root / args.name / "author-work"
    cwd.mkdir(parents=True, exist_ok=True)
    command = args.author_args
    if command and command[0] == "--":
        command = command[1:]
    script = (runtime / args.script).resolve()
    if not script.is_relative_to(runtime.resolve()):
        raise ValueError("Expected an author script")
    subprocess.run([sys.executable, "-u", str(script), *command], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "generate", "extract", "author"])
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("STEP_ROOT", "/mnt/data/wxh/wdl/results/StepCluster")))
    parser.add_argument("--name", default=os.environ.get("RUN_NAME", "qwen25-v1"))
    parser.add_argument("--task", choices=["gsm8k", "math500", "math-full"], default="gsm8k")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--data-jsonl", help="Local file pattern containing {split}; math-full requires an explicit local file")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--stop-mode", choices=["final-line", "eos"], default="final-line")
    parser.add_argument("--state", choices=["author-hidden", "resid-post"], default="author-hidden")
    parser.add_argument("--script")
    args, rest = parser.parse_known_args()
    args.author_args = rest
    if rest and args.phase != "author":
        parser.error("Unexpected arguments: " + " ".join(rest))
    if args.task == "math-full" and args.phase == "prepare" and not args.data_jsonl:
        parser.error("math-full must use the existing exported MATH JSONL")
    if args.limit < 0 or args.max_tokens < 1:
        parser.error("Invalid generation size")
    if args.phase == "author":
        if not args.script:
            parser.error("--script is required")
        author(args)
    else:
        globals()[args.phase](args)


if __name__ == "__main__":
    main()
