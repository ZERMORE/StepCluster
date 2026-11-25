"""Read physical JSONL records without changing existing generation signatures."""
import inspect
import json
from pathlib import Path
import sys

import pipeline


def read_jsonl_records(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSONL record: {path}, physical line {number}: {error}') from error
    return rows


def install_reader():
    """Version-checked, input-reader-only runtime patch; generation is unchanged."""
    source = inspect.getsource(pipeline.generate)
    old = 'records = [json.loads(line) for line in (args.root / "datasets" / args.task / (args.split + ".jsonl")).read_text(encoding="utf-8").splitlines()]'
    new = 'records = read_jsonl_records(args.root / "datasets" / args.task / (args.split + ".jsonl"))'
    if source.count(old) != 1:
        raise ValueError('Upstream generation reader changed; review the input-reader-only patch')
    pipeline.read_jsonl_records = read_jsonl_records
    exec(compile(source.replace(old, new), __file__+'::pipeline.generate', 'exec'), pipeline.__dict__)
    patched = pipeline.generate

    def generate(args):
        patched(args)
        cell = args.root/args.name/args.task/args.split
        pipeline.write_json(cell/'input-reader.json', {
            'policy':'physical JSONL lines; preserve embedded Unicode separators',
            'reader_sha256':pipeline.sha(__file__), 'pipeline_sha256':pipeline.sha(pipeline.__file__),
            'dataset_sha256':pipeline.sha(args.root/'datasets'/args.task/(args.split+'.jsonl'))})
    pipeline.generate = generate


def main():
    if len(sys.argv)<2 or sys.argv[1]!='generate':
        raise SystemExit('Use: python scripts/generate_safe.py generate [pipeline generation arguments]')
    install_reader()
    pipeline.main()


if __name__ == '__main__':
    main()
