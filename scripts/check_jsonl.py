"""Input-reader regression and completed-generation reuse without Transformers."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pipeline
from generate_safe import install_reader, read_jsonl_records
from check import adapter_roundtrip


def main():
    before = hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='stepcluster-jsonl-') as temp:
        root = Path(temp)
        path = root/'records.jsonl'
        records = [{'id':1,'question':'a\u2028b\u2029c\x85d'}, {'id':2,'question':'normal'}]
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        assert read_jsonl_records(path)==records
        try:
            [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError('Fixture must reproduce original splitlines bug')
        path.write_text('{"question":"unfinished}\n',encoding='utf-8')
        try:
            read_jsonl_records(path)
        except ValueError as error:
            assert 'physical line 1' in str(error)
        else:
            raise AssertionError('Real corruption must not be silently accepted')
        # Create a completed legacy run using the existing tiny integration model.
        adapter_roundtrip(pipeline,root)
        original_manifest=(root/'adapter/tiny/gsm8k/test/generation.json').read_bytes()
        install_reader()
        args=SimpleNamespace(root=root/'adapter',name='tiny',task='gsm8k',split='test',
                             model=str(root/'checkpoint'),limit=0,seed=42,max_tokens=256,
                             prompt='auto',stop_mode='final-line',state='author-hidden')
        pipeline.generate(args)
        assert (root/'adapter/tiny/gsm8k/test/generation.json').read_bytes()==original_manifest
        # The existing generation function must now read the entire Unicode fixture.
        path=root/'adapter/datasets/gsm8k/validation.jsonl'
        path.write_text(json.dumps({'id':3,'question':'2+2?\u2028Explain.','answer':'#### 4'},ensure_ascii=False)+'\n',encoding='utf-8')
        args.split='validation'
        pipeline.generate(args)
        row=pipeline.read_json(root/'adapter/tiny/gsm8k/validation/merged/gsm8k_3.json')
        assert row['is_correct'] and '\u2028' in row['question']
    assert hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest()==before
    print('PASS embedded Unicode separators, real corruption diagnostics, unchanged generation signature, completed-run reuse, tiny-model generation')


if __name__=='__main__':
    main()
