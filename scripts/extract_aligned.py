"""Extraction-only fix; keep pipeline.py and all generation signatures unchanged."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pipeline
from token_alignment import original_token_offsets


class AlignedTokenizer:
    def __init__(self, tokenizer, audit):
        self.base, self.audit = tokenizer, audit
        self.original_ids, self.raw = None, None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def decode(self, ids, **kwargs):
        text = self.base.decode(ids, **kwargs)
        if kwargs.get('skip_special_tokens') is False:
            self.original_ids, self.raw = list(ids), text
        return text

    def __call__(self, text, **kwargs):
        if kwargs.get('return_offsets_mapping') and self.original_ids is not None and text == self.raw:
            rebuilt, offsets, info = original_token_offsets(self.base, self.original_ids)
            if rebuilt != text:
                raise ValueError('Original-token alignment changed decoded response')
            self.audit.append({**info, 'original_ids_sha256':hashlib.sha256(str(self.original_ids).encode()).hexdigest()})
            return SimpleNamespace(input_ids=self.original_ids, offset_mapping=offsets)
        return self.base(text, **kwargs)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'extract':
        raise SystemExit('Use: python scripts/extract_aligned.py extract [pipeline extraction arguments]')
    audit = []
    original_load = pipeline.load_lm
    original_extract = pipeline.extract

    def load(path):
        model, tokenizer = original_load(path)
        return model, AlignedTokenizer(tokenizer, audit)

    def extract(args):
        original_extract(args)
        if audit:
            cell = args.root/args.name/args.task/args.split
            pipeline.write_json(cell/'token-alignment.json', {
                'extraction_wrapper_sha256':pipeline.sha(__file__),
                'alignment_implementation_sha256':pipeline.sha(Path(__file__).with_name('token_alignment.py')),
                'original_token_ids_preserved':True, 'trajectories':audit})
            print('Token alignment: original ByteLevel fallback',
                  sum(r['method']=='original_bytelevel_bytes' for r in audit), '/', len(audit), flush=True)

    pipeline.load_lm, pipeline.extract = load, extract
    pipeline.main()


if __name__ == '__main__':
    main()
