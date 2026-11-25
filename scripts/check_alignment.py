"""CPU regression checks for original byte-level token alignment; no model needed."""
import json
import random
from types import SimpleNamespace

from token_alignment import _byte_decoder, _decoded_char_spans, original_token_offsets
from extract_aligned import AlignedTokenizer
from pipeline import marker_positions


class ByteTokenizer:
    all_special_ids = [256]
    backend_tokenizer = SimpleNamespace(decoder=SimpleNamespace(__getstate__=lambda:json.dumps({'type':'ByteLevel'})))

    def __init__(self):
        self.byte_chars = {b:c for c,b in _byte_decoder().items()}

    def convert_ids_to_tokens(self, ids):
        return [self.byte_chars[i] if i < 256 else '<EOS>' for i in ids]

    def decode(self, ids, **kwargs):
        return b''.join(bytes([i]) if i < 256 else b'<EOS>' for i in ids).decode('utf-8',errors='replace')

    def __call__(self, text, **kwargs):
        # Simulate canonical merging: one merged token instead of the byte tokens.
        return {'input_ids':[999], 'offset_mapping':[(0,len(text))]}


def main():
    tokenizer = ByteTokenizer()
    raw = 'Step 1: caf\u00e9\nStep 2: \u4f60\u597d\n#### 42'
    ids = list(raw.encode('utf-8'))+[256]
    text, offsets, info = original_token_offsets(tokenizer, ids)
    assert text == raw+'<EOS>' and len(offsets) == len(ids)
    assert info['method'] == 'original_bytelevel_bytes'
    positions, steps = marker_positions(text, offsets, 10)
    assert steps == [1,2]
    assert positions == [10,10+len('Step 1: caf\u00e9\n'.encode('utf-8')),10+len(raw.split('####')[0].encode('utf-8'))]
    accent = text.index('\u00e9')
    assert sum(a==accent and b==accent+1 for a,b in offsets) == 2
    for bad in (b'\xe2(',b'\xed\xa0\x80',b'\xf0\x9f',b'\xffa\xc3\xa9'):
        result, mapped, _ = original_token_offsets(tokenizer, list(bad))
        assert result == bad.decode('utf-8',errors='replace')
        assert len(mapped) == len(bad)
    rng = random.Random(42)
    for _ in range(300):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1,80)))
        result, spans = _decoded_char_spans(data)
        assert result == data.decode('utf-8',errors='replace')
        assert len(result)==len(spans)
        assert spans[0][0]==0 and spans[-1][1]==len(data)
        assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
    audit=[];wrapped=AlignedTokenizer(tokenizer,audit)
    decoded=wrapped.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=wrapped(decoded,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded.input_ids == ids and encoded.offset_mapping == offsets
    assert len(audit)==1
    class Canonical(ByteTokenizer):
        def __call__(self,text,**kwargs):
            return {'input_ids':[97,98],'offset_mapping':[(0,1),(1,2)]}
    _, mapped, fast = original_token_offsets(Canonical(),[97,98])
    assert fast['method']=='identical_roundtrip' and mapped==[(0,1),(1,2)]
    print('PASS canonical/noncanonical token IDs, Step/answer boundaries, split UTF-8, malformed UTF-8, wrapper, 300 randomized byte strings')


if __name__ == '__main__':
    main()
