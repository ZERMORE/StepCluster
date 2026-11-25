"""CPU integration checks for the server adapter and official analysis schema."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fixture(path, count, offset):
    rng = np.random.default_rng(42 + offset)
    ids = np.arange(offset,offset+count)
    correct = ids % 2 == 0
    steps, answers = [], []
    for layer in range(4):
        states = rng.normal(size=(count,3,8))
        states[:,:,0] += correct[:,None]*2
        states[:,:,1] += np.arange(3)[None]*2
        steps.append(states.reshape(-1,8).astype(np.float32))
        answers.append((states[:,-1]+.1).astype(np.float32))
    np.savez(path,step_activations=np.array(steps),hash_activations=np.array(answers),
         step_numbers=np.tile(np.tile([1,2,3],count),(4,1)),
         question_ids_step=np.tile(np.repeat(ids,3),(4,1)),question_ids_hash=np.tile(ids,(4,1)),
         is_correct_step=np.tile(np.repeat(correct,3),(4,1)),is_correct_hash=np.tile(correct,(4,1)),
         num_layers=4,hidden_dim=8,steering_vectors=np.zeros((4,8)))


def run(command, env):
    result = subprocess.run([sys.executable,"-B",*map(str,command)],cwd=ROOT,env=env,
                            capture_output=True,text=True,encoding="utf-8",errors="replace")
    if result.returncode:
        raise RuntimeError(result.stdout[-2000:]+result.stderr[-2000:])
    return result


def adapter_roundtrip(pipe, temp):
    """Exercise real generation and hook extraction code using a tiny deterministic model."""
    import ast
    import torch
    from torch import nn
    class Inputs(dict):
        def __getattr__(self, key):
            return self[key]
        def to(self, device):
            return Inputs({k:v.to(device) for k,v in self.items()})
    class Tokenizer:
        eos_token_id=0
        def __call__(self,text,return_tensors=None,**kwargs):
            ids=[ord(c) for c in text]
            if return_tensors:
                return Inputs(input_ids=torch.tensor([ids]))
            return SimpleNamespace(input_ids=ids,offset_mapping=[(i,i+1) for i in range(len(text))])
        def decode(self,ids,skip_special_tokens=False,**kwargs):
            return ''.join(chr(int(x)) for x in ids if not(skip_special_tokens and x==0))
    class Block(nn.Module):
        def forward(self,x):
            return (x+1,)
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=nn.ModuleList([Block(),Block()])
            self.norm=nn.Identity()
        def forward(self,input_ids,**kwargs):
            x=input_ids.float()[...,None].expand(-1,-1,8)
            for layer in self.layers:
                x=layer(x)[0]
            return x
    class Model:
        device='cpu'
        config=SimpleNamespace(hidden_size=8)
        generation_config=SimpleNamespace(eos_token_id=0)
        model=Backbone()
        def generate(self,input_ids,**kwargs):
            suffix='Step 1: 2 + 2 = 4.\nStep 2: verify.\n#### 4'+chr(0)
            return torch.cat([input_ids,torch.tensor([[ord(c) for c in suffix]])],dim=1)
    checkpoint=temp/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    dataset=temp/'adapter/datasets/gsm8k'
    dataset.mkdir(parents=True)
    (dataset/'test.jsonl').write_text(json.dumps({'id':0,'question':'2+2?','answer':'#### 4'})+'\n')
    args=SimpleNamespace(root=temp/'adapter',name='tiny',task='gsm8k',split='test',model=str(checkpoint),
                         limit=0,seed=42,max_tokens=256,prompt='plain',state='author-hidden')
    pipe.load_lm=lambda _: (Model(),Tokenizer())
    original=pipe.module
    def load(name,path):
        if name!='official_collector':
            return original(name,path)
        tree=ast.parse(path.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trimmed_mean_by_norm')
        namespace={'np':np,'List':list}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        return SimpleNamespace(trimmed_mean_by_norm=namespace['trimmed_mean_by_norm'])
    pipe.module=load
    pipe.generate(args)
    pipe.generate(args)  # Resume, preserving completed records.
    pipe.extract(args)
    with np.load(temp/'adapter/tiny/gsm8k/test/activations.npz') as data:
        assert data['step_numbers'][0].tolist()==[1,2]
        assert data['is_correct_hash'][0].tolist()==[True]
        row=pipe.read_json(temp/'adapter/tiny/gsm8k/test/merged/gsm8k_0.json')
        expected=row['input_ids'][-1]+1
        assert np.all(data['step_activations'][0,0]==expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true",help="Run official t-SNE and step probe on synthetic data")
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("pipeline",ROOT/"scripts/pipeline.py")
    pipe=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)
    text="Step 1: first.\nStep 2: next.\n#### 4"
    offsets=[(i,i+1) for i in range(len(text))]
    positions,steps=pipe.marker_positions(text,offsets,7)
    assert steps==[1,2] and positions==[7,22,36]
    assert pipe.marker_positions("Step 2: x\n#### 1",[(i,i+1) for i in range(18)],7)==([],[])
    env=dict(os.environ,MPLBACKEND="Agg",PYTHONIOENCODING="utf-8",OMP_NUM_THREADS="2")
    with tempfile.TemporaryDirectory(prefix="stepcluster-check-") as temp:
        temp=Path(temp)
        for split,offset in (("train",0),("validation",100),("test",200)):
            fixture(temp/f"{split}.npz",40,offset)
        run(["scripts/pipeline.py","--help"],env)
        run(["scripts/evaluate.py","--train",temp/"train.npz","--validation",temp/"validation.npz",
             "--test",temp/"test.npz","--output",temp/"heldout","--layer",3,"--pca",8],env)
        report=json.loads((temp/"heldout/metrics.json").read_text())
        assert report["test"]["n"]==40
        adapter_roundtrip(pipe,temp)
        for name in ("scripts/steering/visualize_act_tsne.py","scripts/predictors/train_stepwise_probes.py",
                     "scripts/trajectories/analyze_step_distances.py"):
            run(["scripts/pipeline.py","author","--root",temp/"runtime-root","--script",name,"--","--help"],env)
        if args.full:
            for script,params in [
                ("scripts/steering/visualize_act_tsne.py",["--data",temp/"test.npz","--output-dir",temp/"tsne","--individual"]),
                ("scripts/predictors/train_stepwise_probes.py",["--data",temp/"train.npz","--output",temp/"steps"]),
            ]:
                run(["scripts/pipeline.py","author","--root",temp/"runtime-root","--script",script,"--",*params],env)
            assert list((temp/"tsne").glob("*.png"))
            assert (temp/"steps/results.json").exists()
    print("PASS marker offsets, tiny model generate/extract/resume, official schema, held-out training, official CLI" + (", official t-SNE and step probe" if args.full else ""))
    print("Real checkpoint generation/extraction/steering must be checked on the server with smoke.")


if __name__=="__main__":
    main()
