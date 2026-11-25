"""Held-out-by-question extension to the author's within-training-split predictor."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(path, layer, kind):
    data = np.load(path, allow_pickle=False)
    ids = data["question_ids_hash"][layer]
    labels = data["is_correct_hash"][layer].astype(int)
    steps = data["step_activations"][layer]
    numbers = data["step_numbers"][layer]
    questions = data["question_ids_step"][layer]
    result = []
    for i, q in enumerate(ids):
        ix = np.flatnonzero(questions == q)
        ix = ix[np.argsort(numbers[ix])]
        answer = data["hash_activations"][layer][i]
        if len(ix) < 2:
            result.append(None)
        elif kind == "early":
            result.append(steps[ix[1]] - steps[ix[0]])
        elif kind == "final":
            result.append(answer)
        else:
            result.append(np.concatenate([answer, answer-steps[ix[-1]], steps[ix[-1]]-steps[ix[-2]]]))
    mask = np.array([x is not None for x in result])
    if not mask.any():
        raise ValueError("Need trajectories with two or more steps")
    return ids[mask], np.stack([x for x in result if x is not None]), labels[mask]


def score(model, x, y):
    probability = model.predict_proba(x)[:,1]
    return {"n":len(y),"correct":int(y.sum()),
            "auroc_correct_positive":float(roc_auc_score(y,probability)),
            "ap_error_positive":float(average_precision_score(1-y,1-probability))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", choices=["early","final","late"],default="late")
    parser.add_argument("--pca",type=int,default=128)
    args=parser.parse_args()
    data=[features(p,args.layer,args.feature) for p in (args.train,args.validation,args.test)]
    if set(data[0][0]) & set(data[1][0]):
        raise ValueError("Training and validation question IDs overlap")
    # Train/validation are disjoint partitions of the train dataset. The official
    # test split has its own ID namespace and may reuse numerical indices.
    for _,_,y in data:
        if len(np.unique(y))!=2:
            raise ValueError("Each split needs both correct and incorrect examples; enlarge the pilot")
    best=None
    for c in (.001,.01,.1,1.,10.):
        x,y=data[0][1:]
        model=make_pipeline(PCA(n_components=min(args.pca,len(x)-1,x.shape[1]),random_state=42),
                            StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
        model.fit(x,y)
        validation=score(model,*data[1][1:])
        if best is None or validation["auroc_correct_positive"]>best[0]:
            best=(validation["auroc_correct_positive"],c,model)
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    with (args.output/"probe.pkl").open("wb") as stream:
        pickle.dump(best[2],stream)
    report={"protocol":"held-out official test, C selected on validation; exploratory extension",
            "layer":args.layer,"feature":args.feature,"C":best[1],
            "train":score(best[2],*data[0][1:]),"validation":score(best[2],*data[1][1:]),
            "test":score(best[2],*data[2][1:])}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
