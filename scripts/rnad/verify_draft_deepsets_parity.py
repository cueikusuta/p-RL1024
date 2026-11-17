"""Independent PyTorch reference for cross-language frozen Deep Sets inference."""
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.rnad.train_draft_deepsets import DeepSetsDraft,FEATURE_WIDTH

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--model",required=True)
    a.add_argument("--out",required=True)
    args=a.parse_args()
    payload=json.loads(Path(args.model).read_text())
    if (payload.get("schema")!="rnad-draft-deepsets-weights-v1" or
        payload.get("architecture")!="deepsets-soft" or
        payload.get("feature_width")!=FEATURE_WIDTH):
        raise ValueError("Wrong frozen draft model")
    net=DeepSetsDraft().eval()
    expected=net.state_dict()
    supplied=payload["weights"]
    if set(supplied)!=set(expected):
        raise ValueError("Missing expected frozen tensor")
    new={}
    for k,shape in expected.items():
        p=supplied[k]
        if (p["shape"]!=list(shape.shape) or
            len(p["values"])!=shape.numel() or
            not all(isinstance(v,(int,float)) and math.isfinite(v)
                    for v in p["values"])):
            raise ValueError("Malformed frozen tensor")
        new[k]=torch.tensor(p["values"],dtype=torch.float32).reshape(shape.shape)
    net.load_state_dict(new,strict=True)
    n=7
    pool=torch.linspace(.05,.95,n*FEATURE_WIDTH).reshape(n,FEATURE_WIDTH)
    # A deterministic nontrivial choice set with changing 6th team member.
    codes=torch.tensor([[0,1,2,3,4,5],
                        [0,1,2,3,4,6],
                        [0,1,2,3,5,6]],dtype=torch.long)
    prices=[2,1,2,1,1,1,1]
    carries=[1,0,0,0,0,0,0]
    megas=[0,0,0,1,0,0,1]
    credits=10
    extra=torch.tensor([[
        sum(prices[i] for i in c)/100.,
        (credits-sum(prices[i] for i in c))/100.,
        sum(carries[i] for i in c)/6.,
        sum(megas[i] for i in c)/6.,
    ] for c in codes.tolist()],dtype=torch.float32)
    with torch.no_grad():
        expected=net(pool,codes,extra).tolist()
    Path(args.out).write_text(json.dumps({
        "encoding":{"features":pool.tolist(),"codes":codes.tolist(),
                    "prices":prices,"carries":carries,"megas":megas,
                    "credits":credits},
        "torch_logits":expected
    },allow_nan=False)+"\n")
    print({"reference_logits":expected})
if __name__=="__main__":
    main()
