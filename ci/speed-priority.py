"""Prioritize verified packet work for the existing optimizer order."""
import ast
import json
from pathlib import Path
import random
import sys

path=Path(sys.argv[1])
engine=Path(sys.argv[2])
tree=ast.parse((engine/'scripts/rnad/shared_league_learner.py').read_text())
constants=[ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign)
           and any(isinstance(v,ast.Name) and v.id=='SHUFFLE_SEED' for v in a.targets)]
assert len(constants)==1 and type(constants[0]) is int
p=json.loads(path.read_text())
q={z['task']:z for z in p['packets']}
ids=sorted(row['name'].removesuffix('.json.gz') for z in q.values() for row in z['rows'])
assert len(ids)==len(set(ids))==1280
random.Random(constants[0]).shuffle(ids)
rank={name:i for i,name in enumerate(ids)}
def priority(lease):
    _,u,v=lease
    values=sorted(rank[row['name'].removesuffix('.json.gz')] for k in (u,v) for row in q[k]['rows'])
    return (values[0]//32,values[0],sum(values))
p['prioritizedLeases'].sort(key=priority)
p['optimizer_order']=ids
path.write_text(json.dumps(p))
