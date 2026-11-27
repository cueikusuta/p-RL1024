from pathlib import Path
import hashlib,json,os,sys
work,target=map(Path,sys.argv[1:3]);g,run,old=map(int,sys.argv[3:])
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,doc):path.write_text(json.dumps(doc))
r=json.loads((work/'result.json').read_text());q=json.loads((work/'strict.json').read_text());idx=json.loads((target/'return/checkpoint-index.json').read_text())
cp=work/'output'/f'checkpoint-g{g+1}.pt'
assert r['generation']==g and idx['generation']==g+1 and q['accepted']==r['accepted']==q['audits']==r['audits']==1280 and len(q['pairs'])==1280
assert r['actual_ordered_updates']==40 and idx['full_checkpoint_sha256']==r['checkpoint_sha256']==sha(cp)
used=old+sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and p.name!='boot.tar.gz')+4096
assert used<=1900*1024**2
write(target/'input/index.json',{'generation':g+1,'sha256':sha(target/'boot.tar.gz'),'parts':len(list((target/'input').glob('boot.part-*'))),'prior_checkpoint_sha256':sha(cp),'bytes':used})
write(target/'return/done.json',{'generation':g+1,'input_generation':g,'checkpoint_sha256':sha(cp),'workflow_run':run,'matches':1280,'audits':1280,'updates':40,'actors':20,'bytes':used,'learner_elapsed_s':r['active_elapsed_s']})
for p in target.rglob('*'):
    if p.is_file() and '.git' not in p.parts:
        with p.open('rb') as f:os.fsync(f.fileno())
print(used)
