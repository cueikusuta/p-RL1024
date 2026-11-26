"""Verify all modes against known G421 evaluation receipts."""
from pathlib import Path
import hashlib,json,subprocess,sys
model, fixture = sys.argv[1:]
cases=json.loads(Path(fixture).read_text())
for c in cases:
 for mode in ('off','shadow','public-lethal-priority-v1'):
  out=subprocess.check_output(['node','scripts/rnad/frozen-priority/evaluate.mjs',model,mode,c['opponent'],str(c['seed']),str(c['seat'])],text=True)
  v=json.loads(out)
  assert v['acceptedRecommendations']==int(mode!='off')
  h=hashlib.sha256(json.dumps(v['result'],separators=(',',':')).encode()).hexdigest()
  assert h==c['new' if mode=='public-lethal-priority-v1' else 'old'],v
print(f"Verified {len(cases)*3} full frozen matches, exact off/shadow/on")
