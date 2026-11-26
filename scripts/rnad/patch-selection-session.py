from pathlib import Path
paths=['scripts/rnad/shared_league_actor.mjs','scripts/rnad/evaluate-shared-g012-vs-aq.mjs']
for i,name in enumerate(paths):
    p=Path(name)
    s=p.read_text()
    if i==0:
        src='export function session({modelDir,magnetDir,legacyPath=null,legacySha=null,epsilon=.02,greedy=false,trajectoryBattleSchema=null})'
        assert s.count(src)==1
        s=s.replace(src,src.replace('trajectoryBattleSchema=null','trajectoryBattleSchema=null,selectionReranker=null'))
        src='      const ix=greedy?logits.reduce((best,x,i)=>x>logits[best]?i:best,0):sample(probs,rng);'
        assert s.count(src)==1
        s=s.replace(src,src.replace('const ix','let ix')+'''
      if(selectionReranker&&observation.phase==='selection'){
        assert(greedy&&!legacy,'Selection reranker is frozen eval-only');
        ix=selectionReranker({observation,legalActions,logits,baseline:ix});
        assert(Number.isSafeInteger(ix)&&ix>=0&&ix<legalActions.length,'Invalid reranked selection index');
      }''')
    else:
        src='export function createIntegratedAQSession({modelDir,seat})'
        assert s.count(src)==1
        s=s.replace(src,'export function createIntegratedAQSession({modelDir,seat,selectionReranker=null})')
        src='magnetDir:modelDir,epsilon:.02,greedy:true});'
        assert s.count(src)==1
        s=s.replace(src, 'magnetDir:modelDir,epsilon:.02,greedy:true,selectionReranker});')
    p.write_text(s)
