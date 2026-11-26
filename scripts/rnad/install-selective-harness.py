"""Copy the opt-in harness into a verified separate engine worktree."""
from pathlib import Path
import sys
p=Path(sys.argv[1])
def patched(src,dest,replacements):
    data=Path(src).read_text()
    for old,new in replacements:
        assert data.count(old)==1, (src,old)
        data=data.replace(old,new,1)
    Path(dest).write_text(data)
base="scripts/rnad/"
for file in ("selective-belief-search.mjs","selective-search-match.mjs","check-selective-belief.mjs"):
    (Path(base)/file).write_bytes((p/file).read_bytes())
patched(base+'switch-loop-guard.mjs',base+'selective-switch-loop-guard.mjs',[
 ("choose({observation,legalActions,seat,round,rng}){","choose({observation,legalActions,seat,round,rng,publicContext}){"),
 ("observation,legalActions:allowed,seat,round,rng","observation,legalActions:allowed,seat,round,rng,publicContext"),
])
patched(base+'shared_league_actor.mjs',base+'shared-league-selective-actor.mjs',[
 ("from './switch-loop-guard.mjs'","from './selective-switch-loop-guard.mjs'"),
 ("greedy=false,trajectoryBattleSchema=null})", "greedy=false,trajectoryBattleSchema=null,researchRerank=null})"),
 ("choose({observation,legalActions,seat,round,rng}){","choose({observation,legalActions,seat,round,rng,publicContext}){"),
 ("const ix=greedy?logits.reduce((best,x,i)=>x>logits[best]?i:best,0):sample(probs,rng);",
  "let ix=greedy?logits.reduce((best,x,i)=>x>logits[best]?i:best,0):sample(probs,rng);\n"
  "      if(researchRerank!==null){\n"
  "        assert(greedy&&!legacy,'Selective rerank is research-only frozen inference');\n"
  "        ix=researchRerank({observation,legalActions,logits,index:ix,publicContext});\n"
  "        assert(Number.isInteger(ix)&&ix>=0&&ix<legalActions.length,'Bad selective rerank index');\n"
  "      }"),
])
patched(base+'evaluate-shared-g012-vs-aq.mjs',base+'integrated-selective-session.mjs',[
 ("from './shared_league_actor.mjs'","from './shared-league-selective-actor.mjs'"),
 ("createIntegratedAQSession({modelDir,seat})","createIntegratedAQSession({modelDir,seat,researchRerank=null})"),
 ("epsilon:.02,greedy:true","epsilon:.02,greedy:true,researchRerank"),
 ("function select(view){","function select(view,publicFrame=null){"),
 ("const choice=frozen.func({observation:obs,legalActions,seat,round});",
  "const choice=frozen.func({observation:obs,legalActions,seat,round,\n"
  "      publicContext:publicFrame?{compact:view,publicFrame}:null});"),
 ("return {select,commit,stats,finish};","return {select,commit,stats,finish,needsPublicFrame:researchRerank!==null};"),
])
base="aq-runtime/scripts/dev/simulator/"
patched(base+'strategy-match.js',base+'strategy-selective-match.js',[
 ("""action=measure('pureRlDecision',()=>pureRlSessions[seat].select(view));""",
  """const actor=pureRlSessions[seat];
          // Lazy, whitelisted projection from the SAME seat, only for battle search.
          // No checkpoint, true opposing roster, hit state or pending action.
          const publicFrame=actor.needsPublicFrame?()=>({
            revealed:battle.view(seat).revealed?.map(({active,speciesId,usedMoves}) =>
              ({active,speciesId,usedMoves:Array.isArray(usedMoves)?[...usedMoves]:[]})),
          }):null;
          action=measure('pureRlDecision',()=>actor.select(view,publicFrame));"""),
])
