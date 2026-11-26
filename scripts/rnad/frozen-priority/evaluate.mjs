// Frozen evaluation CLI; no training modules are run and no state is exported.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {createFrozenPrioritySession} from './g421-frozen-priority-session.mjs';
import {runStrategyMatch} from '../../../aq-runtime/scripts/dev/simulator/strategy-frozen-priority.js';
import {createOpponent} from '../../../models/aq-base.mjs';
import provenance from '../../../provenance/manifest.json' with {type:'json'};
import {verifiedRunningSourceSha} from '../../dev/simulator/rnad-match.js';

const digest=b=>createHash('sha256').update(b).digest('hex');
const [modelDir,mode,opponent,seedArg,seatArg]=process.argv.slice(2);
const manifest=JSON.parse(readFileSync(join(modelDir,'manifest.json')));
assert.equal(manifest.source_sha,verifiedRunningSourceSha());
assert.equal(manifest.engine_sha,provenance.sourceCommit);
for(const [head,sha] of Object.entries(manifest.files)){
 assert(['draft','selection','battle','values'].includes(head));
 assert.equal(digest(readFileSync(join(modelDir,head+'.json'))),sha);
}
const seed=Number(seedArg),seat=Number(seatArg),accepted=[];
assert(Number.isSafeInteger(seed)&&seed>0&&seed<=0xffffffff&&[0,1].includes(seat));
assert(['aq-base','strategy-tactical'].includes(opponent));
const actor=createFrozenPrioritySession({modelDir,seat,mode,onAccepted:r=>accepted.push(r)});
const r=runStrategyMatch({seed,policies:[0,1].map(i=>i===seat?'pure-rl-linear':opponent),
 pureRlSessions:[0,1].map(i=>i===seat?actor:null),pureRlAllPhases:true,
 ...(opponent==='aq-base'?{v2Weights:{blend:.28,exposure:.75,continuation:.4}}:{}),
 strategyMode:'integrated',aqBase:opponent==='aq-base'?createOpponent():null});
assert(r.completed&&r.score[r.winner]===4&&actor.stats().updates===0);
const result={winner:r.winner,score:r.score,roundWinners:r.roundWinners,actionCounts:r.actionCounts};
console.log(JSON.stringify({schema:'g421-frozen-priority-eval-v1',mode,opponent,seed,seat,result,acceptedRecommendations:accepted.length}));
