import {createHash} from 'node:crypto';
import {readdirSync,readFileSync} from 'node:fs';
import {join} from 'node:path';
import {mixSeed} from '../models/aq-runner.mjs';

const sha=x=>createHash('sha256').update(x).digest('hex');
const isSha=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
export function verifyTrainingCheckpoints(folder,{model,opponent,count,seed,sourceSha256}) {
  if(!Number.isSafeInteger(count)||count<9||count>256||!Number.isSafeInteger(seed)||seed<1||
     !isSha(sourceSha256)||!['RL-Linear','RL-Deep'].includes(model)||
     !['V2E','RL-Deep','AQ-Base-001'].includes(opponent))throw new Error('Invalid checkpoint provenance');
  const expected=Array.from({length:Math.floor(count/8)},(_,i)=>'training-m'+((i+1)*8)+'.json');
  const actual=readdirSync(folder).filter(x=>/^training-m\d+\.json$/.test(x));
  if(actual.length!==expected.length||expected.some(x=>!actual.includes(x)))
    throw new Error('Missing or unexpected required checkpoint');
  return expected.map((name,i)=>{
    const bytes=readFileSync(join(folder,name)),cp=JSON.parse(bytes),completed=(i+1)*8;
    const isLinear=model==='RL-Linear'&&opponent==='V2E';
    const hashKey=isLinear?'resumeDigestSha256':'checkpointSha256';
    if(!cp||!isSha(cp[hashKey]))throw new Error('Missing checkpoint SHA: '+name);
    const {[hashKey]:recorded,...state}=cp;
    if(sha(JSON.stringify(state))!==recorded)throw new Error('Checkpoint SHA mismatch: '+name);
    let history;
    if(isLinear){
      if(state.resumeState?.schema!=='i975-exact-full-match-resume-v1'||
         state.resumeState.plannedMatches!==count||state.completedMatches!==completed||
         state.seedRoot!==seed||state.sourceSha256!==sourceSha256||state.checkpointEvery!==8||
         !Array.isArray(state.matches)||state.matches.length!==completed)
        throw new Error('Checkpoint configuration mismatch: '+name);
      history=state.matches;
    } else {
      if(state.schema!==(opponent==='AQ-Base-001'?'p-rl1024-aq-checkpoint-v1':'p-rl1024-deep-checkpoint-v1')||
         state.matches!==count||state.seed!==seed||state.sourceSha256!==sourceSha256||
         (opponent==='AQ-Base-001'&&state.modelId!==model)||
         !Array.isArray(state.history)||state.history.length!==completed)
        throw new Error('Checkpoint configuration mismatch: '+name);
      history=state.history;
    }
    const tag=opponent==='AQ-Base-001'?0xa975:model==='RL-Deep'?0x976:0x975;
    for(let j=0;j<history.length;j++) {
      const h=history[j];
      if(!h||h.seed!==mixSeed(seed,j,tag)||h.studentSeat!==j%2||
         ![0,1].includes(h.winner)||!Array.isArray(h.score)||h.score[h.winner]!==4)
        throw new Error('Checkpoint match-seed mismatch: '+name);
    }
    return {name,fileSha256:sha(bytes),stateSha256:recorded,completedMatches:completed};
  });
}
export function verifyReportTiming(report){
  if(!Number.isSafeInteger(report.startedAtMs)||report.startedAtMs<1577836800000||
     report.startedAtMs>9999999999999||typeof report.durationMs!=='number'||
     !Number.isFinite(report.durationMs)||report.durationMs<=0||report.durationMs>86400000||
     !Number.isSafeInteger(report.workers)||report.workers<1||report.workers>4)
    throw new Error('Invalid frozen evaluation timing');
  return report.startedAtMs+report.durationMs;
}
