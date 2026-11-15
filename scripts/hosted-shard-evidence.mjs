// Shared fail-closed evidence verification for the full aggregate and per-runner witness.
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {compare} from './verify-evaluation.mjs';
import {mixSeed,weightsSha256} from '../models/aq-runner.mjs';
import {verifyTrainingCheckpoints,verifyShardClock} from './hosted-proof.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
export function verifyHostedShard({root,artifact,index,offset,model,opponent,base,sourceFingerprint=null,sourceCommit=null}) {
  if(artifact!=='shard-'+index&&!artifact.endsWith('-shard-'+index))
    throw new Error('Missing or duplicate shard index');
  const sub=opponent==='AQ-Base-001'?model+'-vs-aq':model==='RL-Linear'?'i975':'rl-deep';
  const folder=join(root,artifact,sub);
  const trainingPath=join(folder,'training.json'),bytes=readFileSync(trainingPath);
  const training=JSON.parse(bytes),proof=JSON.parse(readFileSync(join(folder,'comparison.json')));
  const actual=compare(...['a','b'].map(x=>JSON.parse(readFileSync(join(folder,'eval-'+x+'.json')))));
  if(actual.normalizedOutcomeSha256!==proof.normalizedOutcomeSha256||actual.modelSha256!==proof.modelSha256||
    training.modelSha256!==sha(JSON.stringify(training.model))||proof.modelSha256!==training.modelSha256||
    sha(bytes)!==proof.modelFileSha256)throw new Error('Weight or evaluation SHA mismatch: '+artifact);
  const t=JSON.parse(readFileSync(join(root,artifact,'wallclock-shard.json')));
  const count=t.count,rootSeed=base+offset,history=training.history||training.matches;
  verifyShardClock(t,{model,opponent,index,offset,count,seed:rootSeed,
    runId:process.env.GITHUB_RUN_ID,runAttempt:process.env.GITHUB_RUN_ATTEMPT});
  if(!Number.isSafeInteger(count)||count<9||count>256||
    training.completedMatches!==count||training.seedRoot!==rootSeed||
    !Array.isArray(history)||history.length!==count||
    !/^[a-f0-9]{64}$/.test(training.sourceSha256??'')||
    (opponent==='AQ-Base-001'&&training.opponentModelSha256!==weightsSha256)||
    (process.env.GITHUB_SHA&&training.sourceCommit!==process.env.GITHUB_SHA))
    throw new Error('Shard source/seed/completion mismatch: '+artifact);
  if(sourceFingerprint===null){sourceFingerprint=training.sourceSha256;sourceCommit=training.sourceCommit??null;}
  if(training.sourceSha256!==sourceFingerprint||(training.sourceCommit??null)!==sourceCommit)
    throw new Error('Incompatible source fingerprint or commit across shards');
  const tag=opponent==='AQ-Base-001'?0xa975:model==='RL-Deep'?0x976:0x975;
  for(let j=0;j<history.length;j++){
    const r=history[j];
    if(!r||r.seed!==mixSeed(rootSeed,j,tag)||r.studentSeat!==j%2)
      throw new Error('Training Match seed/seat mismatch');
    if(![0,1].includes(r.winner)||!Array.isArray(r.score)||r.score.length!==2||
      r.score[r.winner]!==4||!Number.isSafeInteger(r.rounds)||r.rounds<4)
      throw new Error('Invalid completed four-win Match');
  }
  const wins=history.filter(r=>r.studentSeat!==undefined&&r.winner===r.studentSeat).length;
  if(opponent==='AQ-Base-001'&&wins+history.filter(r=>r.studentSeat!==undefined&&r.winner!==r.studentSeat).length!==count)
    throw new Error('Missing AQ outcome');
  const rowWins=opponent==='AQ-Base-001'||model==='RL-Deep'?wins:training.studentWins??wins;
  const checkpoints=verifyTrainingCheckpoints(folder,{
    model,opponent,count,seed:rootSeed,sourceSha256:training.sourceSha256,finalHistory:history
  });
  const row={artifact,offset,count,seedStart:rootSeed,seedEnd:rootSeed+count-1,
    durationSeconds:t.durationSeconds,startedUnix:t.startedUnix,finishedUnix:t.finishedUnix,
    studentWins:rowWins,
    modelSha256:training.modelSha256,modelFileSha256:proof.modelFileSha256,
    normalizedEvaluationDigestSha256:actual.normalizedOutcomeSha256,
    checkpoints};

  const fileDigests={
    trainingSha256:sha(bytes),
    comparisonSha256:sha(readFileSync(join(folder,'comparison.json'))),
    evaluationASha256:sha(readFileSync(join(folder,'eval-a.json'))),
    evaluationBSha256:sha(readFileSync(join(folder,'eval-b.json'))),
    clockSha256:sha(readFileSync(join(root,artifact,'wallclock-shard.json')))
  };
  return {row,sourceFingerprint,sourceCommit,fileDigests};
}
