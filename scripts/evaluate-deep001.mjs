import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {mixSeed} from '../aq-runtime/scripts/dev/simulator/runner.js';
import {runStrategyMatch} from '../aq-runtime/scripts/dev/simulator/strategy-match.js';
import {createOpponent,weightsSha256 as opponentSha} from '../models/aq-base.mjs';
import {loadModel,createSession,weightsSha256} from '../models/deep001.mjs';

const args=new Map(),argv=process.argv.slice(2);
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!['--model','--out','--seed','--purpose','--blocks'].includes(argv[i])||args.has(argv[i]))throw new Error('Invalid option');args.set(argv[i],argv[i+1]);}
const root=Number(args.get('--seed'));
if(!args.get('--model')||!args.get('--out')||args.get('--purpose')!=='smoke'||args.get('--blocks')!=='2'||!Number.isSafeInteger(root)||root<1||root>0xffffffff)throw new Error('Expected a positive seed and two-block smoke');
const sha=x=>createHash('sha256').update(x).digest('hex'),bytes=await readFile(args.get('--model')),model=loadModel(bytes.toString('utf8'));
const seeds=[0,1].map(i=>mixSeed(root,i,0x9764));
const prior=new Set([...Array.from({length:128},(_,i)=>mixSeed(976140,i,0x9761)),...Array.from({length:500},(_,i)=>mixSeed(976140,i,0x9762))]);
if(seeds[0]===seeds[1]||seeds.some(s=>prior.has(s)))throw new Error('Evaluation seed overlaps historical training or screen');
const records=[];
for(let block=0;block<2;block++)for(const rlSeat of [0,1]) {
  const session=createSession(model,{seat:rlSeat,seed:mixSeed(seeds[block],rlSeat,0x9763),train:false});
  const sessions=[null,null],policies=['aq-base','aq-base'];sessions[rlSeat]=session;policies[rlSeat]='pure-rl-linear';
  const result=runStrategyMatch({seed:seeds[block],policies,pureRlSessions:sessions,pureRlAllPhases:true,strategyMode:'integrated',v2Weights:{blend:.28,exposure:.75,continuation:.4},aqBase:createOpponent()});
  if(!result.completed||![0,1].includes(result.winner)||result.score[result.winner]!==4||result.pureRlStats[rlSeat]?.train||result.pureRlStats[rlSeat]?.updates!==0)throw new Error('Invalid frozen RL-Deep-001 evaluation');
  records.push({block,seed:seeds[block],rlSeat,winner:result.winner,won:Number(result.winner===rlSeat),rounds:result.rounds,actionCounts:result.actionCounts,score:result.score});
}
const report={schema:'p-rl1024-deep001-aq-paired-v1',purpose:'smoke',formalEligible:false,comparison:'RL-Deep-001 vs AQ-Base-001; seeded stochastic full-phase PPO',modelId:'RL-Deep-001',modelSha256:weightsSha256,modelFileSha256:sha(bytes),opponentModelSha256:opponentSha,originalSourceCommit:'daafd0ed9442f79f0b72a68b6deb5046cda1fad9',engineSnapshot:'aq-base-300b4450969b',seedRoot:root,blocks:2,completedMatches:records.length,seedDigestSha256:sha(JSON.stringify(seeds)),outcomeDigestSha256:sha(JSON.stringify(records)),records};
await mkdir(dirname(resolve(args.get('--out'))),{recursive:true});await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({modelSha256:weightsSha256,opponentModelSha256:opponentSha,outcomeDigestSha256:report.outcomeDigestSha256,completedMatches:records.length}));
