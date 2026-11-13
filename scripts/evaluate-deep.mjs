import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {mixSeed} from './dev/simulator/runner.js';
import {runStrategyMatch} from './dev/simulator/strategy-match.js';
import {validateModel,createSession} from '../models/deep.mjs';

const args=new Map(),argv=process.argv.slice(2);
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!['--model','--out','--purpose','--blocks'].includes(argv[i])||args.has(argv[i]))throw new Error('Invalid option');args.set(argv[i],argv[i+1]);}
if(!args.get('--model')||!args.get('--out')||args.get('--purpose')!=='smoke'||args.get('--blocks')!=='2')throw new Error('Only two-block smoke is supported');
const hash=x=>createHash('sha256').update(x).digest('hex');
const bytes=await readFile(args.get('--model')),source=JSON.parse(bytes);
if(source.schema!=='p-rl1024-deep-training-v1'||source.completedMatches!==source.history?.length||!Array.isArray(source.history)||source.teacher!=='none')throw new Error('Invalid training result');
const model=validateModel(source.model),modelSha256=hash(JSON.stringify(model));
if(modelSha256!==source.modelSha256)throw new Error('Model digest mismatch');
const trainSeeds=new Set(source.history.map(x=>x.seed));
const seeds=[0,1].map(i=>mixSeed(976442,i,0x9762));
if(seeds.some(s=>trainSeeds.has(s))||seeds[0]===seeds[1])throw new Error('Evaluation seed overlap');
const records=[];
for(let block=0;block<2;block++)for(const rlSeat of [0,1]){
  const result=runStrategyMatch({seed:seeds[block],policies:['pure-rl-linear','pure-rl-linear'],strategyMode:'p2',pureRlSessions:[createSession(model),createSession(model)],pureRlAllPhases:false});
  if(!result.completed||result.score[result.winner]!==4||result.pureRlStats.some(s=>s.train||s.updates))throw new Error('Invalid frozen self-play evaluation');
  records.push({block,seed:seeds[block],rlSeat,winner:result.winner,won:Number(result.winner===rlSeat),rounds:result.rounds,actionCounts:result.actionCounts,score:result.score});
}
const report={schema:'p-rl1024-deep-frozen-paired-v1',purpose:'smoke',formalEligible:false,comparison:'RL-Deep frozen self-play; Basic draft and selection',modelSha256,modelFileSha256:hash(bytes),sourceSha256:source.sourceSha256,blocks:2,completedMatches:records.length,seedDigestSha256:hash(JSON.stringify(seeds)),outcomeDigestSha256:hash(JSON.stringify(records)),records};
await mkdir(dirname(resolve(args.get('--out'))),{recursive:true});await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({modelSha256,outcomeDigestSha256:report.outcomeDigestSha256,completedMatches:records.length}));
