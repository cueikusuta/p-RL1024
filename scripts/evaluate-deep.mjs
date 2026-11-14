import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {mixSeed} from './dev/simulator/runner.js';
import {frozenDeepBatch} from './batch-frozen-deep.mjs';
import {validateModel} from '../models/deep.mjs';

const args=new Map(),argv=process.argv.slice(2);
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!['--model','--out','--purpose','--blocks','--workers'].includes(argv[i])||args.has(argv[i]))throw new Error('Invalid option');args.set(argv[i],argv[i+1]);}
const purpose=args.get('--purpose'),blocks=Number(args.get('--blocks'));
const workers=Number(args.get('--workers')??(blocks>=8?4:1));
if(!args.get('--model')||!args.get('--out')||
  !((purpose==='smoke'&&blocks===2)||(purpose==='throughput'&&Number.isSafeInteger(blocks)&&blocks>=8&&blocks<=128))||
  !Number.isSafeInteger(workers)||workers<1||workers>16)
  throw new Error('Smoke needs two blocks; throughput needs 8–128 blocks and 1–16 workers');
const hash=x=>createHash('sha256').update(x).digest('hex');
const bytes=await readFile(args.get('--model')),source=JSON.parse(bytes);
if(source.schema!=='p-rl1024-deep-training-v1'||source.completedMatches!==source.history?.length||!Array.isArray(source.history)||source.teacher!=='none')throw new Error('Invalid training result');
const model=validateModel(source.model),modelSha256=hash(JSON.stringify(model));
if(modelSha256!==source.modelSha256)throw new Error('Model digest mismatch');
const trainSeeds=new Set(source.history.map(x=>x.seed));
const seeds=Array.from({length:blocks},(_,i)=>mixSeed(976442,i,0x9762));
if(seeds.some(s=>trainSeeds.has(s))||new Set(seeds).size!==blocks)throw new Error('Evaluation seed overlap');
const tasks=[];
for(let block=0;block<blocks;block++)for(const rlSeat of [0,1])
  tasks.push({block,rlSeat,seed:seeds[block]});
const results=await frozenDeepBatch(model,tasks.map(t=>t.seed),{workers});
const records=results.map((result,i)=>{
  const {block,seed,rlSeat}=tasks[i];
  return {block,seed,rlSeat,winner:result.winner,won:Number(result.winner===rlSeat),
    rounds:result.rounds,actionCounts:result.actionCounts,score:result.score};
});
const report={schema:'p-rl1024-deep-frozen-paired-v1',purpose,
  formalEligible:false,comparison:'RL-Deep frozen self-play; Basic draft and selection',
  modelSha256,modelFileSha256:hash(bytes),sourceSha256:source.sourceSha256,
  blocks,completedMatches:records.length,seedDigestSha256:hash(JSON.stringify(seeds)),
  outcomeDigestSha256:hash(JSON.stringify(records)),records};
await mkdir(dirname(resolve(args.get('--out'))),{recursive:true});
await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({modelSha256,outcomeDigestSha256:report.outcomeDigestSha256,
  completedMatches:records.length,workers}));
