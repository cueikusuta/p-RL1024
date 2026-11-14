import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {validateModel,match,mixSeed,weightsSha256} from '../models/aq-runner.mjs';

const args=new Map(),argv=process.argv.slice(2),allowed=['--model','--out','--purpose','--blocks'];
if(argv.length%2)throw new Error('Expected option/value pairs');
for(let i=0;i<argv.length;i+=2){if(!allowed.includes(argv[i])||args.has(argv[i]))throw new Error('Invalid option');args.set(argv[i],argv[i+1]);}
if(!args.get('--model')||!args.get('--out')||args.get('--purpose')!=='smoke'||args.get('--blocks')!=='2')throw new Error('Only two-block smoke is supported');
const sha=x=>createHash('sha256').update(x).digest('hex'),bytes=await readFile(args.get('--model')),source=JSON.parse(bytes);
if(source.schema!=='p-rl1024-aq-training-v1'||!['RL-Linear','RL-Deep'].includes(source.modelId)||source.trainingOpponent!=='AQ-Base-001'||source.opponentModelSha256!==weightsSha256||source.teacher!=='none'||!Array.isArray(source.history)||source.completedMatches!==source.history.length)throw new Error('Invalid AQ training result');
const model=validateModel(source.modelId,source.model),modelSha256=sha(JSON.stringify(model));
if(modelSha256!==source.modelSha256)throw new Error('Student model SHA mismatch');
const trainSeeds=new Set(source.history.map(x=>x.seed)),seeds=[0,1].map(i=>mixSeed(976882,i,0xa976));
if(trainSeeds.size!==source.history.length||seeds.some(s=>trainSeeds.has(s))||seeds[0]===seeds[1])throw new Error('Evaluation seed overlap');
const records=[];
for(let block=0;block<2;block++)for(const rlSeat of [0,1]){
  const result=match({id:source.modelId,model,seed:seeds[block],seat:rlSeat});
  if(result.pureRlStats[rlSeat]?.updates!==0||result.pureRlStats[rlSeat]?.train)throw new Error('Frozen evaluation updated weights');
  records.push({block,seed:seeds[block],rlSeat,winner:result.winner,won:Number(result.winner===rlSeat),rounds:result.rounds,actionCounts:result.actionCounts,score:result.score});
}
const report={schema:'p-rl1024-aq-paired-v1',purpose:'smoke',formalEligible:false,comparison:`${source.modelId} vs AQ-Base-001`,opponentId:'AQ-Base-001',opponentModelSha256:weightsSha256,modelSha256,modelFileSha256:sha(bytes),sourceSha256:source.sourceSha256,blocks:2,completedMatches:records.length,seedDigestSha256:sha(JSON.stringify(seeds)),outcomeDigestSha256:sha(JSON.stringify(records)),records};
await mkdir(dirname(resolve(args.get('--out'))),{recursive:true});await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({modelSha256,opponentModelSha256:weightsSha256,outcomeDigestSha256:report.outcomeDigestSha256,completedMatches:records.length}));
