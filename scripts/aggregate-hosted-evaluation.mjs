import {createHash} from 'node:crypto';
import {readFileSync,readdirSync,mkdirSync,writeFileSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {mixSeed} from '../models/aq-runner.mjs';
import {verifyReportTiming} from './hosted-proof.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
const args=process.argv.slice(2);
if(args.length!==6||args[0]!=='--serial'||args[2]!=='--shards'||args[4]!=='--out')throw new Error('Expected --serial file --shards directory --out file');
const serial=JSON.parse(readFileSync(args[1],'utf8'));
const shardDirs=readdirSync(args[3],{withFileTypes:true}).filter(x=>x.isDirectory()).map(x=>x.name);
const shards=shardDirs.map(x=>JSON.parse(readFileSync(join(args[3],x,'report.json'),'utf8'))).sort((a,b)=>a.offset-b.offset);
function verify(r){
  verifyReportTiming(r);
  if(r.schema!=='p-rl1024-hosted-frozen-evaluation-v1'||r.opponent!=='AQ-Base-001'||
    !['RL-Linear','RL-Deep'].includes(r.model)||r.modelStatus!=='deterministic-initial-frozen'||
    !Number.isSafeInteger(r.offset)||r.offset<0||!Number.isSafeInteger(r.count)||r.count<1||
    !Number.isSafeInteger(r.total)||r.offset+r.count>r.total||
    !Array.isArray(r.records)||r.records.length!==r.count||sha(JSON.stringify(r.records))!==r.outcomeDigestSha256||
    !/^[a-f0-9]{64}$/.test(r.modelSha256)||!/^[a-f0-9]{64}$/.test(r.modelFileSha256)||
    !/^[a-f0-9]{64}$/.test(r.opponentModelSha256))
    throw new Error('Invalid shard schema/digest');
  r.records.forEach((x,i)=>{
    const index=r.offset+i;
    if(x.index!==index||x.seed!==mixSeed(r.seedRoot,index,0xe975)||x.seat!==index%2||
      ![0,1].includes(x.winner)||x.won!==Number(x.winner===x.seat)||
      x.score?.[x.winner]!==4||!Number.isInteger(x.rounds)||x.rounds<4)
      throw new Error('Invalid shard Match '+index);
  });
  if(new Set(r.records.map(x=>x.seed)).size!==r.count||r.wins!==r.records.reduce((n,x)=>n+x.won,0))
    throw new Error('Shard duplicate seeds/wins mismatch');
}
verify(serial);
if(serial.offset!==0||serial.count!==serial.total||shards.length<1||shards.length>20)throw new Error('Invalid serial or shard count');
const all=[];
let next=0;
for(const s of shards) {
  verify(s);
  if(s.offset!==next)throw new Error('Gap or duplicate shard');
  for(const k of ['schema','model','opponent','seedRoot','total','modelSha256','modelFileSha256','opponentModelSha256','sourceCommit'])
    if(s[k]!==serial[k])throw new Error('Mismatched shard '+k);
  next+=s.count;all.push(...s.records);
}
if(next!==serial.total||new Set(all.map(x=>x.seed)).size!==serial.total)
  throw new Error('Missing Match or duplicate global seed');
const digest=sha(JSON.stringify(all));
if(digest!==serial.outcomeDigestSha256||JSON.stringify(all)!==JSON.stringify(serial.records))
  throw new Error('Serial and parallel outcome digest diverged');
const parallelStart=Math.min(...shards.map(x=>x.startedAtMs));
const parallelFinish=Math.max(...shards.map(verifyReportTiming));
const parallelSpanMs=parallelFinish-parallelStart;
const speedup=serial.durationMs/parallelSpanMs;
const report={schema:'p-rl1024-hosted-evaluation-aggregate-v1',model:serial.model,opponent:serial.opponent,
  seedRoot:serial.seedRoot,total:serial.total,modelSha256:serial.modelSha256,modelFileSha256:serial.modelFileSha256,
  opponentModelSha256:serial.opponentModelSha256,sourceCommit:serial.sourceCommit,runId:serial.runId,
  serialDurationMs:serial.durationMs,parallelWallclockMs:parallelSpanMs,speedup,
  serialWins:serial.wins,parallelWins:all.reduce((n,x)=>n+x.won,0),outcomeDigestSha256:digest,
  shards:shards.map(x=>({offset:x.offset,count:x.count,seedStart:x.seedStart,seedEnd:x.seedEnd,workers:x.workers,
    durationMs:x.durationMs,startedAtMs:x.startedAtMs,wins:x.wins,outcomeDigestSha256:x.outcomeDigestSha256}))};
mkdirSync(dirname(args[5]),{recursive:true});writeFileSync(args[5],JSON.stringify(report,null,2)+'\n');
console.log('HOSTED_AGGREGATE '+JSON.stringify(report));
if(!(speedup>1))throw new Error('Parallel wallclock was not faster; inspect evidence');
