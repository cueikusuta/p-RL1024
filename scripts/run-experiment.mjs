import {spawnSync} from 'node:child_process';
import {mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {compare} from './verify-evaluation.mjs';

const matches = Number(process.env.RL_MATCHES || '9');
const seed = Number(process.env.RL_SEED || '975020');
if (!Number.isSafeInteger(matches) || matches < 9 || matches > 256 || !Number.isSafeInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('RL_MATCHES must be 9–256 and RL_SEED must be 1–4294967295');
const dir = 'artifacts/i975';
mkdirSync(dir,{recursive:true});
function run(script,args) {
  const result=spawnSync(process.execPath,[script,...args],{stdio:'inherit',timeout:290*60*1000});
  if (result.error || result.status !== 0) throw result.error || new Error(`${script} exited ${result.status}`);
}
const model=`${dir}/training.json`;
run('scripts/dev/simulator/pure-rl-sarsa-train.mjs',['--all-phases','true','--matches',String(matches),'--seed',String(seed),'--checkpoint-every','8','--out',model]);
for (const name of ['a','b']) run('scripts/dev/simulator/pure-rl-evaluate.mjs',['--model',model,'--purpose','smoke','--blocks','2','--out',`${dir}/eval-${name}.json`]);
const read = name => JSON.parse(readFileSync(`${dir}/${name}`,'utf8'));
const proof=compare(read('eval-a.json'),read('eval-b.json'));
proof.modelFileSha256=createHash('sha256').update(readFileSync(model)).digest('hex');
proof.matches=matches; proof.trainingSeed=seed; proof.node=process.version;
writeFileSync(`${dir}/comparison.json`,JSON.stringify(proof,null,2)+'\n');
console.log(JSON.stringify(proof,null,2));
