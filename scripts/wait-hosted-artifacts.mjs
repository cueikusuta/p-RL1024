// In a 4-to-20 shard research benchmark, shard 0 aggregates on its own
// runner after all three peer artifacts become visible. It never trusts a
// partially uploaded or mismatched artifact, and has read-only Actions access.
const repo=process.env.GITHUB_REPOSITORY;
const runId=process.env.GITHUB_RUN_ID;
const token=process.env.GH_TOKEN;
if(repo!=='cueikusuta/p-RL1024'||!/^\d+$/.test(runId??'')||!token)
  throw new Error('Expected scoped GitHub Actions environment');
const shardCount=Number(process.env.RL_SHARDS||4);
if(!Number.isSafeInteger(shardCount)||shardCount<2||shardCount>20)
  throw new Error('Invalid peer shard count');
const expected=Array.from({length:shardCount-1},(_,i)=>`i1-fast-matrix-${runId}-shard-${i+1}`);
const url=`https://api.github.com/repos/${repo}/actions/runs/${runId}/artifacts?per_page=100`;
let found=false;
for(let attempt=0;attempt<60;attempt++){
  const response=await fetch(url,{headers:{
    Authorization:`Bearer ${token}`,Accept:'application/vnd.github+json',
    'X-GitHub-Api-Version':'2022-11-28'
  }});
  if(!response.ok)throw new Error('Failed to list shard artifacts: '+response.status);
  const payload=await response.json();
  const present=new Set((payload.artifacts??[]).filter(a=>!a.expired&&a.workflow_run?.id===Number(runId)).map(a=>a.name));
  if(expected.every(name=>present.has(name))){
    console.log('FAST_PEER_ARTIFACTS_READY '+JSON.stringify(expected));
    found=true;break;
  }
  await new Promise(resolve=>setTimeout(resolve,2000));
}
if(!found)throw new Error('Timeout waiting for all three finished peer shard artifacts');
