// Read-only GitHub Actions artifact harvest: start downloading each immutable
// peer artifact as soon as GitHub publishes it. Keep all checkpoint files and
// never mark the leader aggregate valid until every download succeeds.
import {spawn} from 'node:child_process';
import {mkdirSync} from 'node:fs';
import {join} from 'node:path';
const repo=process.env.GITHUB_REPOSITORY;
const runId=process.env.GITHUB_RUN_ID;
const token=process.env.GH_TOKEN;
if(repo!=='cueikusuta/p-RL1024'||!/^[0-9]+$/.test(runId??'')||!token)
  throw new Error('Expected scoped GitHub Actions environment');
const shardCount=Number(process.env.RL_SHARDS||4);
if(!Number.isSafeInteger(shardCount)||shardCount<2||shardCount>20)
  throw new Error('Invalid peer shard count');
if(process.argv.length!==4||process.argv[2]!=='--download'||!process.argv[3])
  throw new Error('Expected --download output-directory');
const destination=process.argv[3];
const firstIndex=Number(process.env.RL_FIRST_ARTIFACT_INDEX??1);
if(![0,1].includes(firstIndex))throw new Error('Invalid first artifact index');
const expected=Array.from({length:shardCount-firstIndex},(_,i)=>
  `i1-fast-matrix-${runId}-shard-${i+firstIndex}`);
const expectedNames=new Set(expected);
const downloaded=new Set(),jobs=[];
const url=`https://api.github.com/repos/${repo}/actions/runs/${runId}/artifacts?per_page=100`;
function download(name) {
  const output=join(destination,name);
  mkdirSync(output,{recursive:true});
  return new Promise((resolve,reject)=>{
    const child=spawn('gh',['run','download',runId,'--repo',repo,
      '--name',name,'--dir',output],{stdio:'inherit',env:process.env});
    let failed=false;
    child.once('error',error=>{failed=true;reject(error);});
    child.once('close',(code,signal)=>{
      if(failed)return;
      if(code!==0)reject(new Error('Artifact download failed: '+name+' '+code+'/'+signal));
      else resolve(name);
    });
  });
}
for(let attempt=0;attempt<240&&downloaded.size<expected.length;attempt++) {
  const response=await fetch(url,{headers:{
    Authorization:`Bearer ${token}`,Accept:'application/vnd.github+json',
    'X-GitHub-Api-Version':'2022-11-28',
  }});
  if(!response.ok)throw new Error('Failed to list shard artifacts: '+response.status);
  const payload=await response.json();
  for(const artifact of payload.artifacts??[]){
    if(artifact.expired||artifact.workflow_run?.id!==Number(runId)||
       !expectedNames.has(artifact.name)||downloaded.has(artifact.name))continue;
    downloaded.add(artifact.name);
    console.log('FAST_PEER_READY '+artifact.name);
    // Record errors rather than creating a detached unhandled rejection.
    jobs.push(download(artifact.name).then(()=>({ok:true,name:artifact.name}),
      error=>({ok:false,name:artifact.name,error:String(error)})));
  }
  if(downloaded.size===expected.length)break;
  await new Promise(resolve=>setTimeout(resolve,500));
}
if(downloaded.size!==expected.length)throw new Error('Missing peer artifacts after bounded wait');
const result=await Promise.all(jobs);
if(result.some(x=>!x.ok))throw new Error('Peer artifact download failed: '+JSON.stringify(result.filter(x=>!x.ok)));
console.log('FAST_ALL_PEER_ARTIFACTS_DOWNLOADED '+JSON.stringify(result.map(x=>x.name)));
