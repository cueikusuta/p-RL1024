// Read-only GitHub Actions artifact harvest: start downloading each immutable
// peer artifact as soon as GitHub publishes it. Keep all checkpoint files and
// never mark the leader aggregate valid until every download succeeds.
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdirSync,rmSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
const repo=process.env.GITHUB_REPOSITORY;
const runId=process.env.GITHUB_RUN_ID;
const runAttempt=process.env.GITHUB_RUN_ATTEMPT;
const token=process.env.GH_TOKEN;
if(repo!=='cueikusuta/p-RL1024'||!/^[0-9]+$/.test(runId??'')||!/^[0-9]+$/.test(runAttempt??'')||!token)
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
  `i1-fast-matrix-${runId}-${runAttempt}-shard-${i+firstIndex}`);
const expectedNames=new Set(expected);
const downloaded=new Set(),jobs=[];
const url=`https://api.github.com/repos/${repo}/actions/runs/${runId}/artifacts?per_page=100`;
function unzipArchive(path,output) {
  return new Promise((resolve,reject)=>{
    const child=spawn('unzip',['-q',path,'-d',output],{stdio:'inherit'});
    let failed=false;
    child.once('error',e=>{failed=true;reject(e);});
    child.once('close',(code,signal)=>{
      if(failed)return;
      if(code!==0)reject(new Error('Artifact unzip failed '+code+'/'+signal));
      else resolve();
    });
  });
}
async function downloadOnce(artifact) {
  const {name,digest,size_in_bytes,archive_download_url}=artifact;
  if(typeof archive_download_url!=='string'||
     !archive_download_url.startsWith(`https://api.github.com/repos/${repo}/actions/artifacts/`)||
     !/^sha256:[a-f0-9]{64}$/.test(digest??'')||
     !Number.isSafeInteger(size_in_bytes)||size_in_bytes<1||size_in_bytes>50000000)
    throw new Error('Invalid artifact identity or digest: '+name);
  const response=await fetch(archive_download_url,{headers:{
    Authorization:`Bearer ${token}`,Accept:'application/vnd.github+json',
    'X-GitHub-Api-Version':'2022-11-28'
  }});
  if(!response.ok)throw new Error('Artifact ZIP not yet available '+name+' HTTP '+response.status);
  const bytes=Buffer.from(await response.arrayBuffer());
  if(bytes.length!==size_in_bytes ||
     'sha256:'+createHash('sha256').update(bytes).digest('hex')!==digest)
    throw new Error('Artifact ZIP digest mismatch: '+name);
  const output=join(destination,name);
  const archive=join(tmpdir(),`i1-${runId}-${runAttempt}-${name}.zip`);
  mkdirSync(output,{recursive:true});
  try{
    writeFileSync(archive,bytes,{flag:'wx'});
    await unzipArchive(archive,output);
    return name;
  }finally{rmSync(archive,{force:true});}
}
// GitHub may list the immutable artifact before its ZIP is available.
// A failed attempt must leave no partial extraction accepted as valid.
async function download(artifact) {
  for(let attempt=0;attempt<5;attempt++){
    try{return await downloadOnce(artifact);}
    catch(error){
      rmSync(join(destination,artifact.name),{recursive:true,force:true});
      if(attempt===4)throw error;
      console.warn('FAST_PEER_RETRY '+artifact.name+' '+String(error));
      await new Promise(resolve=>setTimeout(resolve,300*(attempt+1)));
    }
  }
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
    jobs.push(download(artifact).then(()=>({ok:true,name:artifact.name}),
      error=>({ok:false,name:artifact.name,error:String(error)})));
  }
  if(downloaded.size===expected.length)break;
  await new Promise(resolve=>setTimeout(resolve,500));
}
if(downloaded.size!==expected.length)throw new Error('Missing peer artifacts after bounded wait');
const result=await Promise.all(jobs);
if(result.some(x=>!x.ok))throw new Error('Peer artifact download failed: '+JSON.stringify(result.filter(x=>!x.ok)));
console.log('FAST_ALL_PEER_ARTIFACTS_DOWNLOADED '+JSON.stringify(result.map(x=>x.name)));
