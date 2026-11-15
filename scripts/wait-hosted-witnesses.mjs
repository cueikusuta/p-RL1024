// Collect lightweight, prevalidated per-shard witnesses only after GitHub
// records the matching immutable full checkpoint artifact for this run attempt.
import {spawn} from 'node:child_process';
import {mkdirSync,rmSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
const repo=process.env.GITHUB_REPOSITORY,runId=process.env.GITHUB_RUN_ID,
  runAttempt=process.env.GITHUB_RUN_ATTEMPT,token=process.env.GH_TOKEN;
if(repo!=='cueikusuta/p-RL1024'||!/^\d+$/.test(runId??'')||
   !/^\d+$/.test(runAttempt??'')||!token||
   process.argv.length!==3||!process.argv[2])throw Error('Invalid witness collection context');
const destination=process.argv[2],n=18;
const full=i=>`i1-fast-matrix-${runId}-${runAttempt}-shard-${i}`;
const witness=i=>`i1-fast-witness-${runId}-${runAttempt}-shard-${i}`;
const url=`https://api.github.com/repos/${repo}/actions/runs/${runId}/artifacts?per_page=100`;
const downloaded=new Map(),fullMetadata=new Map();
mkdirSync(destination,{recursive:true});
function pull(name){
  return new Promise((resolve,reject)=>{
    const child=spawn('gh',['run','download',runId,'--repo',repo,'--name',name,
      '--dir',join(destination,name)],{env:process.env,stdio:'inherit'});
    let hadError=false;
    child.once('error',e=>{hadError=true;reject(e);});
    child.once('close',(code,signal)=>{
      if(hadError)return;
      if(code!==0)reject(Error('Witness download '+name+' '+code+'/'+signal));
      else resolve(name);
    });
  });
}
async function retry(name){
  for(let i=0;i<5;i++){
    try{return await pull(name);}
    catch(error){
      rmSync(join(destination,name),{recursive:true,force:true});
      if(i===4)throw error;
      await new Promise(resolve=>setTimeout(resolve,300*(i+1)));
    }
  }
}
const jobs=[];
for(let attempt=0;attempt<240&&(downloaded.size<n||fullMetadata.size<n);attempt++){
  const res=await fetch(url,{headers:{Authorization:`Bearer ${token}`,
    Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}});
  if(!res.ok)throw Error('Cannot list artifacts HTTP '+res.status);
  const body=await res.json();
  for(const artifact of body.artifacts??[]){
    if(artifact.expired||artifact.workflow_run?.id!==Number(runId))continue;
    for(let i=0;i<n;i++){
      if(artifact.name===full(i)&&!fullMetadata.has(i)){
        if(!Number.isSafeInteger(artifact.size_in_bytes)||artifact.size_in_bytes<=0||
           !/^sha256:[a-f0-9]{64}$/.test(artifact.digest??''))
          throw Error('Full artifact has invalid immutable digest '+i);
        fullMetadata.set(i,{name:artifact.name,id:artifact.id,
          digest:artifact.digest,size:artifact.size_in_bytes});
      }
      if(artifact.name===witness(i)&&!downloaded.has(i)){
        downloaded.set(i,artifact.name);
        jobs.push(retry(artifact.name).then(()=>({ok:true,name:artifact.name}),
          error=>({ok:false,name:artifact.name,error:String(error)})));
      }
    }
  }
  if(downloaded.size===n&&fullMetadata.size===n)break;
  await new Promise(resolve=>setTimeout(resolve,500));
}
if(downloaded.size!==n||fullMetadata.size!==n)
  throw Error('Missing expected immutable full or witness artifact');
const result=await Promise.all(jobs);
if(result.some(x=>!x.ok))throw Error('Witness artifact download failure '+JSON.stringify(result.filter(x=>!x.ok)));
writeFileSync(join(destination,'full-artifacts.json'),JSON.stringify({
  runId,runAttempt,artifacts:[...fullMetadata.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1])
})+'\n');
console.log('FAST_WITNESSES_COLLECTED '+JSON.stringify({count:result.length,
  fullArtifacts:fullMetadata.size}));
