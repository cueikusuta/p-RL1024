// I975 research-only strict, deterministic continuation of complete Match
// training. These snapshots contain OWN model weights and public aggregate
// results only, never opponent hidden observations or unfinished Matches.
import {createHash,randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,rename,rm} from 'node:fs/promises';
import {dirname} from 'node:path';
import {validatePureRlModel} from '../../../src/ai/pure-rl-linear.js';
import {mixSeed} from './runner.js';

export const PURE_RL_RESUME_SCHEMA='i975-exact-full-match-resume-v1';
const sha256=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const boundedInteger=(n,min,max)=>Number.isSafeInteger(n)&&n>=min&&n<=max;

function validateResults(report,plannedMatches){
  if(!report||report.schema!=='i975-pure-rl-battle-train-result-v1'||
     report.trainingNotHoldout!==true||
     !Array.isArray(report.matches)||
     !boundedInteger(report.completedMatches,1,plannedMatches)||
     report.matches.length!==report.completedMatches||
     !boundedInteger(report.seedRoot,1,0xffffffff)||
     !Number.isFinite(report.epsilon)||
     !Number.isFinite(report.epsilonFinal)||
     !boundedInteger(report.checkpointEvery,1,plannedMatches)||
     !Number.isSafeInteger(report.studentWins)||
     !Number.isSafeInteger(report.trainedDecisions)||
     !Number.isSafeInteger(report.updates)||
     !/^[a-f0-9]{64}$/.test(report.sourceSha256??'')||
     !/^[a-f0-9]{64}$/.test(report.modelSha256??''))
    throw new TypeError('Incomplete pure RL checkpoint report');
  validatePureRlModel(report.model);
  if(sha256(report.model)!==report.modelSha256)
    throw new TypeError('Training model checksum mismatch');
  let wins=0,decisions=0,updates=0;
  for(let i=0;i<report.matches.length;i++){
    const row=report.matches[i],block=Math.floor(i/8);
    const checkpointIndex=block<2?block:
      mixSeed(report.seedRoot,i,0x5e17)%(block+1);
    const expectedEpsilon=report.epsilon+
      (report.epsilonFinal-report.epsilon)*(plannedMatches===1?0:i/(plannedMatches-1));
    if(!row||row.seed!==mixSeed(report.seedRoot,i,0x975)||
       row.studentSeat!==i%2||
       row.opponentCheckpointIndex!==checkpointIndex||
       !boundedInteger(row.winner,0,1)||
       !Array.isArray(row.score)||row.score.length!==2||
       row.score.some(n=>!boundedInteger(n,0,4))||
       row.score[row.winner]!==4||row.won!==(row.winner===row.studentSeat)||
       !boundedInteger(row.rounds,4,64)||
       !boundedInteger(row.decisions,1,300000)||
       !boundedInteger(row.updates,1,300000)||
       !Number.isFinite(row.policyEpsilon)||
       Math.abs(row.policyEpsilon-expectedEpsilon)>1e-12)
      throw new TypeError('Corrupt seed, seat, opponent or result in resume history');
    wins+=Number(row.won);decisions+=row.decisions;updates+=row.updates;
  }
  if(wins!==report.studentWins||decisions!==report.trainedDecisions||
     updates!==report.updates)
    throw new TypeError('Resume history totals do not match the model report');
}

export function makePureRlResumeCheckpoint(report,{plannedMatches,past}){
  if(!boundedInteger(plannedMatches,1,200000)||
     !Array.isArray(past)||
     past.length!==Math.ceil(report.completedMatches/8))
    throw new TypeError('Incomplete checkpoint frozen league state');
  validateResults(report,plannedMatches);
  for(const model of past)validatePureRlModel(model);
  const resumeState={
    schema:PURE_RL_RESUME_SCHEMA,plannedMatches,
    // These snapshots are essential: without them an interrupted run silently
    // trains against a different checkpoint-opponent distribution.
    past:structuredClone(past)
  };
  const result={...report,resumeState};
  return {...result,resumeDigestSha256:sha256(result)};
}

export function restorePureRlResumeCheckpoint(data,expected){
  if(!data||!data.resumeState||
     data.resumeState.schema!==PURE_RL_RESUME_SCHEMA||
     !Array.isArray(data.resumeState.past)||
     !boundedInteger(data.resumeState.plannedMatches,1,200000)||
     data.resumeState.plannedMatches!==expected.plannedMatches||
     data.algorithm!==expected.algorithm||
     data.seedRoot!==expected.seedRoot||
     data.sourceSha256!==expected.sourceSha256||
     data.allPhases!==expected.allPhases||
     data.epsilon!==expected.epsilon||
     data.epsilonFinal!==expected.epsilonFinal||
     data.checkpointEvery!==expected.checkpointEvery)
    throw new TypeError('Resume configuration/source/algorithm mismatch');
  const {resumeDigestSha256,...payload}=data;
  if(!/^[a-f0-9]{64}$/.test(resumeDigestSha256??'')||
     sha256(payload)!==resumeDigestSha256)
    throw new TypeError('Modified or incomplete resume checkpoint');
  validateResults(data,data.resumeState.plannedMatches);
  if(data.resumeState.past.length!==Math.ceil(data.completedMatches/8)||
     data.resumeState.past.length>25000)
    throw new TypeError('Incorrect frozen checkpoint history length');
  for(const frozen of data.resumeState.past)validatePureRlModel(frozen);
  return {
    completedMatches:data.completedMatches,
    model:structuredClone(data.model),
    past:structuredClone(data.resumeState.past),
    matches:structuredClone(data.matches),
    studentWins:data.studentWins,
    trainedDecisions:data.trainedDecisions,
    updates:data.updates
  };
}

export async function readPureRlResumeCheckpoint(path,expected){
  const data=JSON.parse(await readFile(path,'utf8'));
  return restorePureRlResumeCheckpoint(data,expected);
}

export async function writePureRlResumeCheckpoint(path,report,state){
  const content=JSON.stringify(makePureRlResumeCheckpoint(report,state),null,2)+'\n';
  await mkdir(dirname(path),{recursive:true});
  const temporary=path+'.tmp-'+randomUUID();
  try{
    await writeFile(temporary,content,{flag:'wx'});
    await rename(temporary,path);
  }catch(error){
    await rm(temporary,{force:true});
    throw error;
  }
}
