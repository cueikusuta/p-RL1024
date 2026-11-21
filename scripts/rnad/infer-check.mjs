// One-shot parity command for the *persistent-in-memory* inference engine.
// The runner uses this same class repeatedly throughout each live Match.
import {readFileSync} from 'node:fs';
import {PersistentRNADInference} from './infer.mjs';
const [modelFile,inputFile]=process.argv.slice(2);
if(!modelFile||!inputFile)throw new Error('Usage: infer-check MODEL.json INPUT.json');
const fixture=JSON.parse(readFileSync(inputFile,'utf8'));
const net=new PersistentRNADInference(modelFile);
if(fixture.memory){
  if(!Array.isArray(fixture.memory)||fixture.memory.length!==128||
     fixture.memory.some(x=>typeof x!=='number'||!Number.isFinite(x)))
    throw new TypeError('Invalid fixture memory');
  net.memory[0]=[...fixture.memory];
}
const result=net.infer(fixture.state,fixture.actions,0);
process.stdout.write(JSON.stringify(result)+'\n');
