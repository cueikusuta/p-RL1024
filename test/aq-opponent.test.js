import test from 'node:test';
import assert from 'node:assert/strict';
import {createModel,match,weightsSha256} from '../models/aq-runner.mjs';

for(const id of ['RL-Linear','RL-Deep'])for(const seat of [0,1]) {
  test(`${id} vs frozen AQ-Base in seat ${seat}`,{timeout:60000},()=>{
    const model=createModel(id,975020),before=JSON.stringify(model);
    const result=match({id,model,seed:976001+seat,seat});
    assert.equal(result.completed,true);assert.equal(result.score[result.winner],4);
    assert.equal(result.pureRlStats[seat].train,false);
    assert.equal(result.pureRlStats[seat].updates,0);
    assert.equal(JSON.stringify(model),before);
    assert.equal(weightsSha256,'5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e');
  });
}
