import test from 'node:test';
import assert from 'node:assert/strict';
import {legalPureRlActions,pureRlFeatures} from '../src/ai/pure-rl-linear.js';
import {trustedFeatures} from '../models/deep.mjs';

function view(requestType='turn'){
  return {
    phase:requestType,requestType,turn:32,
    self:{active:0,confirmed:false,needsSwitch:false,canSwitch:true,canMega:true,
      team:[
        {hp:55,maxHP:100,status:'par',speciesId:'pikachu',types:['Electric'],
          moves:[{id:'thunderbolt',disabled:false},
            {id:'revivalblessing',disabled:false},
            {id:'protect',disabled:true},{id:'voltswitch',disabled:false}]},
        {hp:71,maxHP:150,status:'',speciesId:'blastoise',types:['Water'],moves:[]},
        {hp:0,maxHP:110,status:'brn',speciesId:'charizard',types:['Fire','Flying'],moves:[]},
      ]},
    opponent:{hpPercent:31,status:'psn',remaining:2,speciesId:'gengar',
      types:['Ghost','Poison']},
    field:{weather:'raindance',terrain:'electricterrain',trickRoomTurns:2},
  };
}
for(const kind of ['turn','replacement','pivot']){
  test('all '+kind+' legal actions yield identical public sparse features',()=>{
    const input=view(kind);
    for(const action of legalPureRlActions(input)){
      assert.deepEqual(trustedFeatures(input,action),pureRlFeatures(input,action));
      assert.deepEqual(Object.keys(trustedFeatures(input,action)),
        Object.keys(pureRlFeatures(input,action)));
    }
  });
}
test('trick room, statuses and hidden lack of information keep exact keys',()=>{
  const input=view();
  input.field={};input.opponent={hpPercent:0,remaining:0};
  input.self.team[0].status=null;
  for(const action of legalPureRlActions(input))
    assert.deepEqual(trustedFeatures(input,action),pureRlFeatures(input,action));
});
