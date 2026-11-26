import {createIntegratedAQSession} from '../integrated-selective-session.mjs';
import {createFrozenPriorityProfile} from './frozen-priority-profile.mjs';
import {estimateAttackPercent} from '../../../src/ai/matchup-assessment.js';
import {calculateChampionsStats} from '../../../src/champions-stats.js';
import data from '../../../aq-runtime/data/catalog/battle-data.json' with {type:'json'};

export function createFrozenPrioritySession({modelDir,seat,mode='off',onAccepted}){
 const profile=createFrozenPriorityProfile({mode,onAccepted,deps:{data,estimateAttackPercent,calculateChampionsStats}});
 const actor=createIntegratedAQSession({modelDir,seat,researchRerank:profile.needsPublicFrame?profile.decide:null});
 if(mode==='off')return actor;
 let proposal=null;
 return {...actor,
    select(...args){proposal=actor.select(...args);return proposal;},
    commit(){actor.commit();profile.accept(proposal);proposal=null;}
 };
}
