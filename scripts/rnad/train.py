"""RNAD-001 bounded actual Match → NeuRD/V-trace → PyTorch update loop.

Research smoke, not a Champion promotion. Requires a GitHub checkout and
only complete authority-accepted two-seat Match trajectories.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch

from scripts.rnad.neural import (RNADNetwork, freeze_magnet, export_json,
                                 load_exported_json, BATTLE_FEATURE_SCHEMA,
                                 THREE_PHASE_FEATURE_SCHEMA,
                                 THREE_PHASE_FEATURE_SCHEMA_V2,
                                 THREE_PHASE_FEATURE_SCHEMA_V3,
                                 STATE_FEATURES_V3,ACTION_FEATURES_V3,
                                 migrate_v2_export_to_v3)
from scripts.rnad.two_player import rnad_targets, neurd_policy_loss
from scripts.rnad.checkpoint import save_checkpoint, restore_checkpoint


def current_git_sha():
    return subprocess.check_output(
        ['git','rev-parse','HEAD'], text=True, timeout=5
    ).strip()


def frozen_engine_sha():
    return json.loads(Path('provenance/manifest.json').read_text())['sourceCommit']


def state_hash(net):
    digest=hashlib.sha256()
    for key,value in sorted(net.state_dict().items()):
        digest.update(key.encode('utf8')+b'\\0')
        digest.update(str(tuple(value.shape)).encode('ascii')+b'\\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_match(seed, model, magnet, directory, *,
              three_phase=False,features_v2=False,features_v3=False,
              switch_guard='none',switch_cost='none'):
    directory.mkdir(parents=True,exist_ok=True)
    model_path=directory/'actor.json'
    magnet_path=directory/'magnet.json'
    output=directory/(f'accepted-seed{seed}.json')
    feature_schema=(THREE_PHASE_FEATURE_SCHEMA_V3 if features_v3 else
                    THREE_PHASE_FEATURE_SCHEMA_V2 if features_v2 else
                    THREE_PHASE_FEATURE_SCHEMA if three_phase
                    else BATTLE_FEATURE_SCHEMA)
    export_json(model,model_path,feature_schema=feature_schema)
    export_json(magnet,magnet_path,feature_schema=feature_schema)
    if output.exists():
        raise ValueError('Refusing to overwrite a prior authoritative Match')
    command=['node','scripts/rnad/export-trajectory.mjs',
             '--model',str(model_path),'--magnet',str(magnet_path),
             '--seed',str(seed),'--out',str(output)]
    if three_phase:
        command.extend(['--three-phase','true'])
    if features_v2:
        command.extend(['--features-v2','true'])
    if features_v3:
        command.extend(['--features-v3','true'])
    if switch_guard!='none':
        command.extend(['--switch-guard',switch_guard])
    if switch_cost!='none':
        command.extend(['--switch-cost',switch_cost])
    subprocess.run(command,check=True,timeout=1800)
    row=json.loads(output.read_text())
    guarded=switch_guard=='anti-loop-v1'
    opportunity=switch_cost=='opportunity-v1'
    if (row.get('schema')!=(
        'rnad-accepted-three-phase-v3-training-match-v1' if features_v3 else
        'rnad-accepted-three-phase-v2-opportunity-training-match-v1'
            if opportunity else
        'rnad-accepted-three-phase-v2-anti-loop-training-match-v1' if guarded
        else 'rnad-accepted-three-phase-v2-training-match-v1' if features_v2
        else 'rnad-accepted-three-phase-training-match-v1' if three_phase
        else 'rnad-accepted-training-match-v1') or
        (row.get('deployment_policy') if guarded else
         row.get('deployment_policy',None))!=(
            {'switch_guard':'anti-loop-v1'} if guarded else None) or
        row.get('feature_schema')!=feature_schema or
        row.get('completed') is not True or
        row.get('source_sha')!=current_git_sha() or
        row.get('engine_sha')!=frozen_engine_sha() or
        row.get('seed')!=seed or
        row.get('score',[0,0])[row['winner']]!=4 or
        len(row.get('events',[]))!=sum(row.get('action_counts',[]))+
        (4*row.get('rounds',0) if three_phase else 0) or
        not row.get('events')):
        raise ValueError('Untrusted, partial or mismatched Match training record')
    if guarded:
        for event in row['events']:
            actual=event.get('policy_eligible_action_count')
            authority=event.get('authoritative_legal_action_count')
            if (not isinstance(actual,int) or not isinstance(authority,int) or
                actual<1 or authority<actual or
                actual!=len(event.get('actions',[]))):
                raise ValueError('Incorrect guarded policy-support cardinality')
            if event.get('phase')!='battle' and actual!=authority:
                raise ValueError('Unauthorized nonbattle policy-support mask')
    if three_phase:
        counts={phase:sum(e.get('phase')==phase for e in row['events'])
                for phase in ('draft','selection','battle')}
        if (counts['draft']!=2*row['rounds'] or
            counts['selection']!=2*row['rounds'] or
            counts['battle']!=sum(row['action_counts'])):
            raise ValueError('Invalid three-phase accepted action coverage')
    elif any(e.get('phase')!='battle' for e in row['events']):
        raise ValueError('Unexpected non-battle decision in legacy learner')
    rewards=[entry['reward'] for entry in row['events']]
    if opportunity:
        shaping=row.get('reward_shaping')
        if (row.get('reward_objective')!='opportunity-v1' or
            not isinstance(shaping,dict) or
            shaping.get('objective')!='opportunity-v1' or
            shaping.get('max_per_switch')!=0.02 or
            shaping.get('max_per_seat')!=0.20 or
            not isinstance(shaping.get('cost_per_seat'),list) or
            len(shaping['cost_per_seat'])!=2):
            raise ValueError('Incorrect opportunity reward objective metadata')
        observed=[0.0,0.0]
        for i,e in enumerate(row['events']):
            c=e.get('switch_opportunity_cost')
            seat=e['actor']
            if (type(c) not in (float,int) or not math.isfinite(c) or
                not -1e-12<=c<=0.02+1e-12 or
                (c>1e-12 and (e.get('phase')!='battle' or
                    e.get('request_type')!='turn' or
                    e.get('action_kind')!='switch')) or
                len(e.get('reward',[]))!=2):
                raise ValueError('Invalid accepted opportunity switch cost')
            expected=[-c,c] if seat==0 else [c,-c]
            if i==len(rewards)-1:
                payoff=1 if row['winner']==0 else -1
                expected=[expected[0]+payoff,expected[1]-payoff]
            if any(abs(x-y)>1e-8 for x,y in zip(e['reward'],expected)):
                raise ValueError('Non-zero-sum or incorrect per-step switch reward')
            observed[seat]+=c
        if (any(abs(observed[seat]-shaping['cost_per_seat'][seat])>1e-8
                for seat in (0,1)) or
            any(x>0.20+1e-8 for x in observed)):
            raise ValueError('Opportunity reward budget overflow')
    elif (row.get('reward_objective') is not None or
          any(any(abs(x)>1e-10 for x in r) for r in rewards[:-1]) or
          rewards[-1] not in ([1,-1],[-1,1])):
        raise ValueError('Unexpected nonterminal or nonzero-sum reward')
    return row


def train_match(net, target, magnet, optimizer, scheduler, data, eta=.2):
    # The two policy states and the critic target are rolled forward only
    # on their own seat's information states. Hidden opponent moves never
    # become inputs to either recurrent Actor.
    net.train()
    target.eval()
    magnet.eval()
    memories=[net.initial_memory() for _ in range(2)]
    target_mem=[target.initial_memory() for _ in range(2)]
    magnet_mem=[magnet.initial_memory() for _ in range(2)]
    inputs=[]
    terms=[]
    for event in data['events']:
        seat=event['actor']
        if seat not in (0,1):
            raise ValueError('Invalid seat in accepted timeline')
        state=event['state'];actions=event['actions']
        if (len(state)!=net.state_features or not actions or
            any(len(action)!=net.action_features for action in actions) or
            not all(math.isfinite(float(x)) for x in state) or
            not all(math.isfinite(float(x)) for action in actions for x in action) or
            not 0 <= event['action_index'] < len(actions) or
            not 0 < event['behavior_prob'] <= 1):
            raise ValueError('Invalid feature/action/probability data')
        x=torch.tensor([state],dtype=torch.float32)
        a=torch.tensor([actions],dtype=torch.float32)
        logits,value,next_h=net(x,a,memories[seat])
        memories[seat]=next_h.detach()  # bounded truncated BPTT
        with torch.no_grad():
            t_logits,t_value,t_h=target(x,a,target_mem[seat])
            m_logits,_,m_h=magnet(x,a,magnet_mem[seat])
            target_mem[seat]=t_h
            magnet_mem[seat]=m_h
            t_pi=torch.softmax(t_logits[0],dim=-1)
            m_pi=torch.softmax(m_logits[0],dim=-1)
            if abs(math.log(float(m_pi[event['action_index']]))-
                   float(event['magnet_log_prob']))>1e-3:
                raise ValueError('Frozen magnet policy / Actor mismatch')
        terms.append({
            'actor':seat, 'reward':event['reward'],
            'pi':t_pi,'magnet':m_pi,
            'behavior_prob':float(event['behavior_prob']),
            'value':float(t_value[0]),'action_index':event['action_index']})
        inputs.append((logits[0],value[0]))
    targets=rnad_targets(terms,eta=eta,rho_bar=1,c_bar=1)
    policy_losses=[]
    value_losses=[]
    for (logits,value),est in zip(inputs,targets):
        policy_losses.append(neurd_policy_loss(logits,est['q_targets'],
                                                beta=2.0,q_clip=10.0,
                                                is_c=est['c']))
        value_losses.append((value-torch.tensor(
            est['value_target'],dtype=value.dtype))**2)
    loss=sum(policy_losses)/len(policy_losses)+(
         .5*sum(value_losses)/len(value_losses))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite NeuRD/V-trace objective')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),10.0,error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        for dst,src in zip(target.parameters(),net.parameters()):
            dst.mul_(.99).add_(src,alpha=.01)
    if any(not torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Nonfinite RNAD parameter update')
    return {'loss':float(loss.detach()),
            'critic_loss':float(sum(value_losses).detach()/len(value_losses)),
            'policy_loss':float(sum(policy_losses).detach()/len(policy_losses)),
            'accepted_decisions':len(terms),
            'behavior_ratio_max':max(x['ratio'] for x in targets)}


def train(*,seed=975021,matches=1,out='artifacts/rnad-001',resume=False,
          eta=.2,magnet_period=4,initial_model=None,three_phase=False,
          features_v2=False,features_v3=False,switch_guard='none',
          switch_cost='none',updates_per_match=1,migrate_from_v2=False):
    if (not isinstance(seed,int) or not 1 <= seed <= 0xffffffff or
        not isinstance(matches,int) or not 1 <= matches <= 16 or
        not isinstance(eta,(float,int)) or not math.isfinite(float(eta)) or
        not 0 < eta <= 1 or
        type(magnet_period) is not int or not 1 <= magnet_period <= 256 or
        type(three_phase) is not bool or
        type(features_v2) is not bool or (features_v2 and not three_phase) or
        type(features_v3) is not bool or
        (features_v3 and (features_v2 or not three_phase)) or
        type(updates_per_match) is not int or updates_per_match not in (1,4) or
        type(migrate_from_v2) is not bool or
        (migrate_from_v2 and not features_v3) or
        switch_guard not in ('none','anti-loop-v1') or
        (switch_guard!='none' and not
         (three_phase and (features_v2 or features_v3))) or
        switch_cost not in ('none','opportunity-v1') or
        (switch_cost!='none' and
            (switch_guard!='anti-loop-v1' or not
             (features_v2 or features_v3)))):
        raise ValueError('Training must bound seed, matches, eta and magnet period')
    torch.set_num_threads(1)
    path=Path(out)
    path.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);random.seed(seed)
    net=(RNADNetwork(state_features=STATE_FEATURES_V3,
                     action_features=ACTION_FEATURES_V3)
         if features_v3 else RNADNetwork())
    feature_schema=(THREE_PHASE_FEATURE_SCHEMA_V3 if features_v3 else
                    THREE_PHASE_FEATURE_SCHEMA_V2 if features_v2 else
                    THREE_PHASE_FEATURE_SCHEMA if three_phase
                    else BATTLE_FEATURE_SCHEMA)
    model_id=('RNAD-004-3phase-v3' if features_v3 else
              'RNAD-003-3phase-v2' if features_v2 else
              'RNAD-002-3phase' if three_phase else 'RNAD-001')+(
                  '-anti-loop-v1' if switch_guard!='none' else '')+(
                  '-opportunity-v1' if switch_cost!='none' else '')+(
                  '-u4' if updates_per_match==4 else '')
    initialization_sha256=(
        (migrate_v2_export_to_v3(net,initial_model)
         if migrate_from_v2 else
         load_exported_json(net,initial_model,
                            expected_feature_schema=feature_schema))
        if initial_model is not None else "none")
    target=copy.deepcopy(net)
    magnet=freeze_magnet(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=.98)
    checkpoint_dir=path/'checkpoints'
    prior=None
    completed=0
    counter=seed
    ids=[]
    if resume:
        restored=restore_checkpoint(
            checkpoint_dir,policy=net,value=target,magnet=magnet,
            optimizer=optimizer,scheduler=scheduler,
            expected={'model_id':model_id,
                      'initialization_sha256':initialization_sha256,
                      'regularization_eta':float(eta),
                      'magnet_period':magnet_period,
                      'source_sha':current_git_sha(),
                      'engine_snapshot_sha':frozen_engine_sha(),
                      'training_code_sha':current_git_sha(),
                      'observation_schema_version':feature_schema,
                      **({'optimizer_updates_per_match':updates_per_match,
                          'initialization_feature_schema':
                            THREE_PHASE_FEATURE_SCHEMA_V2 if migrate_from_v2
                            else feature_schema} if features_v3 else {}),
                      **({'switch_guard':switch_guard}
                         if switch_guard!='none' else {}),
                      **({'reward_objective':switch_cost}
                         if switch_cost!='none' else {})})
        metadata=restored['metadata']
        if metadata.get('initial_seed')!=seed:
            raise ValueError('Initial seed differs from previous run')
        completed=metadata['completed_matches']
        counter=metadata['next_match_seed']
        ids=list(metadata.get('completed_match_ids',[]))
        if len(ids)!=completed:raise ValueError('Completed-match manifest mismatch')
        prior=restored['checkpoint_sha256']
    summaries=[]
    for _ in range(matches):
        data=run_match(counter,net,magnet,path/'rollouts',
                       three_phase=three_phase,features_v2=features_v2,
                       features_v3=features_v3,
                       switch_guard=switch_guard,switch_cost=switch_cost)
        # Within a Match, later passes reuse accepted data off-policy with
        # recomputed clipped importance weights, not new self-play outcomes.
        epochs=[train_match(net,target,magnet,optimizer,scheduler,data,eta=eta)
                for _ in range(updates_per_match)]
        summary={**epochs[-1],
                 'optimizer_updates_this_match':updates_per_match,
                 'first_epoch_loss':epochs[0]['loss']}
        if switch_cost=='opportunity-v1':
            events=data['events']
            summary.update({
              'accepted_voluntary_switches':sum(
                e['action_kind']=='switch' and e['request_type']=='turn'
                for e in events),
              'shaped_switches':sum(
                e['switch_opportunity_cost']>1e-12 for e in events),
              'switch_cost_by_seat':data['reward_shaping']['cost_per_seat']})
        completed+=1
        ids.append(data['match_id'])
        if completed%magnet_period==0:
            magnet.load_state_dict(target.state_dict())
        prior_result=save_checkpoint(
            checkpoint_dir,policy=net,value=target,magnet=magnet,
            optimizer=optimizer,scheduler=scheduler,expected_prior_sha=prior,
            metadata={
                'model_id':model_id,
                'initialization_sha256':initialization_sha256,
                'regularization_eta':float(eta),
                'magnet_period':magnet_period,
                'source_sha':current_git_sha(),
                'engine_snapshot_sha':frozen_engine_sha(),
                'training_code_sha':current_git_sha(),
                'observation_schema_version':feature_schema,
                **({'optimizer_updates_per_match':updates_per_match,
                    'initialization_feature_schema':
                      THREE_PHASE_FEATURE_SCHEMA_V2 if migrate_from_v2
                      else feature_schema} if features_v3 else {}),
                **({'switch_guard':switch_guard}
                    if switch_guard!='none' else {}),
                **({'reward_objective':switch_cost}
                    if switch_cost!='none' else {}),
                'opponent_policy_version':(
                    'RNAD-004-v3-selfplay' if features_v3 else
                    'RNAD-003-v2-selfplay' if features_v2
                    else 'RNAD-001-selfplay')+
                    ('-anti-loop-v1' if switch_guard!='none' else '')+
                    ('-opportunity-v1' if switch_cost!='none' else ''),
                'completed_matches':completed,
                'accepted_decisions':summary['accepted_decisions']+
                    (sum(x['accepted_decisions'] for x in summaries)
                     if not resume else
                     metadata['accepted_decisions']+sum(x['accepted_decisions'] for x in summaries)),
                'updates':completed*updates_per_match,'next_match_seed':
                    ((counter%0xffffffff)+1),
                'last_match_id':data['match_id'],
                'completed_match_ids':ids,
                'initial_seed':seed,'boundary':'completed_match'})
        prior=prior_result['checkpoint_sha256']
        counter=(counter%0xffffffff)+1
        summaries.append({'seed':data['seed'],'winner':data['winner'],
                          'rounds':data['rounds'],**summary})
    export_json(net,path/'final-model.json',feature_schema=feature_schema)
    report={'schema':'rnad-three-phase-v3-training-report-v1' if features_v3
            else 'rnad-three-phase-v2-training-report-v1' if features_v2
            else 'rnad-three-phase-training-report-v1' if three_phase
            else 'rnad-smoke-training-report-v1',
            'feature_schema':feature_schema,'model_id':model_id,
            **({'switch_guard':switch_guard,
                'deployment_policy':{'switch_guard':switch_guard}}
               if switch_guard!='none' else {}),
            **({'reward_objective':switch_cost}
               if switch_cost!='none' else {}),
            'initialization_sha256':initialization_sha256,
            'regularization_eta':float(eta),
            'magnet_period':magnet_period,
            **({'optimizer_updates_per_match':updates_per_match,
                'initialization_feature_schema':
                  THREE_PHASE_FEATURE_SCHEMA_V2 if migrate_from_v2
                  else feature_schema} if features_v3 else {}),
            'source_sha':current_git_sha(),'engine_sha':frozen_engine_sha(),
            'completed_matches':completed,'updates':completed*updates_per_match,
            'last_match_checkpoint_sha':prior,
            'model_sha256':state_hash(net),
            'magnet_sha256':state_hash(magnet),
            'results':summaries,'is_champion':False}
    (path/'report.json').write_text(json.dumps(
        report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=975021)
    parser.add_argument('--matches',type=int,default=1)
    parser.add_argument('--out',default='artifacts/rnad-001')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--initial-model',default=None)
    parser.add_argument('--three-phase',action='store_true')
    parser.add_argument('--features-v2',action='store_true')
    parser.add_argument('--features-v3',action='store_true')
    parser.add_argument('--updates-per-match',type=int,choices=(1,4),
                        default=1)
    parser.add_argument('--migrate-from-v2',action='store_true')
    parser.add_argument('--switch-guard',choices=['none','anti-loop-v1'],
                        default='none')
    parser.add_argument('--switch-cost',choices=['none','opportunity-v1'],
                        default='none')
    args=parser.parse_args()
    print(json.dumps(train(seed=args.seed,matches=args.matches,
                           out=args.out,resume=args.resume,
                           initial_model=args.initial_model,
                           three_phase=args.three_phase,
                           features_v2=args.features_v2,
                           features_v3=args.features_v3,
                           updates_per_match=args.updates_per_match,
                           migrate_from_v2=args.migrate_from_v2,
                           switch_guard=args.switch_guard,
                           switch_cost=args.switch_cost),
                     indent=2))
