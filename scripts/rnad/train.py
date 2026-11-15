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

from scripts.rnad.neural import RNADNetwork, freeze_magnet, export_json
from scripts.rnad.two_player import rnad_targets, neurd_policy_loss
from scripts.rnad.checkpoint import save_checkpoint, restore_checkpoint


def current_git_sha():
    return subprocess.check_output(
        ['git','rev-parse','HEAD'], text=True, timeout=5
    ).strip()


def frozen_engine_sha():
    return json.loads(Path('provenance/manifest.json').read_text())['sourceCommit']


def state_hash(net):
    import io
    buffer=io.BytesIO()
    torch.save(net.state_dict(),buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def run_match(seed, model, magnet, directory):
    directory.mkdir(parents=True,exist_ok=True)
    model_path=directory/'actor.json'
    magnet_path=directory/'magnet.json'
    output=directory/(f'accepted-seed{seed}.json')
    export_json(model,model_path)
    export_json(magnet,magnet_path)
    if output.exists():
        raise ValueError('Refusing to overwrite a prior authoritative Match')
    subprocess.run(
        ['node','scripts/rnad/export-trajectory.mjs',
         '--model',str(model_path),'--magnet',str(magnet_path),
         '--seed',str(seed),'--out',str(output)],
        check=True,timeout=1800)
    row=json.loads(output.read_text())
    if (row.get('schema')!='rnad-accepted-training-match-v1' or
        row.get('feature_schema')!='rnad-battle-features-v1' or
        row.get('completed') is not True or
        row.get('source_sha')!=current_git_sha() or
        row.get('engine_sha')!=frozen_engine_sha() or
        row.get('seed')!=seed or
        row.get('score',[0,0])[row['winner']]!=4 or
        len(row.get('events',[]))!=sum(row.get('action_counts',[])) or
        not row.get('events')):
        raise ValueError('Untrusted, partial or mismatched Match training record')
    rewards=[entry['reward'] for entry in row['events']]
    if (any(any(abs(x)>1e-10 for x in r) for r in rewards[:-1]) or
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
        if (len(state)!=32 or not actions or
            any(len(action)!=16 for action in actions) or
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
                                                beta=2.0,q_clip=10.0))
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
          eta=.2,magnet_period=4):
    if not 1 <= seed <= 0xffffffff or not 1 <= matches <= 16 or not 0 < eta <= 1:
        raise ValueError('Training smoke must be bounded (1–16 completed Matches)')
    torch.set_num_threads(1)
    path=Path(out)
    path.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);random.seed(seed)
    net=RNADNetwork()
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
            expected={'model_id':'RNAD-001',
                      'source_sha':current_git_sha(),
                      'engine_snapshot_sha':frozen_engine_sha(),
                      'training_code_sha':current_git_sha(),
                      'observation_schema_version':'rnad-battle-features-v1'})
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
        data=run_match(counter,net,magnet,path/'rollouts')
        summary=train_match(net,target,magnet,optimizer,scheduler,data,eta=eta)
        completed+=1
        ids.append(data['match_id'])
        if completed%magnet_period==0:
            magnet.load_state_dict(target.state_dict())
        prior_result=save_checkpoint(
            checkpoint_dir,policy=net,value=target,magnet=magnet,
            optimizer=optimizer,scheduler=scheduler,expected_prior_sha=prior,
            metadata={
                'model_id':'RNAD-001',
                'source_sha':current_git_sha(),
                'engine_snapshot_sha':frozen_engine_sha(),
                'training_code_sha':current_git_sha(),
                'observation_schema_version':'rnad-battle-features-v1',
                'opponent_policy_version':'RNAD-001-selfplay',
                'completed_matches':completed,
                'accepted_decisions':summary['accepted_decisions']+
                    (sum(x['accepted_decisions'] for x in summaries)
                     if not resume else
                     metadata['accepted_decisions']+sum(x['accepted_decisions'] for x in summaries)),
                'updates':completed,'next_match_seed':
                    ((counter%0xffffffff)+1),
                'last_match_id':data['match_id'],
                'completed_match_ids':ids,
                'initial_seed':seed,'boundary':'completed_match'})
        prior=prior_result['checkpoint_sha256']
        counter=(counter%0xffffffff)+1
        summaries.append({'seed':data['seed'],'winner':data['winner'],
                          'rounds':data['rounds'],**summary})
    export_json(net,path/'final-model.json')
    report={'schema':'rnad-smoke-training-report-v1',
            'source_sha':current_git_sha(),'engine_sha':frozen_engine_sha(),
            'completed_matches':completed,'updates':completed,
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
    args=parser.parse_args()
    print(json.dumps(train(seed=args.seed,matches=args.matches,
                           out=args.out,resume=args.resume),
                     indent=2))
