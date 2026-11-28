"""Native ops for an isolated learner experiment. No training or I/O at import."""
import os
import torch
def bind_native(net,reuse):
    b=net.battle
    native=b.forward
    params=(*b.state_proj.parameters(), *b.gru.parameters(), *b.policy_hidden.parameters(),
            *b.policy_out.parameters(), *b.value_hidden.parameters(), *b.value_out.parameters())
    assert len(params)==14
    op=torch.ops.speed_native_battle.forward
    def forward(state,candidates,memory):
        if state.ndim!=2 or state.shape[-1]!=b.state_features or candidates.ndim!=3 or candidates.shape[0]!=state.shape[0] or candidates.shape[-1]!=b.action_features or candidates.shape[1]<1:
            return native(state,candidates,memory)
        return op(state,candidates,memory,params,reuse)
    b.forward=forward
    return net
def native_target(terms,eta=.2,rho_bar=1.,c_bar=1.):
    assert torch.get_default_dtype()==torch.float32
    payload=torch.tensor([(r['actor'],*r['reward'],r['value'],r['behavior_prob'],r['action_index']) for r in terms],dtype=torch.float64)
    q,metrics=torch.ops.speed_native_targets.targets([r['pi'] for r in terms],[r['magnet'] for r in terms],payload,eta,rho_bar,c_bar)
    return [dict(value_target=v,rho=r,c=c,ratio=i,actor=int(a),regularized_reward=(x,y),q_targets=t)
            for t,(v,r,c,i,a,x,y) in zip(q,metrics.tolist(),strict=True)]

def initialize(model,target,magnet,learner):
    path=os.environ['LEAGUE_NATIVE_LIBRARY']
    assert os.path.isabs(path) and path.endswith('.so')
    torch.ops.load_library(path)
    for net in (model,target,magnet):bind_native(net,False)
    learner.rnad_targets=native_target
