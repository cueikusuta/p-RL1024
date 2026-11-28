"""Native ops for an isolated learner experiment. No training or I/O at import."""
import os
import torch
def bind_native(net,reuse,enabled):
    b=net.battle
    native=b.forward
    params=(*b.state_proj.parameters(), *b.gru.parameters(), *b.policy_hidden.parameters(),
            *b.policy_out.parameters(), *b.value_hidden.parameters(), *b.value_out.parameters())
    assert len(params)==14
    op=torch.ops.speed_native_battle.forward
    def forward(state,candidates,memory):
        if not enabled[0]:return native(state,candidates,memory)
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
    enabled=[False]
    eager_target=learner.rnad_targets
    eager_loss=learner.neurd_policy_loss
    import importlib,inspect,re,textwrap
    cls=type(model)
    assert type(target) is cls and type(magnet) is cls
    module=importlib.import_module(cls.__module__)
    validation={}
    for key in ('forward','selection_forward_batch'):
        original=getattr(cls,key)
        code=textwrap.dedent(inspect.getsource(original))
        code,n=re.subn(r'bool\(torch\.isfinite\(logits\)\.all\(\)\) and\s*bool\(torch\.isfinite\(value\)(?:\.all\(\))?\)',
                       'fast_finite(logits,value)',code)
        assert n==1
        space=dict(vars(module),fast_finite=torch.ops.speed_native_loss.both_finite)
        exec(compile(code,'<fused-integration-validation>','exec'),space)
        validation[key]=(original,space[key])
    def fast_loss(logits,q_targets,*,beta=2.,q_clip=10.,logit_step=3e-4,is_c=1.):
        if not isinstance(is_c,(int,float)) or logits.dtype!=torch.float32 or logits.device.type!='cpu':
            return eager_loss(logits,q_targets,beta=beta,q_clip=q_clip,logit_step=logit_step,is_c=is_c)
        return torch.ops.speed_native_loss.policy_loss(logits,q_targets,beta,q_clip,logit_step,is_c)
    for net in (model,target,magnet):bind_native(net,False,enabled)
    def select(enabled_now):
        enabled[0]=enabled_now
        learner.rnad_targets=native_target if enabled_now else eager_target
        learner.neurd_policy_loss=fast_loss if enabled_now else eager_loss
        for k,choices in validation.items():
            fn=choices[int(enabled_now)]
            if getattr(cls,k) is not fn:setattr(cls,k,fn)
    return select
