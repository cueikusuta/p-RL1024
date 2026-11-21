import math
import torch
from scripts.rnad.train_teacher import phase_balanced_mean_loss

def test_phase_balance():
    x=torch.tensor(1.0,requires_grad=True)
    y=torch.tensor(2.0,requires_grad=True)
    z=torch.tensor(3.0,requires_grad=True)
    events=[('draft',x),('selection',y)]+[('battle',z)]*10
    balanced=phase_balanced_mean_loss(events,enabled=True)
    baseline=phase_balanced_mean_loss(events,enabled=False)
    assert math.isclose(balanced.item(),2.0,abs_tol=1e-6)
    assert math.isclose(baseline.item(),(1+2+30)/12,abs_tol=1e-6)
    assert not math.isclose(balanced.item(),baseline.item())
    balanced.backward()
    assert math.isclose(x.grad.item(),1/3,abs_tol=1e-6)
    assert math.isclose(y.grad.item(),1/3,abs_tol=1e-6)
    assert math.isclose(z.grad.item(),1/3,abs_tol=1e-6)
    try:
        phase_balanced_mean_loss([('draft',x),('battle',z)],enabled=True)
        raise AssertionError('Must fail closed when a phase is missing')
    except ValueError as e:
        assert 'all three phases' in str(e)
    print('PASS: phase-balanced gradients are independent of battle volume')

if __name__=='__main__':
    test_phase_balance()
