"""Research-only A/B selection actor, teacher-label and capacity contracts."""
from __future__ import annotations
import copy
import tempfile
import unittest
import torch
from scripts.rnad.train_selection_specialist import TRIPLES, TRIO_IDS, LEAD_IDS
from scripts.rnad.selection_ab import (
    DecomposedTeacher, DetailedAttention, read_events, validate_event,
    pack, objective, evaluated_logits, clustered_delta, KEYS_VOCAB,
)

def fixture():
    vocab={"schema":"rnad-i5-selection-rich-vocabulary-v1",
           "sha256":"a"*64,
           "sizes":{k:32 for k in KEYS_VOCAB}}
    core=[float(i)/100 for i in range(20)]
    lead=[float(i)/100 for i in range(6)]
    utilities=[core[t]+lead[l] for t,l in zip(TRIO_IDS,LEAD_IDS)]
    choice=max(range(60),key=lambda i:utilities[i])
    actor={
        "schema":"rnad-i5-selection-public-actor-tensors-v1",
        "vocab_sha256":vocab["sha256"],
        "own_cats":[[1,2,3,4,5,6,7,8,9] for _ in range(6)],
        "own_numeric":[[.5]*16 for _ in range(6)],
        "foe_cats":[[1,8,9] for _ in range(6)],
        "foe_numeric":[[.5]*6 for _ in range(6)],
        "legal":[list(x) for x in TRIPLES],
    }
    event={"seat":0,"round":1,"actor":actor,"choice":choice,
           "state":[0.]*96,"actions":[[0.]*48 for _ in range(60)],
           "targets":{"core":core,"lead":lead,"utilities":utilities}}
    validate_event(event,vocab)
    return event,vocab

class SelectionABContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.event,self.vocab=fixture()
        self.rows=[(10,self.event),(11,self.event)]

    def test_models_are_finite_legal_and_trainable(self):
        for name,cls in (("A",DecomposedTeacher),("B",DetailedAttention)):
            with self.subTest(name=name):
                torch.manual_seed(301)
                model=cls(self.vocab)
                net=pack(self.rows)
                logits,core,lead=model(net)
                self.assertEqual(tuple(logits.shape),(2,60))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                if name=="A":
                    self.assertEqual(tuple(core.shape),(2,20))
                    self.assertEqual(tuple(lead.shape),(2,6))
                    self.assertTrue(torch.allclose(
                        logits,core[:,model.trio_index]+lead[:,model.lead_index]))
                loss=objective(model,self.rows,name)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertGreater(sum(float(t.grad.abs().sum())
                    for t in model.parameters() if t.grad is not None),0.)
                self.assertEqual(len(evaluated_logits(model,self.rows)),2)

    def test_targets_never_used_in_model_forward(self):
        clean,_=fixture()
        corrupted=copy.deepcopy(clean)
        corrupted["targets"]["core"]=[12.]*20
        corrupted["targets"]["lead"]=[3.]*6
        corrupted["targets"]["utilities"]=[12.]*60
        corrupted["choice"]=0
        self.assertEqual(pack([(1,clean)])["own_cat"].tolist(),
                         pack([(1,corrupted)])["own_cat"].tolist())
        for cls in (DecomposedTeacher,DetailedAttention):
            model=cls(self.vocab).eval()
            with torch.no_grad():
                first=model(pack([(1,clean)]))[0]
                second=model(pack([(1,corrupted)]))[0]
            self.assertTrue(torch.equal(first,second))

    def test_foe_permutation_invariant_and_own_item_sensitive(self):
        event,vocab=fixture()
        # Opponent preview has no privileged ordering; both models are set-based.
        changed=copy.deepcopy(event)
        for k in ("foe_cats","foe_numeric"):
            changed["actor"][k]=changed["actor"][k][::-1]
        changed["actor"]["foe_cats"]=[[i+1,8,9] for i in range(6)][::-1]
        event["actor"]["foe_cats"]=[[i+1,8,9] for i in range(6)]
        for cls in (DecomposedTeacher,DetailedAttention):
            torch.manual_seed(93)
            m=cls(vocab).eval()
            with torch.no_grad():
                lhs=m(pack([(1,event)]))[0]
                rhs=m(pack([(1,changed)]))[0]
            self.assertLess((lhs-rhs).abs().max().item(),1e-4)
            other=copy.deepcopy(event)
            other["actor"]["own_cats"][0][2]=17
            with torch.no_grad():
                perturb=m(pack([(1,other)]))[0]
            self.assertFalse(torch.equal(lhs,perturb))

    def test_fail_closed_on_illegal_actions_out_of_vocab_or_teacher_leak(self):
        event,vocab=fixture()
        for mutate in (
            lambda e: e["actor"]["legal"].reverse(),
            lambda e: e["actor"]["own_cats"][0].__setitem__(2,999),
            lambda e: e["actor"]["foe_numeric"][0].__setitem__(1,float("nan")),
            lambda e: e["targets"]["core"].__setitem__(0,500),
        ):
            malformed=copy.deepcopy(event)
            mutate(malformed)
            with self.assertRaises(ValueError):
                validate_event(malformed,vocab)

    def test_independent_seed_clusters_not_decisions(self):
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            clustered_delta({1:0.,2:0.},{1:.1,2:.1})
        a={s:.1 for s in range(128)}
        b={s:.2 for s in range(128)}
        delta=clustered_delta(a,b)
        self.assertAlmostEqual(delta["paired_cluster_mean_top1_delta"],.1)
        self.assertAlmostEqual(delta["approximate_95pct_two_sided_ci"][0],.1)

if __name__=="__main__":
    unittest.main()
