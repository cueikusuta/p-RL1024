"""Contracts for selection-only 60-action structured teacher imitation."""
from __future__ import annotations
import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.rnad.train_teacher import AQ_WEIGHTS_SHA256

from scripts.rnad.train_selection_specialist import (
    TRIPLES,TRIO_ROWS,LEAD_ROWS,SelectionSpecialist,teacher_loss,
    feature_sufficiency_audit,paired_bootstrap80,save_model,load_traces,
    TEACHER_SCHEMA,TEACHER,SOURCE_SHA,
)

class SelectionSpecialistTests(unittest.TestCase):
    def test_frozen_authority_engine_commit_hex_40_and_64_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            e={"phase":"selection","state":[0.]*96,
               "actions":[[0.]*48 for _ in range(60)],
               "candidate_utilities":[float(i) for i in range(60)],
               "choice":59}
            t={"schema":TEACHER_SCHEMA,
               "source_sha":SOURCE_SHA,"feature_schema":"rnad-three-phase-features-v5",
               "teacher":TEACHER,"teacher_model_sha256":AQ_WEIGHTS_SHA256,
               "utility_labels":"aq-public-phase-rank-v1",
               "completed":True,"winner":0,"score":[4,0],
               "seed":2471001,"action_counts":[0,0],"rounds":1,
               "examples":[e,e,e,e],"engine_sha":"e"*40}
            f=root/'teacher-2471001.json'
            f.write_text(json.dumps(t))
            extracted,engine=load_traces(root,2471001,1)
            self.assertEqual(engine,"e"*40)
            self.assertEqual(len(extracted[0]["examples"]),4)
            t['engine_sha']="e"*64
            f.write_text(json.dumps(t))
            self.assertEqual(load_traces(root,2471001,1)[1],"e"*64)
            t['engine_sha']="not-a-digest"
            f.write_text(json.dumps(t))
            with self.assertRaisesRegex(ValueError,"engine"):
                load_traces(root,2471001,1)

    def test_60_actions_have_20_unordered_trios_three_leads_each(self):
        self.assertEqual(len(TRIPLES),60)
        self.assertEqual(len(TRIO_ROWS),20)
        self.assertTrue(all(len(x)==3 for x in TRIO_ROWS))
        self.assertEqual(len(LEAD_ROWS),6)
        self.assertTrue(all(len(x)==10 for x in LEAD_ROWS))
        for rows in TRIO_ROWS:
            self.assertEqual(len({TRIPLES[j][0] for j in rows}),3)
            self.assertEqual(len({tuple(sorted(TRIPLES[j])) for j in rows}),1)

    def test_both_models_emit_legal_finite_logits_and_gradients(self):
        for structured in (False,True):
            with self.subTest(structured=structured):
                torch.manual_seed(17)
                model=SelectionSpecialist(structured=structured)
                s=torch.randn(3,96)
                a=torch.randn(3,60,48)
                out=model(s,a)
                self.assertEqual(tuple(out.shape),(3,60))
                self.assertTrue(bool(torch.isfinite(out).all()))
                utility=torch.randn(3,60)
                utility[:,4]=10
                choice=torch.tensor([4,4,4])
                for mixed in (False,True):
                    model.zero_grad(set_to_none=True)
                    loss=teacher_loss(model(s,a),choice,utility,mixed=mixed)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assertGreater(sum(float(x.grad.abs().sum()) for x in model.parameters()),0)
                for tensor in (s[:,:-1],a[:,:59,:]):
                    with self.assertRaises(ValueError):
                        if tensor.shape[-1]==95:
                            model(tensor,a)
                        else:
                            model(s,tensor)

    def test_python_js_inference_parity_and_tamper_rejection(self):
        torch.manual_seed(454)
        state=torch.randn(96)
        actions=torch.randn(60,48)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            inputfile=root/'inputs.json'
            inputfile.write_text(json.dumps({
                'state':state.tolist(),'actions':actions.tolist()}))
            for structured in (False,True):
                arm='structured-hard' if structured else 'flat-hard'
                net=SelectionSpecialist(structured=structured).eval()
                with torch.no_grad():
                    expected=net(state[None,:],actions[None,:,:])[0].tolist()
                model=root/(arm+'.json')
                save_model(net,model,arm)
                sha=hashlib.sha256(model.read_bytes()).hexdigest()
                child=subprocess.run(['node','scripts/rnad/selection-specialist-infer.mjs',
                                      '--model',str(model),'--sha256',sha,
                                      '--input',str(inputfile)],
                    capture_output=True,text=True,check=True,timeout=30)
                got=json.loads(child.stdout)['logits']
                self.assertEqual(len(got),60)
                self.assertLess(max(abs(a-b) for a,b in zip(expected,got)),2e-4)
                broken=json.loads(model.read_text())
                # A shape-valid finite weight corruption must be rejected using the
                # separately pinned SHA, not just schema / tensor checks.
                broken['weights']['state.bias']['values'][0]+=0.125
                model.write_text(json.dumps(broken))
                bad=subprocess.run(['node','scripts/rnad/selection-specialist-infer.mjs',
                                    '--model',str(model),'--sha256',sha,
                                    '--input',str(inputfile)],
                    capture_output=True,text=True,timeout=30)
                self.assertNotEqual(bad.returncode,0)
                self.assertIn('SHA-256 mismatch',bad.stderr)
                # Even if caller explicitly pins a new, structurally invalid
                # model, the canonical action-order fail-closed check still runs.
                broken['weights']['trio_ids']['values'][0]=5
                model.write_text(json.dumps(broken))
                broken_sha=hashlib.sha256(model.read_bytes()).hexdigest()
                bad=subprocess.run(['node','scripts/rnad/selection-specialist-infer.mjs',
                                    '--model',str(model),'--sha256',broken_sha,
                                    '--input',str(inputfile)],
                    capture_output=True,text=True,timeout=30)
                self.assertNotEqual(bad.returncode,0)
                self.assertIn('buffer mismatch',bad.stderr)
                # No pinned digest is never an implicit trust mode.
                missing=subprocess.run(['node','scripts/rnad/selection-specialist-infer.mjs',
                    '--model',str(model),'--input',str(inputfile)],
                    capture_output=True,text=True,timeout=30)
                self.assertNotEqual(missing.returncode,0)

    def test_hard_only_and_hardsoft_targets_are_different(self):
        v=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
        labels=torch.tensor([2])
        util=torch.tensor([[5.,1.,6.,0.]])
        h=teacher_loss(v,labels,util,mixed=False)
        m=teacher_loss(v,labels,util,mixed=True)
        # At identical logits the losses happen to match; gradients differ.
        h.backward(retain_graph=True)
        hard_grad=v.grad.detach().clone()
        v.grad.zero_()
        m.backward()
        self.assertFalse(torch.allclose(hard_grad,v.grad))

    def test_feature_alias_audit_flags_utility_conflict(self):
        options=[[0.0]*48 for _ in range(60)]
        utils=[float(i) for i in range(60)]
        event={"phase":"selection","actions":options,
               "candidate_utilities":utils}
        out=feature_sufficiency_audit([{"seed":101,"examples":[event]}])
        self.assertEqual(out["selection_teacher_decisions"],1)
        self.assertEqual(out["decisions_with_action_feature_utility_collision"],1)
        self.assertGreater(out["trio_groups_with_indistinguishable_leads"],0)
        self.assertFalse(out["input_sufficiency_proven"])

    def test_full_actor_input_alias_detects_cross_match_disagreement(self):
        base={"phase":"selection","state":[0.]*96,
              "actions":[[float(i)]*48 for i in range(60)],
              "candidate_utilities":[float(i) for i in range(60)],
              "choice":59}
        altered=copy.deepcopy(base)
        altered["choice"]=58
        altered["candidate_utilities"][58]=61.
        # Duplicate full actor inputs (state plus ordered 60 candidates) with
        # different teacher labels across distinct accepted Match seeds.
        records=[{"seed":101,"examples":[base]},
                 {"seed":102,"examples":[altered]}]
        report=feature_sufficiency_audit(records)
        self.assertEqual(report["full_input_reused_across_independent_matches"],1)
        self.assertEqual(report["crossmatch_full_input_discordant_teacher_top1_groups"],1)
        records[1]["examples"][0]["state"][0]=0.5
        changed=feature_sufficiency_audit(records)
        self.assertEqual(changed["full_input_reused_across_independent_matches"],0)
        self.assertFalse(changed["input_sufficiency_proven"])

    def test_width_ablation_preserves_60_choice_contract(self):
        for width in (64,96,128):
            net=SelectionSpecialist(structured=True,width=width)
            self.assertEqual(tuple(net(torch.randn(2,96),torch.randn(2,60,48)).shape),
                             (2,60))
            if width != 96:
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaisesRegex(ValueError,"96-wide"):
                        save_model(net,Path(folder)/"experimental.json","structured-hard")
        with self.assertRaisesRegex(ValueError,"Unregistered"):
            SelectionSpecialist(structured=True,width=77)

    def test_gap_bins_and_trio_lead_breakdown(self):
        from scripts.rnad.selection_specialist_followup import (
            gap_bucket, gap_diagnostics)
        self.assertEqual(gap_bucket(.01),"gap<=1%")
        self.assertEqual(gap_bucket(.05),"1%<gap<=5%")
        self.assertEqual(gap_bucket(.10),"5%<gap<=10%")
        self.assertEqual(gap_bucket(.50),"gap>10%")
        class FixedNet:
            def eval(self):return self
            def __call__(self, state, actions):
                return torch.tensor([[60.]+[0.]*59])
        close=[0.]*60
        close[0]=1.
        close[1]=.995
        wide=[0.]*60
        wide[1]=1.
        wide[0]=.3
        event=lambda choice, utility: {
            "phase":"selection","choice":choice,
            "state":[0.]*96,"actions":[[0.]*48 for _ in range(60)],
            "candidate_utilities":utility}
        rows=gap_diagnostics(FixedNet(),[
            {"seed":101,"examples":[event(0,close)]},
            {"seed":102,"examples":[event(1,wide)]}])
        self.assertEqual(rows["gap<=1%"]["decisions"],1)
        self.assertEqual(rows["gap<=1%"]["top1"],1.)
        self.assertEqual(rows["gap>10%"]["decisions"],1)
        self.assertEqual(rows["gap>10%"]["top1"],0.)
        self.assertEqual(rows["gap>10%"]["independent_match_seed_clusters"],1)

    def test_paired_seed_bootstrap_is_clustered_and_deterministic(self):
        ids=list(range(128))
        old={i:{"top1":.25} for i in ids}
        new={i:{"top1":.40} for i in ids}
        a=paired_bootstrap80(old,new,ids)
        b=paired_bootstrap80(old,new,ids)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["per_match_cluster_top1_delta"],.15,places=6)
        self.assertAlmostEqual(a["approximate_familywise80_CI"][0],.15,places=6)

if __name__=="__main__":
    unittest.main()
