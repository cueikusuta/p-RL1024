"""Run a width-one or width-four Learner against the unmodified pinned runtime."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

PLAN_SHA = "f96c7431217de2323d6aa4d5a5e8f5039eac1c9b6ac7c9d8ccc304bdfb3e997e"
LEARNER_SHA = "3abafcb478d5755b66fa0e28c0b58c06a99195060e9902a691f925c119ae1ed0"

def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def advance_memory(memory, steps, seat, width):
    """The window is indexed per seat and never crosses a Match."""
    steps[seat] += 1
    return memory.detach() if steps[seat] % width == 0 else memory

def self_test(torch):
    for width in (1, 4):
        hidden = [torch.tensor(0.), torch.tensor(7.)]
        steps = [0, 0]
        inputs = [torch.tensor(float(i), requires_grad=True) for i in range(1, 6)]
        for n, x in enumerate(inputs):
            output = hidden[0] * 2 + x
            if n == 2:
                actual = torch.autograd.grad(output, inputs[:n+1], retain_graph=True, allow_unused=True)
                assert [None if g is None else g.item() for g in actual] == (
                    [None, None, 1.] if width == 1 else [4., 2., 1.])
            if n == 4:
                assert torch.autograd.grad(output,inputs[3],allow_unused=True)[0] is None
            hidden[0] = advance_memory(output, steps, 0, width)
            hidden[1] = advance_memory(hidden[1], steps, 1, width)
        assert steps == [5, 5] and hidden[1].item() == 7.

def run(plan_path, rep, arm, bootstrap, actors, out, experiment):
    import torch
    # The archival code checkout stays clean. The only experimental code path
    # is the in-memory replacement of this exact, digest-checked loss function.
    sys.path.insert(0, str(Path.cwd()))
    from scripts.rnad import shared_league_learner as original
    from scripts.rnad import mix_bootstrap
    plan = json.loads(Path(plan_path).read_text())
    require(sha(plan_path) == PLAN_SHA, "Wrong plan")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(actual == plan["baselineSource"], "Wrong unchanged runtime/collector source")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=True)
    require(sha(original.__file__) == LEARNER_SHA, "Wrong pinned Learner source")
    require(os.environ.get("RNAD_LEARNER_BACKEND", "reference") == "reference", "Only reference Learner is supported")
    self_test(torch)
    unchanged = original._match_loss
    width = plan["training"]["widths"][arm]
    before = inspect.getsource(unchanged)
    require(before.count("    inputs, terms = [], []") == 1 and before.count("memories[\"main\"][seat] = next_memory.detach()") == 1, "Pinned loss changed")
    if width == 4:
        after = before.replace("    inputs, terms = [], []", "    inputs, terms = [], []\n    battle_steps = [0, 0]")
        after = after.replace('memories["main"][seat] = next_memory.detach()',
            'memories["main"][seat] = SHORT_BPTT_ADVANCE(next_memory, battle_steps, seat, 4)')
        scope = {**original.__dict__, "SHORT_BPTT_ADVANCE": advance_memory}
        exec(compile(after, "<research-only four-decision BPTT>", "exec"), scope)
        original._match_loss = scope["_match_loss"]
    else:
        require(width == 1 and original._match_loss is unchanged, "Control loss must be exact")
    b = Path(bootstrap)
    initial = sha(b/"checkpoint-g0.pt")
    old_plan = Path("docs/research/rnad/G422_OPPONENT_MIX_PILOT.json")
    require(sha(old_plan) == plan["sourcePlanSha"], "Wrong archival plan")
    report = original.learn(bootstrap=b, actors=actors, out=out,
        original_draft=b/"original-draft.json", original_selection_js=b/"original-selection-js.json",
        max_steps=40)
    mix_bootstrap.receipt(old_plan, b, Path(out), rep, "control")
    receipt_path = Path(out)/"training-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["originalCheckpoint"] == plan["baseCheckpoint"], "Wrong G421 start")
    final = torch.load(Path(out)/"checkpoint-g423.pt", map_location="cpu", weights_only=True)
    ids = final["training_match_ids"]
    require(len(ids) == len(set(ids)) == 1280, "Wrong unique training Match IDs")
    receipt.update({"arm":arm, "width":width, "experimentHead":experiment,
        "planSha":PLAN_SHA, "startCheckpoint":initial,
        "oldPlanSha":plan["sourcePlanSha"], "sourceRun":plan["sourceRun"],
        "exactActorArchives":[plan["artifacts"][f"i5-mix-actors-{rep}-control-{k}"]["sha256"] for k in range(5)],
        "trainingMatchOrderSha":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()})
    receipt_path.write_text(json.dumps(receipt,sort_keys=True)+"\n")
    print(json.dumps(receipt,sort_keys=True),flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("plan","rep","arm","bootstrap","actors","out","experiment"):
        p.add_argument(key)
    x=p.parse_args()
    run(x.plan,x.rep,x.arm,x.bootstrap,x.actors,x.out,x.experiment)
