import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

plan, root = Path(sys.argv[1]), Path(sys.argv[2])
p = json.loads(plan.read_text())
repo = "cueikusuta/p-RL1024"
def sha(data):
    return hashlib.sha256(data).hexdigest()
def gh(url):
    return subprocess.check_output(["gh", "api", url])
files = ("battle", "draft", "selection", "values")
for model, spec in p["models"].items():
    path = f"repos/{repo}/actions/artifacts/{spec['artifact']}"
    metadata = json.loads(gh(path))
    expected_name = "rnad-i5-cont-next-bootstrap-after-" + model.lower()
    assert metadata["id"] == spec["artifact"] and metadata["workflow_run"]["id"] == spec["run"]
    assert metadata["name"] == expected_name and metadata["expired"] is False
    assert metadata["digest"] == "sha256:" + spec["zip"]
    archive = root / (model + ".zip")
    with archive.open("xb") as stream:
        subprocess.run(["gh", "api", path+"/zip"], stdout=stream, check=True)
    assert sha(archive.read_bytes()) == spec["zip"]
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        manifest_raw = z.read("actor/manifest.json")
        m = json.loads(manifest_raw)
        handoff = json.loads(z.read("handoff-report.json"))
        assert m["generation"] == handoff["generation"] == spec["generation"]
        assert m["source_sha"] == spec["source"] and m["engine_sha"] == p["engine"]
        assert m["schema"] == "rnad-integrated-policy-actor-bundle-v1"
        assert set(m["files"]) == set(files)
        assert sha(manifest_raw) == handoff["main_sha256"] == spec["actor"]
        pins = {}
        for line in z.read("file-pins.sha256").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            assert name not in pins
            pins[name] = digest
        assert pins["actor/manifest.json"] == spec["actor"]
        result = root / "models" / model
        result.mkdir(parents=True)
        for key in files:
            name = "actor/" + key + ".json"
            raw = z.read(name)
            assert sha(raw) == pins[name] == m["files"][key]
            (result / (key + ".json")).write_bytes(raw)
        assert json.loads((result / "battle.json").read_bytes())["feature_schema"] == "rnad-three-phase-features-v6"
        assert m["source_sha"] == p["source"]
        (result / "manifest.json").write_bytes(manifest_raw)
    archive.unlink()
    print(model, "verified", spec["artifact"], "runtime manifest", sha(manifest_raw), flush=True)

# Evaluation-only one-phase transplants. Keep immutable phase payloads byte-for-byte;
# use the associated value head and retain the baseline for the other phases.
for name, phase in p["swaps"].items():
    assert phase in ("draft", "selection", "battle")
    base = root / "models" / p["modelBase"]
    donor = root / "models" / p["modelDonor"]
    m = json.loads((base / "manifest.json").read_bytes())
    d = json.loads((donor / "manifest.json").read_bytes())
    assert m["engine_sha"] == d["engine_sha"] and m["source_sha"] == d["source_sha"]
    result = root / "models" / name
    result.mkdir()
    for key in ("draft", "selection", "battle"):
        a = (base / (key + ".json")).read_bytes()
        b = (donor / (key + ".json")).read_bytes()
        a0, b0 = json.loads(a), json.loads(b)
        fields = {"draft": ("schema", "architecture", "feature_width"),
                  "selection": ("schema", "engine_sha", "public_actor_vocabulary"),
                  "battle": ("schema", "feature_schema", "state_features", "action_features", "hidden")}[key]
        assert all(a0[field] == b0[field] for field in fields)
        payload = b if key == phase else a
        (result / (key + ".json")).write_bytes(payload)
        m["files"][key] = sha(payload)
    va = json.loads((base / "values.json").read_bytes())
    vb = json.loads((donor / "values.json").read_bytes())
    assert va["schema"] == vb["schema"] and va["tensors"].keys() == vb["tensors"].keys()
    if phase != "battle":
        for k in va["tensors"]:
            if k.startswith(phase + "_value."):
                va["tensors"][k] = vb["tensors"][k]
    m["policy_id"] = va["policy_id"] = "evaluation-only-" + p["modelBase"] + "-with-" + p["modelDonor"] + "-" + phase
    def encoded(obj):
        return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()
    values = encoded(va)
    (result / "values.json").write_bytes(values)
    m["files"]["values"] = sha(values)
    (result / "manifest.json").write_bytes(encoded(m))
    print(name, "verified isolated transplant", phase, flush=True)
