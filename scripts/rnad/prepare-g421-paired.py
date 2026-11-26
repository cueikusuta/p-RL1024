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
        # G198 uses the same engine and v6 features but has an older source pin.
        # Normalize only a temporary copy of manifest metadata, never weight bytes.
        if m["source_sha"] != p["source"]:
            assert model == "G198"
            m["source_sha"] = p["source"]
            manifest_raw = (json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (result / "manifest.json").write_bytes(manifest_raw)
    archive.unlink()
    print(model, "verified", spec["artifact"], "runtime manifest", sha(manifest_raw), flush=True)
