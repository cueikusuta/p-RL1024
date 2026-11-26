"""Materialize a frozen evaluation-only adapter in a disposable checkout."""
import argparse, hashlib, json, subprocess
from pathlib import Path
PIN = "6a11eee1a011dcc65ac80e3b5f2c3c0b57341494"
sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
def install(spec):
    assert subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip() == PIN
    for entry in json.loads(Path(spec).read_text()):
        source=Path(entry["source"]).read_text()
        assert sha(source)==entry["source_sha256"]
        lines=source.splitlines(keepends=True)
        for first,last,old,new in reversed(entry["edits"]):
            assert "".join(lines[first:last])==old
            lines[first:last]=[new]
        result="".join(lines)
        assert sha(result)==entry["result_sha256"]
        with Path(entry["destination"]).open("x") as f:
            f.write(result)
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("overlay");install(p.parse_args().overlay)
