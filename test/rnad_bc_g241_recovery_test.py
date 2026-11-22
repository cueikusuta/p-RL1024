"""Prove that the 100h G241+ recovery override changes only the bound.

No training-weight, policy, learner or random-number source may change.
"""
from __future__ import annotations

import hashlib
import subprocess
import unittest
from pathlib import Path

BASE="6a11eee1a011dcc65ac80e3b5f2c3c0b57341494"
OLD_BLOB="78691dfecab481a8ef9e233e08f3daf383eb58f4"
NEW_BLOB="a29ae5e46cff00598d84440071594275686029f2"
NAME="scripts/rnad/shared_league_artifact_recovery.py"
OLD=("# Source-pinned G137 continuation is bounded to G240; "
     "reject unapproved later generations.\nMAX_RECOVERY_GENERATION = 240")
NEW=("# B+C 100h approved continuation has fixed waves through G598.\n"
     "# Keep the bound explicit: generation 599 and beyond fail closed.\n"
     "MAX_RECOVERY_GENERATION = 598")


def git_blob(data):
    return hashlib.sha1(
        ("blob "+str(len(data))+"\0").encode()+data).hexdigest()


class RecoveryBoundTest(unittest.TestCase):
    def test_exact_blob_diff_and_actor_bounds(self):
        old=subprocess.check_output(["git","show",f"{BASE}:{NAME}"])
        latest=Path(NAME).read_bytes()
        self.assertEqual(git_blob(old),OLD_BLOB)
        self.assertEqual(git_blob(latest),NEW_BLOB)
        self.assertEqual(old.count(OLD.encode()),1)
        self.assertEqual(latest,old.replace(OLD.encode(),NEW.encode()))
        from scripts.rnad.shared_league_artifact_recovery import (
            artifact_name,missing_actor_ids,accepted_artifact_names)
        for g in (240,241,258,598):
            names=[artifact_name(g,i) for i in range(20)]
            rows=[{"name":name,"expired":False} for name in names]
            self.assertEqual(accepted_artifact_names(g,rows),names)
            self.assertEqual(missing_actor_ids(g,rows),[])
        with self.assertRaises(ValueError):artifact_name(599,0)

    def test_both_frozen_jobs_apply_identical_pinned_patch(self):
        flow=Path(".github/workflows/rnad-i5-bc-100h-generation-reusable.yml").read_text()
        self.assertEqual(flow.count("git/blobs/"+NEW_BLOB),2)
        self.assertEqual(flow.count("git hash-object \"$target\""),2)
        self.assertEqual(flow.count("Non-boundary changes to validation module forbidden"),2)
        self.assertEqual(flow.count("git rev-parse HEAD"),4)
        self.assertIn("ref: ${{ inputs.source_ref || github.sha }}",flow)
        self.assertIn("source_sha",Path("scripts/rnad/shared_league_contract.py").read_text())

if __name__=="__main__":
    unittest.main()
