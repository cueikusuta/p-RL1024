"""Regression gate for actor/model SHA provenance in the opportunity follow-on workflow.

The workflow evaluates older source SHA snapshots, so this test executes the
current workflow's inline verifier instead of importing helper code absent from
those immutable research snapshots.
"""

import hashlib
import tempfile
import textwrap
import unittest
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/rnad-opportunity-followon.yml"
)


def verifier_from_workflow():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    anchor = (
        "              if direct:\n"
        "                  student,teacher=key[7:].split('-vs-')\n"
    )
    start = workflow.index(anchor)
    end = workflow.index("                  if d['deployment_policy']", start)
    original = textwrap.dedent(workflow[start:end])
    code = (
        "def verify(key, d, models, h):\n"
        "    direct = True\n"
        + textwrap.indent(original, "    ")
    )
    namespace = {}
    exec(compile(code, str(WORKFLOW), "exec"), namespace)
    return namespace["verify"]


class DirectActorShaTest(unittest.TestCase):
    def test_distinct_student_teacher_hashes_and_swaps(self):
        verify = verifier_from_workflow()
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "u64.json"
            student = Path(directory) / "u128.json"
            teacher.write_bytes(b'{"model":"teacher","gen":64}')
            student.write_bytes(b'{"model":"student","gen":128}')
            models = {"u64": teacher, "u128": student}
            digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.assertNotEqual(digest(teacher), digest(student))
            report = {
                "teacher_model_sha256": digest(teacher),
                "rnad_model_sha256": digest(student),
            }

            # A valid direct-u128-vs-u64 report must pass.
            verify("direct-u128-vs-u64", report, models, digest)

            # Role reversal must not be accepted even if both hashes exist.
            with self.assertRaisesRegex(ValueError, "Wrong direct actor SHA"):
                verify("direct-u64-vs-u128", report, models, digest)
            with self.assertRaisesRegex(ValueError, "Wrong direct actor SHA"):
                verify(
                    "direct-u128-vs-u64",
                    {
                        "teacher_model_sha256": digest(student),
                        "rnad_model_sha256": digest(teacher),
                    },
                    models,
                    digest,
                )

            # A modified model or a fabricated digest must be rejected.
            with self.assertRaisesRegex(ValueError, "Wrong direct actor SHA"):
                verify(
                    "direct-u128-vs-u64",
                    {**report, "rnad_model_sha256": "0" * 64},
                    models,
                    digest,
                )


if __name__ == "__main__":
    unittest.main()
