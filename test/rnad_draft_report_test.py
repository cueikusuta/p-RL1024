"""Fast draft-report validation: review metadata must be authority-consistent."""
import unittest
from scripts.rnad.report_draft_picks import _validate, render_markdown

def fixture():
    roster=[{"id":str(i),"species":f"Mon-{i}","set_id":str(i),
             "credit_cost":1,"is_carry":False,"roles":[]}
            for i in range(8)]
    codes=[list(range(6)),[0,1,2,3,4,6]]
    view={"schema":"rnad-accepted-draft-pick-review-v1",
          "credits":10,"roster":roster,"candidate_codes":codes,
          "accepted_choice_index":0,"accepted_roster_codes":codes[0],
          "accepted_spent":6}
    event={"phase":"draft","choice":0,"actions":[[0],[1]],
           "candidate_utilities":[7.,6.]}
    record={"score":[4,2],"winner":0}
    return record,event,view

class DraftReportTests(unittest.TestCase):
    def test_accepted_validity(self):
        _validate(*fixture())

    def test_reject_forged_candidate_or_budget(self):
        record,event,view=fixture()
        view["candidate_codes"][0][0]=100
        with self.assertRaises(ValueError):
            _validate(record,event,view)
        record,event,view=fixture()
        view["credits"]=1
        with self.assertRaises(ValueError):
            _validate(record,event,view)

    def test_reject_mismatched_teacher_utility(self):
        record,event,view=fixture()
        event["candidate_utilities"]=[6.,7.]
        with self.assertRaises(ValueError):
            _validate(record,event,view)

    def test_markdown_explicitly_marks_counterfactual(self):
        report={"teacher_id":"teacher-v1","student_model_sha256":"a",
                "source_sha":"source","engine_sha":"engine","teacher_matches":1,
                "total_draft_decisions":0,"shown_draft_decisions":0,"picks":[]}
        text=render_markdown(report)
        self.assertIn("教師が訪れた局面",text)
        self.assertIn("未受理",text)

if __name__=="__main__":
    unittest.main()
