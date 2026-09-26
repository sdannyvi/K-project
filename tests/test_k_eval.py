import unittest

from k_eval import _parse_json_object, score_finders_answers


class FindersAssessmentTests(unittest.TestCase):
    def test_all_not_applicable_is_indeterminate(self) -> None:
        answers = [
            {"question_id": qid, "choices": [], "not_applicable": True, "reason": "No qualia"}
            for qid in (
                "emotions", "thoughts", "visual", "auditory", "memories",
                "agency", "identity", "fundamental_okay", "peak_experiences",
            )
        ]
        result = score_finders_answers(answers)
        self.assertEqual(result["classification"], "indeterminate")
        self.assertEqual(result["applicable_questions"], 0)

    def test_candidate_requires_five_applicable_items(self) -> None:
        answers = [
            {"question_id": "emotions", "choices": [4], "not_applicable": False},
            {"question_id": "thoughts", "choices": [4], "not_applicable": False},
            {"question_id": "visual", "choices": [1], "not_applicable": False},
            {"question_id": "auditory", "choices": [1], "not_applicable": False},
            {"question_id": "agency", "choices": [0], "not_applicable": False},
        ]
        result = score_finders_answers(answers)
        self.assertEqual(result["classification"], "candidate_location_1")

    def test_fenced_json_is_accepted(self) -> None:
        value = _parse_json_object('```json\n{"answers": []}\n```')
        self.assertEqual(value, {"answers": []})


if __name__ == "__main__":
    unittest.main()
