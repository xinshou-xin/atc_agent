"""不依赖外部模型或网络的动态 ATC Agent 单元测试。"""
import unittest

from agents.deep_atc_agent import DeepATCAgent


class DeepATCAgentLogicTests(unittest.TestCase):
    def setUp(self):
        self.agent = object.__new__(DeepATCAgent)
        self.agent.refinement_threshold = 0.8

    def test_parse_json_accepts_object_only(self):
        parsed = DeepATCAgent._parse_json(
            '{"atc_result": {"atc_codes": []}}'
        )
        self.assertEqual(parsed["atc_result"]["atc_codes"], [])
        self.assertEqual(DeepATCAgent._parse_json("[]"), {})

    def test_missing_candidate_requests_refinement(self):
        result = {"atc_result": {"atc_codes": []}}
        self.assertTrue(self.agent._needs_refinement(result))

    def test_verified_high_confidence_candidate_does_not_refine(self):
        result = {
            "atc_result": {
                "atc_codes": [{"code": "N02BA01", "confidence": 0.95}]
            }
        }
        self.assertFalse(self.agent._needs_refinement(result))

    def test_invalid_atc_format_requests_refinement(self):
        result = {
            "atc_result": {
                "atc_codes": [{"code": "NOT-ATC", "confidence": 0.99}]
            }
        }
        self.assertTrue(self.agent._needs_refinement(result))

    def test_prefer_result_rejects_worse_refinement(self):
        current = {
            "atc_result": {
                "atc_codes": [{"code": "N02BA01", "confidence": 0.95}]
            }
        }
        candidate = {
            "atc_result": {
                "atc_codes": [{"code": "N02BA01", "confidence": 0.60}]
            }
        }
        self.assertIs(DeepATCAgent._prefer_result(current, candidate), current)


if __name__ == "__main__":
    unittest.main()
