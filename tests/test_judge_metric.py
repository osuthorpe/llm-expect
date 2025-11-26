import pytest
from unittest.mock import MagicMock
from llm_expect.metrics import CustomJudgeMetric, MetricResult
from llm_expect.judges import BaseJudgeProvider

class MockJudgeProvider(BaseJudgeProvider):
    def evaluate_instruction_adherence(self, actual, instruction, test_id):
        return 0.8
        
    def evaluate_safety(self, actual, test_id):
        return 0.9
        
    def evaluate_custom(self, actual, prompt, test_id):
        if "polite" in prompt.lower():
            return 0.95
        return 0.5

def test_custom_judge_metric_supports():
    metric = CustomJudgeMetric()
    assert metric.supports({"judge": {"prompt": "foo"}})
    assert not metric.supports({"reference": "bar"})

def test_custom_judge_metric_evaluate():
    provider = MockJudgeProvider(MagicMock())
    metric = CustomJudgeMetric(judge_provider=provider)
    
    expected = {
        "judge": {
            "prompt": "Is this polite?"
        },
        "threshold": 0.9
    }
    
    result = metric.evaluate("Thank you very much", expected, "test_1")
    
    assert isinstance(result, MetricResult)
    assert result.name == "custom_judge"
    assert result.score == 0.95
    assert result.passed is True
    assert result.details["method"] == "llm_judge"
    assert result.details["prompt"] == "Is this polite?"

def test_custom_judge_metric_evaluate_fail():
    provider = MockJudgeProvider(MagicMock())
    metric = CustomJudgeMetric(judge_provider=provider)
    
    expected = {
        "judge": {
            "prompt": "Is this rude?" # Mock returns 0.5 for non-polite prompts
        },
        "threshold": 0.8
    }
    
    result = metric.evaluate("Whatever", expected, "test_2")
    
    assert result.score == 0.5
    assert result.passed is False

def test_custom_judge_metric_no_provider():
    metric = CustomJudgeMetric(judge_provider=None)
    
    expected = {"judge": {"prompt": "foo"}}
    
    # Should raise error or return failed result depending on implementation
    # Current impl raises MetricCalculationError which is caught and returned as failed result in evaluator
    # But calling evaluate directly raises it
    from llm_expect.errors import MetricCalculationError
    with pytest.raises(MetricCalculationError):
        metric.evaluate("actual", expected, "test_3")
