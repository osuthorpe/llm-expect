import pytest
from llm_expect.metrics import MetricEvaluator, AccuracyMetric, SchemaFidelityMetric, SafetyMetric

def test_metric_supports():
    """Test that metrics correctly identify supported expectations."""
    accuracy = AccuracyMetric()
    schema = SchemaFidelityMetric()
    safety = SafetyMetric()
    
    # Accuracy supports
    assert accuracy.supports({"reference": "foo"})
    assert accuracy.supports({"contains": ["foo"]})
    assert accuracy.supports({"regex": "foo"})
    assert not accuracy.supports({"schema": {}})
    assert not accuracy.supports({"safe": True})
    
    # Schema supports
    assert schema.supports({"schema": {}})
    assert not schema.supports({"reference": "foo"})
    
    # Safety supports
    assert safety.supports({"safe": True})
    assert safety.supports({"no_harmful_content": True})
    assert not safety.supports({"reference": "foo"})

def test_evaluator_auto_selection():
    """Test that evaluator selects correct metrics based on expectations."""
    evaluator = MetricEvaluator()
    
    # Case 1: Reference only -> Accuracy
    expected = {"reference": "foo"}
    metrics = evaluator.evaluate_metrics("foo", expected, "test1", None, {})
    assert len(metrics) == 1
    assert metrics[0].name == "accuracy"
    assert metrics[0].passed
    
    # Case 2: Schema only -> Schema Fidelity
    expected = {"schema": {"type": "string"}}
    metrics = evaluator.evaluate_metrics('"foo"', expected, "test2", None, {})
    assert len(metrics) == 1
    assert metrics[0].name == "schema_fidelity"
    assert metrics[0].passed
    
    # Case 3: Mixed -> Both
    # Use a JSON string that is both valid JSON and matches the reference
    input_val = '{"a": 1}'
    expected = {"reference": '{"a": 1}', "schema": {"type": "object"}}
    metrics = evaluator.evaluate_metrics(input_val, expected, "test3", None, {})
    assert len(metrics) == 2
    names = {m.name for m in metrics}
    assert "accuracy" in names
    assert "schema_fidelity" in names
    assert all(m.passed for m in metrics)

def test_evaluator_explicit_selection():
    """Test that explicit metric selection overrides auto-selection."""
    evaluator = MetricEvaluator()
    
    # Even though schema is present, only run accuracy
    expected = {"reference": "foo", "schema": {"type": "string"}}
    metrics = evaluator.evaluate_metrics("foo", expected, "test4", ["accuracy"], {})
    assert len(metrics) == 1
    assert metrics[0].name == "accuracy"
