import pytest
from pathlib import Path
from llm_expect import llm_expect

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset for testing."""
    dataset_path = tmp_path / "test_data.jsonl"
    with open(dataset_path, "w") as f:
        for i in range(5):
            f.write(f'{{"id": "{i}", "input": "test", "expected": {{"reference": "test"}}}}\n')
    return str(dataset_path)

def test_html_report_generation(temp_dataset, tmp_path):
    """Test that HTML report is generated."""
    
    results_dir = tmp_path / "runs"
    
    @llm_expect(dataset=temp_dataset, results_dir=str(results_dir))
    def simple_func(input_str: str) -> str:
        return "test"
        
    result = simple_func.run_eval()
    
    run_dir = Path(result["run_dir"])
    html_report = run_dir / "report.html"
    
    assert html_report.exists()
    assert html_report.stat().st_size > 0
    
    # Check content
    content = html_report.read_text()
    assert "LLM Expect Report" in content
    assert "simple_func" in content
