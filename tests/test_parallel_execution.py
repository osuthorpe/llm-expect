import time
import pytest
from llm_expect import llm_expect

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset for testing."""
    dataset_path = tmp_path / "test_data.jsonl"
    with open(dataset_path, "w") as f:
        for i in range(5):
            f.write(f'{{"id": "{i}", "input": "test", "expected": {{"reference": "test"}}}}\n')
    return str(dataset_path)

def test_parallel_execution(temp_dataset):
    """Test that parallel execution is faster than serial for slow functions."""
    
    @llm_expect(dataset=temp_dataset, parallel=True)
    def slow_parallel(input_str: str) -> str:
        time.sleep(0.5)
        return "test"
        
    @llm_expect(dataset=temp_dataset, parallel=False)
    def slow_serial(input_str: str) -> str:
        time.sleep(0.5)
        return "test"
    
    # Measure parallel time
    start = time.time()
    slow_parallel.run_eval()
    parallel_time = time.time() - start
    
    # Measure serial time
    start = time.time()
    slow_serial.run_eval()
    serial_time = time.time() - start
    
    # Parallel should be significantly faster (at least 2x for 5 items)
    # 5 * 0.5s = 2.5s serial
    # Parallel should be ~0.5s + overhead
    assert parallel_time < serial_time
    assert parallel_time < 1.5  # Should be well under 2.5s
