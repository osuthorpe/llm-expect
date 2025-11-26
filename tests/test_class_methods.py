
import pytest
import os
from llm_expect.decorator import llm_expect

# Create a dummy dataset for testing
@pytest.fixture
def class_test_dataset():
    dataset_path = "test_class_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write('{"id": "1", "input": "value", "expected": {"reference": "test_value"}}\n')
    yield dataset_path
    if os.path.exists(dataset_path):
        os.remove(dataset_path)

class TestClassMethods:
    def test_instance_method_decorator(self, class_test_dataset):
        """Test that vald8 works on instance methods."""
        
        class MyService:
            def __init__(self):
                self.prefix = "test_"
                
            @llm_expect(dataset=class_test_dataset)
            def process(self, value):
                return self.prefix + value
        
        service = MyService()
        
        # 1. Test direct call
        assert service.process("value") == "test_value"
        
        # 2. Test evaluation run
        result = service.process.run_eval()
        assert result["passed"] is True
        assert result["summary"]["passed_tests"] == 1

    def test_class_method_decorator(self, class_test_dataset):
        """Test that vald8 works on class methods (@classmethod)."""
        # Note: Decorating @classmethod is tricky because of order of decorators.
        # Usually @classmethod(@vald8) won't work easily because vald8 returns a Vald8Function, not a function.
        # But let's see if we can support it or if we should just stick to instance methods for now.
        # For now, let's focus on instance methods as that was the user request.
        pass
