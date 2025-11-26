"""
Dataset loading and validation for Vald8.

Handles JSONL file parsing, validation, and preprocessing for evaluation datasets.
"""

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional

from .errors import DatasetValidationError, ValidationError
from .models import DatasetExample


class DatasetLoader:
    """Loads and validates JSONL datasets for evaluation."""
    
    def __init__(self, file_path: str, sample_size: Optional[int] = None, shuffle: bool = False):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.shuffle = shuffle
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that the dataset file exists and is readable."""
        if not self.file_path.exists():
            raise DatasetValidationError(
                f"Dataset file not found: {self.file_path}",
                file_path=str(self.file_path)
            )
        
        if not self.file_path.is_file():
            raise DatasetValidationError(
                f"Dataset path is not a file: {self.file_path}",
                file_path=str(self.file_path)
            )
        
        if not self.file_path.suffix == '.jsonl':
            raise DatasetValidationError(
                f"Dataset file must have .jsonl extension, got: {self.file_path.suffix}",
                file_path=str(self.file_path)
            )
    
    def load(self) -> List[DatasetExample]:
        """
        Load and validate the complete dataset.
        
        Returns:
            List of validated DatasetExample objects
            
        Raises:
            DatasetValidationError: If file format or content is invalid
        """
        examples = list(self._load_raw_examples())
        
        if not examples:
            raise DatasetValidationError(
                "Dataset is empty",
                file_path=str(self.file_path)
            )
        
        # Apply sampling if specified
        if self.sample_size is not None:
            if self.sample_size > len(examples):
                raise DatasetValidationError(
                    f"Sample size {self.sample_size} is larger than dataset size {len(examples)}",
                    file_path=str(self.file_path)
                )
            
            if self.shuffle:
                random.shuffle(examples)
            
            examples = examples[:self.sample_size]
        elif self.shuffle:
            random.shuffle(examples)
        
        return examples
    
    def _load_raw_examples(self) -> Iterator[DatasetExample]:
        """
        Load raw examples from JSONL file with line-by-line validation.
        
        Yields:
            DatasetExample objects
            
        Raises:
            DatasetValidationError: If JSON parsing or validation fails
        """
        seen_ids = set()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON
                        raw_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetValidationError(
                            f"Invalid JSON on line {line_num}: {str(e)}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    
                    # Validate with Pydantic
                    try:
                        example = DatasetExample(**raw_data)
                    except Exception as e:
                        # Convert Pydantic validation errors to our format
                        if hasattr(e, 'errors'):
                            error_details = []
                            for error in e.errors():
                                error_details.append({
                                    "field": " -> ".join(str(x) for x in error.get("loc", [])),
                                    "message": error.get("msg", ""),
                                    "value": error.get("input")
                                })
                            
                            error_msg = "Validation errors:\n" + "\n".join(
                                f"  {err['field']}: {err['message']}" for err in error_details
                            )
                        else:
                            error_msg = str(e)
                        
                        raise DatasetValidationError(
                            f"Validation failed on line {line_num}: {error_msg}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    
                    # Check for duplicate IDs
                    if example.id in seen_ids:
                        raise DatasetValidationError(
                            f"Duplicate test ID '{example.id}' on line {line_num}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    
                    seen_ids.add(example.id)
                    yield example
                    
        except FileNotFoundError:
            raise DatasetValidationError(
                f"Dataset file not found: {self.file_path}",
                file_path=str(self.file_path)
            )
        except PermissionError:
            raise DatasetValidationError(
                f"Permission denied reading dataset file: {self.file_path}",
                file_path=str(self.file_path)
            )
        except Exception as e:
            if isinstance(e, DatasetValidationError):
                raise
            raise DatasetValidationError(
                f"Unexpected error loading dataset: {str(e)}",
                file_path=str(self.file_path)
            )
    
    def count_examples(self) -> int:
        """
        Count total examples in the dataset without loading them all into memory.
        
        Returns:
            Number of non-empty lines in the dataset
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            raise DatasetValidationError(
                f"Error counting examples: {str(e)}",
                file_path=str(self.file_path)
            )
    
    def validate_format(self) -> List[str]:
        """
        Validate dataset format without loading all data.
        
        Returns:
            List of validation warnings (empty if no issues)
            
        Raises:
            DatasetValidationError: If critical format errors are found
        """
        warnings = []
        seen_ids = set()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Quick JSON validation
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DatasetValidationError(
                            f"Invalid JSON on line {line_num}: {str(e)}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    
                    # Check required fields
                    required_fields = ['id', 'input', 'expected']
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        raise DatasetValidationError(
                            f"Missing required fields on line {line_num}: {missing_fields}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    
                    # Check for duplicate IDs
                    test_id = data.get('id')
                    if test_id in seen_ids:
                        raise DatasetValidationError(
                            f"Duplicate test ID '{test_id}' on line {line_num}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                            line_content=line
                        )
                    seen_ids.add(test_id)
                    
                    # Check expected format
                    expected = data.get('expected', {})
                    if not isinstance(expected, dict) or not expected:
                        warnings.append(
                            f"Line {line_num}: 'expected' should be a non-empty dictionary"
                        )
                    
                    # Warn about unknown expected types
                    known_expected_keys = {
                        'reference', 'contains', 'regex', 'schema', 'safe', 
                        'no_harmful_content', 'instruction_adherence'
                    }
                    unknown_keys = set(expected.keys()) - known_expected_keys
                    if unknown_keys:
                        warnings.append(
                            f"Line {line_num}: Unknown expected keys: {unknown_keys}"
                        )
        
        except Exception as e:
            if isinstance(e, DatasetValidationError):
                raise
            raise DatasetValidationError(
                f"Error validating dataset format: {str(e)}",
                file_path=str(self.file_path)
            )
        
        return warnings


def load_dataset(
    file_path: str, 
    sample_size: Optional[int] = None, 
    shuffle: bool = False
) -> List[DatasetExample]:
    """
    Convenience function to load a dataset.
    
    Args:
        file_path: Path to the JSONL dataset file
        sample_size: Optional number of examples to sample
        shuffle: Whether to shuffle examples
    
    Returns:
        List of DatasetExample objects
        
    Raises:
        DatasetValidationError: If dataset loading fails
    """
    loader = DatasetLoader(file_path, sample_size, shuffle)
    return loader.load()


def validate_dataset_format(file_path: str) -> List[str]:
    """
    Convenience function to validate dataset format.
    
    Args:
        file_path: Path to the JSONL dataset file
    
    Returns:
        List of validation warnings
        
    Raises:
        DatasetValidationError: If critical validation errors are found
    """
    loader = DatasetLoader(file_path)
    return loader.validate_format()