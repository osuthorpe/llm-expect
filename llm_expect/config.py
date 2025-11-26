"""
Configuration management for LLM Expect.

Handles loading configuration from various sources (environment, files, parameters)
with proper validation and defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import ConfigurationError, ValidationError
from .models import JudgeConfig, LLMExpectConfig


class ConfigManager:
    """Manages LLM Expect configuration from multiple sources."""
    
    def __init__(self):
        self.env_prefix = "LLM_EXPECT_"
    
    def create_config(
        self,
        dataset: str,
        tests: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
        sample_size: Optional[int] = None,
        shuffle: bool = False,
        cache: bool = True,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        fail_fast: bool = False,
        timeout: int = 60,
        **kwargs
    ) -> LLMExpectConfig:
        """
        Create a LLMExpectConfig from parameters with environment variable fallbacks.
        
        Args:
            dataset: Path to dataset file
            tests: List of metrics to evaluate
            thresholds: Threshold values for metrics
            judge_provider: LLM judge provider ("openai", "anthropic", "bedrock")
            judge_model: Model name for judge
            sample_size: Number of examples to sample
            shuffle: Whether to shuffle examples
            cache: Whether to cache results
            cache_dir: Cache directory path
            results_dir: Results directory path
            fail_fast: Whether to stop on first failure
            timeout: Function execution timeout
            **kwargs: Additional configuration parameters
            
        Returns:
            Validated LLMExpectConfig object
            
        Raises:
            ConfigurationError: If configuration is invalid
            ValidationError: If Pydantic validation fails
        """
        try:
            # Build configuration dictionary with defaults and environment fallbacks
            config_dict = {
                "dataset": dataset,
                "tests": tests or self._get_env_list("TESTS") or [],  # Convert None to []
                "thresholds": thresholds or self._build_thresholds_from_env(tests),
                "sample_size": sample_size or self._get_env_int("SAMPLE_SIZE"),
                "shuffle": shuffle or self._get_env_bool("SHUFFLE", False),
                "cache": cache and self._get_env_bool("CACHE", True),  # Allow disabling cache
                "cache_dir": cache_dir or self._get_env_str("CACHE_DIR") or ".llm_expect_cache",
                "results_dir": results_dir or self._get_env_str("RESULTS_DIR") or "runs",
                "fail_fast": fail_fast or self._get_env_bool("FAIL_FAST", False),
                "timeout": timeout or self._get_env_int("TIMEOUT") or 60,
                "save_results": kwargs.get("save_results", True),
                "parallel": kwargs.get("parallel", False)
            }
            
            # Handle judge configuration if provider specified
            if judge_provider:
                judge_config = self._create_judge_config(judge_provider, judge_model)
                config_dict["judge"] = judge_config
            
            # Create and validate config
            config = LLMExpectConfig(**config_dict)
            
            # Post-creation validation
            self._validate_paths(config)
            
            return config
            
        except Exception as e:
            if hasattr(e, 'errors'):
                # Pydantic validation error
                raise ValidationError.from_pydantic(e)
            else:
                raise ConfigurationError(f"Configuration creation failed: {str(e)}")
    
    def _create_judge_config(
        self, 
        provider: str, 
        model: Optional[str] = None
    ) -> JudgeConfig:
        """Create judge configuration with environment fallbacks."""
        
        # Default models for each provider
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
            "bedrock": "anthropic.claude-v2"
        }
        
        if provider not in default_models:
            raise ConfigurationError(
                f"Unsupported judge provider: {provider}",
                config_key="judge_provider",
                config_value=provider,
                valid_options=list(default_models.keys())
            )
        
        judge_dict = {
            "provider": provider,
            "model": model or self._get_env_str(f"JUDGE_MODEL") or default_models[provider],
            "api_key": self._get_env_str(f"JUDGE_API_KEY") or self._get_provider_api_key(provider),
            "base_url": self._get_env_str("JUDGE_BASE_URL"),
            "timeout": self._get_env_int("JUDGE_TIMEOUT") or 30,
            "max_retries": self._get_env_int("JUDGE_MAX_RETRIES") or 3,
            "temperature": self._get_env_float("JUDGE_TEMPERATURE") or 0.0
        }
        
        return JudgeConfig(**judge_dict)
    
    def _get_provider_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider from standard environment variables."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID"  # Bedrock uses AWS credentials
        }
        
        if provider in env_vars:
            return os.getenv(env_vars[provider])
        return None
    
    def _build_thresholds_from_env(self, tests: Optional[List[str]]) -> Dict[str, float]:
        """Build thresholds dictionary from environment variables."""
        thresholds = {}
        
        # Global threshold
        global_threshold = self._get_env_float("THRESHOLD")
        if global_threshold is not None:
            if tests:
                for test in tests:
                    thresholds[test] = global_threshold
        
        # Per-metric thresholds
        for test_name in ["accuracy", "schema_fidelity", "instruction_adherence", "safety"]:
            threshold = self._get_env_float(f"THRESHOLD_{test_name.upper()}")
            if threshold is not None:
                thresholds[test_name] = threshold
        
        # Default thresholds if nothing specified
        if not thresholds and tests:
            for test in tests:
                if test == "safety":
                    thresholds[test] = 1.0  # Safety should be perfect by default
                else:
                    thresholds[test] = 0.8  # Good default for other metrics
        
        return thresholds
    
    def _validate_paths(self, config: LLMExpectConfig) -> None:
        """Validate file and directory paths in configuration."""
        
        # Check dataset file exists
        dataset_path = Path(config.dataset)
        if not dataset_path.exists():
            raise ConfigurationError(
                f"Dataset file not found: {config.dataset}",
                config_key="dataset",
                config_value=config.dataset
            )
        
        # Ensure directories exist or can be created
        for dir_name, dir_path in [("cache_dir", config.cache_dir), ("results_dir", config.results_dir)]:
            path = Path(dir_path)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ConfigurationError(
                    f"Cannot create {dir_name}: {dir_path} (permission denied)",
                    config_key=dir_name,
                    config_value=dir_path
                )
            except Exception as e:
                raise ConfigurationError(
                    f"Cannot create {dir_name}: {dir_path} ({str(e)})",
                    config_key=dir_name,
                    config_value=dir_path
                )
    
    def _get_env_str(self, key: str) -> Optional[str]:
        """Get string value from environment variable."""
        value = os.getenv(f"{self.env_prefix}{key}")
        return value if value else None
    
    def _get_env_int(self, key: str) -> Optional[int]:
        """Get integer value from environment variable."""
        value = self._get_env_str(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                raise ConfigurationError(
                    f"Environment variable {self.env_prefix}{key} must be an integer, got: {value}",
                    config_key=key,
                    config_value=value
                )
        return None
    
    def _get_env_float(self, key: str) -> Optional[float]:
        """Get float value from environment variable."""
        value = self._get_env_str(key)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                raise ConfigurationError(
                    f"Environment variable {self.env_prefix}{key} must be a number, got: {value}",
                    config_key=key,
                    config_value=value
                )
        return None
    
    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = self._get_env_str(key)
        if value is None:
            return default
        
        # Convert string to boolean
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False
        else:
            raise ConfigurationError(
                f"Environment variable {self.env_prefix}{key} must be a boolean (true/false), got: {value}",
                config_key=key,
                config_value=value
            )
    
    def _get_env_list(self, key: str) -> Optional[List[str]]:
        """Get list value from environment variable (comma-separated)."""
        value = self._get_env_str(key)
        if value is not None:
            # Split by comma and strip whitespace
            return [item.strip() for item in value.split(",") if item.strip()]
        return None
    
    def load_from_file(self, file_path: str) -> LLMExpectConfig:
        """
        Load configuration from JSON or TOML file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Validated LLMExpectConfig object
            
        Raises:
            ConfigurationError: If file loading fails
        """
        config_path = Path(file_path)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_key="config_file",
                config_value=file_path
            )
        
        try:
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    data = json.load(f)
            elif config_path.suffix.lower() in ['.toml', '.tml']:
                try:
                    import tomllib  # Python 3.11+
                except ImportError:
                    try:
                        import tomli as tomllib
                    except ImportError:
                        raise ConfigurationError(
                            "TOML support requires 'tomli' package for Python < 3.11",
                            config_key="config_file",
                            config_value=file_path
                        )
                
                with open(config_path, 'rb') as f:
                    data = tomllib.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}",
                    config_key="config_file",
                    config_value=file_path,
                    valid_options=[".json", ".toml"]
                )
            
            # Extract judge config if present
            judge_data = data.pop("judge", None)
            if judge_data:
                data["judge"] = JudgeConfig(**judge_data)
            
            return LLMExpectConfig(**data)
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration from {file_path}: {str(e)}",
                config_key="config_file",
                config_value=file_path
            )
    
    def save_to_file(self, config: LLMExpectConfig, file_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration to save
            file_path: Path where to save the configuration
            
        Raises:
            ConfigurationError: If file saving fails
        """
        try:
            config_path = Path(file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and handle judge config
            data = config.model_dump()
            if data.get("judge"):
                # Exclude sensitive information like API keys
                judge_data = data["judge"].copy()
                if "api_key" in judge_data:
                    judge_data["api_key"] = "[REDACTED]"
                data["judge"] = judge_data
            
            import json
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {file_path}: {str(e)}",
                config_key="config_file",
                config_value=file_path
            )


# Global config manager instance
config_manager = ConfigManager()