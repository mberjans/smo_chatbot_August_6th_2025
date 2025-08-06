"""
LightRAGConfig dataclass for Clinical Metabolomics Oracle LightRAG integration.

This module provides comprehensive configuration management for LightRAG integration
with the Clinical Metabolomics Oracle chatbot. It includes environment variable
handling, validation, directory management, and factory methods for creating
configurations from various sources.

Classes:
    - LightRAGConfigError: Custom exception for configuration errors
    - LightRAGConfig: Main configuration dataclass with validation and utility methods

The configuration system supports:
    - Environment variable loading with defaults
    - Configuration validation with detailed error messages
    - Directory creation and path management
    - Factory methods for different configuration sources
    - Secure string representations that mask API keys
    - Serialization and deserialization support
"""

import os
import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union


class LightRAGConfigError(Exception):
    """Custom exception for LightRAG configuration errors."""
    pass


@dataclass
class LightRAGConfig:
    """
    Comprehensive configuration class for LightRAG integration.
    
    This dataclass manages all configuration parameters for the LightRAG system,
    including API keys, model settings, directory paths, and performance limits.
    It supports environment variable loading, validation, and various factory methods.
    
    Attributes:
        api_key: OpenAI API key (from OPENAI_API_KEY env var)
        model: LLM model to use (from LIGHTRAG_MODEL env var, default: "gpt-4o-mini")
        embedding_model: Embedding model (from LIGHTRAG_EMBEDDING_MODEL env var, default: "text-embedding-3-small")
        working_dir: Working directory path (from LIGHTRAG_WORKING_DIR env var, default: current directory)
        graph_storage_dir: Graph storage directory (derived from working_dir / "lightrag")
        max_async: Maximum async operations (from LIGHTRAG_MAX_ASYNC env var, default: 16)
        max_tokens: Maximum token limit (from LIGHTRAG_MAX_TOKENS env var, default: 32768)
    """
    
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    working_dir: Path = field(default_factory=lambda: Path(os.getenv("LIGHTRAG_WORKING_DIR", Path.cwd())))
    graph_storage_dir: Optional[Path] = None
    max_async: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ASYNC", "16")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_TOKENS", "32768")))
    
    def __post_init__(self):
        """Post-initialization processing to handle Path objects and derived values."""
        # Ensure working_dir is a Path object
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)
        elif self.working_dir is None:
            self.working_dir = Path.cwd()
        
        # Set default graph_storage_dir if not provided
        if self.graph_storage_dir is None:
            self.graph_storage_dir = self.working_dir / "lightrag"
        elif isinstance(self.graph_storage_dir, str):
            self.graph_storage_dir = Path(self.graph_storage_dir)
        
        # Handle None values for string fields by using defaults
        if self.model is None:
            self.model = "gpt-4o-mini"
        if self.embedding_model is None:
            self.embedding_model = "text-embedding-3-small"
    
    def validate(self) -> None:
        """
        Validate the configuration and raise LightRAGConfigError if invalid.
        
        Validates:
            - API key is present and not empty
            - Numeric values are positive
            - Working directory exists or can be created
        
        Raises:
            LightRAGConfigError: If any validation check fails
        """
        # Validate API key
        if not self.api_key or not self.api_key.strip():
            raise LightRAGConfigError("API key is required and cannot be empty")
        
        # Validate numeric parameters
        if self.max_async <= 0:
            raise LightRAGConfigError("max_async must be positive")
        
        if self.max_tokens <= 0:
            raise LightRAGConfigError("max_tokens must be positive")
        
        # Validate working directory
        if not self.working_dir.exists():
            try:
                # Try to create the directory to see if it's possible
                self.working_dir.mkdir(parents=True, exist_ok=True)
                # Remove it if we just created it for testing
                if not any(self.working_dir.iterdir()):
                    self.working_dir.rmdir()
            except (OSError, PermissionError):
                raise LightRAGConfigError(f"Working directory does not exist and cannot be created: {self.working_dir}")
        
        if not self.working_dir.is_dir():
            raise LightRAGConfigError(f"Working directory path is not a directory: {self.working_dir}")
    
    def ensure_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        
        Creates:
            - Working directory (with parent directories as needed)
            - Graph storage directory (with parent directories as needed)
        
        Raises:
            OSError: If directories cannot be created due to permissions or other issues
        """
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create graph storage directory
        self.graph_storage_dir.mkdir(parents=True, exist_ok=True)
    
    def get_absolute_path(self, path: Union[str, Path]) -> Path:
        """
        Convert a path to an absolute path, resolving relative paths from working_dir.
        
        Args:
            path: The path to make absolute (string or Path object)
        
        Returns:
            Path: Absolute path object
        """
        path_obj = Path(path) if isinstance(path, str) else path
        
        if path_obj.is_absolute():
            return path_obj
        else:
            return (self.working_dir / path_obj).resolve()
    
    @classmethod
    def from_environment(cls) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from environment variables.
        
        This factory method creates a configuration by reading all relevant
        environment variables. It's equivalent to calling the default constructor
        but makes the intent explicit.
        
        Returns:
            LightRAGConfig: Configuration instance with values from environment
        """
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        
        Returns:
            LightRAGConfig: Configuration instance with values from dictionary
        """
        # Handle Path objects in the dictionary
        if 'working_dir' in config_dict:
            config_dict = config_dict.copy()  # Don't modify original
            config_dict['working_dir'] = Path(config_dict['working_dir'])
        
        if 'graph_storage_dir' in config_dict:
            if config_dict not in locals():  # Only copy if we haven't already
                config_dict = config_dict.copy()
            config_dict['graph_storage_dir'] = Path(config_dict['graph_storage_dir'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from a JSON configuration file.
        
        Args:
            file_path: Path to the JSON configuration file
        
        Returns:
            LightRAGConfig: Configuration instance with values from file
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            LightRAGConfigError: If the configuration is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise LightRAGConfigError(f"Invalid JSON in configuration file {file_path}: {e}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return {
            'api_key': self.api_key,
            'model': self.model,
            'embedding_model': self.embedding_model,
            'working_dir': str(self.working_dir),
            'graph_storage_dir': str(self.graph_storage_dir),
            'max_async': self.max_async,
            'max_tokens': self.max_tokens
        }
    
    def copy(self) -> 'LightRAGConfig':
        """
        Create a deep copy of the configuration.
        
        Returns:
            LightRAGConfig: Deep copy of this configuration instance
        """
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        """
        String representation with masked API key for security.
        
        Returns:
            str: Human-readable string representation
        """
        masked_key = "***masked***" if self.api_key else None
        return (
            f"LightRAGConfig("
            f"api_key={masked_key}, "
            f"model={self.model}, "
            f"embedding_model={self.embedding_model}, "
            f"working_dir={self.working_dir}, "
            f"graph_storage_dir={self.graph_storage_dir}, "
            f"max_async={self.max_async}, "
            f"max_tokens={self.max_tokens})"
        )
    
    def __repr__(self) -> str:
        """
        Detailed representation with masked API key for security.
        
        Returns:
            str: Detailed string representation suitable for debugging
        """
        masked_key = "***masked***" if self.api_key else None
        return (
            f"LightRAGConfig("
            f"api_key='{masked_key}', "
            f"model='{self.model}', "
            f"embedding_model='{self.embedding_model}', "
            f"working_dir=Path('{self.working_dir}'), "
            f"graph_storage_dir=Path('{self.graph_storage_dir}'), "
            f"max_async={self.max_async}, "
            f"max_tokens={self.max_tokens})"
        )