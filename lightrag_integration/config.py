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
import logging
import logging.handlers
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
        auto_create_dirs: Whether to automatically create directories in __post_init__ (default: True)
        log_level: Logging level (from LIGHTRAG_LOG_LEVEL env var, default: "INFO")
        log_dir: Log directory path (from LIGHTRAG_LOG_DIR env var, default: "logs")
        enable_file_logging: Whether to enable file logging (from LIGHTRAG_ENABLE_FILE_LOGGING env var, default: True)
        log_max_bytes: Maximum log file size in bytes (from LIGHTRAG_LOG_MAX_BYTES env var, default: 10MB)
        log_backup_count: Number of backup log files to keep (from LIGHTRAG_LOG_BACKUP_COUNT env var, default: 5)
        log_filename: Name of the log file (default: "lightrag_integration.log")
    """
    
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    working_dir: Path = field(default_factory=lambda: Path(os.getenv("LIGHTRAG_WORKING_DIR", Path.cwd())))
    graph_storage_dir: Optional[Path] = None
    max_async: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ASYNC", "16")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_TOKENS", "32768")))
    auto_create_dirs: bool = True
    
    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LIGHTRAG_LOG_DIR", "logs")))
    enable_file_logging: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_FILE_LOGGING", "true").lower() in ("true", "1", "yes", "t", "on"))
    log_max_bytes: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_LOG_MAX_BYTES", "10485760")))
    log_backup_count: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_LOG_BACKUP_COUNT", "5")))
    log_filename: str = "lightrag_integration.log"
    
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
        
        # Ensure log_dir is a Path object and handle defaults
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        elif self.log_dir is None:
            self.log_dir = Path("logs")
        
        # Handle log_level validation and normalization
        if self.log_level is None:
            self.log_level = "INFO"
        else:
            # Normalize log level to uppercase
            self.log_level = self.log_level.upper()
            # Validate log level
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if self.log_level not in valid_levels:
                self.log_level = "INFO"  # Fall back to INFO for invalid levels
        
        # Automatically create necessary directories if requested
        if self.auto_create_dirs:
            try:
                # Create working directory
                self.working_dir.mkdir(parents=True, exist_ok=True)
                
                # Create graph storage directory
                self.graph_storage_dir.mkdir(parents=True, exist_ok=True)
                
                # Create log directory if file logging is enabled
                if self.enable_file_logging:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError, ValueError, TypeError) as e:
                # Handle errors gracefully but don't raise - let validation handle this
                # This allows the config to be created even if directories can't be created immediately
                # ValueError/TypeError can occur with invalid path characters
                pass
    
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
        
        # Validate logging parameters
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise LightRAGConfigError(f"log_level must be one of {valid_log_levels}, got: {self.log_level}")
        
        if self.log_max_bytes <= 0:
            raise LightRAGConfigError("log_max_bytes must be positive")
        
        if self.log_backup_count < 0:
            raise LightRAGConfigError("log_backup_count must be non-negative")
        
        # Validate log filename
        if not self.log_filename or not self.log_filename.strip():
            raise LightRAGConfigError("log_filename cannot be empty")
        
        # Check if log filename has valid extension
        if not self.log_filename.endswith('.log'):
            raise LightRAGConfigError("log_filename should end with '.log' extension")
        
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
            - Log directory (if file logging is enabled, with parent directories as needed)
        
        Raises:
            OSError: If directories cannot be created due to permissions or other issues
        """
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create graph storage directory
        self.graph_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory if file logging is enabled
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def setup_lightrag_logging(self, logger_name: str = "lightrag_integration") -> logging.Logger:
        """
        Set up LightRAG integration logging using the configuration parameters.
        
        This method creates a logger with both console and file handlers (if enabled),
        implements log rotation, and integrates with LightRAG's native logging patterns.
        
        Args:
            logger_name: Name of the logger to create/configure (default: "lightrag_integration")
            
        Returns:
            logging.Logger: Configured logger instance
            
        Raises:
            LightRAGConfigError: If logging setup fails due to configuration issues
        """
        try:
            # Get or create logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(self.log_level)
            logger.handlers = []  # Clear existing handlers
            logger.propagate = False
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
            
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(self.log_level)
            logger.addHandler(console_handler)
            
            # Add file handler if enabled
            if self.enable_file_logging:
                # Ensure log directory exists
                self.log_dir.mkdir(parents=True, exist_ok=True)
                
                # Construct log file path
                log_file_path = self.log_dir / self.log_filename
                
                try:
                    # Create rotating file handler
                    file_handler = logging.handlers.RotatingFileHandler(
                        filename=str(log_file_path),
                        maxBytes=self.log_max_bytes,
                        backupCount=self.log_backup_count,
                        encoding="utf-8",
                    )
                    file_handler.setFormatter(detailed_formatter)
                    file_handler.setLevel(self.log_level)
                    logger.addHandler(file_handler)
                    
                except (OSError, PermissionError) as e:
                    # Log warning but don't fail - continue with console logging only
                    logger.warning(f"Could not create log file at {log_file_path}: {e}")
                    logger.warning("Continuing with console logging only")
            
            return logger
            
        except Exception as e:
            raise LightRAGConfigError(f"Failed to set up logging: {e}") from e
    
    @classmethod
    def get_config(cls, 
                   source: Optional[Union[str, Path, Dict[str, Any]]] = None,
                   validate_config: bool = True,
                   ensure_dirs: bool = True,
                   **overrides) -> 'LightRAGConfig':
        """
        Primary factory function for creating and configuring LightRAGConfig instances.
        
        This is the recommended entry point for creating LightRAG configurations.
        It provides intelligent source detection, automatic validation, and
        directory creation with comprehensive error handling.
        
        Args:
            source: Configuration source. Can be:
                - None: Load from environment variables (default)
                - str/Path: Load from JSON file
                - dict: Load from dictionary
            validate_config: Whether to validate the configuration before returning
            ensure_dirs: Whether to ensure directories exist before returning
            **overrides: Additional configuration values to override
        
        Returns:
            LightRAGConfig: Fully configured and validated instance
        
        Raises:
            LightRAGConfigError: If configuration is invalid or cannot be created
            FileNotFoundError: If source file doesn't exist
            TypeError: If source type is unsupported
        
        Examples:
            # Load from environment with defaults
            config = LightRAGConfig.get_config()
            
            # Load from file with overrides
            config = LightRAGConfig.get_config(
                source="/path/to/config.json",
                max_async=32
            )
            
            # Load from dict with validation disabled
            config = LightRAGConfig.get_config(
                source={"api_key": "test"},
                validate_config=False
            )
        """
        try:
            # Determine source and create base configuration
            # Set auto_create_dirs based on ensure_dirs parameter
            if source is None:
                # Load from environment variables
                config = cls.from_environment(auto_create_dirs=ensure_dirs)
            elif isinstance(source, (str, Path)):
                # Load from file
                config = cls.from_file(source, auto_create_dirs=ensure_dirs)
            elif isinstance(source, dict):
                # Load from dictionary
                config = cls.from_dict(source, auto_create_dirs=ensure_dirs)
            else:
                raise TypeError(
                    f"Unsupported source type: {type(source)}. "
                    f"Expected None, str, Path, or dict."
                )
            
            # Apply any override values
            if overrides:
                working_dir_overridden = False
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        if key == "working_dir":
                            working_dir_overridden = True
                    else:
                        raise LightRAGConfigError(
                            f"Invalid configuration parameter: '{key}'. "
                            f"Valid parameters are: {', '.join(config.__dataclass_fields__.keys())}"
                        )
                
                # If working_dir was overridden and graph_storage_dir wasn't explicitly set,
                # reset graph_storage_dir to None so it gets recalculated based on new working_dir
                if working_dir_overridden and "graph_storage_dir" not in overrides:
                    config.graph_storage_dir = None
                
                # Re-run post-init processing to handle any Path conversions
                # and derived values after applying overrides
                config.__post_init__()
            
            # Ensure directories exist if requested
            if ensure_dirs:
                try:
                    config.ensure_directories()
                except OSError as e:
                    raise LightRAGConfigError(
                        f"Failed to create required directories: {e}"
                    ) from e
            
            # Validate configuration if requested
            if validate_config:
                config.validate()
            
            return config
            
        except (FileNotFoundError, TypeError) as e:
            # Re-raise these as they are already appropriate
            raise
        except LightRAGConfigError as e:
            # Re-raise LightRAGConfigError as-is
            raise
        except Exception as e:
            # Wrap any other exceptions in LightRAGConfigError
            raise LightRAGConfigError(
                f"Failed to create configuration: {e}"
            ) from e

    @classmethod
    def from_environment(cls, auto_create_dirs: bool = True) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from environment variables.
        
        This factory method creates a configuration by reading all relevant
        environment variables. It's equivalent to calling the default constructor
        but makes the intent explicit.
        
        Args:
            auto_create_dirs: Whether to automatically create directories during initialization
        
        Returns:
            LightRAGConfig: Configuration instance with values from environment
        """
        return cls(auto_create_dirs=auto_create_dirs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], auto_create_dirs: bool = True) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            auto_create_dirs: Whether to automatically create directories during initialization
        
        Returns:
            LightRAGConfig: Configuration instance with values from dictionary
        """
        # Handle Path objects in the dictionary
        config_dict = config_dict.copy()  # Don't modify original
        
        if 'working_dir' in config_dict:
            config_dict['working_dir'] = Path(config_dict['working_dir'])
        
        if 'graph_storage_dir' in config_dict:
            config_dict['graph_storage_dir'] = Path(config_dict['graph_storage_dir'])
        
        # Handle log_dir path object
        if 'log_dir' in config_dict:
            config_dict['log_dir'] = Path(config_dict['log_dir'])
        
        # Set auto_create_dirs if not already specified in the dictionary
        if 'auto_create_dirs' not in config_dict:
            config_dict['auto_create_dirs'] = auto_create_dirs
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], auto_create_dirs: bool = True) -> 'LightRAGConfig':
        """
        Create a LightRAGConfig instance from a JSON configuration file.
        
        Args:
            file_path: Path to the JSON configuration file
            auto_create_dirs: Whether to automatically create directories during initialization
        
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
        
        return cls.from_dict(config_dict, auto_create_dirs=auto_create_dirs)
    
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
            'max_tokens': self.max_tokens,
            'auto_create_dirs': self.auto_create_dirs,
            'log_level': self.log_level,
            'log_dir': str(self.log_dir),
            'enable_file_logging': self.enable_file_logging,
            'log_max_bytes': self.log_max_bytes,
            'log_backup_count': self.log_backup_count,
            'log_filename': self.log_filename
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
            f"max_tokens={self.max_tokens}, "
            f"auto_create_dirs={self.auto_create_dirs}, "
            f"log_level={self.log_level}, "
            f"log_dir={self.log_dir}, "
            f"enable_file_logging={self.enable_file_logging}, "
            f"log_max_bytes={self.log_max_bytes}, "
            f"log_backup_count={self.log_backup_count}, "
            f"log_filename={self.log_filename})"
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
            f"max_tokens={self.max_tokens}, "
            f"auto_create_dirs={self.auto_create_dirs}, "
            f"log_level='{self.log_level}', "
            f"log_dir=Path('{self.log_dir}'), "
            f"enable_file_logging={self.enable_file_logging}, "
            f"log_max_bytes={self.log_max_bytes}, "
            f"log_backup_count={self.log_backup_count}, "
            f"log_filename='{self.log_filename}')"
        )


def setup_lightrag_logging(
    config: Optional[LightRAGConfig] = None,
    logger_name: str = "lightrag_integration"
) -> logging.Logger:
    """
    Standalone utility function to set up LightRAG integration logging.
    
    This function provides a convenient way to set up logging for LightRAG integration
    without needing to instantiate a LightRAGConfig object first. It can use an existing
    configuration or create one from environment variables.
    
    Args:
        config: LightRAGConfig instance to use. If None, creates config from environment variables.
        logger_name: Name of the logger to create/configure (default: "lightrag_integration")
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        LightRAGConfigError: If logging setup fails due to configuration issues
        
    Examples:
        # Use with existing config
        config = LightRAGConfig.get_config()
        logger = setup_lightrag_logging(config)
        
        # Create config from environment and use it
        logger = setup_lightrag_logging()
        
        # Use with custom logger name
        logger = setup_lightrag_logging(logger_name="my_lightrag_app")
    """
    if config is None:
        # Create config from environment variables with minimal validation
        config = LightRAGConfig.get_config(validate_config=False, ensure_dirs=False)
    
    return config.setup_lightrag_logging(logger_name)