# Makefile for Factual Accuracy Validation Test Suite
# Clinical Metabolomics Oracle - LightRAG Integration Project
#
# Usage:
#   make test-all          # Run all validation tests
#   make test-unit         # Run unit tests
#   make test-integration  # Run integration tests
#   make test-performance  # Run performance tests
#   make test-coverage     # Run tests with coverage analysis
#   make validate-coverage # Validate coverage meets requirements
#   make clean            # Clean test artifacts
#   make help             # Show available targets

# Configuration
PYTHON = python3
TEST_DIR = .
PROJECT_ROOT = ..
MIN_COVERAGE = 90
PARALLEL_JOBS = auto

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)Factual Accuracy Validation Test Suite$(NC)"
	@echo "$(CYAN)======================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  $(GREEN)make test-all$(NC)          - Run comprehensive test suite"
	@echo "  $(GREEN)make test-coverage$(NC)     - Run tests with coverage analysis"
	@echo "  $(GREEN)make validate-coverage$(NC) - Check coverage meets requirements"
	@echo "  $(GREEN)make test-performance$(NC)  - Run performance benchmarks"

.PHONY: setup
setup: ## Install test dependencies
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	pip install -r test_requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

.PHONY: test-all
test-all: ## Run all validation test suites
	@echo "$(YELLOW)Running comprehensive validation test suite...$(NC)"
	$(PYTHON) run_validation_tests.py --suite all --verbose
	@echo "$(GREEN)All tests completed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTHON) run_validation_tests.py --suite unit --verbose
	@echo "$(GREEN)Unit tests completed$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTHON) run_validation_tests.py --suite integration --verbose
	@echo "$(GREEN)Integration tests completed$(NC)"

.PHONY: test-performance
test-performance: ## Run performance tests with benchmarking
	@echo "$(YELLOW)Running performance tests with benchmarking...$(NC)"
	$(PYTHON) run_validation_tests.py --suite performance --benchmark --verbose
	@echo "$(GREEN)Performance tests completed$(NC)"

.PHONY: test-error-handling
test-error-handling: ## Run error handling tests
	@echo "$(YELLOW)Running error handling tests...$(NC)"
	$(PYTHON) run_validation_tests.py --suite error_handling --verbose
	@echo "$(GREEN)Error handling tests completed$(NC)"

.PHONY: test-mocks
test-mocks: ## Run mock-based isolation tests
	@echo "$(YELLOW)Running mock-based isolation tests...$(NC)"
	$(PYTHON) run_validation_tests.py --suite mock --verbose
	@echo "$(GREEN)Mock tests completed$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with comprehensive coverage analysis
	@echo "$(YELLOW)Running tests with coverage analysis...$(NC)"
	$(PYTHON) run_validation_tests.py --suite all --coverage --verbose
	@echo "$(GREEN)Coverage analysis completed$(NC)"

.PHONY: test-parallel
test-parallel: ## Run tests in parallel for faster execution
	@echo "$(YELLOW)Running tests in parallel...$(NC)"
	$(PYTHON) run_validation_tests.py --suite all --parallel $(PARALLEL_JOBS) --verbose
	@echo "$(GREEN)Parallel tests completed$(NC)"

.PHONY: test-fast
test-fast: ## Run fast test suite (unit + mocks)
	@echo "$(YELLOW)Running fast test suite...$(NC)"
	$(PYTHON) run_validation_tests.py --suite unit --parallel $(PARALLEL_JOBS)
	$(PYTHON) run_validation_tests.py --suite mock --parallel $(PARALLEL_JOBS)
	@echo "$(GREEN)Fast tests completed$(NC)"

.PHONY: validate-coverage
validate-coverage: ## Validate coverage meets minimum requirements
	@echo "$(YELLOW)Validating test coverage (minimum: $(MIN_COVERAGE)%)...$(NC)"
	$(PYTHON) validate_test_coverage.py --validate --min-coverage $(MIN_COVERAGE)
	@echo "$(GREEN)Coverage validation completed$(NC)"

.PHONY: analyze-coverage
analyze-coverage: ## Run comprehensive coverage analysis
	@echo "$(YELLOW)Running comprehensive coverage analysis...$(NC)"
	$(PYTHON) validate_test_coverage.py --analyze --min-coverage $(MIN_COVERAGE) --format html
	@echo "$(GREEN)Coverage analysis completed$(NC)"

.PHONY: report-coverage
report-coverage: ## Generate coverage reports in multiple formats
	@echo "$(YELLOW)Generating coverage reports...$(NC)"
	$(PYTHON) validate_test_coverage.py --analyze --format json
	$(PYTHON) validate_test_coverage.py --analyze --format html
	$(PYTHON) validate_test_coverage.py --analyze --format text
	@echo "$(GREEN)Coverage reports generated$(NC)"

.PHONY: test-ci
test-ci: ## Run tests suitable for CI/CD (with validation)
	@echo "$(YELLOW)Running CI/CD test suite...$(NC)"
	$(PYTHON) run_validation_tests.py --suite all --coverage --fail-fast
	$(PYTHON) validate_test_coverage.py --validate --min-coverage $(MIN_COVERAGE)
	@echo "$(GREEN)CI/CD tests completed successfully$(NC)"

.PHONY: test-debug
test-debug: ## Run tests in debug mode with detailed output
	@echo "$(YELLOW)Running tests in debug mode...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -m validation -v -s --tb=long --showlocals
	@echo "$(GREEN)Debug tests completed$(NC)"

.PHONY: test-specific
test-specific: ## Run specific test file (usage: make test-specific FILE=test_file.py)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: Please specify FILE parameter$(NC)"; \
		echo "$(YELLOW)Usage: make test-specific FILE=test_accuracy_scorer_comprehensive.py$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running specific test file: $(FILE)...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/$(FILE) -v
	@echo "$(GREEN)Specific test completed$(NC)"

.PHONY: benchmark
benchmark: ## Run performance benchmarks only
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -m performance_validation --benchmark-only
	@echo "$(GREEN)Benchmarks completed$(NC)"

.PHONY: test-memory
test-memory: ## Run tests with memory profiling
	@echo "$(YELLOW)Running tests with memory profiling...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -m performance_validation --profile-mem
	@echo "$(GREEN)Memory profiling completed$(NC)"

.PHONY: test-stress
test-stress: ## Run stress tests for system limits
	@echo "$(YELLOW)Running stress tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -m "performance_validation and not slow" -v
	@echo "$(GREEN)Stress tests completed$(NC)"

.PHONY: clean
clean: ## Clean test artifacts and temporary files
	@echo "$(YELLOW)Cleaning test artifacts...$(NC)"
	find $(TEST_DIR) -type f -name "*.pyc" -delete
	find $(TEST_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(TEST_DIR) -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find $(TEST_DIR) -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf $(TEST_DIR)/.pytest_cache 2>/dev/null || true
	rm -rf $(TEST_DIR)/validation_test_results/* 2>/dev/null || true
	rm -rf $(TEST_DIR)/coverage_results/* 2>/dev/null || true
	rm -rf $(TEST_DIR)/performance_test_results/* 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed$(NC)"

.PHONY: clean-reports
clean-reports: ## Clean only test reports and results
	@echo "$(YELLOW)Cleaning test reports...$(NC)"
	rm -rf $(TEST_DIR)/validation_test_results/* 2>/dev/null || true
	rm -rf $(TEST_DIR)/coverage_results/* 2>/dev/null || true
	rm -rf $(TEST_DIR)/performance_test_results/* 2>/dev/null || true
	@echo "$(GREEN)Report cleanup completed$(NC)"

.PHONY: lint-tests
lint-tests: ## Run linting on test files
	@echo "$(YELLOW)Linting test files...$(NC)"
	flake8 $(TEST_DIR)/*.py --max-line-length=120 --ignore=E203,W503
	@echo "$(GREEN)Linting completed$(NC)"

.PHONY: format-tests
format-tests: ## Format test files using black
	@echo "$(YELLOW)Formatting test files...$(NC)"
	black $(TEST_DIR)/*.py --line-length=120
	@echo "$(GREEN)Formatting completed$(NC)"

.PHONY: check-deps
check-deps: ## Check test dependencies
	@echo "$(YELLOW)Checking test dependencies...$(NC)"
	$(PYTHON) -c "import pytest, pytest_asyncio, pytest_cov, psutil; print('All dependencies available')"
	@echo "$(GREEN)Dependency check completed$(NC)"

.PHONY: docs
docs: ## Generate test documentation
	@echo "$(YELLOW)Generating test documentation...$(NC)"
	@echo "$(BLUE)Test documentation available in:$(NC)"
	@echo "  - VALIDATION_TESTING_README.md"
	@echo "  - Coverage reports in coverage_results/"
	@echo "  - Test results in validation_test_results/"
	@echo "$(GREEN)Documentation ready$(NC)"

.PHONY: status
status: ## Show test suite status
	@echo "$(CYAN)Test Suite Status$(NC)"
	@echo "$(CYAN)=================$(NC)"
	@echo "$(YELLOW)Test Files:$(NC)"
	@find $(TEST_DIR) -name "test_*.py" -exec basename {} \; | sort
	@echo ""
	@echo "$(YELLOW)Latest Test Results:$(NC)"
	@if [ -d "$(TEST_DIR)/validation_test_results" ]; then \
		ls -la $(TEST_DIR)/validation_test_results/ | head -5; \
	else \
		echo "  No test results found"; \
	fi
	@echo ""
	@echo "$(YELLOW)Coverage Reports:$(NC)"
	@if [ -d "$(TEST_DIR)/coverage_results" ]; then \
		ls -la $(TEST_DIR)/coverage_results/ | head -3; \
	else \
		echo "  No coverage reports found"; \
	fi

# Configuration targets
.PHONY: config
config: ## Show current configuration
	@echo "$(CYAN)Test Suite Configuration$(NC)"
	@echo "$(CYAN)========================$(NC)"
	@echo "$(YELLOW)Python:$(NC) $(PYTHON)"
	@echo "$(YELLOW)Test Directory:$(NC) $(TEST_DIR)"
	@echo "$(YELLOW)Project Root:$(NC) $(PROJECT_ROOT)"
	@echo "$(YELLOW)Minimum Coverage:$(NC) $(MIN_COVERAGE)%"
	@echo "$(YELLOW)Parallel Jobs:$(NC) $(PARALLEL_JOBS)"

# Utility targets
.PHONY: watch
watch: ## Watch for file changes and run tests automatically (requires entr)
	@echo "$(YELLOW)Watching for file changes (press Ctrl+C to stop)...$(NC)"
	find $(TEST_DIR) $(PROJECT_ROOT) -name "*.py" | entr -c make test-fast

# Docker targets (if using Docker)
.PHONY: docker-test
docker-test: ## Run tests in Docker container
	@echo "$(YELLOW)Running tests in Docker container...$(NC)"
	docker run --rm -v $(PWD):/workspace -w /workspace/tests python:3.9 bash -c "pip install -r test_requirements.txt && make test-all"
	@echo "$(GREEN)Docker tests completed$(NC)"

# Advanced targets
.PHONY: test-regression
test-regression: ## Run regression tests against baseline
	@echo "$(YELLOW)Running regression tests...$(NC)"
	$(PYTHON) run_validation_tests.py --suite performance --benchmark
	# Compare against baseline (implement comparison logic)
	@echo "$(GREEN)Regression tests completed$(NC)"

.PHONY: test-load
test-load: ## Run load testing scenarios
	@echo "$(YELLOW)Running load tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -m "performance_validation" -k "load" -v
	@echo "$(GREEN)Load tests completed$(NC)"

# Safety checks
.PHONY: pre-commit
pre-commit: lint-tests format-tests test-fast ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed$(NC)"

.PHONY: pre-push
pre-push: test-ci ## Run pre-push validation
	@echo "$(GREEN)Pre-push validation passed$(NC)"

# Information targets
.PHONY: version
version: ## Show version information
	@echo "$(CYAN)Factual Accuracy Validation Test Suite$(NC)"
	@echo "Version: 1.0.0"
	@echo "Author: Claude Code (Anthropic)"
	@echo "Created: August 7, 2025"
	@echo "Project: Clinical Metabolomics Oracle - LightRAG Integration"

# Include custom targets if they exist
-include Makefile.local