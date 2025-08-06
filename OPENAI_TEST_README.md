# OpenAI API Connectivity Test

This document describes how to use the `test_openai_connectivity.py` script for verifying OpenAI API connectivity before LightRAG integration.

## Purpose

The script tests OpenAI API connectivity for both chat completions and embeddings functionality required for LightRAG integration, specifically testing:

- **Chat Completion API** using `gpt-4o-mini` model
- **Embeddings API** using `text-embedding-3-small` model

## Prerequisites

### 1. Install OpenAI Library

The script requires the `openai` library, which is not included in `requirements_lightrag.txt`:

```bash
# In your lightrag_env virtual environment
pip install openai
```

### 2. Configure API Key

Set your OpenAI API key in one of these locations:

**Option A: Project root `.env` file:**
```
OPENAI_API_KEY=your_api_key_here
```

**Option B: `src/.env` file:**
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```bash
# From project root directory
python test_openai_connectivity.py

# Or make it executable and run directly
chmod +x test_openai_connectivity.py
./test_openai_connectivity.py
```

### Expected Output

The script will run 5 comprehensive tests:

1. **Environment Configuration** - Loads .env files and checks for API key
2. **Client Initialization** - Creates OpenAI client instance  
3. **Model Availability** - Verifies required models are accessible
4. **Chat Completion** - Tests gpt-4o-mini with a simple prompt
5. **Embeddings** - Tests text-embedding-3-small with sample text

### Success Example

```
OPENAI API CONNECTIVITY TEST REPORT
====================================================================
Timestamp: 2025-08-06 12:34:56
Project: Clinical Metabolomics Oracle - LightRAG Integration
Task: CMO-LIGHTRAG-001-T06

TEST RESULTS:
------------------------------
Environment Configuration      ✓ PASS
Client Initialization         ✓ PASS
Model Availability            ✓ PASS
Chat Completion (gpt-4o-mini) ✓ PASS
Embeddings (text-embedding-3-small) ✓ PASS

OVERALL STATUS:
------------------------------
🎉 ALL TESTS PASSED - OpenAI API is ready for LightRAG integration!

RECOMMENDATIONS:
------------------------------
• ✅ Ready to proceed with LightRAG integration
• Both chat completion and embeddings APIs are working
• Required models are available and accessible
```

## Troubleshooting

### Common Issues

**1. Missing API Key**
```
✗ OPENAI_API_KEY not set
```
**Solution:** Add your OpenAI API key to `.env` or `src/.env` file

**2. Empty API Key**  
```
✗ OPENAI_API_KEY is empty
```
**Solution:** Ensure the API key value is not empty in your .env file

**3. Invalid API Key**
```
Chat completion attempt 1 failed: Error code: 401 - Unauthorized
```
**Solution:** Verify your API key is correct and has sufficient credits

**4. Network Issues**
```
Client initialization failed: Connection error
```
**Solution:** Check internet connectivity and firewall settings

**5. Missing OpenAI Library**
```
✗ ERROR: openai library not found.
```
**Solution:** Install with `pip install openai`

## Exit Codes

- **0**: All tests passed - Ready for LightRAG integration
- **1**: One or more tests failed - Do not proceed with integration

## Integration with LightRAG

Once all tests pass, you can proceed with LightRAG integration knowing that:

- ✅ OpenAI API key is properly configured
- ✅ Required models (gpt-4o-mini, text-embedding-3-small) are accessible  
- ✅ Both chat completion and embeddings functionality work correctly
- ✅ Network connectivity and authentication are working

## Script Features

- **Comprehensive Testing**: Tests all required OpenAI functionality for LightRAG
- **Retry Logic**: Automatically retries failed tests up to 3 times
- **Error Handling**: Graceful error handling with detailed error messages
- **Environment Detection**: Automatically loads from multiple .env locations
- **Detailed Reporting**: Comprehensive test reports with recommendations
- **Security**: API keys are masked in output for security
- **Timeout Protection**: 30-second timeout prevents hanging