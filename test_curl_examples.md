# FastAPI Testing Examples

## Prerequisites
1. Make sure your FastAPI server is running: `python api.py`
2. Add your OpenRouter API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

## Test Commands

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Test Query (Simple)
```bash
curl -X POST "http://localhost:8000/process_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the coverage for medical expenses?"}'
```

### 3. Test Query (Complex)
```bash
curl -X POST "http://localhost:8000/process_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How much can I claim for hospitalization under this policy?"}'
```

### 4. Test with Invalid JSON
```bash
curl -X POST "http://localhost:8000/process_query" \
  -H "Content-Type: application/json" \
  -d '{"query": ""}'
```

### 5. Test with Missing Field
```bash
curl -X POST "http://localhost:8000/process_query" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Expected Responses

### Success Response (200)
```json
{
  "decision": "Approved",
  "amount": "50000",
  "justification": "Based on policy clause 3.2...",
  "clauses_used": ["Clause 3.2", "Clause 4.1"]
}
```

### Error Response (500)
```json
{
  "detail": "Internal server error: [specific error message]"
}
```

## Common Issues and Solutions

### 1. Missing OPENROUTER_API_KEY
**Error**: `OPENROUTER_API_KEY not found in environment variables`
**Solution**: Add to `.env` file:
```
OPENROUTER_API_KEY=your_actual_api_key_here
```

### 2. Invalid JSON from LLM
**Error**: `LLM did not return a valid JSON object`
**Solution**: The API now handles this gracefully with fallback responses

### 3. Network/Connection Issues
**Error**: Connection refused or timeout
**Solution**: Make sure the server is running on port 8000

### 4. Chroma Database Issues
**Error**: Database not found or corrupted
**Solution**: Run the ingest script first to populate the database

## Debug Steps

1. **Check server logs** when making requests
2. **Verify environment variables** are loaded correctly
3. **Test with simple queries** first
4. **Check Chroma database** exists and has data
5. **Verify OpenRouter API key** is valid and has credits
