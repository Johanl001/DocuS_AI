from typing import List, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Initialize the FastAPI app instance
app = FastAPI()

# 2. Pydantic models for the structured JSON query
# These models define the expected structure for the POST /process_query endpoint
class Expense(BaseModel):
    expense_type: str
    amount: float
    currency: str

class ClaimDetails(BaseModel):
    incident_type: str
    date_of_incident: str
    description: str
    expenses: List[Expense]
    supporting_documents: List[str]

class ClaimQuery(BaseModel):
    type: str
    claim_id: str
    policy_number: str
    claim_details: ClaimDetails

class StructuredQueryRequest(BaseModel):
    query: ClaimQuery

# 3. The POST endpoint to handle the structured eligibility/claim checks
@app.post("/process_query")
async def process_structured_query(request: StructuredQueryRequest):
    """
    Processes a structured claim query and returns a mock response.
    This endpoint is designed to handle the complex JSON body.
    """
    # This is a mock response demonstrating the server received and parsed the data
    # In a real system, you would now use this data to perform a lookup or calculation
    claim_details = request.query.claim_details
    
    return {
        "status": "received",
        "message": "Claim details successfully received and validated.",
        "claim_id": request.query.claim_id,
        "policy_number": request.query.policy_number,
        "incident_type": claim_details.incident_type
    }

# 4. The GET endpoint to handle simple search queries
@app.get("/search/")
async def search_items(query: str, limit: Optional[int] = 10):
    """
    Searches for items based on a query string parameter.
    This endpoint is designed to handle URL query parameters.
    """
    return {
        "query": query,
        "limit": limit,
        "results": [f"Result for {query} {i}" for i in range(limit)]
    }