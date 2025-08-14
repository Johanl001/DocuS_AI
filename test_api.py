import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_query(query_text):
    """Test the query endpoint with a sample query"""
    try:
        # Prepare the request
        payload = {"query": query_text}
        headers = {"Content-Type": "application/json"}
        
        print(f"Sending query: {query_text}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make the request
        response = requests.post(
            f"{API_URL}/process_query",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Success! Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def check_environment():
    """Check if required environment variables are set"""
    print("=== Environment Check ===")
    
    required_vars = ["OPENROUTER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value and value != "your_openrouter_api_key_here":
            print(f"✓ {var}: Set")
        else:
            print(f"✗ {var}: Missing or invalid")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\nMissing environment variables: {missing_vars}")
        print("Please add these to your .env file:")
        for var in missing_vars:
            print(f"{var}=your_actual_api_key_here")
    
    return len(missing_vars) == 0

def main():
    print("=== FastAPI Debug Test ===")
    
    # Check environment
    env_ok = check_environment()
    print()
    
    # Test health check
    health_ok = test_health_check()
    print()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the API server is running.")
        print("Run: python api.py")
        return
    
    # Test with a sample query
    sample_queries = [
        "What is the coverage for medical expenses?",
        "How much is the claim amount for hospitalization?",
        "What are the terms and conditions for claim approval?"
    ]
    
    for query in sample_queries:
        print(f"\n=== Testing Query: {query} ===")
        success = test_query(query)
        if success:
            print("✅ Query successful!")
        else:
            print("❌ Query failed!")
        print("-" * 50)

if __name__ == "__main__":
    main()
