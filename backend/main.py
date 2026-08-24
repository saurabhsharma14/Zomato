from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import pandas as pd
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

app = FastAPI(title="Zomato Restaurant Recommendation API")

class RecommendationRequest(BaseModel):
    location: str
    budget: str  # low, medium, high
    cuisine: str
    min_rating: float = 0.0
    preferences: Optional[str] = ""

# Load data on startup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "restaurants.db")
df = pd.DataFrame()

@app.on_event("startup")
def startup_event():
    global df
    if not os.path.exists(DB_PATH):
        print(f"Warning: Database not found at {DB_PATH}")
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM restaurants", conn)
        conn.close()
        print(f"Loaded {len(df)} restaurants from database.")

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    if df.empty:
        raise HTTPException(status_code=500, detail="Database is empty or not loaded.")
    
    # Pre-filtering logic
    filtered_df = df.copy()
    
    # Filter by location (exact match or contains)
    if request.location:
        filtered_df = filtered_df[filtered_df['location'].str.contains(request.location.lower(), na=False, case=False)]
    
    # Filter by budget
    if request.budget:
        filtered_df = filtered_df[filtered_df['budget'].str.lower() == request.budget.lower()]
    
    # Filter by cuisine
    if request.cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'].str.contains(request.cuisine.lower(), na=False, case=False)]
    
    # Filter by min_rating
    if request.min_rating > 0:
        filtered_df = filtered_df[filtered_df['rating'] >= request.min_rating]
    
    # Sort by rating (highest first) and get top 15
    top_15 = filtered_df.sort_values(by="rating", ascending=False).head(15)
    
    if top_15.empty:
        return {"restaurants": []}
    
    # Convert to list of dicts, replace NaN with None for JSON serialization
    top_15 = top_15.where(pd.notnull(top_15), None)
    results = top_15.to_dict(orient="records")
    
    # --- PHASE 4: LLM Integration ---
    system_prompt = """You are an expert AI restaurant concierge. Your goal is to select the top 5 best matching restaurants from the provided pre-filtered list based on the user's explicit preferences and vibe.

CRITICAL RULES:
1. You MUST ONLY recommend restaurants from the provided candidate list. Do not invent or retrieve outside restaurants.
2. If there are fewer than 5 candidates provided, return all of them.
3. You must output ONLY a valid JSON object. Do not include conversational text, markdown formatting, or preamble before or after the JSON.

JSON SCHEMA:
Return a JSON object with a single key 'restaurants' mapping to a list of objects. Each object must have:
- 'Name': String, the exact name of the restaurant from the list.
- 'Cuisine': String, the cuisine it serves.
- 'Rating': Number, the rating.
- 'Cost': String or Number, the cost.
- 'AI_Explanation': String, a highly engaging, personalized 1-2 sentence explanation of why this specific restaurant perfectly matches the user's vibe and preferences. Be persuasive and sound like a local foodie!
"""
    
    # Only send relevant fields to LLM to save tokens
    candidates_for_llm = []
    for r in results:
        # truncating review lists heavily to avoid large token counts and parse errors
        rev_str = str(r.get('reviews_list', ''))
        candidates_for_llm.append({
            'name': r.get('name'),
            'cuisine': r.get('cuisine'),
            'rating': r.get('rating'),
            'cost': r.get('cost'),
            'reviews': rev_str[:500] + "..." if len(rev_str) > 500 else rev_str
        })
    
    user_prompt = f"""
User Preferences:
Location: {request.location}
Budget: {request.budget}
Cuisine: {request.cuisine}
Minimum Rating: {request.min_rating}
Additional Preferences: {request.preferences}

Pre-filtered Candidates:
{json.dumps(candidates_for_llm, indent=2)}

Please return the best 5 restaurants from this list in the requested JSON format.
"""

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        
        llm_response = completion.choices[0].message.content
        llm_json = json.loads(llm_response)
        
        return {"restaurants": llm_json.get("restaurants", [])}
        
    except Exception as e:
        print(f"LLM Error: {e}")
        # Fallback to returning the top 5 results directly from DB
        return {"restaurants": results[:5]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
