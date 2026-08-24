# Evaluation Plan: AI-Powered Restaurant Recommendation System

This document outlines the comprehensive evaluation strategy for the AI-Powered Restaurant Recommendation System, referencing the established system architecture and implementation plan. The goal is to ensure the system delivers accurate, performant, and personalized recommendations.

## 1. Architectural Component Evaluation

### 1.1. Data Ingestion & Preprocessing
*   **Data Integrity & Completeness:** Verify the Hugging Face dataset (`ManikaSaini/zomato-restaurant-recommendation`) is fully loaded. Check for data loss during null value handling and text standardization (e.g., locations, cuisines).
*   **Normalization Accuracy:** Ensure categorical features (like numerical costs) are accurately mapped to the designated budget tiers (Low, Medium, High).
*   **Storage Performance:** Measure the latency of loading the cleaned dataset into the chosen in-memory engine (Pandas DataFrame or DuckDB/SQLite).

### 1.2. Backend Pre-filtering Engine
*   **Constraint Accuracy:** Evaluate if the filtering engine strictly obeys user-defined *hard constraints* (Location, Budget, Cuisine, Minimum Rating). The resulting candidate pool must not violate these parameters.
*   **Query Latency:** Ensure the querying mechanism against the Data Store is highly optimized, returning the Top `N` (e.g., 10-15) candidate restaurants in milliseconds to prevent bottlenecking the LLM stage.

### 1.3. LLM Recommendation Engine (Groq + Llama 3)
*   **Reasoning & Soft Constraints:** Assess how effectively the LLM incorporates the user's *soft constraints* (e.g., "Good for dates", "Quiet atmosphere") to rank the pre-filtered candidates.
*   **Structured Output Adherence:** Verify that the LLM consistently returns a strictly formatted response matching the expected JSON schema (e.g., `restaurant_id`, `name`, `explanation`, `rank`). Failure to adhere to the schema should trigger a retry or fallback mechanism.
*   **Explanation Quality (Prompt Tuning):** Conduct qualitative reviews of the AI-generated explanations. They must sound like an "expert food critic" (as defined in the system prompt), clearly explaining *why* the restaurant fits the specific user preferences.

## 2. Phase-wise Implementation Testing

### Phases 1 & 2: Environment & Data Pipeline
*   **Testing:** Run the data fetching and cleaning scripts from scratch.
*   **Validation:** Check the output artifact (CSV, Parquet, or DB file) to confirm schema validity and data cleanliness.

### Phase 3: Backend API Validation
*   **Testing:** Send direct HTTP requests to the `/recommend` FastAPI endpoint using tools like Postman.
*   **Validation:** Ensure Pydantic models correctly validate input payloads and reject malformed requests with standard HTTP 422 errors. Confirm the pre-filtered dataset size aligns with expectations.

### Phase 4: LLM Integration Quality
*   **Testing:** Monitor the requests sent to the Groq API and parse the responses.
*   **Validation:** Implement test suites covering edge cases, such as conflicting user preferences or malformed inputs, verifying the backend gracefully handles LLM hallucinations or parsing failures.

### Phase 5: Frontend Usability
*   **Testing:** Interact with the Streamlit/React UI using various combinations of inputs.
*   **Validation:** Ensure the UI successfully sends API requests, correctly parses the JSON payload, and renders the ranked list and explanations clearly without layout issues.

### Phase 6 & 7: End-to-End Performance & Deployment
*   **Latency Testing (Performance):** Measure the total round-trip time from user click to UI update. The goal is rapid response times, capitalizing on Groq's fast inference.
*   **Edge Case Handling:** Test restrictive filters that yield 0 candidates from the Data Store. Verify the system does not crash and instead displays a user-friendly "No restaurants found" message.
*   **Deployment Stability:** Verify the connectivity between the Vercel-hosted frontend and the Railway-hosted backend in a production-like environment.
