# 💜 District — AI Restaurant Concierge

> An AI-powered restaurant recommendation system inspired by Zomato, combining structured data filtering with a Large Language Model to deliver personalized, human-like dining recommendations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Powered_by-Groq_LLM-F55036?logo=groq&logoColor=white)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-district.streamlit.app-purple?logo=streamlit&logoColor=white)](https://district.streamlit.app/)

---

## 🌐 Live Demo

**Try it now:** [https://district.streamlit.app/](https://district.streamlit.app/)

---

## ✨ What it does

**District** is an AI restaurant concierge that takes your dining preferences—location, budget, cuisine, and vibe—and returns a curated shortlist of the best-matching restaurants with personalized AI-generated explanations for each pick.

| Step | What happens |
|------|-------------|
| 🎛️ **You set your preferences** | Location, budget tier, cuisine, minimum rating, and any extra vibe notes (e.g., "Dimly lit and cozy for a date") |
| 🔍 **Hard filter** | The engine queries the Zomato dataset and narrows it down to the top 20 candidates, extracting URL and raw Review data for context. |
| 🤖 **LLM ranking & Vibe Match** | Groq (Llama 3) reads the candidates' reviews and your preferences, then ranks the top 5, mirroring your language/tone, and extracts a 2-3 word `Vibe_Match`. |
| 🍽️ **Results** | A beautiful card-based UI displays each restaurant with its name, a direct **Zomato URL link**, rating, cost, interactive Google Maps chip, and an AI-crafted explanation. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│  (Two-screen flow: Preference Form → Results View)   │
└───────────────────────┬─────────────────────────────┘
                        │ In-process call
┌───────────────────────▼─────────────────────────────┐
│               Backend Logic (in app.py)              │
│                                                      │
│  1. load_data()  — reads cleaned_restaurants.parquet │
│  2. Pre-filter   — location / budget / cuisine /     │
│                    min rating → top 15 candidates    │
│  3. Prompt build — serialize candidates + prefs      │
│  4. Groq API     — Llama 3 ranks & explains top 5   │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┴──────────────┐
        ▼                              ▼
  Parquet Dataset               Groq Cloud API
  (data/cleaned_restaurants)    (llama3-8b-8192)
```

> The FastAPI layer from the original design was merged directly into `app.py` for zero-dependency deployment on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) + custom CSS (Inter & Outfit fonts, glassmorphism dark theme) |
| Data | [Pandas](https://pandas.pydata.org) + Parquet via [PyArrow](https://arrow.apache.org/docs/python/) |
| LLM | [Groq Python SDK](https://console.groq.com) — `llama3-8b-8192` (configurable) |
| Dataset | [`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) on Hugging Face |

---

## 📁 Project Structure

```
Zomato/
├── frontend/
│   ├── app.py                  # Main Streamlit app (UI + all backend logic)
│   ├── logo.jpeg               # App logo
│   └── .streamlit/
│       └── secrets.toml        # Local secrets (not committed)
├── data/
│   └── cleaned_restaurants.parquet  # Pre-processed dataset
├── notebooks/                  # Exploratory data analysis notebooks
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── requirements.txt            # Python dependencies
├── architecture.md             # System architecture doc
├── implementation-plan.md      # Phase-by-phase build plan
└── .env.example                # Environment variable template
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone the repo

```bash
git clone https://github.com/saurabhsharma14/Zomato.git
cd Zomato
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure secrets

Create `frontend/.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
GROQ_MODEL   = "llama3-8b-8192"   # optional, this is the default
```

### 4. Run the app

```bash
streamlit run frontend/app.py
```

The app will be available at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your code to GitHub (ensure `data/cleaned_restaurants.parquet` is included or loaded at runtime).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `frontend/app.py`.
4. Under **Advanced settings → Secrets**, add your `GROQ_API_KEY` (and optionally `GROQ_MODEL`).
5. Click **Deploy** 🚀

---

## 🤖 How the AI Works

The recommendation engine uses a **two-stage retrieval + ranking** pattern:

1. **Hard filtering (deterministic)** — Pandas filters the Zomato dataset by location, budget tier, cuisine, and minimum rating, returning up to 20 candidates.
2. **LLM ranking (generative)** — The candidates (including up to 300 characters of user reviews and liked dishes) are serialized as JSON and injected into a structured prompt. Groq's Llama 3 then acts as an *expert food critic and personalized concierge*, returning a ranked JSON array of the top 5 restaurants, each with a persuasive, personalized explanation and a concise `Vibe_Match`.

```
System: "You are an expert AI restaurant concierge..."
User:   "Location: Delhi | Budget: High | Cuisine: Italian
         Candidates: [ {name, cuisine, rating, cost, reviews, url, dish_liked}, ... ]"

→ Response: { "restaurants": [ {Name, Cuisine, Rating, Cost, Vibe_Match, AI_Explanation}, ... ] }
```

### 🛡️ Enterprise-Grade Robustness
- **Prompt Injection Defense:** The system prompt explicitly guards against malicious user inputs (e.g., "ignore all previous instructions").
- **Graceful Fallbacks:** If the API rate limits or hallucinates an invalid JSON response, the system automatically retries. If 0 results are returned from the hard filter, it seamlessly falls back to the top-rated spots overall.
- **Tone Mirroring:** The AI automatically detects the user's language (e.g., Hindi, Hinglish, formal English, slang) and mirrors it in the `AI_Explanation`.

---

## 📦 Dependencies

```
streamlit
pandas
groq
pyarrow
numpy
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.
