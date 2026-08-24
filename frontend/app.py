import streamlit as st
import pandas as pd
import json
import os
import base64
from groq import Groq

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="District — AI Restaurant Concierge",
    page_icon="💜",
    layout="centered",          # centered keeps content tight on all screens
    initial_sidebar_state="collapsed"
)


# ══════════════════════════════════════════════
# DATA LOADING & BACKEND LOGIC
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load the pre-cleaned Zomato dataset from local storage."""
    file_path = "data/cleaned_restaurants.parquet"
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_restaurants.parquet")
    df = pd.read_parquet(file_path)
    return df


@st.cache_data(show_spinner=False)
def get_unique_locations(df):
    """Return sorted list of unique locations, title-cased for display."""
    locs = sorted(df['location'].dropna().unique())
    return [loc.title() for loc in locs]


@st.cache_data(show_spinner=False)
def get_unique_cuisines(df):
    """Return sorted list of unique individual cuisines extracted from all entries."""
    all_cuisines = set()
    for c in df['cuisine'].dropna():
        for item in str(c).split(','):
            stripped = item.strip().title()
            if stripped:
                all_cuisines.add(stripped)
    return sorted(all_cuisines)


def extract_intent_from_text(text, all_locations, all_cuisines):
    """Parse a free-text query to extract location and cuisine hints."""
    text_lower = text.lower()
    found_location = ""
    found_cuisine = ""

    # Check for location keywords in the text (longest match first to avoid false positives)
    for loc in sorted(all_locations, key=len, reverse=True):
        if loc.lower() in text_lower:
            found_location = loc
            break

    # Check for cuisine keywords in the text
    for cui in sorted(all_cuisines, key=len, reverse=True):
        if cui.lower() in text_lower:
            found_cuisine = cui
            break

    return found_location, found_cuisine


def get_recommendations(df, location, budget, cuisine, min_rating, preferences, all_locations, all_cuisines):
    """Filter restaurants and call Groq LLM to rank + explain them."""
    filtered = df.copy()

    # Auto-extract location/cuisine from free-text prompt when dropdowns are empty
    extracted_location, extracted_cuisine = "", ""
    if preferences and (not location or not cuisine):
        extracted_location, extracted_cuisine = extract_intent_from_text(
            preferences, all_locations, all_cuisines
        )

    effective_location = location or extracted_location
    effective_cuisine = cuisine or extracted_cuisine

    if effective_location:
        filtered = filtered[filtered['location'].str.contains(effective_location.lower(), na=False, case=False)]
    if budget:
        filtered = filtered[filtered['budget'].fillna('').str.lower() == budget.lower()]
    if effective_cuisine:
        filtered = filtered[filtered['cuisine'].str.contains(effective_cuisine.lower(), na=False, case=False)]
    rating_num = pd.to_numeric(filtered['rating'].astype(str).str.split('/').str[0], errors='coerce')
    if min_rating > 0:
        filtered = filtered[rating_num >= min_rating]

    # If filters leave nothing, fall back to top-rated across full dataset
    if filtered.empty and (effective_location or effective_cuisine):
        filtered = df.copy()
        if effective_cuisine:
            filtered = filtered[filtered['cuisine'].str.contains(effective_cuisine.lower(), na=False, case=False)]
        rating_num = pd.to_numeric(filtered['rating'].astype(str).str.split('/').str[0], errors='coerce')

    filtered['rating_num_sort'] = rating_num.reindex(filtered.index).fillna(0)
    top_15 = filtered.sort_values(by="rating_num_sort", ascending=False).head(15)
    top_15 = top_15.drop(columns=['rating_num_sort'])

    if top_15.empty:
        return []

    top_15 = top_15.where(pd.notnull(top_15), None)
    results = top_15.to_dict(orient="records")

    system_prompt = """You are an expert AI restaurant concierge. Your goal is to select the top 5 best matching restaurants from the provided pre-filtered list based on the user's full request — including any location, cuisine, vibe, or food preferences mentioned.

CRITICAL RULES:
1. You MUST ONLY recommend restaurants from the provided candidate list. Do not invent or retrieve outside restaurants.
2. If there are fewer than 5 candidates provided, return all of them.
3. You must output ONLY a valid JSON object. Do not include conversational text, markdown formatting, or preamble before or after the JSON.
4. If the user's prompt mentions specific food or a vibe, prioritize candidates that match it best.

JSON SCHEMA:
Return a JSON object with a single key 'restaurants' mapping to a list of objects. Each object must have:
- 'Name': String, the exact name of the restaurant from the list.
- 'Cuisine': String, the cuisine it serves.
- 'Rating': Number, the rating.
- 'Cost': String or Number, the cost.
- 'AI_Explanation': String, a highly engaging, personalized 1-2 sentence explanation of why this specific restaurant perfectly matches the user's request. Be persuasive and sound like a local foodie!
"""

    candidates_for_llm = []
    for r in results:
        rev_str = str(r.get('reviews_list', ''))
        candidates_for_llm.append({
            'name': r.get('name'),
            'cuisine': r.get('cuisine'),
            'location': r.get('location'),
            'rating': r.get('rating'),
            'cost': r.get('cost'),
            'reviews': rev_str[:500] + "..." if len(rev_str) > 500 else rev_str
        })

    user_prompt = f"""
User's full request: "{preferences or 'Show me great restaurants'}"

Additional filters applied:
- Location: {effective_location or 'Any'}
- Budget: {budget or 'Any'}
- Cuisine: {effective_cuisine or 'Any'}
- Minimum Rating: {min_rating}

Pre-filtered Candidates:
{json.dumps(candidates_for_llm, indent=2)}

Please return the best 5 restaurants from this list in the requested JSON format.
"""

    try:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        model = st.secrets.get("GROQ_MODEL", "llama3-8b-8192")
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        llm_response = completion.choices[0].message.content
        llm_json = json.loads(llm_response)
        return llm_json.get("restaurants", [])

    except Exception as e:
        st.toast(f"⚠️ AI ranking unavailable, showing top results. ({e})", icon="⚠️")
        fallback = []
        for r in results[:5]:
            fallback.append({
                "Name": r.get("name", "Unknown"),
                "Cuisine": r.get("cuisine", "N/A"),
                "Rating": r.get("rating", "N/A"),
                "Cost": r.get("cost", "N/A"),
                "AI_Explanation": "Highly rated based on your constraints."
            })
        return fallback


# ══════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg-deep: #0D0B1A;
    --bg-surface: #141026;
    --bg-card: rgba(22, 17, 44, 0.75);
    --purple: #A020F0;
    --purple-bright: #B44FFF;
    --purple-neon: #C77DFF;
    --violet: #7000FF;
    --text-1: #F0EAFF;
    --text-2: #B0A3CC;
    --text-3: #6B5F8A;
    --border: rgba(160, 32, 240, 0.18);
    --border-hover: rgba(160, 32, 240, 0.4);
    --gold: #FFD700;
    --radius: 16px;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
footer { display: none !important; }
.stDeployButton,
[data-testid="stDeployButton"],
[data-testid="stGitHubIcon"],
[data-testid="manage-app-button"],
[data-testid="stAppViewerStatus"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Header: transparent ── */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
}
[data-testid="stHeader"]::after { display: none !important; }

/* ── App Background ── */
.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 5%, rgba(112,0,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 90%, rgba(160,32,240,0.06) 0%, transparent 60%) !important;
}

/* ── Container ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 720px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(160,32,240,0.3); border-radius: 3px; }

/* ══════════════════════════════════════════════
   HERO SECTION
   ══════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 32px 20px 24px;
    position: relative;
}

.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 250px;
    background: radial-gradient(circle, rgba(160,32,240,0.15) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: glowPulse 4s ease-in-out infinite alternate;
}
@keyframes glowPulse {
    from { opacity: 0.4; transform: translateX(-50%) scale(0.95); }
    to   { opacity: 1;   transform: translateX(-50%) scale(1.05); }
}

.hero-content { position: relative; z-index: 1; }

.hero-logo {
    display: inline-block;
    border-radius: 22px;
    padding: 3px;
    background: linear-gradient(135deg, var(--purple), var(--violet), var(--purple-bright));
    box-shadow: 0 0 35px rgba(160,32,240,0.25);
    animation: logoBreathe 3s ease-in-out infinite alternate;
    margin-bottom: 20px;
}
.hero-logo img {
    display: block;
    width: 100px; height: 100px;
    border-radius: 19px;
    object-fit: cover;
}
@keyframes logoBreathe {
    from { box-shadow: 0 0 30px rgba(160,32,240,0.25); }
    to   { box-shadow: 0 0 50px rgba(160,32,240,0.45), 0 0 90px rgba(112,0,255,0.15); }
}

.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 0 0 8px;
    background: linear-gradient(135deg, #fff 0%, #D4A8FF 50%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}

.hero p {
    font-size: 1.05rem;
    color: var(--text-2);
    margin: 0;
    font-weight: 400;
}

.hero-line {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--purple), transparent);
    margin: 18px auto 0;
    border-radius: 1px;
}

/* ══════════════════════════════════════════════
   FORM CARD
   ══════════════════════════════════════════════ */
[data-testid="stForm"] {
    background: rgba(22, 17, 44, 0.65) !important;
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    padding: 32px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3), 0 0 0 1px rgba(160,32,240,0.08) !important;
}

/* Form labels */
label, .stTextInput label, .stSelectbox label, .stSlider label, .stTextArea label {
    color: var(--text-2) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    margin-bottom: 2px !important;
}

/* Form inputs */
input, textarea,
[data-baseweb="select"] > div {
    background: rgba(13, 11, 26, 0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

input:focus, textarea:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 3px rgba(160,32,240,0.12) !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: var(--text-3) !important;
    font-size: 0.85rem !important;
}

/* Slider thumb value */
[data-testid="stThumbValue"] {
    color: var(--purple-neon) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
}

/* Reduce spacing between form fields */
.stTextInput, .stSelectbox, .stSlider, .stTextArea {
    margin-bottom: -4px !important;
}

/* Hide "Press Enter to submit" text */
[data-testid="InputInstructions"],
[data-testid="stInputInstructions"],
div[class*="InputInstructions"],
div[data-baseweb="input"] > div:not(:first-child),
div[data-baseweb="textarea"] > div:not(:first-child) {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}

/* ── Form Submit Button ── */
button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, var(--violet), var(--purple), var(--purple-bright)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(112,0,255,0.4) !important;
    transition: all 0.25s ease !important;
    margin-top: 12px !important;
}

button[kind="primaryFormSubmit"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(112,0,255,0.55) !important;
}

/* ── Refine Search Button ── */
.refine-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.refine-summary {
    font-size: 0.82rem;
    color: var(--text-3);
    flex: 1;
}
.refine-summary b { color: var(--purple-neon); font-weight: 600; }

/* Style "Refine" button — targets regular st.button */
button[kind="secondary"] {
    background: rgba(160,32,240,0.08) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    color: var(--purple-neon) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 6px 18px !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"]:hover {
    background: rgba(160,32,240,0.18) !important;
    border-color: var(--border-hover) !important;
    transform: none !important;
}

/* ══════════════════════════════════════════════
   RESTAURANT CARDS
   ══════════════════════════════════════════════ */
.r-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    animation: fadeUp 0.5s ease-out backwards;
}

.r-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--purple), transparent);
    opacity: 0.5;
}

.r-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-hover);
    box-shadow: 0 10px 35px rgba(112,0,255,0.15);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.r-top {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}

.r-rank {
    flex-shrink: 0;
    width: 38px; height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--violet), var(--purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem;
    font-weight: 800;
    color: white;
    box-shadow: 0 3px 12px rgba(112,0,255,0.35);
}

.r-info { flex: 1; min-width: 0; }

.r-name {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-1);
    margin: 0 0 4px;
    line-height: 1.25;
}

.r-stars {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,215,0,0.08);
    border: 1px solid rgba(255,215,0,0.15);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.82rem;
    color: var(--gold);
    font-weight: 600;
}

.r-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}

.r-tag {
    background: rgba(112,0,255,0.06);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.82rem;
    color: var(--text-2);
    font-weight: 500;
}

.r-ai {
    background: rgba(112,0,255,0.05);
    border: 1px solid var(--border);
    border-left: 3px solid var(--purple);
    border-radius: 10px;
    padding: 14px 18px;
}

.r-ai-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--purple-neon);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.r-ai-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--purple-neon);
    box-shadow: 0 0 8px var(--purple-neon);
    animation: dotPulse 2s ease-in-out infinite;
}
@keyframes dotPulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; box-shadow: 0 0 12px var(--purple-neon); }
}

.r-ai-text {
    color: var(--text-2);
    font-size: 0.9rem;
    line-height: 1.65;
}

/* ══════════════════════════════════════════════
   RESULTS HEADER
   ══════════════════════════════════════════════ */
.results-hdr {
    margin-bottom: 24px;
}
.results-hdr h2 {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-1);
    margin: 0 0 4px;
    letter-spacing: -0.3px;
}
.results-hdr p {
    color: var(--text-3);
    font-size: 0.88rem;
    margin: 0;
}

/* ══════════════════════════════════════════════
   NO RESULTS
   ══════════════════════════════════════════════ */
.no-match {
    text-align: center;
    padding: 50px 30px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}
.no-match h3 {
    color: var(--purple-neon);
    font-weight: 700;
    margin: 12px 0 8px;
    font-size: 1.2rem;
}
.no-match p {
    color: var(--text-3);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Footer ── */
.ft {
    text-align: center;
    padding: 36px 20px 16px;
    color: var(--text-3);
    font-size: 0.75rem;
    letter-spacing: 0.3px;
}
.ft b { color: var(--purple-neon); font-weight: 600; }

/* ── Global Image ── */
img { border-radius: 20px !important; box-shadow: none !important; }

/* ── Responsive ── */
@media (max-width: 600px) {
    .hero h1 { font-size: 2rem; }
    .r-card { padding: 18px 20px; }
    .r-name { font-size: 1.1rem; }
    [data-testid="stForm"] { padding: 20px !important; }
}

/* Column gap tightening for form layout */
[data-testid="stHorizontalBlock"] {
    gap: 12px !important;
}

/* ══════════════════════════════════════════════
   SEARCHABLE DROPDOWN
   ══════════════════════════════════════════════ */

/* Dropdown option list */
[role="listbox"] {
    background: rgba(20, 16, 38, 0.98) !important;
    border: 1px solid var(--border-hover) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
    padding: 4px !important;
}

/* Individual option */
[role="option"] {
    border-radius: 8px !important;
    color: var(--text-2) !important;
    font-size: 0.88rem !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: rgba(160, 32, 240, 0.15) !important;
    color: var(--purple-neon) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'restaurants' not in st.session_state:
    st.session_state.restaurants = []
if 'search_params' not in st.session_state:
    st.session_state.search_params = {}


# ══════════════════════════════════════════════
# LOAD DATA (cached)
# ══════════════════════════════════════════════
with st.spinner("Loading restaurant database..."):
    df = load_data()


# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
logo_path = os.path.join(os.path.dirname(__file__), "logo.jpeg")
hero_img = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    hero_img = f'<div class="hero-logo"><img src="data:image/jpeg;base64,{b64}" alt="District"/></div>'

st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        {hero_img}
        <h1>district.</h1>
        <p>Your AI-Powered Dining Concierge</p>
        <div class="hero-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: FORM
# ══════════════════════════════════════════════
if st.session_state.page == 'form':
    all_locations = get_unique_locations(df)
    all_cuisines = get_unique_cuisines(df)

    with st.form("preferences_form", clear_on_submit=False):
        st.markdown(
            '<p style="font-family:Outfit,sans-serif; font-size:1.15rem; font-weight:700; '
            'color:#F0EAFF; margin:0 0 4px;">Your Preferences</p>'
            '<p style="color:#6B5F8A; font-size:0.82rem; margin:0 0 18px; line-height:1.5;">'
            'Tell us what you\'re craving.</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox(
                "📍 Location",
                options=[""] + all_locations,
                index=0,
                placeholder="Search area... e.g. BTM",
            )
        with col2:
            cuisine = st.selectbox(
                "🥘 Cuisine",
                options=[""] + all_cuisines,
                index=0,
                placeholder="Search cuisine... e.g. North Indian",
            )

        col3, col4 = st.columns(2)
        with col3:
            budget = st.selectbox("💸 Budget", ["low", "medium", "high"], index=1)
        with col4:
            min_rating = st.slider("⭐ Min Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

        preferences = st.text_area(
            "✨ Your Request",
            placeholder="e.g. best pizza in Koramangala, romantic rooftop dinner, great biryani near HSR...",
            height=90
        )

        submitted = st.form_submit_button("✨  Curate My Experience", use_container_width=True)

    if submitted:
        # At least the prompt or one filter must be provided
        if not preferences and not location and not cuisine:
            st.warning("Please describe what you're looking for — or select a location / cuisine.")
        else:
            with st.spinner("Our AI concierge is curating the perfect spots for you..."):
                restaurants = get_recommendations(
                    df, location, budget, cuisine, min_rating, preferences,
                    all_locations, all_cuisines
                )
            st.session_state.restaurants = restaurants
            st.session_state.search_params = {
                'location': location,
                'cuisine': cuisine,
                'budget': budget,
                'min_rating': min_rating,
                'preferences': preferences,
            }
            st.session_state.page = 'results'
            st.rerun()


# ══════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════
elif st.session_state.page == 'results':
    params = st.session_state.search_params
    restaurants = st.session_state.restaurants

    # ── Refine / back row ──
    col_back, col_summary = st.columns([1, 3])
    with col_back:
        if st.button("← Refine", key="refine_btn", type="secondary"):
            st.session_state.page = 'form'
            st.rerun()
    with col_summary:
        # Build a compact summary showing only non-empty params
        parts = []
        if params.get('preferences'):
            parts.append(f'<b>"{params["preferences"][:40]}{"..." if len(params["preferences"]) > 40 else ""}"</b>')
        if params.get('location'):
            parts.append(f'<b>📍 {params["location"]}</b>')
        if params.get('cuisine'):
            parts.append(f'<b>🥘 {params["cuisine"]}</b>')
        parts.append(f'<b>💸 {params.get("budget", "medium")}</b>')
        parts.append(f'<b>≥{params.get("min_rating", 3.5)}</b>★')
        st.markdown(
            f'<p class="refine-summary">{" · ".join(parts)}</p>',
            unsafe_allow_html=True
        )

    if not restaurants:
        st.markdown("""
        <div class="no-match">
            <div style="font-size:2.5rem;">😕</div>
            <h3>No Matches Found</h3>
            <p>We couldn't find spots matching your exact vibe.<br>
            Try broadening your location, budget, or lowering the rating.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="results-hdr">
            <h2>✨ Top {len(restaurants)} Curated Picks</h2>
            <p>Hand-picked by our AI concierge based on your preferences</p>
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(restaurants):
            name = r.get("Name") or r.get("name", "Unknown")
            cuis = r.get("Cuisine") or r.get("cuisine", "N/A")
            rating = r.get("Rating") or r.get("rating", "N/A")
            cost = r.get("Cost") or r.get("cost", "N/A")
            explanation = r.get("AI_Explanation") or "Highly rated based on your constraints."

            try:
                rn = float(rating)
                stars = "★" * int(rn) + ("½" if rn - int(rn) >= 0.5 else "")
            except (ValueError, TypeError):
                stars = "★★★"

            cuis_display = cuis.title() if isinstance(cuis, str) else cuis

            st.markdown(f"""
            <div class="r-card" style="animation-delay:{i*0.08}s;">
                <div class="r-top">
                    <div class="r-rank">{i+1}</div>
                    <div class="r-info">
                        <p class="r-name">{name}</p>
                        <span class="r-stars">{stars} {rating}</span>
                    </div>
                </div>
                <div class="r-tags">
                    <span class="r-tag">🥘 {cuis_display}</span>
                    <span class="r-tag">💸 ₹{cost} for two</span>
                </div>
                <div class="r-ai">
                    <div class="r-ai-label"><span class="r-ai-dot"></span> AI Concierge Insight</div>
                    <div class="r-ai-text">{explanation}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ft">Made with 💜 by <b>District</b> — AI-Curated Dining Experiences</div>
    """, unsafe_allow_html=True)
