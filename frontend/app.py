import streamlit as st
import pandas as pd
import json
import os
import base64
import streamlit.components.v1 as components
from groq import Groq

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="District — AI Restaurant Concierge",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════
# DATA LOADING & BACKEND LOGIC (merged from backend/main.py)
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load the pre-cleaned Zomato dataset from local storage.
    Cached so it only runs once per session / deployment.
    (Cache invalidated to load new 'budget' column)"""
    import os
    
    # Check both potential paths depending on where streamlit is run from
    file_path = "data/cleaned_restaurants.parquet"
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_restaurants.parquet")
        
    df = pd.read_parquet(file_path)
    return df


def get_recommendations(df, location, budget, cuisine, min_rating, preferences):
    """Filter restaurants and call Groq LLM to rank + explain them.
    Returns a list of restaurant dicts."""

    # ── Pre-filtering (from backend/main.py) ──
    filtered = df.copy()

    if location:
        filtered = filtered[filtered['location'].str.contains(location.lower(), na=False, case=False)]
    if budget:
        filtered = filtered[filtered['budget'].fillna('').str.lower() == budget.lower()]
    if cuisine:
        filtered = filtered[filtered['cuisine'].str.contains(cuisine.lower(), na=False, case=False)]
    rating_num = pd.to_numeric(filtered['rating'].astype(str).str.split('/').str[0], errors='coerce')
    if min_rating > 0:
        filtered = filtered[rating_num >= min_rating]
        # Re-subset rating_num to match filtered if we want to use it for sorting, but easier to just assign it
    
    # Assign for reliable sorting, fill NaNs with 0
    filtered['rating_num_sort'] = rating_num.reindex(filtered.index).fillna(0)
    top_15 = filtered.sort_values(by="rating_num_sort", ascending=False).head(15)
    top_15 = top_15.drop(columns=['rating_num_sort'])

    if top_15.empty:
        return []

    top_15 = top_15.where(pd.notnull(top_15), None)
    results = top_15.to_dict(orient="records")

    # ── LLM Ranking (from backend/main.py) ──
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

    candidates_for_llm = []
    for r in results:
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
Location: {location}
Budget: {budget}
Cuisine: {cuisine}
Minimum Rating: {min_rating}
Additional Preferences: {preferences}

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
        # Fallback: return top 5 raw results
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

/* ── Hide Streamlit Chrome (keep sidebar toggle visible) ── */
#MainMenu {visibility: hidden !important;}
footer {display: none !important;}

/* Kill deploy button — every known selector */
.stDeployButton,
[data-testid="stDeployButton"],
button[kind="header"],
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}
[data-testid="stHeader"]::after { display: none !important; }

/* ── Force sidebar expanded and ideal size ── */
section[data-testid="stSidebar"] {
    width: 380px !important;
    min-width: 380px !important;
    max-width: 380px !important;
    transform: none !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: none !important;
}

/* ── App Background ── */
.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 5%, rgba(112,0,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 90%, rgba(160,32,240,0.06) 0%, transparent 60%) !important;
}

/* ── Container ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(160,32,240,0.3); border-radius: 3px; }

/* ══════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem 1.5rem !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: var(--text-2) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    text-transform: none !important;
    margin-bottom: 2px !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(22, 17, 44, 0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] textarea:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 2px rgba(160,32,240,0.12) !important;
    outline: none !important;
}

section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: var(--text-3) !important;
    font-size: 0.85rem !important;
}

/* Hide Input Instructions (Press Enter to submit) */
[data-testid="InputInstructions"],
[data-testid="stInputInstructions"],
[data-testid="stWidgetInstructions"],
[data-testid="stFormSubmitInstructions"],
.stTextInput small,
.stTextArea small,
div[class*="InputInstructions"] {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* Force hide the suffix container inside baseweb inputs where Streamlit injects 'Press Enter...' */
div[data-baseweb="input"] > div:not(:first-child),
div[data-baseweb="textarea"] > div:not(:first-child) {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}

/* Slider */
section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: var(--purple-neon) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
}

/* Sidebar reduce inner gaps */
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stTextArea {
    margin-bottom: -4px !important;
}

/* ── Sidebar Submit Button ── */
section[data-testid="stSidebar"] button,
button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--violet), var(--purple), var(--purple-bright)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 18px rgba(112,0,255,0.35) !important;
    transition: all 0.25s ease !important;
    margin-top: 8px !important;
}

section[data-testid="stSidebar"] button:hover,
button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(112,0,255,0.5) !important;
}

/* ── Sidebar divider ── */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ══════════════════════════════════════════════
   HERO SECTION
   ══════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 40px 20px 30px;
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
    margin-bottom: 24px;
}
.hero-logo img {
    display: block;
    width: 120px; height: 120px;
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
    margin: 22px auto 0;
    border-radius: 1px;
}

/* ══════════════════════════════════════════════
   EMPTY STATE — The Page Load View
   ══════════════════════════════════════════════ */
.welcome {
    text-align: center;
    padding: 80px 40px 90px;
    margin-top: 10px;
    background: linear-gradient(160deg, rgba(22,17,44,0.5) 0%, rgba(13,11,26,0.6) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    position: relative;
    overflow: hidden;
}

.welcome::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 20%, rgba(160,32,240,0.07) 0%, transparent 70%);
    pointer-events: none;
}

.welcome-emoji {
    font-size: 4.5rem;
    display: block;
    margin-bottom: 24px;
    animation: float 3.5s ease-in-out infinite;
    position: relative;
    z-index: 1;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-14px); }
}

.welcome h2 {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-1);
    margin: 0 0 12px;
    letter-spacing: -0.5px;
    position: relative; z-index: 1;
}

.welcome p {
    color: var(--text-3);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 400px;
    margin: 0 auto;
    position: relative; z-index: 1;
}

.welcome-features {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 40px;
    position: relative; z-index: 1;
}

.welcome-feat {
    text-align: center;
}

.welcome-feat-icon {
    width: 48px; height: 48px;
    border-radius: 14px;
    background: rgba(160,32,240,0.1);
    border: 1px solid var(--border);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 10px;
}

.welcome-feat-text {
    color: var(--text-2);
    font-size: 0.8rem;
    font-weight: 500;
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

/* Card Top Row */
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

/* Tags Row */
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

/* AI Insight */
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
@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    .r-card { padding: 18px 20px; }
    .r-name { font-size: 1.1rem; }
    .welcome { padding: 50px 24px 60px; }
    .welcome-features { flex-direction: column; align-items: center; gap: 20px; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# LOAD DATA (cached)
# ══════════════════════════════════════════════
with st.spinner("Loading restaurant database..."):
    df = load_data()


# ══════════════════════════════════════════════
# SIDEBAR — Preference Form
# ══════════════════════════════════════════════
logo_path = os.path.join(os.path.dirname(__file__), "logo.jpeg")

with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=60)
    st.markdown(
        '<p style="font-family:Outfit,sans-serif; font-size:1.15rem; font-weight:700; '
        'color:#F0EAFF; margin:4px 0 2px;">Your Preferences</p>'
        '<p style="color:#6B5F8A; font-size:0.82rem; margin:0 0 18px; line-height:1.5;">'
        'Tell us what you\'re craving.</p>',
        unsafe_allow_html=True
    )

    with st.form("preferences_form"):
        location = st.text_input("Location", placeholder="e.g. Indiranagar, Koramangala")
        cuisine = st.text_input("Cuisine", placeholder="e.g. North Indian, Italian")
        budget = st.selectbox("Budget", ["low", "medium", "high"], index=1)
        min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
        preferences = st.text_area(
            "Vibe & Preferences",
            placeholder="e.g. romantic rooftop, craft cocktails, live music...",
            height=90
        )
        submitted = st.form_submit_button("✨  Curate My Experience")

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#6B5F8A; font-size:0.72rem;">'
        'Powered by <b style="color:#A020F0;">District AI</b> × Groq</p>',
        unsafe_allow_html=True
    )

# Inject JS to forcefully remove the "Press Enter to submit" text which Streamlit injects dynamically
components.html(
    """
    <script>
    const doc = window.parent.document;
    const hideInstructions = () => {
        // Find all elements that might contain the instruction text
        const elements = doc.querySelectorAll('div, small, span, p');
        elements.forEach(el => {
            if (el.innerText && (el.innerText.includes('Press Enter to') || el.innerText.includes('Press enter to'))) {
                el.style.display = 'none';
                el.style.opacity = '0';
                el.style.visibility = 'hidden';
            }
        });
    };
    
    // Run initially
    hideInstructions();
    
    // Set an observer to catch elements as they are dynamically added by Streamlit
    const observer = new MutationObserver(hideInstructions);
    observer.observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
    height=0,
    width=0
)


# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
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
# MAIN CONTENT
# ══════════════════════════════════════════════
if submitted:
    if not location or not cuisine or not preferences:
        st.warning("Please fill out all the mandatory fields (Location, Cuisine, Vibe & Preferences) to begin your search.")
    else:
        with st.spinner("Our AI concierge is curating the perfect spots for you..."):
            restaurants = get_recommendations(df, location, budget, cuisine, min_rating, preferences)

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

else:
    # ── Welcome / Empty State ──
    st.markdown("""
    <div class="welcome">
        <span class="welcome-emoji">🍽️</span>
        <h2>Ready to discover your next meal?</h2>
        <p>Set your preferences in the sidebar and let our AI concierge curate the perfect dining experience — tailored just for you.</p>
        <div class="welcome-features">
            <div class="welcome-feat">
                <div class="welcome-feat-icon">📍</div>
                <div class="welcome-feat-text">Any Location</div>
            </div>
            <div class="welcome-feat">
                <div class="welcome-feat-icon">🤖</div>
                <div class="welcome-feat-text">AI Ranked</div>
            </div>
            <div class="welcome-feat">
                <div class="welcome-feat-icon">⭐</div>
                <div class="welcome-feat-text">Top Rated</div>
            </div>
            <div class="welcome-feat">
                <div class="welcome-feat-icon">💜</div>
                <div class="welcome-feat-text">Curated Vibes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="ft">Made with 💜 by <b>District</b> — AI-Curated Dining Experiences</div>
""", unsafe_allow_html=True)
