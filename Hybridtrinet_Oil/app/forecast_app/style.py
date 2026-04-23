# forecast_app/style.py
import streamlit as st


def inject_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">

<style>
:root{
  --bg: #F5FAFF;
  --card: #FFFFFF;
  --text: #0B1220;
  --muted: #5B6B82;
  --border: #E3EDF8;
  --accent: #2563EB;
  --accent2: #14B8A6;
  --shadow: 0 10px 26px rgba(2, 6, 23, 0.06);
}

html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1260px; padding-top: 1.6rem; padding-bottom: 2.0rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
section[data-testid="stSidebar"] {display: none !important;}

/* Containers (border=True) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 18px !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 0.15rem 0.15rem 0.40rem 0.15rem;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px !important;
  padding: 0.62rem 1.0rem !important;
  font-weight: 800 !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 12px 26px rgba(2, 6, 23, 0.08);
  border-color: rgba(37, 99, 235, 0.28) !important;
}
button[kind="primary"]{
  background: linear-gradient(135deg, rgba(37,99,235,1), rgba(20,184,166,0.95)) !important;
  color: white !important;
  border: 1px solid rgba(37,99,235,0.22) !important;
}
button[kind="primary"]:hover{
  border-color: rgba(20,184,166,0.45) !important;
}

/* Dataframe */
.stDataFrame{
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid var(--border);
}

/* Metrics */
[data-testid="stMetric"]{
  border-radius: 14px;
  padding: 12px 14px;
  background: #FFFFFF;
  border: 1px solid var(--border);
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
  border-radius: 12px !important;
  padding: 10px 14px !important;
  font-weight: 900 !important;
}
.stTabs [aria-selected="true"]{
  border: 1px solid rgba(37,99,235,0.18) !important;
  background: rgba(37,99,235,0.06) !important;
}

/* Hero header */
.hero{
  border-radius: 20px;
  border: 1px solid var(--border);
  background:
    radial-gradient(900px circle at 16% 12%, rgba(37,99,235,0.13), transparent 55%),
    radial-gradient(900px circle at 86% 10%, rgba(20,184,166,0.12), transparent 55%),
    linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.86));
  box-shadow: var(--shadow);
  padding: 26px 18px 18px 18px;
  margin-bottom: 18px;
}

.page-header{ text-align: center; }
.page-title{
  font-size: 42px;
  font-weight: 900;
  line-height: 1.18;
  margin: 0;
  letter-spacing: 0.2px;
}
.page-title::after{
  content:"";
  display:block;
  height: 5px;
  width: 180px;
  margin: 12px auto 0 auto;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(20,184,166,0.95));
}
.page-subtitle{
  font-size: 15px;
  color: var(--muted);
  margin: 10px 0 0 0;
}

.hr-soft{
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, rgba(2,6,23,0.12), transparent);
  border: 0;
  margin: 1.0rem 0 1.0rem 0;
}

/* Section header */
.section-h{
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0.10rem 0 0.8rem 0;
}
.section-ico{
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display:flex;
  align-items:center;
  justify-content:center;
  border: 1px solid var(--border);
  background: #FFFFFF;
}
.section-ico i{ font-size: 18px; color: var(--accent2); }
.section-txt{
  font-size: 20px;
  font-weight: 900;
  color: var(--text);
}

/* Nice stat cards */
.stat-card{
  border-radius: 16px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(37,99,235,0.06), rgba(255,255,255,1));
  box-shadow: 0 8px 18px rgba(2,6,23,0.05);
  padding: 14px 14px 12px 14px;
}
.stat-card .row{
  display:flex; align-items:center; justify-content:space-between;
}
.stat-card .ttl{
  font-size: 13px;
  font-weight: 800;
  color: var(--muted);
}
.stat-card .val{
  font-size: 30px;
  font-weight: 900;
  color: var(--text);
  margin-top: 6px;
  line-height: 1.1;
}
.stat-card .sub{
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}
.stat-card .badge{
  width: 34px; height: 34px;
  border-radius: 12px;
  border: 1px solid rgba(37,99,235,0.18);
  background: rgba(20,184,166,0.08);
  display:flex; align-items:center; justify-content:center;
}
.stat-card .badge i{ font-size: 18px; color: var(--accent); }

details summary { font-weight: 900 !important; }
</style>
""",
        unsafe_allow_html=True,
    )
