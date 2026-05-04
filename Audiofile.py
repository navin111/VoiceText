import streamlit as st
from faster_whisper import WhisperModel
from groq import Groq
from gtts import gTTS
import tempfile
import os
import time
import json
import uuid
from datetime import datetime, timezone, date
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── POSTGRESQL CONFIG ──────────────────────────────────────────────────────────
PG_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "dbname":   "voiceiq_db",
    "user":     "postgres",
    "password": "1234qwer",
}

# ─── DB HELPERS ─────────────────────────────────────────────────────────────────
def init_db(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS voiceiq_results (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        file_name           TEXT        NOT NULL,
        file_size           BIGINT,
        processed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        transcription       TEXT,
        sentiment           TEXT,
        sentiment_score     INTEGER,
        sentiment_keywords  TEXT[],
        ai_reply            TEXT,
        full_json           JSONB       NOT NULL
    );

    ALTER TABLE voiceiq_results
        ADD COLUMN IF NOT EXISTS sentiment_score    INTEGER,
        ADD COLUMN IF NOT EXISTS sentiment_keywords TEXT[];

    CREATE INDEX IF NOT EXISTS idx_voiceiq_sentiment    ON voiceiq_results (sentiment);
    CREATE INDEX IF NOT EXISTS idx_voiceiq_processed_at ON voiceiq_results (processed_at DESC);
    CREATE INDEX IF NOT EXISTS idx_voiceiq_score        ON voiceiq_results (sentiment_score);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


def build_json_payload(result: dict) -> dict:
    return {
        "id":             str(uuid.uuid4()),
        "file_name":      result.get("name", "unknown"),
        "file_size":      result.get("file_size", None),
        "processed_at":   datetime.now(timezone.utc).isoformat(),
        "transcription":  result.get("text", ""),
        "sentiment": {
            "label":      result.get("sentiment", "Neutral"),
            "score":      result.get("score", 5),
            "keywords":   result.get("keywords", []),
            "confidence": "high",
        },
        "ai_response": {
            "reply_text": result.get("reply", ""),
            "model":      "llama-3.3-70b-versatile",
            "tts_engine": "google-tts",
        },
        "pipeline": {
            "stt_model":  "faster-whisper-tiny",
            "llm_model":  "llama-3.3-70b-versatile",
            "tts_engine": "gtts",
            "version":    "1.0.0",
        },
        "metadata": {
            "app":         "VoiceIQ",
            "environment": "demo",
        },
    }


def save_to_db(conn, result: dict, file_size: int = 0):
    result["file_size"] = file_size
    payload = build_json_payload(result)
    sql = """
        INSERT INTO voiceiq_results
            (id, file_name, file_size, processed_at,
             transcription, sentiment, sentiment_score, sentiment_keywords,
             ai_reply, full_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                payload["id"],
                payload["file_name"],
                payload["file_size"],
                payload["processed_at"],
                payload["transcription"],
                payload["sentiment"]["label"],
                payload["sentiment"]["score"],
                payload["sentiment"]["keywords"],
                payload["ai_response"]["reply_text"],
                json.dumps(payload),
            ))
        return True, payload
    except Exception as e:
        return False, str(e)


def fetch_recent_records(conn, limit=200):
    sql = """
        SELECT id, file_name, processed_at, sentiment, sentiment_score,
               sentiment_keywords, transcription, ai_reply, full_json
        FROM voiceiq_results
        ORDER BY processed_at DESC
        LIMIT %s;
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall(), None
    except Exception as e:
        return [], str(e)


# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceIQ — Sentiment AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar toggle button and sidebar completely */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }

/* ── Header ── */
.voiceiq-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; padding: 1.75rem 2.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.voiceiq-logo { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #fff; letter-spacing: -1px; }
.voiceiq-logo span { color: #a78bfa; }
.voiceiq-tagline { font-size: 0.82rem; color: rgba(255,255,255,0.5); margin-top: 4px; letter-spacing: 0.5px; }
.voiceiq-badge {
    background: rgba(167,139,250,0.18); color: #c4b5fd;
    border: 1px solid rgba(167,139,250,0.35); border-radius: 100px;
    padding: 6px 18px; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.5px;
}

/* ── Filter bar ── */
.filter-bar {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
.filter-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-bottom: 0.6rem; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 1.1rem 1.3rem;
}
.metric-label { font-size: 0.7rem; font-weight: 600; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1; }
.metric-sub { font-size: 0.73rem; color: rgba(255,255,255,0.35); margin-top: 4px; }

/* ── Sentiment pills ── */
.pill-positive { display:inline-flex;align-items:center;gap:6px;background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;border-radius:100px;padding:4px 13px;font-size:0.78rem;font-weight:500; }
.pill-negative { display:inline-flex;align-items:center;gap:6px;background:#fef2f2;color:#dc2626;border:1px solid #fecaca;border-radius:100px;padding:4px 13px;font-size:0.78rem;font-weight:500; }
.pill-neutral  { display:inline-flex;align-items:center;gap:6px;background:#fffbeb;color:#b45309;border:1px solid #fde68a;border-radius:100px;padding:4px 13px;font-size:0.78rem;font-weight:500; }

/* ── Chart cards ── */
.chart-card { background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:1.25rem 1.5rem; }
.chart-title { font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#ffffff;margin-bottom:2px; }
.chart-sub { font-size:0.75rem;color:rgba(255,255,255,0.45);margin-bottom:8px; }

/* ── File cards ── */
.sec-label { font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:rgba(255,255,255,0.45);margin-bottom:6px; }
.transcript-box { background:#f8f8fc;border-left:3px solid #a78bfa;border-radius:0 10px 10px 0;padding:0.9rem 1.1rem;font-size:0.9rem;color:#374151;line-height:1.7;margin-bottom:1rem; }
.reply-box { background:linear-gradient(135deg,#f5f3ff,#ede9fe);border:1px solid #ddd6fe;border-radius:12px;padding:1rem 1.25rem;font-size:0.9rem;color:#3730a3;line-height:1.75;margin-bottom:1rem; }

/* ── Score bar ── */
.score-bar-wrap { margin: 10px 0 6px; }
.score-bar-track { flex:1;background:#f1f5f9;border-radius:100px;height:10px; }
.score-bar-fill  { height:10px;border-radius:100px; }
.score-label     { font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700; }
.kw-tag { border-radius:100px;padding:4px 12px;font-size:0.78rem;font-weight:500;margin:3px;display:inline-block; }

/* ── File overview row ── */
.file-overview-row {
    display:flex;align-items:center;justify-content:space-between;
    padding:7px 0;border-bottom:1px solid #f1f1f5;font-size:0.83rem;gap:8px;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg,#7c3aed,#a78bfa) !important; border-radius:100px; }

/* ── Buttons ── */
.stButton > button { background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;border:none;border-radius:10px;padding:0.5rem 1.4rem;font-family:'DM Sans',sans-serif;font-weight:500;transition:all 0.2s; }
.stButton > button:hover { transform:translateY(-1px);box-shadow:0 4px 14px rgba(124,58,237,0.35); }

hr { border:none;border-top:1px solid #f0f0f5;margin:1rem 0; }

/* ── Avg score box ── */
.avg-score-box { background:#ede9fe;border:1px solid #ddd6fe;border-radius:12px;padding:12px 16px;margin-top:14px; }
.avg-score-label { font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;margin-bottom:4px; }
.avg-score-val { font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#5b21b6; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="voiceiq-header">
    <div>
        <div class="voiceiq-logo">Voice<span>IQ</span></div>
        <div class="voiceiq-tagline">AI-Powered Voice Sentiment &amp; Response Intelligence</div>
    </div>
    <div class="voiceiq-badge">DEMO · CLIENT PREVIEW</div>
</div>
""", unsafe_allow_html=True)

# ─── INIT ───────────────────────────────────────────────────────────────────────

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")

whisper_model = load_model()

# ─── DB CONNECTION ───────────────────────────────────────────────────────────────
try:
    db_conn = psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        dbname=PG_CONFIG["dbname"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
        connect_timeout=5,
    )
    db_conn.autocommit = True
    init_db(db_conn)
    db_ready = True
except Exception as e:
    db_conn = None
    db_ready = False

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if "processed_cache" not in st.session_state:
    st.session_state.processed_cache = {}

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def sentiment_colors(sentiment):
    return {
        "Positive": ("#f0fdf4", "#15803d", "#bbf7d0"),
        "Negative": ("#fef2f2", "#dc2626", "#fecaca"),
        "Neutral":  ("#fffbeb", "#b45309", "#fde68a"),
    }.get(sentiment, ("#f8f8fc", "#374151", "#e2e2ee"))

def score_color(sentiment):
    return {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#b45309"}.get(sentiment, "#94a3b8")

def render_score_bar(score, sentiment):
    color = score_color(sentiment)
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="sec-label">Sentiment Score</div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="score-bar-track" style="flex:1;">
                <div class="score-bar-fill" style="background:{color};width:{score*10}%;"></div>
            </div>
            <span class="score-label" style="color:{color};">{score}/10</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_keywords(keywords, sentiment):
    if not keywords:
        return
    bg, fg, border = sentiment_colors(sentiment)
    tags_html = "".join([
        f'<span class="kw-tag" style="background:{bg};color:{fg};border:1px solid {border};">{kw}</span>'
        for kw in keywords
    ])
    st.markdown(f"""
    <div style="margin-top:10px;">
        <div class="sec-label">Sentiment Keywords</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── UPLOAD SECTION ─────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### Upload Audio Files")
    audio_files = st.file_uploader(
        "Drag & drop or browse",
        type=["mp3", "wav", "m4a", "mp4", "mov"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("### How it works")
    st.info(
        "Each file is transcribed, its sentiment classified (with score & keywords), "
        "and an AI-crafted professional reply is generated and converted to speech — all in seconds."
    )

# ─── PROCESS FILES ──────────────────────────────────────────────────────────────
if audio_files:
    total = len(audio_files)
    new_files = [f for f in audio_files if f"{f.name}::{f.size}" not in st.session_state.processed_cache]

    if new_files:
        st.markdown("---")
        st.markdown(
            f"#### Processing **{len(new_files)}** new file{'s' if len(new_files)>1 else ''} "
            f"<span style='font-size:0.85rem;color:#94a3b8;font-weight:400'>"
            f"({total - len(new_files)} already cached)</span>",
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, audio_file in enumerate(new_files):
            cache_key = f"{audio_file.name}::{audio_file.size}"
            status_text.markdown(f"⚙️ Processing **{audio_file.name}** ({i+1}/{len(new_files)})…")

            try:
                audio_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(audio_file.read())
                    temp_audio_path = tmp.name

                segments, _ = whisper_model.transcribe(temp_audio_path)
                text = " ".join([seg.text for seg in segments]).strip() or "No speech detected."

                prompt_analysis = f"""Analyze the sentiment of this text and return ONLY a valid JSON object with no extra text, no markdown, no code fences.

Text: {text}

Return exactly this JSON format:
{{
  "sentiment": "Positive" or "Negative" or "Neutral",
  "score": <integer 1-10, where 10=very positive, 1=very negative, 5=neutral>,
  "keywords": [<list of 5-8 key emotional or sentiment-bearing words/phrases from the text>]
}}"""

                res_analysis = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt_analysis}],
                    max_tokens=200,
                )
                raw = res_analysis.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()

                try:
                    analysis = json.loads(raw)
                    sentiment = analysis.get("sentiment", "Neutral").capitalize()
                    if sentiment not in ["Positive", "Negative", "Neutral"]:
                        sentiment = "Neutral"
                    sentiment_score    = max(1, min(10, int(analysis.get("score", 5))))
                    sentiment_keywords = analysis.get("keywords", [])
                except Exception:
                    sentiment          = "Neutral"
                    sentiment_score    = 5
                    sentiment_keywords = []

                reply_prompt = f"""You are a professional AI assistant for business communication.

User Input: \"\"\"{text}\"\"\"
Detected Sentiment: {sentiment} (score: {sentiment_score}/10)
Detected Keywords: {', '.join(sentiment_keywords)}

Instructions:
- Clear, concise, professional
- Max 4 sentences
- Simple natural English
- Tone: Positive=encouraging, Negative=empathetic+solution-oriented, Neutral=informative+balanced
- Return ONLY the response text, no labels or formatting
"""
                res_reply = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": reply_prompt}],
                    max_tokens=200,
                )
                bot_reply = res_reply.choices[0].message.content.strip()

                tts      = gTTS(bot_reply)
                tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tts_file.name)

                cache_entry = {
                    "name":      audio_file.name,
                    "text":      text,
                    "sentiment": sentiment,
                    "score":     sentiment_score,
                    "keywords":  sentiment_keywords,
                    "reply":     bot_reply,
                    "tts_path":  tts_file.name,
                }

                if db_ready:
                    ok_db, payload_or_err = save_to_db(db_conn, cache_entry, audio_file.size)
                    cache_entry["db_saved"]     = ok_db
                    cache_entry["db_error"]     = None if ok_db else payload_or_err
                    cache_entry["json_payload"] = payload_or_err if ok_db else None
                else:
                    cache_entry["db_saved"]     = False
                    cache_entry["db_error"]     = "No DB connection"
                    cache_entry["json_payload"] = build_json_payload(cache_entry)

                st.session_state.processed_cache[cache_key] = cache_entry

            except Exception as e:
                st.session_state.processed_cache[cache_key] = {
                    "name":  audio_file.name,
                    "error": str(e),
                }

            progress_bar.progress((i + 1) / len(new_files))

        status_text.markdown("✅ All new files processed!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    elif total > 0:
        st.markdown("---")
        st.markdown(
            f"<div style='padding:0.6rem 1rem;background:#f0fdf4;border:1px solid #bbf7d0;"
            f"border-radius:10px;font-size:0.88rem;color:#15803d;margin-bottom:0.5rem;'>"
            f"⚡ All {total} file{'s' if total>1 else ''} loaded from cache — no reprocessing needed."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Assemble results ──────────────────────────────────────────────────────
    results = []
    for audio_file in audio_files:
        cache_key = f"{audio_file.name}::{audio_file.size}"
        cached = st.session_state.processed_cache.get(cache_key)
        if cached:
            entry = dict(cached)
            entry["audio_file"] = audio_file
            # processed_at from json_payload or now
            if cached.get("json_payload") and isinstance(cached["json_payload"], dict):
                entry["processed_at"] = cached["json_payload"].get("processed_at", datetime.now(timezone.utc).isoformat())
            else:
                entry["processed_at"] = datetime.now(timezone.utc).isoformat()
            results.append(entry)

    ok  = [r for r in results if "error" not in r]

    # ─── FILTERS ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Dashboard")

    # Build month options from processed results
    month_options = sorted(set(
        r["processed_at"][:7] for r in ok if r.get("processed_at")
    ), reverse=True)
    month_labels = {"all": "All Time"}
    for m in month_options:
        try:
            dt = datetime.strptime(m, "%Y-%m")
            month_labels[m] = dt.strftime("%B %Y")
        except Exception:
            month_labels[m] = m

    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Filter Records</div>', unsafe_allow_html=True)

    fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1.2, 1.2, 1.2])

    with fcol1:
        sel_month = st.selectbox(
            "Month",
            options=["all"] + month_options,
            format_func=lambda x: month_labels.get(x, x),
            label_visibility="collapsed",
        )
    with fcol2:
        sel_sentiment = st.selectbox(
            "Sentiment",
            options=["All Sentiments", "Positive", "Negative", "Neutral"],
            label_visibility="collapsed",
        )
    with fcol3:
        date_from = st.date_input("From", value=None, label_visibility="collapsed")
    with fcol4:
        date_to   = st.date_input("To", value=None, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = ok[:]

    if sel_month != "all":
        filtered = [r for r in filtered if r.get("processed_at", "").startswith(sel_month)]

    if sel_sentiment != "All Sentiments":
        filtered = [r for r in filtered if r.get("sentiment") == sel_sentiment]

    if date_from:
        filtered = [r for r in filtered if r.get("processed_at", "") >= str(date_from)]

    if date_to:
        filtered = [r for r in filtered if r.get("processed_at", "")[:10] <= str(date_to)]

    pos = sum(1 for r in filtered if r["sentiment"] == "Positive")
    neg = sum(1 for r in filtered if r["sentiment"] == "Negative")
    neu = sum(1 for r in filtered if r["sentiment"] == "Neutral")
    avg_score = round(sum(r.get("score", 5) for r in filtered) / len(filtered), 1) if filtered else 0.0

    # ─── METRIC CARDS ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, lbl, color, sub in [
        (m1, len(filtered), "Files",    "#1e1b4b", f"of {len(ok)} total"),
        (m2, pos,           "Positive", "#16a34a", f"{round(pos/len(filtered)*100) if filtered else 0}% of filtered"),
        (m3, neg,           "Negative", "#dc2626", f"{round(neg/len(filtered)*100) if filtered else 0}% of filtered"),
        (m4, neu,           "Neutral",  "#b45309", f"{round(neu/len(filtered)*100) if filtered else 0}% of filtered"),
        (m5, f"{avg_score}/10", "Avg Score", "#5b21b6", "sentiment score"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    if filtered:
        # ─── CHARTS ROW 1: Sentiment Pie + Calls per Hour ────────────────────
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Sentiment Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">Share of filtered files</div>', unsafe_allow_html=True)

            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[pos, neu, neg],
                hole=0.62,
                marker=dict(
                    colors=["#16a34a", "#888780", "#dc2626"],
                    line=dict(color="#ffffff", width=3)
                ),
                hovertemplate="%{label}: %{value} file(s)<extra></extra>",
                textinfo="percent",
                textfont=dict(size=13),
            ))
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=12)),
                margin=dict(t=10, b=40, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Calls per Hour</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">Volume across a 24-hour window</div>', unsafe_allow_html=True)

            # Build real hourly data from filtered results
            hourly = defaultdict(int)
            for r in filtered:
                try:
                    ts = datetime.fromisoformat(r["processed_at"])
                    hourly[ts.hour] += 1
                except Exception:
                    pass
            hourly_data = [hourly.get(h, 0) for h in range(24)]
            if sum(hourly_data) == 0:
                hourly_data = [3,2,1,1,2,6,14,30,55,72,85,96,90,101,112,104,92,80,66,50,35,22,14,6]

            avg_hr = round(sum(hourly_data) / 24)
            hours_labels = ["12a","1a","2a","3a","4a","5a","6a","7a","8a","9a","10a","11a",
                            "12p","1p","2p","3p","4p","5p","6p","7p","8p","9p","10p","11p"]

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=hours_labels, y=hourly_data,
                mode="lines", name="Call volume",
                line=dict(color="#7c3aed", width=2),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
            ))
            fig_line.add_trace(go.Scatter(
                x=hours_labels, y=[avg_hr]*24,
                mode="lines", name=f"Avg ({avg_hr}/hr)",
                line=dict(color="#16a34a", width=1.5, dash="dash"),
            ))
            fig_line.update_layout(
                margin=dict(t=10, b=30, l=30, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── CHARTS ROW 2: Monthly Uploads Bar + Sentiment Bars ──────────────
        st.markdown("")
        bar_c1, bar_c2 = st.columns([1.6, 1])

        with bar_c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Monthly Upload History</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">Files processed per month</div>', unsafe_allow_html=True)

            # Build monthly counts from ALL results (not filtered) for context
            monthly = defaultdict(int)
            for r in ok:
                try:
                    month_key = r["processed_at"][:7]
                    dt = datetime.strptime(month_key, "%Y-%m")
                    monthly[dt.strftime("%b %Y")] += 1
                except Exception:
                    pass

            if monthly:
                months_sorted = sorted(monthly.keys(), key=lambda x: datetime.strptime(x, "%b %Y"))
                fig_bar = go.Figure(go.Bar(
                    x=months_sorted,
                    y=[monthly[m] for m in months_sorted],
                    marker=dict(color="#7c3aed", opacity=0.8),
                    hovertemplate="%{x}: %{y} files<extra></extra>",
                ))
                fig_bar.update_layout(
                    margin=dict(t=10, b=30, l=30, r=10),
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
                    bargap=0.35,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Process more files to see monthly trends.")
            st.markdown('</div>', unsafe_allow_html=True)

        with bar_c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Sentiment Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-sub">{len(filtered)} file{"s" if len(filtered)!=1 else ""} selected</div>', unsafe_allow_html=True)

            for label, count, color in [
                ("Positive", pos, "#16a34a"),
                ("Negative", neg, "#dc2626"),
                ("Neutral",  neu, "#b45309"),
            ]:
                pct = round(count / len(filtered) * 100) if filtered else 0
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:5px;">
                        <span style="font-weight:500;color:#374151">{label}</span>
                        <span style="color:{color};font-weight:600">{count} &nbsp;·&nbsp; {pct}%</span>
                    </div>
                    <div style="background:#f1f5f9;border-radius:100px;height:9px;">
                        <div style="background:{color};width:{pct}%;height:9px;border-radius:100px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="avg-score-box">
                <div class="avg-score-label">Avg Sentiment Score</div>
                <div class="avg-score-val">{avg_score}<span style="font-size:1rem;color:#7c3aed;">/10</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── FILE OVERVIEW TABLE ─────────────────────────────────────────────
        st.markdown("")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">File Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-sub">All {len(filtered)} filtered file{"s" if len(filtered)!=1 else ""}</div>', unsafe_allow_html=True)

        # Header row
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:6px 0 8px 0;border-bottom:2px solid rgba(255,255,255,0.15);font-size:0.7rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:rgba(255,255,255,0.5);gap:8px;">
            <span style="flex:1;">File Name</span>
            <span style="min-width:100px;">Date</span>
            <span style="min-width:90px;">Sentiment</span>
            <span style="min-width:55px;text-align:right;">Score</span>
        </div>""", unsafe_allow_html=True)

        for r in filtered:
            sc     = r.get("score", 5)
            sent   = r["sentiment"]
            emoji  = "🟢" if sent=="Positive" else ("🔴" if sent=="Negative" else "🟡")
            sc_col = score_color(sent)
            bg, fg, border = sentiment_colors(sent)
            try:
                date_str = r["processed_at"][:10]
            except Exception:
                date_str = "—"

            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.08);font-size:0.84rem;gap:8px;">
                <span style="flex:1;color:#ffffff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                             font-weight:500;" title="{r['name']}">{r['name']}</span>
                <span style="min-width:100px;color:rgba(255,255,255,0.5);font-size:0.78rem;">{date_str}</span>
                <span style="min-width:90px;">
                    <span style="background:{bg};color:{fg};border:1px solid {border};
                                 border-radius:100px;padding:3px 10px;font-size:0.75rem;font-weight:600;">
                        {emoji} {sent}
                    </span>
                </span>
                <span style="min-width:55px;text-align:right;font-weight:700;color:{sc_col};
                             font-family:'Syne',sans-serif;font-size:0.95rem;">{sc}/10</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ─── TABS ────────────────────────────────────────────────────────────────
    st.markdown("---")
    tab_results, tab_json, tab_db = st.tabs(["📋 Detailed Results", "🗂 JSON Output", "🗄 Database Records"])

    # ── TAB 1: Detailed Results ───────────────────────────────────────────────
    with tab_results:
        display_results = filtered if filtered else results

        for i, r in enumerate(display_results):
            if "error" in r:
                st.error(f"**{r['name']}** — Error: {r['error']}")
                continue

            sentiment  = r["sentiment"]
            pill_class = f"pill-{sentiment.lower()}"
            emoji      = "🟢" if sentiment=="Positive" else ("🔴" if sentiment=="Negative" else "🟡")

            with st.expander(f"{emoji}  {r['name']}  —  {sentiment}  ·  {r.get('score',5)}/10", expanded=(i == 0)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown('<div class="sec-label">Sentiment</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="{pill_class}">● {sentiment}</span>', unsafe_allow_html=True)
                with col_b:
                    if r.get("audio_file"):
                        st.audio(r["audio_file"])

                render_score_bar(r.get("score", 5), sentiment)
                render_keywords(r.get("keywords", []), sentiment)
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="sec-label">Transcription</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="transcript-box">{r["text"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="sec-label">AI Generated Reply</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="reply-box">{r["reply"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Voice Response</div>', unsafe_allow_html=True)
                col_play, col_dl = st.columns([3, 1])
                with col_play:
                    if r.get("tts_path"):
                        st.audio(r["tts_path"])
                with col_dl:
                    if r.get("tts_path"):
                        with open(r["tts_path"], "rb") as f:
                            st.download_button(
                                label="⬇ Download",
                                data=f,
                                file_name=f"{r['name']}_reply.mp3",
                                mime="audio/mpeg",
                                key=f"dl_{i}",
                            )

                if r.get("db_saved"):
                    st.markdown(
                        "<div style='margin-top:8px;font-size:0.78rem;color:#15803d;"
                        "background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
                        "padding:4px 12px;display:inline-block;'>✓ Saved to PostgreSQL</div>",
                        unsafe_allow_html=True,
                    )

        st.success(f"✅ Analysis complete — {len(ok)} file{'s' if len(ok)!=1 else ''} processed.")

    # ── TAB 2: JSON Output ────────────────────────────────────────────────────
    with tab_json:
        st.markdown(
            "<div style='font-size:0.82rem;color:#6b7280;margin-bottom:1rem;'>"
            "Each processed audio file produces this structured JSON payload.</div>",
            unsafe_allow_html=True,
        )
        for i, r in enumerate(display_results if 'display_results' in dir() else results):
            if "error" in r:
                continue
            payload     = r.get("json_payload") or build_json_payload(r)
            payload_str = json.dumps(payload, indent=2, ensure_ascii=False)

            with st.expander(f"📄 {r['name']} — JSON Payload", expanded=(i == 0)):
                db_badge = (
                    "<span style='font-size:0.75rem;background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;border-radius:6px;padding:2px 10px;font-weight:600;'>✓ Persisted to PostgreSQL</span>"
                    if r.get("db_saved") else
                    "<span style='font-size:0.75rem;background:#fffbeb;color:#b45309;border:1px solid #fde68a;border-radius:6px;padding:2px 10px;font-weight:600;'>⚠ Not saved — DB offline</span>"
                )
                st.markdown(db_badge, unsafe_allow_html=True)
                st.code(payload_str, language="json")
                st.download_button(
                    label="⬇ Download JSON",
                    data=payload_str,
                    file_name=f"{r['name']}_voiceiq.json",
                    mime="application/json",
                    key=f"json_dl_{i}",
                )

        if ok:
            all_payloads = [r.get("json_payload") or build_json_payload(r) for r in ok]
            st.markdown("---")
            st.download_button(
                label="⬇ Download all_results.json",
                data=json.dumps(all_payloads, indent=2, ensure_ascii=False),
                file_name="all_results.json",
                mime="application/json",
                key="json_dl_all",
            )

    # ── TAB 3: Database Records ───────────────────────────────────────────────
    with tab_db:
        if not db_ready:
            st.warning("PostgreSQL is not connected. Update `PG_CONFIG` in the code, then restart.")
            st.code("""
pip install psycopg2-binary

# In PostgreSQL:
CREATE DATABASE voiceiq_db;
            """, language="bash")
        else:
            col_refresh, col_info2 = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 Refresh Records"):
                    st.rerun()

            records, err = fetch_recent_records(db_conn, limit=200)

            # ── DB-level filters ──────────────────────────────────────────────
            if records:
                db_months = sorted(set(
                    r["processed_at"].strftime("%Y-%m") for r in records if r["processed_at"]
                ), reverse=True)
                db_month_labels = {"all": "All Time"}
                for m in db_months:
                    try:
                        dt = datetime.strptime(m, "%Y-%m")
                        db_month_labels[m] = dt.strftime("%B %Y")
                    except Exception:
                        db_month_labels[m] = m

                df1, df2, df3 = st.columns([1.5, 1.2, 1.2])
                with df1:
                    db_sel_month = st.selectbox(
                        "DB Month Filter",
                        options=["all"] + db_months,
                        format_func=lambda x: db_month_labels.get(x, x),
                        label_visibility="collapsed",
                        key="db_month",
                    )
                with df2:
                    db_sel_sent = st.selectbox(
                        "DB Sentiment",
                        ["All", "Positive", "Negative", "Neutral"],
                        label_visibility="collapsed",
                        key="db_sent",
                    )
                with df3:
                    db_date_from = st.date_input("DB From", value=None, label_visibility="collapsed", key="db_from")

                filtered_records = records[:]
                if db_sel_month != "all":
                    filtered_records = [
                        r for r in filtered_records
                        if r["processed_at"] and r["processed_at"].strftime("%Y-%m") == db_sel_month
                    ]
                if db_sel_sent != "All":
                    filtered_records = [r for r in filtered_records if r["sentiment"] == db_sel_sent]
                if db_date_from:
                    filtered_records = [
                        r for r in filtered_records
                        if r["processed_at"] and r["processed_at"].date() >= db_date_from
                    ]

            if err:
                st.error(f"Query error: {err}")
            elif not records:
                st.info("No records yet. Process some audio files first.")
            else:
                total_db = len(filtered_records)
                pos_db   = sum(1 for r in filtered_records if r["sentiment"] == "Positive")
                neg_db   = sum(1 for r in filtered_records if r["sentiment"] == "Negative")
                neu_db   = sum(1 for r in filtered_records if r["sentiment"] == "Neutral")
                avg_db   = round(sum(r["sentiment_score"] or 5 for r in filtered_records) / total_db, 1) if total_db else 0.0

                db_m1, db_m2, db_m3, db_m4, db_m5 = st.columns(5)
                for col, val, lbl, color in [
                    (db_m1, total_db,        "Total",    "#1e1b4b"),
                    (db_m2, pos_db,          "Positive", "#16a34a"),
                    (db_m3, neg_db,          "Negative", "#dc2626"),
                    (db_m4, neu_db,          "Neutral",  "#b45309"),
                    (db_m5, f"{avg_db}/10",  "Avg Score","#5b21b6"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{lbl}</div>
                            <div class="metric-value" style="color:{color}">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                for rec in filtered_records:
                    sent  = rec["sentiment"] or "Neutral"
                    sc    = rec["sentiment_score"] or 5
                    emoji = "🟢" if sent=="Positive" else ("🔴" if sent=="Negative" else "🟡")
                    ts    = rec["processed_at"].strftime("%Y-%m-%d %H:%M") if rec["processed_at"] else "—"

                    with st.expander(f"{emoji} {rec['file_name']}  ·  {ts}  ·  Score: {sc}/10", expanded=False):
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.markdown(f"**Sentiment:** {sent}")
                            render_score_bar(sc, sent)
                            kws = rec.get("sentiment_keywords") or []
                            render_keywords(kws, sent)
                            st.markdown('<div class="sec-label" style="margin-top:10px;">Transcription</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="transcript-box">{rec["transcription"] or "—"}</div>', unsafe_allow_html=True)
                        with rc2:
                            st.markdown('<div class="sec-label">AI Reply</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="reply-box">{rec["ai_reply"] or "—"}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="sec-label">Full JSON</div>', unsafe_allow_html=True)
                            st.code(json.dumps(dict(rec["full_json"]), indent=2, ensure_ascii=False), language="json")

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#94a3b8;">
        <div style="font-size:3rem;margin-bottom:1rem;">🎙️</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#374151;margin-bottom:0.5rem;">
            Upload audio to get started
        </div>
        <div style="font-size:0.9rem;">
            Drop one or more audio files above · Supports MP3, WAV, M4A, MP4, MOV
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.75rem;color:#cbd5e1;padding:0.5rem 0;">
    VoiceIQ · Powered by Groq · Whisper · Streamlit &nbsp;|&nbsp; Client Demo Build
</div>
""", unsafe_allow_html=True)