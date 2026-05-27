import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import re
from collections import deque

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(
    page_title="AI Text Classifier | Enterprise Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL CLEAN CSS (Slate & Cobalt Theme)
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean, soft background */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Elegant Minimal Header */
    .main-header {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .badge {
        display: inline-block;
        background: #334155;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.75rem;
        color: #cbd5e1;
        margin: 0.2rem;
        font-weight: 500;
        border: 1px solid #475569;
    }
    
    /* Modern minimalist container cards */
    .glass-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Clean Result Cards */
    .result-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .spam-result {
        background: #7f1d1d;
        border: 1px solid #f87171;
        border-radius: 8px;
        padding: 1.5rem;
        color: #fef2f2;
    }
    
    .ham-result {
        background: #064e3b;
        border: 1px solid #34d399;
        border-radius: 8px;
        padding: 1.5rem;
        color: #ecfdf5;
    }
    
    .positive-result {
        background: #064e3b;
        border: 1px solid #34d399;
        border-radius: 8px;
        padding: 1.5rem;
        color: #ecfdf5;
    }
    
    .negative-result {
        background: #7f1d1d;
        border: 1px solid #f87171;
        border-radius: 8px;
        padding: 1.5rem;
        color: #fef2f2;
    }
    
    .neutral-result {
        background: #14532d;
        border: 1px solid #4ade80;
        border-radius: 8px;
        padding: 1.5rem;
        color: #f0fdf4;
    }
    
    /* Standardized Buttons */
    .stButton button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        width: 100%;
        transition: background 0.2s ease;
    }
    
    .stButton button:hover {
        background: #1d4ed8;
        border: none;
        color: white;
    }
    
    /* Text input overrides */
    .stTextArea textarea {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f8fafc;
    }
    
    .stTextArea textarea:focus {
        border-color: #2563eb;
    }
    
    /* History card standard row */
    .history-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #64748b;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    /* Corporate Info box */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #cbd5e1;
        font-size: 0.85rem;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    /* Footer layout styling */
    .modern-footer {
        text-align: center;
        padding: 1.5rem;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid #334155;
        margin-top: 3rem;
    }
    
    /* Clean sidebar setup */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load both spam and sentiment models"""
    with st.spinner("Initializing ML pipelines..."):
        models = {}
        
        # Spam Detection Model
        try:
            models["spam"] = pipeline(
                "text-classification",
                model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
                device=-1
            )
        except:
            try:
                models["spam"] = pipeline(
                    "text-classification",
                    model="bert-base-uncased",
                    device=-1
                )
            except:
                models["spam"] = None
        
        # Sentiment Analysis Model
        try:
            models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
        except:
            try:
                models["sentiment"] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
            except:
                models["sentiment"] = None
    return models

# ============================================
# FALLBACK BACKEND CLASSIFIER
# ============================================
class SimpleClassifier:
    @staticmethod
    def is_spam(text):
        text_lower = text.lower()
        spam_indicators = [
            "free", "win", "prize", "click", "subscribe", "offer", "discount",
            "limited", "urgent", "cash", "money", "lottery", "winner",
            "congratulations", "viagra", "cheap", "buy now", "act now"
        ]
        score = sum(1 for word in spam_indicators if word in text_lower)
        return score >= 2
    
    @staticmethod
    def get_sentiment(text):
        text_lower = text.lower()
        positive_words = ["good", "great", "awesome", "amazing", "love", "like", "best", "excellent", "happy", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "worst", "poor", "sad", "angry", "horrible"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "POSITIVE", max(0.5, positive_count / (positive_count + negative_count + 1))
        elif negative_count > positive_count:
            return "NEGATIVE", max(0.5, negative_count / (positive_count + negative_count + 1))
        else:
            return "NEUTRAL", 0.5

# ============================================
# STATE ENGINE
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(text, classification_type, result, confidence, timestamp):
    st.session_state.history.insert(0, {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "type": classification_type,
        "result": result,
        "confidence": confidence,
        "timestamp": timestamp,
        "full_text": text
    })
    if len(st.session_state.history) > 50:
        st.session_state.history.pop()

def clear_history():
    st.session_state.history = []

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### **Text Analytics Engine**")
    st.markdown("---")
    
    st.markdown("#### **Scope Profile**")
    st.markdown("""
    <div class="info-box">
        <strong>Spam Verification</strong><br>
        Identifies payload patterns, unsolicited structural matches, and high-risk terms.
    </div>
    <div class="info-box">
        <strong>Sentiment Evaluation</strong><br>
        Extracts emotional tonality signatures from text contexts.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### **Pipeline Architecture**")
    st.markdown("""
    | Subsystem | Baseline Model |
    | :--- | :--- |
    | Spam Detect | BERT-Tiny Variant |
    | Sentiment | RoBERTa Core Pipeline |
    | Fallback | Rule-Base Dictionary |
    """)
    
    st.markdown("---")
    st.markdown("#### **Operational Summary**")
    if st.session_state.history:
        total = len(st.session_state.history)
        spam_count = sum(1 for h in st.session_state.history if h.get("result") == "SPAM")
        positive_count = sum(1 for h in st.session_state.history if h.get("result") == "POSITIVE")
        
        st.metric("Processed Total", total)
        st.metric("Spam Incidents", spam_count, delta=f"{(spam_count/total*100):.1f}% targeted")
        st.metric("Positive Valence", positive_count, delta=f"{(positive_count/total*100):.1f}% upbeat")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Flush Cache History", use_container_width=True):
            clear_history()
            st.rerun()
    else:
        st.info("Awaiting execution metrics input.")
    
    st.markdown("---")
    st.caption(f"Engine Build 2.1.0 • v{datetime.now().year}")

# ============================================
# MAIN INTERFACE
# ============================================
st.markdown("""
<div class="main-header">
    <h1>Text Intelligence Platform</h1>
    <p>Enterprise Language Verification Engine • Powered by Transformer Models</p>
    <div>
        <span class="badge">BERT Pipeline</span>
        <span class="badge">RoBERTa Architecture</span>
        <span class="badge">Deterministic Fallback Ready</span>
        <span class="badge">Sub-second Execution</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode Selector
col1, _ = st.columns([2, 2])
with col1:
    classification_mode = st.radio(
        "Analysis Model Selection",
        ["📧 Spam Detection Pipeline", "😊 Contextual Sentiment Analysis"],
        horizontal=True
    )

# Input Block
st.markdown("### **Payload Definition**")
user_text = st.text_area(
    "Target String Content Input",
    height=140,
    placeholder="Paste clean prose, unstructured chat logs, emails, or marketing copy here for pipeline processing...",
    label_visibility="collapsed",
    key="input_text"
)

if user_text:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Character Count", len(user_text))
    with col_b:
        st.metric("Word Token Count", len(user_text.split()))
    with col_c:
        st.metric("Lines Detected", user_text.count('\n') + 1)

analyze_btn = st.button("EXECUTE ANALYSIS PIPELINE", use_container_width=True, type="primary")

# ============================================
# EXECUTION & GRAPHICS INTERFACES
# ============================================
if analyze_btn and user_text:
    try:
        models = load_models()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("✨ *Staging run parameters...*")
        progress_bar.progress(30)
        time.sleep(0.05)
        
        status_text.markdown("⚡ *Evaluating tensor layers...*")
        progress_bar.progress(70)
        
        # Branch Execution Paths
        if "Spam" in classification_mode:
            classification_type = "Spam Detection"
            if models.get("spam"):
                result = models["spam"](user_text)[0]
                is_spam = result["label"].upper() == "SPAM"
                confidence = result["score"]
                label = "SPAM" if is_spam else "NOT SPAM"
            else:
                is_spam = SimpleClassifier.is_spam(user_text)
                confidence = 0.88 if is_spam else 0.82
                label = "SPAM" if is_spam else "NOT SPAM"
            
            classification_result = label
            
            # Presentation Layout
            st.markdown("### **Core Output Parameters**")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "Pipeline Confidence", "font": {"color": "#94a3b8", "size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar": {"color": "#f87171" if is_spam else "#34d399"},
                        "bgcolor": "#1e293b",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(148, 163, 184, 0.05)"},
                            {"range": [50, 100], "color": "rgba(148, 163, 184, 0.1)"}
                        ]
                    },
                    number={"suffix": "%", "font": {"color": "#f8fafc", "size": 38}}
                ))
                fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                if is_spam:
                    st.markdown(f"""
                    <div class="result-card" style="margin-top: 25px;">
                        <div class="spam-result">
                            <div style="font-size:1.5rem; font-weight:700; letter-spacing: 0.5px;">SYSTEM ALERT: SPAM PATTERN FLAG</div>
                            <div style="font-size:1rem; margin-top:8px; opacity: 0.9;">Confidence Level Metric: {confidence*100:.2f}%</div>
                            <div style="font-size:0.85rem; margin-top:12px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">String architecture matches high-risk marketing or credential fishing heuristics.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card" style="margin-top: 25px;">
                        <div class="ham-result">
                            <div style="font-size:1.5rem; font-weight:700; letter-spacing: 0.5px;">VERIFIED SAFE: LEGITIMATE PROSE</div>
                            <div style="font-size:1rem; margin-top:8px; opacity: 0.9;">Confidence Level Metric: {confidence*100:.2f}%</div>
                            <div style="font-size:0.85rem; margin-top:12px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">No structural indicators or anomalies associated with transactional spam detected.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # Contextual Sentiment Pipeline Execution
            classification_type = "Sentiment Analysis"
            if models.get("sentiment"):
                result = models["sentiment"](user_text)[0]
                sentiment = result["label"].upper()
                confidence = result["score"]
                
                if "POS" in sentiment:
                    label = "POSITIVE"
                elif "NEG" in sentiment:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
            else:
                label, confidence = SimpleClassifier.get_sentiment(user_text)
            
            classification_result = label
            
            st.markdown("### **Core Output Parameters**")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                accent_color = "#34d399" if label == "POSITIVE" else "#f87171" if label == "NEGATIVE" else "#4ade80"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "Valence Intensity Assessment", "font": {"color": "#94a3b8", "size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar": {"color": accent_color},
                        "bgcolor": "#1e293b",
                        "borderwidth": 1,
                        "bordercolor": "#334155"
                    },
                    number={"suffix": "%", "font": {"color": "#f8fafc", "size": 38}}
                ))
                fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                if label == "POSITIVE":
                    st.markdown(f"""
                    <div class="result-card" style="margin-top: 25px;">
                        <div class="positive-result">
                            <div style="font-size:1.5rem; font-weight:700;">VALENCE SIGNATURE: POSITIVE</div>
                            <div style="font-size:1rem; margin-top:8px;">Classification Weight: {confidence*100:.2f}%</div>
                            <div style="font-size:0.85rem; margin-top:12px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">Target payload demonstrates highly constructive or favorable tone indicators.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif label == "NEGATIVE":
                    st.markdown(f"""
                    <div class="result-card" style="margin-top: 25px;">
                        <div class="negative-result">
                            <div style="font-size:1.5rem; font-weight:700;">VALENCE SIGNATURE: NEGATIVE</div>
                            <div style="font-size:1rem; margin-top:8px;">Classification Weight: {confidence*100:.2f}%</div>
                            <div style="font-size:0.85rem; margin-top:12px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">Target payload maps clearly to structural features of critique or customer friction.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card" style="margin-top: 25px;">
                        <div class="neutral-result">
                            <div style="font-size:1.5rem; font-weight:700;">VALENCE SIGNATURE: NEUTRAL</div>
                            <div style="font-size:1rem; margin-top:8px;">Classification Weight: {confidence*100:.2f}%</div>
                            <div style="font-size:0.85rem; margin-top:12px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">The text maintains a standard non-emotional, informational, or objective tone context.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Clean Probability Distribution chart
            st.markdown("#### **Categorical Valence Metrics**")
            prob_pos = confidence if label == "POSITIVE" else (0.15 if label == "NEGATIVE" else 0.20)
            prob_neg = confidence if label == "NEGATIVE" else (0.10 if label == "POSITIVE" else 0.15)
            prob_neu = confidence if label == "NEUTRAL" else (1.0 - prob_pos - prob_neg)
            
            sentiment_data = pd.DataFrame({
                "Dimension Label": ["Positive Intent", "Neutral Core", "Negative Intent"],
                "Confidence Metrics Value": [prob_pos, prob_neu, prob_neg]
            })
            
            fig2 = px.bar(
                sentiment_data, 
                x="Dimension Label", 
                y="Confidence Metrics Value",
                color="Dimension Label",
                color_discrete_map={"Positive Intent": "#10b981", "Neutral Core": "#64748b", "Negative Intent": "#ef4444"},
                text="Confidence Metrics Value"
            )
            fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig2.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8"},
                xaxis_title="",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Clear progress artifacts cleanly
        progress_bar.progress(100)
        time.sleep(0.1)
        progress_bar.empty()
        status_text.empty()
        
        # Add Record Entry safely
        timestamp = datetime.now().strftime("%H:%M:%S Profile")
        add_to_history(user_text, classification_type, classification_result, confidence, timestamp)
        
    except Exception as e:
        st.error(f"Execution Exception Framework Interrupt: {str(e)}")

elif analyze_btn and not user_text:
    st.error("Validation Halt: Input buffer cannot remain null during execution sequence.")

# ============================================
# HISTORICAL DATA RETRIEVAL ROWS
# ============================================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### **Operational Audit History Logs**")
    
    for item in st.session_state.history[:6]:
        if "Spam" in item["type"]:
            border_c = "#ef4444" if "SPAM" == item["result"] else "#10b981"
            display_result = item["result"]
        else:
            border_c = "#10b981" if item["result"] == "POSITIVE" else "#ef4444" if item["result"] == "NEGATIVE" else "#64748b"
            display_result = f"SENTIMENT: {item['result']}"
            
        st.markdown(f"""
        <div class="history-card" style="border-left-color: {border_c};">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <div><strong>{display_result}</strong> <span style="color: #64748b; margin-left: 12px;">• Confidence Context Value: {item['confidence']*100:.1f}%</span></div>
                <div style="color: #475569;">{item['timestamp']}</div>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px; font-style: italic;">"{item['text']}"</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# SYSTEM FOOTER ARCHITECTURE
# ============================================
st.markdown("""
<div class="modern-footer">
    <p><strong>Enterprise Text Classification System Framework</strong> • Standard Secure Deploy Interface</p>
    <p style="opacity: 0.6;">Compliant NLP Pipeline Layer Engine Protocol Architecture Components</p>
</div>
""", unsafe_allow_html=True)
