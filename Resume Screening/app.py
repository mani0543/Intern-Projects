import streamlit as st
import pandas as pd
import PyPDF2
import docx  # Changed from docx2txt
import re
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight: bold; }
.score-good { color: #00CC96; font-size: 24px; font-weight: bold; }
.score-avg { color: #FFA500; font-size: 24px; font-weight: bold; }
.score-bad { color: #EF553B; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return ""

def extract_text_from_docx(file):
    try:
        # Read the docx file
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return ""

def extract_text_from_txt(file):
    try:
        return file.getvalue().decode("utf-8")
    except Exception as e:
        return ""

def extract_resume_text(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_type == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_type == 'txt':
        return extract_text_from_txt(uploaded_file)
    return ""

def calculate_match_score(resume_text, job_description):
    # Simple keyword matching
    resume_words = set(re.findall(r'\b[a-z]{3,}\b', resume_text.lower()))
    job_words = set(re.findall(r'\b[a-z]{3,}\b', job_description.lower()))
    
    stopwords = {'the', 'and', 'for', 'are', 'with', 'have', 'this', 'that', 'from', 'they', 'will', 'what', 'your', 'you', 'can', 'has', 'was', 'were', 'been', 'being', 'but', 'not', 'all', 'any', 'each', 'which', 'their', 'there', 'could', 'would', 'should'}
    resume_words = resume_words - stopwords
    job_words = job_words - stopwords
    
    if not job_words:
        return 0
    
    common = len(resume_words.intersection(job_words))
    score = (common / len(job_words)) * 100
    return min(100, score)

def extract_skills(text, skill_list):
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        if skill.lower() in text_lower:
            found.append(skill)
    return found

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="screening_results.csv">📥 Download Results CSV</a>'

# Main app
def main():
    st.title("📄 Resume Screening System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Skills database
        default_skills = "Python, SQL, Machine Learning, Deep Learning, NLP, Computer Vision, Data Analysis, TensorFlow, PyTorch, AWS, Docker, Git, React, JavaScript, Java, C++, Communication, Leadership, Project Management, Excel, Tableau, Power BI, Hadoop, Spark, Kafka, MongoDB, PostgreSQL"
        skills = st.text_area("Skills to look for:", default_skills, height=150)
        skill_list = [s.strip() for s in skills.split(',') if s.strip()]
        
        # Threshold
        threshold = st.slider("Shortlist threshold (%)", 0, 100, 60)
        
        st.markdown("---")
        st.info("💡 **Tips:**\n- Write detailed job descriptions\n- Upload PDF, DOCX, or TXT files\n- Adjust threshold to filter candidates")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Job Description")
        job_desc = st.text_area("Paste job description here:", height=300, 
                               placeholder="Example:\n\nWe are looking for a Data Scientist with experience in Python, Machine Learning, and SQL...")
        
        if job_desc:
            st.success(f"✅ Job description loaded ({len(job_desc.split())} words)")
    
    with col2:
        st.subheader("📎 Upload Resumes")
        files = st.file_uploader("Choose files (PDF, DOCX, TXT)", 
                                type=['pdf', 'docx', 'txt'], 
                                accept_multiple_files=True,
                                help="Upload multiple resumes at once")
        
        if files:
            st.success(f"✅ {len(files)} file(s) uploaded")
    
    # Process
    if files and job_desc:
        st.markdown("---")
        st.subheader("📊 Screening Results")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}... ({i+1}/{len(files)})")
            progress_bar.progress((i + 1) / len(files))
            
            text = extract_resume_text(file)
            if not text or len(text.strip()) < 50:
                st.warning(f"⚠️ Could not extract enough text from {file.name}")
                continue
                
            score = calculate_match_score(text, job_desc)
            found_skills = extract_skills(text, skill_list)
            skill_percent = (len(found_skills) / len(skill_list)) * 100 if skill_list else 0
            
            # Determine score color
            if score >= 70:
                score_color = "🟢"
            elif score >= 40:
                score_color = "🟡"
            else:
                score_color = "🔴"
            
            results.append({
                "Resume": file.name,
                "Match Score": round(score, 1),
                "Skills Found": len(found_skills),
                "Skill Match %": round(skill_percent, 1),
                "Status": "✅ Shortlisted" if score >= threshold else "❌ Rejected",
                "Top Skills": ", ".join(found_skills[:5]) if found_skills else "None found"
            })
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("Match Score", ascending=False)
            
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Resumes", len(results))
            with col2:
                shortlisted = len([r for r in results if r['Status'] == '✅ Shortlisted'])
                st.metric("Shortlisted", shortlisted)
            with col3:
                st.metric("Avg Score", f"{df['Match Score'].mean():.1f}%")
            with col4:
                st.metric("Top Score", f"{df['Match Score'].max():.1f}%")
            with col5:
                rejection_rate = ((len(results) - shortlisted) / len(results)) * 100
                st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
            
            st.markdown("---")
            
            # Display results table
            st.dataframe(
                df[["Resume", "Match Score", "Skill Match %", "Skills Found", "Status", "Top Skills"]], 
                use_container_width=True,
                hide_index=True
            )
            
            # Tabs for visualizations
            tab1, tab2 = st.tabs(["📊 Charts", "📈 Analysis"])
            
            with tab1:
                # Bar chart
                fig = px.bar(df, x="Resume", y="Match Score", color="Status",
                            title="Match Scores by Candidate",
                            labels={"Match Score": "Match Score (%)", "Resume": "Candidate"},
                            color_discrete_map={
                                "✅ Shortlisted": "#00CC96",
                                "❌ Rejected": "#EF553B"
                            },
                            height=500)
                fig.update_layout(showlegend=True, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Horizontal bar chart for skills
                if df['Skills Found'].sum() > 0:
                    fig2 = px.bar(df, x="Skills Found", y="Resume", orientation='h',
                                 title="Number of Skills Found per Candidate",
                                 labels={"Skills Found": "Number of Skills", "Resume": "Candidate"},
                                 color="Match Score",
                                 color_continuous_scale="Viridis",
                                 height=400)
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                # Score distribution histogram
                fig3 = px.histogram(df, x="Match Score", nbins=20,
                                   title="Match Score Distribution",
                                   labels={"Match Score": "Match Score (%)", "count": "Number of Candidates"},
                                   color_discrete_sequence=["#1E88E5"])
                fig3.add_vline(x=threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {threshold}%")
                st.plotly_chart(fig3, use_container_width=True)
                
                # Pie chart for shortlisted vs rejected
                fig4 = px.pie(df, names="Status", title="Shortlisting Summary",
                             color="Status",
                             color_discrete_map={"✅ Shortlisted": "#00CC96", "❌ Rejected": "#EF553B"})
                st.plotly_chart(fig4, use_container_width=True)
            
            # Word cloud
            if job_desc:
                st.markdown("---")
                st.subheader("☁️ Job Description Word Cloud")
                try:
                    wordcloud = WordCloud(width=800, height=300, background_color='white', 
                                        colormap='viridis', random_state=42).generate(job_desc)
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.info("Could not generate word cloud")
            
            # Download button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(get_download_link(df), unsafe_allow_html=True)
                st.caption("Click above to download results as CSV")
            
        else:
            st.error("❌ No resumes could be processed. Please check file formats and content.")
            
    elif files and not job_desc:
        st.warning("⚠️ Please paste a job description first")
    elif job_desc and not files:
        st.info("📎 Please upload resume files to start screening")
    else:
        st.info("👈 **Get Started:** Paste a job description and upload resumes to begin screening")

if __name__ == "__main__":
    main()