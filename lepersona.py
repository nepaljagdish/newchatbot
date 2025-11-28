import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# -----------------------------
# 0. Configuration
# -----------------------------
st.set_page_config(page_title="AI Learning Concierge (Gemini)", layout="wide")

# Configure Google Gemini Key
# NOTE: In production, use st.secrets["GOOGLE_API_KEY"] instead of hardcoding!
API_KEY = "AIzaSyDPh4ttMMdcFR9NCmhMzvJxVSG9LZxw6vw" 
genai.configure(api_key=API_KEY)

# -----------------------------
# 1. Data Initialization
# -----------------------------
learners = [
    {"user_id": "E0001", "name": "Passive Explorer – A", "persona": "Passive Explorer", "cluster": 0, "dominant_format": "Video", "avg_score": 68, "description": "Starts many modules but completes few."},
    {"user_id": "E0002", "name": "Structured Performer – B", "persona": "Structured Performer", "cluster": 1, "dominant_format": "Quiz", "avg_score": 78, "description": "Engages consistently and performs well."},
    {"user_id": "E0003", "name": "Deep Diver – C", "persona": "Deep Diver", "cluster": 2, "dominant_format": "Video", "avg_score": 88, "description": "Highly engaged, prefers depth."},
]
learners_df = pd.DataFrame(learners)

modules = [
    ["M1",  "Workplace Communication Basics",       "Video", 4.4, 10, "Communication soft skills talk"],
    ["M5",  "Security Awareness Deep Dive",         "Video", 4.7, 20, "Security cyber protection"],
    ["M6",  "Cybersecurity Essentials Quiz",        "Quiz",  4.6, 8,  "Security cyber hacking defense"],
    ["M7",  "Conflict Resolution Scenarios",        "Video", 4.2, 12, "Communication arguments hr"],
    ["M10", "Effective Email Writing",              "Video", 4.5, 15, "Communication writing text"],
    ["M11", "Policy Overview Article",              "Article",4.3, 10, "Policy compliance rules legal"],
    ["M12", "Intermediate Security Quiz",           "Quiz",  4.4, 15, "Security network passwords"],
    ["M13", "Phishing Detection Basics",            "Quiz",  4.1, 10, "Security email scam hack"],
    ["M14", "Secure Authentication Webinar",        "Webinar",4.5, 30, "Security mfa login access"],
    ["M22", "Understanding Data Privacy Policies",  "Article",4.3, 12, "Policy gdpr data legal"],
]

modules_df = pd.DataFrame(modules, columns=["module_id", "title", "type", "rating", "duration_min", "tags"])

# -----------------------------
# 2. Retrieval Engine (The "R" in RAG)
# -----------------------------
class LearningIntelligence:
    def __init__(self, df):
        self.df = df
        self.df['semantic_soup'] = self.df['title'] + " " + self.df['type'] + " " + self.df['tags']
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['semantic_soup'])

    def retrieve_context(self, query, top_n=3):
        """
        Finds the most relevant courses to feed to the LLM.
        """
        query_lower = query.lower()
        
        # 1. Check for metadata filters (Rule-based layer)
        if "long" in query_lower:
            return self.df.sort_values("duration_min", ascending=False).head(top_n)
        elif "short" in query_lower or "quick" in query_lower:
            return self.df.sort_values("duration_min", ascending=True).head(top_n)
        elif "best" in query_lower or "top" in query_lower:
            return self.df.sort_values("rating", ascending=False).head(top_n)
        
        # 2. Semantic Search (Vector Search)
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = cosine_sim.argsort()[:-top_n-1:-1]
        
        # Return the actual dataframe rows
        return self.df.iloc[related_indices]

ai_engine = LearningIntelligence(modules_df)

# -----------------------------
# 3. Gemini LLM Generation (The "G" in RAG)
# -----------------------------
def generate_gemini_response(user_query, relevant_courses, user_profile):
    """
    Sends the user query + retrieved course data to Google Gemini.
    """
    # Initialize Model
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Construct the Context String from Dataframe
    context_str = ""
    for _, row in relevant_courses.iterrows():
        context_str += (f"- ID: {row['module_id']} | Title: {row['title']} | "
                        f"Type: {row['type']} | Duration: {row['duration_min']}m | "
                        f"Rating: {row['rating']}\n")

    # Construct the Full Prompt
    prompt = f"""
    You are an expert Learning Concierge for a corporate training portal.
    
    USER PROFILE:
    Name: {user_profile['name']}
    Preferred Format: {user_profile['dominant_format']}
    
    USER REQUEST: "{user_query}"
    
    AVAILABLE COURSES (Context):
    {context_str}
    
    INSTRUCTIONS:
    1. Recommend the best course from the list above.
    2. Explain WHY it fits the user's request and their profile (e.g. if they like videos, mention that).
    3. If the user asked for "long" or "short", highlight the duration.
    4. Keep the tone professional, encouraging, and concise.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error contacting Google Gemini: {str(e)}"

# -----------------------------
# 4. Interface (Sidebar & Dashboard)
# -----------------------------
st.sidebar.header("👤 User Profile")
selected_user_name = st.sidebar.selectbox("Select User:", learners_df["name"])
user_row = learners_df[learners_df["name"] == selected_user_name].iloc[0]

st.sidebar.markdown(f"""
**Persona:** {user_row['persona']}  
**Format:** {user_row['dominant_format']}  
**Skill Level:** {user_row['avg_score']}/100
""")
st.sidebar.info("System is running on **Google Gemini 1.5 Flash** (Free Tier)")

# Dashboard Top
st.title("🎓 Enterprise Learning Portal")
st.subheader(f"Dashboard for {user_row['name'].split('–')[0].strip()}")

# Static Recs (Visuals)
rec_df = modules_df[modules_df['type'] == user_row['dominant_format']].head(3)
cols = st.columns(3)
for index, (col, row) in enumerate(zip(cols, rec_df.iterrows())):
    module = row[1]
    with col:
        st.info(f"**{module['title']}**")
        st.caption(f"⏱ {module['duration_min']} mins | ⭐ {module['rating']}")
        if st.button(f"Start {module['module_id']}", key=f"btn_{index}"):
            st.success(f"Launched {module['title']}")

st.markdown("---")
st.markdown("### 💬 Ask the Course Concierge")
st.caption("Scroll down to chat. The AI is pinned to the bottom.")

# -----------------------------
# 5. Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm powered by Google Gemini. Ask me for a course recommendation (e.g., 'short video on security')."}]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("What do you want to learn?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Logic (Retrieve + Generate)
    with st.spinner("Gemini is thinking..."):
        # A. Search Database
        relevant_data = ai_engine.retrieve_context(prompt)
        
        # B. Ask LLM
        ai_response = generate_gemini_response(prompt, relevant_data, user_row)
    
    # 3. AI Response
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
