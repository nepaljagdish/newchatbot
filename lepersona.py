import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 0. Page config
# -----------------------------
st.set_page_config(page_title="AI Learning Concierge", layout="wide")

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
# 2. Synthetic Intelligence Engine
# -----------------------------
class LearningIntelligence:
    def __init__(self, df):
        self.df = df
        self.df['semantic_soup'] = self.df['title'] + " " + self.df['type'] + " " + self.df['tags']
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['semantic_soup'])

    def find_recommendations(self, query, top_n=2):
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices of top matches
        related_indices = cosine_sim.argsort()[:-top_n-1:-1]
        
        results = []
        for idx in related_indices:
            score = cosine_sim[idx]
            if score > 0.1: # Only return relevant matches
                results.append((self.df.iloc[idx], score))
        return results

ai_engine = LearningIntelligence(modules_df)

# -----------------------------
# 3. Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your AI Course Concierge. Tell me what you want to learn today (e.g., 'security', 'writing skills') and I'll find the best module for you."}
    ]

# -----------------------------
# 4. Sidebar (User Profile)
# -----------------------------
st.sidebar.header("👤 User Profile")
selected_user_name = st.sidebar.selectbox("Select User:", learners_df["name"])
user_row = learners_df[learners_df["name"] == selected_user_name].iloc[0]
st.sidebar.write(f"**Persona:** {user_row['persona']}")
st.sidebar.write(f"**Format:** {user_row['dominant_format']}")
st.sidebar.write(f"**Skill Level:** {user_row['avg_score']}/100")

# -----------------------------
# 5. Main Layout (Dashboard)
# -----------------------------
st.title("🎓 Enterprise Learning Portal")

# Top Section: Static Recommendations (The "Netflix" view)
st.subheader(f"Recommended for {user_row['name'].split('–')[0].strip()}")

# Simple logic for static display (filter by preferred format)
rec_df = modules_df[modules_df['type'] == user_row['dominant_format']].head(3)
cols = st.columns(3)

for index, (col, row) in enumerate(zip(cols, rec_df.iterrows())):
    module = row[1]
    with col:
        st.info(f"**{module['title']}**")
        st.caption(f"⏱ {module['duration_min']} mins | ⭐ {module['rating']}")
        st.button(f"Start Module {module['module_id']}", key=f"btn_{index}")

st.markdown("---")
st.markdown("### 💬 Ask the Course Concierge")
st.caption("Scroll down to chat. The AI is pinned to the bottom.")

# -----------------------------
# 6. Chatbot Interface (Pinned to Bottom)
# -----------------------------

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input (This pins to the bottom of the screen automatically)
if prompt := st.chat_input("What do you want to learn?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI Logic
    recommendations = ai_engine.find_recommendations(prompt)
    
    response_text = ""
    if recommendations:
        top_rec, score = recommendations[0]
        response_text = f"I found a great match for **'{prompt}'**:\n\n" \
                        f"### 🚀 **{top_rec['title']}**\n" \
                        f"- **Type:** {top_rec['type']}\n" \
                        f"- **Match Confidence:** {score*100:.0f}%\n\n" \
                        f"Would you like to start this now?"
        
        # If we have a second good match, mention it
        if len(recommendations) > 1:
            sec_rec, sec_score = recommendations[1]
            response_text += f"\n\n*Alternatively, you might like '{sec_rec['title']}' ({sec_rec['type']}).*"
    else:
        response_text = "I couldn't find a specific course matching that topic. " \
                        "Try searching for 'Security', 'Communication', or 'Compliance'."

    # 3. AI Response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
