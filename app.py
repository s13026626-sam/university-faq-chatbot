import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-user {
        background-color: #EFF6FF;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 0;
        border: 1px solid #BFDBFE;
    }
    .chat-assistant {
        background-color: #F0FDF4;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 0;
        border: 1px solid #BBF7D0;
    }
</style>
""", unsafe_allow_html=True)

class UniversityFAQAssistant:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self):
        """Load FAQ data from CSV file"""
        try:
            self.df = pd.read_csv('university_faq_expanded.csv')
            st.success(f"✅ Loaded {len(self.df)} FAQs successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading FAQ data: {e}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text for better matching"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_model(self):
        """Train the TF-IDF model"""
        if self.df is None:
            st.error("❌ No data loaded. Please load data first.")
            return False
            
        with st.spinner("🤖 Training AI model..."):
            # Preprocess all questions
            processed_questions = [self.preprocess_text(q) for q in self.df['question']]
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
            
        st.success("✅ AI model trained successfully!")
        return True
    
    def get_answer(self, user_question, similarity_threshold=0.2):
        """Find the best matching answer for a user question"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return "Sorry, the AI model is not ready yet.", 0.0, "System"
        
        # Preprocess user question
        processed_question = self.preprocess_text(user_question)
        
        if not processed_question:
            return "Please ask a valid question.", 0.0, "System"
        
        # Convert user question to TF-IDF vector
        user_tfidf = self.vectorizer.transform([processed_question])
        
        # Calculate similarity with all FAQ questions
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]
        
        # Get the best matching question and answer
        best_question = self.df.iloc[best_match_idx]['question']
        best_answer = self.df.iloc[best_match_idx]['answer']
        
        # Return answer based on similarity threshold
        if best_score >= similarity_threshold:
            return best_answer, best_score, best_question
        else:
            return "I'm sorry, I don't have a specific answer for that question. Please contact the university administration for more assistance.", best_score, best_question

@st.cache_resource
def initialize_assistant():
    """Initialize and train the assistant"""
    assistant = UniversityFAQAssistant()
    if assistant.load_data():
        assistant.train_model()
        return assistant
    return None

def main():
    # Header section
    st.markdown('<h1 class="main-header">🎓 University AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Your 24/7 FAQ Helper</h2>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI Assistant can help you with:
        - 📚 Academic queries
        - ⏰ Deadlines & schedules  
        - 🏫 Campus facilities
        - 📋 Administrative processes
        - 🎯 Project guidelines
        """)
        
        st.markdown("---")
        st.header("💡 Sample Questions")
        st.markdown("""
        Try asking:
        - *"Project submission deadline"*
        - *"Library opening hours"*
        - *"How to get transcript"*
        - *"Course registration period"*
        - *"Student cafeteria location"*
        """)
    
    # Initialize assistant
    assistant = initialize_assistant()
    
    if assistant is None:
        st.error("Failed to initialize the AI Assistant. Please check if 'university_faq_expanded.csv' exists.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your University AI Assistant. How can I help you today? 🎓",
            "confidence": 1.0,
            "matched_question": "Welcome"
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="chat-assistant">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                if "confidence" in message and message["confidence"] < 1.0:
                    st.caption(f"Confidence: {message['confidence']:.3f}")
    
    # Chat input
    if prompt := st.chat_input("Ask your university question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-user">👤 {prompt}</div>', unsafe_allow_html=True)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                time.sleep(0.3)  # Small delay for better UX
                answer, confidence, matched_question = assistant.get_answer(prompt)
                
                # Display assistant response
                st.markdown(f'<div class="chat-assistant">🤖 {answer}</div>', unsafe_allow_html=True)
                
                # Show confidence
                if confidence < 1.0:
                    st.caption(f"Confidence: {confidence:.3f}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "confidence": confidence,
            "matched_question": matched_question
        })

if __name__ == "__main__":
    main()
