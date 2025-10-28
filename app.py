import streamlit as st
import pandas as pd
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
    .confidence-bar {
        background: linear-gradient(90deg, #10B981 0%, #EF4444 100%);
        height: 5px;
        border-radius: 2px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_assistant():
    try:
        # Try to load the pre-trained assistant
        model_data = joblib.load('faq_model.joblib')
        
        # Create a simple assistant class for inference
        class SimpleAssistant:
            def __init__(self, model_data):
                self.vectorizer = model_data['vectorizer']
                self.tfidf_matrix = model_data['tfidf_matrix']
                self.df = pd.DataFrame(model_data['faq_data'])
            
            def preprocess_text(self, text):
                import re
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            def get_answer(self, user_question, similarity_threshold=0.2):
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                processed_question = self.preprocess_text(user_question)
                user_tfidf = self.vectorizer.transform([processed_question])
                similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
                best_match_idx = np.argmax(similarities)
                best_score = similarities[0, best_match_idx]
                
                best_question = self.df.iloc[best_match_idx]['question']
                best_answer = self.df.iloc[best_match_idx]['answer']
                
                if best_score >= similarity_threshold:
                    return best_answer, best_score, best_question
                else:
                    return "I'm sorry, I don't have a specific answer for that question. Please contact the university administration for more assistance.", best_score, best_question
        
        assistant = SimpleAssistant(model_data)
        st.success("✅ AI Assistant loaded successfully!")
        return assistant
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
        
        st.markdown("---")
        st.header("📊 System Info")
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write("Powered by: Scikit-learn + Streamlit")
    
    # Load assistant
    assistant = load_assistant()
    
    if assistant is None:
        st.error("Please make sure 'faq_model.joblib' is in the same directory.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
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
                    st.progress(message["confidence"])
                    st.caption(f"Confidence: {message['confidence']:.3f} | Matched: '{message['matched_question']}'")
    
    # Chat input
    if prompt := st.chat_input("Ask your university question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-user">👤 {prompt}</div>', unsafe_allow_html=True)
        
        # Get AI response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Simulate thinking time for better UX
                time.sleep(0.5)
                answer, confidence, matched_question = assistant.get_answer(prompt)
                
                # Display assistant response
                st.markdown(f'<div class="chat-assistant">🤖 {answer}</div>', unsafe_allow_html=True)
                
                # Show confidence and matching info (for debugging/transparency)
                if confidence < 1.0:
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.3f} | Matched: '{matched_question}'")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "confidence": confidence,
            "matched_question": matched_question
        })
    
    # FAQ Preview Section
    st.markdown("---")
    st.subheader("📋 Frequently Asked Questions")
    
    try:
        faq_data = assistant.df
        cols = st.columns(2)
        
        for i, row in faq_data.iterrows():
            with cols[i % 2]:
                with st.expander(f"❓ {row['question']}"):
                    st.write(f"**Answer:** {row['answer']}")
    except:
        st.info("FAQ preview not available")

if __name__ == "__main__":
    main()
