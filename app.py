import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
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
    .stSpinner > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# EMBED YOUR FAQ DATA DIRECTLY IN THE CODE
FAQ_DATA = {
    'question': [
        'What is the deadline for project submission?',
        'How many members in a project group?',
        'Where to submit internship evaluation?',
        'What are the library opening hours?',
        'How to contact computer science department?',
        'When is the course registration period?',
        'How to apply for a transcript?',
        'What is the minimum GPA requirement?',
        'Where can I find the academic calendar?',
        'How to reset my student portal password?',
        'Are there any programming workshops?',
        'What textbooks are required for CS courses?',
        'How to join student clubs?',
        'What are the exam rules and regulations?',
        'Can I take courses from other departments?',
        'Where is the student cafeteria located?',
        'How to apply for graduation?',
        'What is the tuition fee payment deadline?',
        'Are there scholarships available?',
        'How to book a study room in the library?',
        'What is the procedure for course withdrawal?',
        'Where can I find research paper resources?',
        'How to access online learning materials?',
        'What are the computer lab operating hours?',
        'How to report technical issues?'
    ],
    'answer': [
        'Project submissions are due by November 3, 2025 for Final Year Project I.',
        'Each group can have maximum 5 members for the Final Year Project.',
        'Submit your internship evaluation through your class representative by October 31, 2025.',
        'The library is open from 8:00 AM to 8:00 PM from Monday to Saturday.',
        'You can visit the CS department office in Room 205, or email csdepartment@university.edu',
        'Course registration opens on November 10, 2025 and closes on November 20, 2025.',
        'You can apply for transcripts online through the student portal or visit the registrar\'s office.',
        'The minimum GPA requirement to graduate is 2.0 on a 4.0 scale.',
        'The academic calendar is available on the university website under "Academic Affairs".',
        'Visit the IT help desk in Room 101 or use the "Forgot Password" feature on the login page.',
        'Yes, the CS department holds programming workshops every Friday at 3 PM in Lab 3.',
        'Textbook lists are available on the department website and at the university bookstore.',
        'Visit the Student Affairs office to learn about available clubs and registration process.',
        'Exam rules are posted on the university website and announced before each exam period.',
        'Yes, with approval from both department advisors. Maximum 2 cross-department courses per semester.',
        'The student cafeteria is located in the Student Center building, ground floor.',
        'Graduation applications must be submitted through the student portal by March 15, 2026.',
        'Tuition fees for the semester are due by January 30, 2026.',
        'Yes, various scholarships are available. Check the Financial Aid office for details.',
        'Study rooms can be booked online through the library website or at the front desk.',
        'Course withdrawal requires advisor approval and must be done before the mid-term deadline.',
        'Research papers and journals are available through the library digital database.',
        'Online materials can be accessed through the university LMS (Learning Management System).',
        'Computer labs are open from 7:00 AM to 10:00 PM daily, including weekends.',
        'Technical issues can be reported to the IT help desk in Room 101 or via email at it-support@university.edu'
    ]
}

class UniversityFAQAssistant:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self):
        """Load FAQ data from embedded data"""
        try:
            self.df = pd.DataFrame(FAQ_DATA)
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
            
        with st.spinner("🤖 Training AI model... This may take a few seconds."):
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
        if assistant.train_model():
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
        - *"Computer lab hours"*
        - *"Scholarship information"*
        """)
        
        st.markdown("---")
        st.header("📊 System Info")
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write("Powered by: Scikit-learn + Streamlit")
    
    # Initialize assistant
    assistant = initialize_assistant()
    
    if assistant is None:
        st.error("Failed to initialize the AI Assistant. Please refresh the page.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your University AI Assistant. I can help with academic queries, deadlines, campus facilities, and more! How can I assist you today? 🎓",
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
    
    # FAQ Preview Section
    st.markdown("---")
    st.subheader("📋 Frequently Asked Questions")
    
    # Display some FAQs from the embedded data
    faq_df = pd.DataFrame(FAQ_DATA)
    cols = st.columns(2)
    
    for i, row in faq_df.head(10).iterrows():  # Show first 10 FAQs
        with cols[i % 2]:
            with st.expander(f"❓ {row['question']}"):
                st.write(f"**Answer:** {row['answer']}")

if __name__ == "__main__":
    main()
