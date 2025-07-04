import streamlit as st
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
try:
    from chatbot.llm_handler import LLMHandler
    from chatbot.knowledge_base import KnowledgeBase  
    from chatbot.response_handler import ResponseHandler
    from config import Config
except ImportError:
    # Fallback if modules are in different structure
    try:
        from chatbot import LLMHandler, KnowledgeBase, ResponseHandler
        from config import Config
    except ImportError:
        st.error("Required modules not found. Please check your project structure.")
        st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
.sidebar-info {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.quick-question {
    background-color: #fff3e0;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    border: 1px solid #ffcc02;
}
.quick-question:hover {
    background-color: #ffe0b2;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! 👋 I'm your College Admission Assistant. I can help you with admission requirements, deadlines, programs, fees, and more. What would you like to know?"
        })
    
    if 'llm_handler' not in st.session_state:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            st.session_state.llm_handler = LLMHandler()
    
    if 'knowledge_base' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.knowledge_base = KnowledgeBase()
    
    if 'response_handler' not in st.session_state:
        st.session_state.response_handler = ResponseHandler()
    
    if 'show_faq' not in st.session_state:
        st.session_state.show_faq = False

# Load college information for display
@st.cache_data
def load_college_info():
    try:
        with open('data/college_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_faqs():
    try:
        with open('data/faqs.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"faqs": []}

def display_sidebar():
    """Display sidebar with college information and quick actions"""
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.header("🏫 College Information")
        
        college_info = load_college_info()
        if college_info:
            general_info = college_info.get('general_info', {})
            st.markdown(f"**Name:** {general_info.get('name', 'N/A')}")
            st.markdown(f"**Established:** {general_info.get('established', 'N/A')}")
            st.markdown(f"**Location:** {general_info.get('location', 'N/A')}")
            st.markdown(f"**Type:** {general_info.get('type', 'N/A')}")
            
            # Important dates
            important_dates = college_info.get('important_dates', {})
            if important_dates:
                st.subheader("📅 Important Dates")
                for semester, dates in important_dates.items():
                    st.markdown(f"**{semester.replace('_', ' ').title()}:**")
                    if isinstance(dates, dict):
                        for date_type, date_value in dates.items():
                            st.markdown(f"• {date_type.replace('_', ' ').title()}: {date_value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Knowledge base statistics
        if hasattr(st.session_state, 'knowledge_base'):
            st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
            st.header("📊 Knowledge Base Stats")
            stats = st.session_state.knowledge_base.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", stats['total_entries'])
                st.metric("FAQs", stats['faqs'])
            with col2:
                st.metric("Programs", stats['programs'])
                st.metric("Categories", stats['categories'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick questions
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.header("🚀 Quick Questions")
        
        quick_questions = [
            "What are the admission requirements?",
            "What programs do you offer?",
            "What are the application deadlines?",
            "How much are the tuition fees?",
            "How can I contact the admission office?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{hash(question)}", help="Click to ask this question"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Control buttons
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.header("🔧 Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat", help="Clear all messages"):
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "Hello! 👋 I'm your College Admission Assistant. How can I help you today?"
                }]
                st.rerun()
        
        with col2:
            if st.button("❓ Show FAQ", help="Display frequently asked questions"):
                st.session_state.show_faq = not st.session_state.show_faq
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_response(user_message):
    """Generate response using the improved pipeline"""
    try:
        # Detect intent
        intent = st.session_state.response_handler.detect_intent(user_message)
        
        # Search for relevant information
        relevant_info = st.session_state.knowledge_base.search_similar(user_message, top_k=3)
        
        # Handle specific intents with direct responses
        if intent in ['admission_requirements', 'deadlines', 'programs', 'contact', 'fees']:
            # Try to get a quick response first
            quick_response = st.session_state.llm_handler.get_quick_response(intent)
            if quick_response:
                return st.session_state.response_handler.format_response(quick_response, intent, relevant_info)
        
        # Generate response using LLM
        conversation_history = st.session_state.messages[-6:]  # Last 6 messages for context
        
        # Create a more specific prompt based on intent and relevant info
        enhanced_prompt = create_enhanced_prompt(user_message, intent, relevant_info)
        
        response = st.session_state.llm_handler.get_response(
            enhanced_prompt, 
            conversation_history, 
            relevant_info
        )
        
        # Format and enhance the response
        formatted_response = st.session_state.response_handler.format_response(
            response, intent, relevant_info
        )
        
        # Validate response quality
        if not st.session_state.response_handler.validate_response_quality(formatted_response):
            # Use fallback response
            formatted_response = st.session_state.response_handler.get_fallback_response(intent)
        
        return formatted_response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm experiencing technical difficulties. Please try asking your question again, or contact our admission office directly for assistance."

def create_enhanced_prompt(user_message, intent, relevant_info):
    """Create an enhanced prompt with context"""
    prompt = f"Question about college admissions: {user_message}\n\n"
    
    if relevant_info:
        prompt += "Relevant information:\n\n"
        for info in relevant_info[:2]:  # Use top 2 relevant pieces
            if info['knowledge']['type'] == 'faq':
                prompt += f"FAQ: {info['knowledge']['question']} - {info['knowledge']['answer']}\n"
            else:
                prompt += f"Info: {str(info['knowledge']['data'])}\n"
    
    prompt += f"\nPlease provide a helpful, friendly response about {intent if intent != 'general' else 'this topic'}."
    
    return prompt

def display_chat():
    """Display the chat interface"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 🎓 College Admission Assistant")
    st.markdown("### Get instant answers to your admission questions!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Success message
    st.success("✅ All components loaded successfully!")
    
    # Display chat messages
    st.markdown("## 💬 Chat with our Admission Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about admissions, programs, deadlines, fees, or anything else!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def display_faq():
    """Display FAQ section"""
    if st.session_state.show_faq:
        st.markdown("---")
        st.markdown("## ❓ Frequently Asked Questions")
        
        faqs_data = load_faqs()
        faqs = faqs_data.get('faqs', [])
        
        if faqs:
            for i, faq in enumerate(faqs):
                with st.expander(f"**Q{i+1}:** {faq['question']}"):
                    st.markdown(faq['answer'])
                    if st.button(f"💬 Ask about this", key=f"faq_ask_{i}"):
                        st.session_state.messages.append({"role": "user", "content": faq['question']})
                        st.rerun()
        else:
            st.info("No FAQs available. Please check the data/faqs.json file.")

def display_footer():
    """Display footer with contact information"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
            <h3>🎓 College Admission Chatbot</h3>
            <p><strong>Need more help?</strong> Our admission office is here to assist you!</p>
            <p>
                📧 <strong>Email:</strong> admissions@college.edu<br>
                📞 <strong>Phone:</strong> (555) 123-4567<br>
                🏢 <strong>Office Hours:</strong> Monday-Friday, 9 AM - 5 PM
            </p>
            <p><em>Powered by AI | Need immediate help? Call us at (555) 123-4567</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display sidebar
        display_sidebar()
        
        # Display main chat interface
        display_chat()
        
        # Display FAQ section if requested
        display_faq()
        
        # Display footer
        display_footer()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check that all required files are in place and try refreshing the page.")

if __name__ == "__main__":
    main()