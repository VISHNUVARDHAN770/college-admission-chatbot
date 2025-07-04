import os
from pathlib import Path

class Config:
    """Configuration settings for the College Admission Chatbot"""
    
    # Page Configuration
    PAGE_TITLE = "College Admission Assistant"
    PAGE_ICON = "🎓"
    LAYOUT = "wide"
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CHATBOT_DIR = BASE_DIR / "chatbot"
    
    # Data Files
    COLLEGE_INFO_FILE = DATA_DIR / "college_info.json"
    FAQS_FILE = DATA_DIR / "faqs.json"
    PROGRAMS_FILE = DATA_DIR / "programs.json"
    
    # LLM Configuration
    MODEL_NAME = "microsoft/DialoGPT-medium"
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Knowledge Base Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.3
    MAX_CONTEXT_LENGTH = 1000
    
    # Response Configuration
    MAX_RESPONSE_LENGTH = 500
    MIN_RESPONSE_LENGTH = 20
    
    # Intent Categories
    INTENTS = {
        'admission_requirements': ['requirements', 'eligibility', 'criteria', 'qualify', 'admit'],
        'deadlines': ['deadline', 'last date', 'application date', 'closing date', 'when to apply'],
        'programs': ['courses', 'programs', 'departments', 'majors', 'degrees', 'studies'],
        'fees': ['fees', 'cost', 'tuition', 'expenses', 'payment', 'scholarship'],
        'contact': ['contact', 'phone', 'email', 'address', 'office', 'reach'],
        'general': ['hello', 'hi', 'help', 'about', 'information']
    }
    
    # Quick Response Templates
    QUICK_RESPONSES = {
        'admission_requirements': "For admission requirements, you typically need to submit your academic transcripts, entrance exam scores, and complete the application form. Specific requirements may vary by program.",
        'deadlines': "Application deadlines vary by program and semester. Please check our website or contact the admission office for specific dates.",
        'programs': "We offer a wide range of undergraduate and graduate programs across various disciplines. You can find detailed information about each program on our website.",
        'fees': "Tuition fees vary by program and level of study. Please contact our financial aid office for detailed fee structure and scholarship opportunities.",
        'contact': "You can reach our admission office at admissions@college.edu or call us at (555) 123-4567. Our office hours are Monday-Friday, 9 AM - 5 PM."
    }
    
    # Default Responses
    DEFAULT_GREETING = "Hello! 👋 I'm your College Admission Assistant. I can help you with admission requirements, deadlines, programs, fees, and more. What would you like to know?"
    
    FALLBACK_RESPONSE = "I understand you're asking about college admissions. While I don't have specific information about your question right now, I'd be happy to help you with information about admission requirements, deadlines, programs, fees, or contact details. Could you please rephrase your question or ask about one of these topics?"
    
    ERROR_RESPONSE = "I apologize, but I'm experiencing technical difficulties. Please try asking your question again, or contact our admission office directly for assistance."
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = BASE_DIR / "app.log"
