__version__ = "1.0.0"
__author__ = "College Admission Chatbot Team"
__email__ = "admissions@college.edu"

# Import main classes for easy access
from .llm_handler import LLMHandler
from .knowledge_base import KnowledgeBase
from .response_handler import ResponseHandler

__all__ = [
    'LLMHandler',
    'KnowledgeBase', 
    'ResponseHandler'
]
