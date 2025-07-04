"""from .llm_handler import LLMHandler
from .knowledge_base import KnowledgeBase
from .response_handler import ResponseHandler

__version__ = "1.0.0"
__author__ = "College Admission Team"

__all__ = [
    "LLMHandler",
    "KnowledgeBase", 
    "ResponseHandler"
]

"""






"""
College Admission Chatbot Package

This package contains the core components for the College Admission Chatbot:
- LLMHandler: Handles language model interactions
- KnowledgeBase: Manages knowledge base operations and similarity search
- ResponseHandler: Processes and formats responses
"""

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