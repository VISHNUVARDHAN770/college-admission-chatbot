import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Knowledge base manager for the College Admission Chatbot"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the knowledge base
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.knowledge_data = {}
        self.embeddings = {}
        self.texts = []
        self.metadata = []
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Load all data
        self._load_data()
        self._create_embeddings()
    
    def _load_data(self):
        """Load all data files from the data directory"""
        try:
            # Load college info
            college_info_path = self.data_dir / "college_info.json"
            if college_info_path.exists():
                with open(college_info_path, 'r', encoding='utf-8') as f:
                    self.knowledge_data['college_info'] = json.load(f)
            else:
                self.knowledge_data['college_info'] = self._get_default_college_info()
            
            # Load FAQs
            faqs_path = self.data_dir / "faqs.json"
            if faqs_path.exists():
                with open(faqs_path, 'r', encoding='utf-8') as f:
                    self.knowledge_data['faqs'] = json.load(f)
            else:
                self.knowledge_data['faqs'] = self._get_default_faqs()
            
            # Load programs
            programs_path = self.data_dir / "programs.json"
            if programs_path.exists():
                with open(programs_path, 'r', encoding='utf-8') as f:
                    self.knowledge_data['programs'] = json.load(f)
            else:
                self.knowledge_data['programs'] = self._get_default_programs()
            
            logger.info("Knowledge base data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._load_default_data()
    
    def _get_default_college_info(self):
        """Get default college information"""
        return {
            "general_info": {
                "name": "ABC University",
                "established": "1990",
                "location": "New York, USA",
                "type": "Public University"
            },
            "important_dates": {
                "fall_semester": {
                    "application_deadline": "March 1, 2024",
                    "semester_start": "August 15, 2024"
                },
                "spring_semester": {
                    "application_deadline": "October 1, 2024",
                    "semester_start": "January 15, 2025"
                }
            },
            "contact_info": {
                "email": "admissions@college.edu",
                "phone": "(555) 123-4567",
                "office_hours": "Monday-Friday, 9 AM - 5 PM"
            }
        }
    
    def _get_default_faqs(self):
        """Get default FAQs"""
        return {
            "faqs": [
                {
                    "question": "What are the admission requirements?",
                    "answer": "You need to submit completed application form, academic transcripts, entrance exam scores (SAT/ACT), letters of recommendation, and personal statement."
                },
                {
                    "question": "What is the application deadline?",
                    "answer": "Fall semester deadline is March 1st, Spring semester deadline is October 1st."
                },
                {
                    "question": "How much are the tuition fees?",
                    "answer": "Tuition fees vary by program. Undergraduate programs start from $15,000 per year. Contact our financial aid office for detailed information."
                },
                {
                    "question": "What programs do you offer?",
                    "answer": "We offer undergraduate and graduate programs in Computer Science, Business Administration, Engineering, Arts, and Sciences."
                },
                {
                    "question": "How can I contact the admission office?",
                    "answer": "You can reach us at admissions@college.edu or call (555) 123-4567. Our office hours are Monday-Friday, 9 AM - 5 PM."
                }
            ]
        }
    
    def _get_default_programs(self):
        """Get default programs"""
        return {
            "programs": [
                {
                    "name": "Computer Science",
                    "degree": "Bachelor's",
                    "duration": "4 years",
                    "description": "Comprehensive program covering programming, algorithms, data structures, and software engineering."
                },
                {
                    "name": "Business Administration",
                    "degree": "Bachelor's",
                    "duration": "4 years",
                    "description": "Business management, marketing, finance, and organizational behavior."
                },
                {
                    "name": "Engineering",
                    "degree": "Bachelor's",
                    "duration": "4 years",
                    "description": "Various engineering disciplines including mechanical, electrical, and civil engineering."
                }
            ]
        }
    
    def _load_default_data(self):
        """Load default data if files are missing"""
        self.knowledge_data = {
            'college_info': self._get_default_college_info(),
            'faqs': self._get_default_faqs(),
            'programs': self._get_default_programs()
        }
    
    def _create_embeddings(self):
        """Create embeddings for all text data"""
        if not self.sentence_model:
            logger.warning("Sentence model not available, using simple text matching")
            return
        
        try:
            # Collect all text for embedding
            for data_type, data in self.knowledge_data.items():
                if data_type == 'faqs':
                    for faq in data.get('faqs', []):
                        text = f"{faq['question']} {faq['answer']}"
                        self.texts.append(text)
                        self.metadata.append({
                            'type': 'faq',
                            'question': faq['question'],
                            'answer': faq['answer'],
                            'source': 'faqs'
                        })
                
                elif data_type == 'programs':
                    for program in data.get('programs', []):
                        text = f"{program['name']} {program['degree']} {program['description']}"
                        self.texts.append(text)
                        self.metadata.append({
                            'type': 'program',
                            'data': program,
                            'source': 'programs'
                        })
                
                elif data_type == 'college_info':
                    # Add general info
                    general_info = data.get('general_info', {})
                    text = f"{general_info.get('name', '')} {general_info.get('location', '')} {general_info.get('type', '')}"
                    self.texts.append(text)
                    self.metadata.append({
                        'type': 'general_info',
                        'data': general_info,
                        'source': 'college_info'
                    })
            
            # Create embeddings
            if self.texts:
                embeddings = self.sentence_model.encode(self.texts)
                self.embeddings = embeddings
                logger.info(f"Created embeddings for {len(self.texts)} text entries")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for similar content in the knowledge base
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar content with metadata
        """
        if not self.sentence_model or len(self.texts) == 0:
            return self._simple_text_search(query, top_k)
        
        try:
            # Encode the query
            query_embedding = self.sentence_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        'text': self.texts[idx],
                        'knowledge': self.metadata[idx],
                        'similarity': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self._simple_text_search(query, top_k)
    
    def _simple_text_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple text-based search fallback"""
        query_lower = query.lower()
        results = []
        
        # Search FAQs
        for faq in self.knowledge_data.get('faqs', {}).get('faqs', []):
            if any(word in faq['question'].lower() or word in faq['answer'].lower() 
                   for word in query_lower.split()):
                results.append({
                    'text': f"{faq['question']} {faq['answer']}",
                    'knowledge': {
                        'type': 'faq',
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'source': 'faqs'
                    },
                    'similarity': 0.8
                })
        
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge base statistics"""
        stats = {
            'total_entries': len(self.texts),
            'faqs': len(self.knowledge_data.get('faqs', {}).get('faqs', [])),
            'programs': len(self.knowledge_data.get('programs', {}).get('programs', [])),
            'categories': len(self.knowledge_data.keys())
        }
        return stats
    
    def get_all_faqs(self) -> List[Dict]:
        """Get all FAQs"""
        return self.knowledge_data.get('faqs', {}).get('faqs', [])
    
    def get_all_programs(self) -> List[Dict]:
        """Get all programs"""
        return self.knowledge_data.get('programs', {}).get('programs', [])
    
    def get_college_info(self) -> Dict:
        """Get college information"""
        return self.knowledge_data.get('college_info', {})
