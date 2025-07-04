'''import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = self.load_knowledge()
        self.embeddings = self.create_embeddings()
    
    def load_knowledge(self):
        """Load and combine all knowledge sources"""
        knowledge = []
        
        # Load FAQs
        with open('data/faqs.json', 'r') as f:
            faqs = json.load(f)['faqs']
            for faq in faqs:
                knowledge.append({
                    'text': faq['question'] + ' ' + faq['answer'],
                    'type': 'faq',
                    'category': faq['category'],
                    'question': faq['question'],
                    'answer': faq['answer']
                })
        
        # Load college info
        with open('data/college_info.json', 'r') as f:
            college_info = json.load(f)
            # Convert nested JSON to searchable text
            for key, value in college_info.items():
                knowledge.append({
                    'text': f"{key}: {json.dumps(value)}",
                    'type': 'college_info',
                    'category': key,
                    'data': value
                })
        
        return knowledge
    
    def create_embeddings(self):
        """Create embeddings for knowledge base"""
        texts = [item['text'] for item in self.knowledge_base]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def search_similar(self, query, top_k=3):
        """Find most similar knowledge base entries"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'knowledge': self.knowledge_base[idx],
                'similarity': similarities[idx]
            })
        
        return results

        '''
'''
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from config import Config

class KnowledgeBase:
    def __init__(self):
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(Config.EMBEDDINGS_MODEL, cache_folder=Config.CACHE_DIR)
        print("Sentence transformer loaded successfully!")
        
        self.knowledge_base = self.load_knowledge()
        self.embeddings = self.create_embeddings()
        
    def load_knowledge(self):
        """Load and combine all knowledge sources"""
        knowledge = []
        
        # Load FAQs
        try:
            with open('data/faqs.json', 'r', encoding='utf-8') as f:
                faqs_data = json.load(f)
                faqs = faqs_data.get('faqs', [])
                
            for faq in faqs:
                knowledge.append({
                    'text': faq['question'] + ' ' + faq['answer'],
                    'type': 'faq',
                    'category': faq.get('category', 'general'),
                    'question': faq['question'],
                    'answer': faq['answer']
                })
            print(f"FAQs loaded: {len([item for item in knowledge if item['type'] == 'faq'])}")
        except FileNotFoundError:
            print("Warning: faqs.json not found")
        
        # Load college info
        try:
            with open('data/college_info.json', 'r', encoding='utf-8') as f:
                college_info = json.load(f)
                
            # Convert nested JSON to searchable text
            for key, value in college_info.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        knowledge.append({
                            'text': f"{key} {sub_key}: {json.dumps(sub_value) if isinstance(sub_value, (dict, list)) else str(sub_value)}",
                            'type': 'college_info',
                            'category': key,
                            'subcategory': sub_key,
                            'data': sub_value
                        })
                else:
                    knowledge.append({
                        'text': f"{key}: {json.dumps(value) if isinstance(value, (dict, list)) else str(value)}",
                        'type': 'college_info',
                        'category': key,
                        'data': value
                    })
        except FileNotFoundError:
            print("Warning: college_info.json not found")
        
        # Load programs
        try:
            with open('data/programs.json', 'r', encoding='utf-8') as f:
                programs = json.load(f)
                
            for program_level, program_list in programs.items():
                if isinstance(program_list, list):
                    for program in program_list:
                        if isinstance(program, dict):
                            program_text = f"{program_level} program: {program.get('name', '')} - {program.get('duration', '')} - {program.get('career_prospects', '')}"
                            knowledge.append({
                                'text': program_text,
                                'type': 'program',
                                'category': program_level,
                                'data': program
                            })
        except FileNotFoundError:
            print("Warning: programs.json not found")
        
        print(f"Loaded {len(knowledge)} knowledge base entries")
        return knowledge  
    
    def create_embeddings(self):
        """Create embeddings for knowledge base"""
        if not self.knowledge_base:
            return np.array([])
            
        print("Creating embeddings for knowledge base...")
        texts = [item['text'] for item in self.knowledge_base]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Embeddings created successfully!")
        return embeddings
    
    def search_similar(self, query, top_k=None):
        """Find most similar knowledge base entries"""
        if top_k is None:
            top_k = Config.TOP_K_SIMILAR
            
        if len(self.knowledge_base) == 0 or len(self.embeddings) == 0:
            return []
            
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Filter by similarity threshold
            valid_indices = np.where(similarities >= Config.SIMILARITY_THRESHOLD)[0]
            
            if len(valid_indices) == 0:
                top_idx = int(np.argmax(similarities))
                return [{
                    'knowledge': self.knowledge_base[top_idx],
                    'similarity': float(similarities[top_idx])
                }]
            
            # Get top-k most similar indices from valid ones
            valid_similarities = similarities[valid_indices]
            top_valid_indices = np.argsort(valid_similarities)[-top_k:][::-1]
            top_indices = valid_indices[top_valid_indices]
            
            results = []
            for idx in top_indices:
                results.append({
                    'knowledge': self.knowledge_base[idx],
                    'similarity': float(similarities[idx])
                })
            
            return results
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def get_category_info(self, category):
        """Get all information for a specific category"""
        results = []
        for item in self.knowledge_base:
            if item.get('category', '').lower() == category.lower():
                results.append(item)
        return results
    
    def get_faq_by_category(self, category=None):
        """Get FAQs, optionally filtered by category"""
        faqs = []
        for item in self.knowledge_base:
            if item['type'] == 'faq':
                if category is None or item.get('category', '').lower() == category.lower():
                    faqs.append(item)
        return faqs
    
    def get_programs_by_level(self, level=None):
        """Get programs, optionally filtered by level (undergraduate/graduate)"""
        programs = []
        for item in self.knowledge_base:
            if item['type'] == 'program':
                if level is None or item.get('category', '').lower() == level.lower():
                    programs.append(item)
        return programs
    
    def search_by_keywords(self, keywords):
        """Search knowledge base by keywords"""
        results = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for item in self.knowledge_base:
            text_lower = item['text'].lower()
            score = sum(1 for kw in keywords_lower if kw in text_lower)
            
            if score > 0:
                results.append({
                    'knowledge': item,
                    'keyword_score': score
                })
        
        # Sort by keyword score
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:Config.TOP_K_SIMILAR]
    
    def get_all_categories(self):
        """Get all available categories"""
        categories = set()
        for item in self.knowledge_base:
            if 'category' in item:
                categories.add(item['category'])
        return list(categories)
    
    def get_statistics(self):
        """Get knowledge base statistics"""
        stats = {
            'total_entries': len(self.knowledge_base),
            'faqs': len([item for item in self.knowledge_base if item['type'] == 'faq']),
            'college_info': len([item for item in self.knowledge_base if item['type'] == 'college_info']),
            'programs': len([item for item in self.knowledge_base if item['type'] == 'program']),
            'categories': len(self.get_all_categories())
        }
        return stats    '''


'''
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

class KnowledgeBase:
    """Handles knowledge base operations for college admission information"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.knowledge_data = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load all knowledge base data from JSON files"""
        try:
            # Load college information
            college_info = self._load_json_file(self.config.COLLEGE_INFO_FILE)
            if college_info:
                self._process_college_info(college_info)
            
            # Load FAQs
            faqs = self._load_json_file(self.config.FAQS_FILE)
            if faqs:
                self._process_faqs(faqs)
            
            # Load programs
            programs = self._load_json_file(self.config.PROGRAMS_FILE)
            if programs:
                self._process_programs(programs)
            
            self.logger.info(f"Loaded {len(self.knowledge_data)} knowledge entries")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
            # Create default knowledge base if files don't exist
            self._create_default_knowledge_base()
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load JSON data from file"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def _process_college_info(self, college_info: Dict):
        """Process college information and add to knowledge base"""
        # Process general information
        general_info = college_info.get('general_info', {})
        if general_info:
            self.knowledge_data.append({
                'type': 'general_info',
                'category': 'college_info',
                'keywords': ['about', 'college', 'university', 'information', 'general'],
                'data': general_info
            })
        
        # Process important dates
        important_dates = college_info.get('important_dates', {})
        if important_dates:
            self.knowledge_data.append({
                'type': 'dates',
                'category': 'deadlines',
                'keywords': ['dates', 'deadlines', 'important', 'schedule'],
                'data': important_dates
            })
        
        # Process admission requirements
        admission_req = college_info.get('admission_requirements', {})
        if admission_req:
            self.knowledge_data.append({
                'type': 'admission_requirements',
                'category': 'requirements',
                'keywords': ['admission', 'requirements', 'eligibility', 'criteria'],
                'data': admission_req
            })
        
        # Process contact information
        contact_info = college_info.get('contact_info', {})
        if contact_info:
            self.knowledge_data.append({
                'type': 'contact',
                'category': 'contact',
                'keywords': ['contact', 'phone', 'email', 'address', 'office'],
                'data': contact_info
            })
    
    def _process_faqs(self, faqs_data: Dict):
        """Process FAQ data and add to knowledge base"""
        faqs = faqs_data.get('faqs', [])
        for faq in faqs:
            # Extract keywords from question and answer
            keywords = self._extract_keywords(faq['question'] + ' ' + faq['answer'])
            
            self.knowledge_data.append({
                'type': 'faq',
                'category': 'faq',
                'keywords': keywords,
                'question': faq['question'],
                'answer': faq['answer'],
                'data': faq
            })
    
    def _process_programs(self, programs_data: Dict):
        """Process program data and add to knowledge base"""
        programs = programs_data.get('programs', [])
        for program in programs:
            # Extract keywords from program information
            keywords = self._extract_keywords(
                program.get('name', '') + ' ' + 
                program.get('description', '') + ' ' + 
                program.get('degree_type', '')
            )
            keywords.extend(['program', 'course', 'degree', 'study'])
            
            self.knowledge_data.append({
                'type': 'program',
                'category': 'programs',
                'keywords': keywords,
                'data': program
            })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words and filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = [word for word in cleaned_text.split() if word not in stop_words and len(word) > 2]
        return list(set(words))
    
    def _create_default_knowledge_base(self):
        """Create a default knowledge base with sample data"""
        self.knowledge_data = [
            {
                'type': 'faq',
                'category': 'admission_requirements',
                'keywords': ['admission', 'requirements', 'eligibility', 'criteria', 'apply'],
                'question': 'What are the admission requirements?',
                'answer': 'General admission requirements include high school diploma, minimum GPA of 3.0, standardized test scores, letters of recommendation, and completed application form.',
                'data': {
                    'question': 'What are the admission requirements?',
                    'answer': 'General admission requirements include high school diploma, minimum GPA of 3.0, standardized test scores, letters of recommendation, and completed application form.'
                }
            },
            {
                'type': 'faq',
                'category': 'deadlines',
                'keywords': ['deadline', 'date', 'application', 'submit', 'when'],
                'question': 'What are the application deadlines?',
                'answer': 'Application deadlines vary by semester: Fall semester - January 15, Spring semester - October 1, Summer session - March 15.',
                'data': {
                    'question': 'What are the application deadlines?',
                    'answer': 'Application deadlines vary by semester: Fall semester - January 15, Spring semester - October 1, Summer session - March 15.'
                }
            },
            {
                'type': 'faq',
                'category': 'programs',
                'keywords': ['programs', 'courses', 'majors', 'degrees', 'study'],
                'question': 'What programs do you offer?',
                'answer': 'We offer undergraduate and graduate programs in Business, Computer Science, Engineering, Liberal Arts, Sciences, Education, Nursing, and Psychology.',
                'data': {
                    'question': 'What programs do you offer?',
                    'answer': 'We offer undergraduate and graduate programs in Business, Computer Science, Engineering, Liberal Arts, Sciences, Education, Nursing, and Psychology.'
                }
            },
            {
                'type': 'faq',
                'category': 'fees',
                'keywords': ['fees', 'tuition', 'cost', 'payment', 'scholarship'],
                'question': 'How much are the tuition fees?',
                'answer': 'Tuition fees vary by program and residency status. Undergraduate in-state: $12,000-$15,000, out-of-state: $18,000-$25,000. Graduate programs range from $15,000-$35,000.',
                'data': {
                    'question': 'How much are the tuition fees?',
                    'answer': 'Tuition fees vary by program and residency status. Undergraduate in-state: $12,000-$15,000, out-of-state: $18,000-$25,000. Graduate programs range from $15,000-$35,000.'
                }
            },
            {
                'type': 'faq',
                'category': 'contact',
                'keywords': ['contact', 'phone', 'email', 'address', 'office', 'reach'],
                'question': 'How can I contact the admission office?',
                'answer': 'You can contact our admission office at admissions@college.edu or call (555) 123-4567. Office hours are Monday-Friday, 9 AM - 5 PM.',
                'data': {
                    'question': 'How can I contact the admission office?',
                    'answer': 'You can contact our admission office at admissions@college.edu or call (555) 123-4567. Office hours are Monday-Friday, 9 AM - 5 PM.'
                }
            }
        ]
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar content using keyword matching"""
        try:
            query_keywords = self._extract_keywords(query.lower())
            
            if not query_keywords:
                return []
            
            # Score each knowledge entry based on keyword overlap
            scored_entries = []
            
            for entry in self.knowledge_data:
                score = self._calculate_similarity_score(query_keywords, entry['keywords'])
                
                if score > 0:
                    scored_entries.append({
                        'knowledge': entry,
                        'score': score
                    })
            
            # Sort by score and return top k
            scored_entries.sort(key=lambda x: x['score'], reverse=True)
            return scored_entries[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in search_similar: {str(e)}")
            return []
    
    def _calculate_similarity_score(self, query_keywords: List[str], entry_keywords: List[str]) -> float:
        """Calculate similarity score between query and entry keywords"""
        if not query_keywords or not entry_keywords:
            return 0.0
        
        # Count matching keywords
        matches = len(set(query_keywords) & set(entry_keywords))
        
        # Calculate Jaccard similarity
        union_size = len(set(query_keywords) | set(entry_keywords))
        jaccard_score = matches / union_size if union_size > 0 else 0.0
        
        # Boost score for exact matches
        exact_matches = sum(1 for qk in query_keywords if qk in entry_keywords)
        exact_match_boost = exact_matches * 0.2
        
        return jaccard_score + exact_match_boost
    
    def get_by_category(self, category: str) -> List[Dict]:
        """Get all knowledge entries by category"""
        return [entry for entry in self.knowledge_data if entry.get('category') == category]
    
    def get_by_type(self, entry_type: str) -> List[Dict]:
        """Get all knowledge entries by type"""
        return [entry for entry in self.knowledge_data if entry.get('type') == entry_type]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_entries': len(self.knowledge_data),
            'faqs': len([e for e in self.knowledge_data if e.get('type') == 'faq']),
            'programs': len([e for e in self.knowledge_data if e.get('type') == 'program']),
            'categories': len(set(e.get('category', 'unknown') for e in self.knowledge_data))
        }
        return stats
    
    def add_knowledge_entry(self, entry: Dict):
        """Add a new knowledge entry"""
        self.knowledge_data.append(entry)
        self.logger.info(f"Added new knowledge entry: {entry.get('type', 'unknown')}")
    
    def update_knowledge_entry(self, entry_id: str, updated_entry: Dict):
        """Update an existing knowledge entry"""
        # Implementation depends on how you want to identify entries
        # For now, this is a placeholder
        pass
    
    def delete_knowledge_entry(self, entry_id: str):
        """Delete a knowledge entry"""
        # Implementation depends on how you want to identify entries
        # For now, this is a placeholder
        pass   


import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

class KnowledgeBase:
    """Knowledge base for college admission information"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.knowledge_data = []
        self.embeddings = None
        self.model = None
        
        # Initialize and load data
        self._load_embedding_model()
        self._load_knowledge_base()
        self._create_embeddings()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to simple text matching
            self.model = None
    
    def _load_knowledge_base(self):
        """Load all knowledge base files"""
        try:
            # Load college info
            self._load_college_info()
            
            # Load FAQs
            self._load_faqs()
            
            # Load programs
            self._load_programs()
            
            self.logger.info(f"Knowledge base loaded with {len(self.knowledge_data)} entries")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
            # Create fallback knowledge base
            self._create_fallback_knowledge_base()
    
    def _load_college_info(self):
        """Load college information"""
        try:
            if self.config.COLLEGE_INFO_FILE.exists():
                with open(self.config.COLLEGE_INFO_FILE, 'r', encoding='utf-8') as f:
                    college_info = json.load(f)
                    
                # Add general info
                if 'general_info' in college_info:
                    self.knowledge_data.append({
                        'type': 'general_info',
                        'data': college_info['general_info'],
                        'text': f"College information: {json.dumps(college_info['general_info'])}"
                    })
                
                # Add important dates
                if 'important_dates' in college_info:
                    self.knowledge_data.append({
                        'type': 'important_dates',
                        'data': college_info['important_dates'],
                        'text': f"Important dates: {json.dumps(college_info['important_dates'])}"
                    })
                    
        except Exception as e:
            self.logger.error(f"Error loading college info: {str(e)}")
    
    def _load_faqs(self):
        """Load FAQs"""
        try:
            if self.config.FAQS_FILE.exists():
                with open(self.config.FAQS_FILE, 'r', encoding='utf-8') as f:
                    faqs_data = json.load(f)
                    
                for faq in faqs_data.get('faqs', []):
                    self.knowledge_data.append({
                        'type': 'faq',
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'text': f"Question: {faq['question']} Answer: {faq['answer']}"
                    })
                    
        except Exception as e:
            self.logger.error(f"Error loading FAQs: {str(e)}")
    
    def _load_programs(self):
        """Load programs information"""
        try:
            if self.config.PROGRAMS_FILE.exists():
                with open(self.config.PROGRAMS_FILE, 'r', encoding='utf-8') as f:
                    programs_data = json.load(f)
                    
                for program in programs_data.get('programs', []):
                    self.knowledge_data.append({
                        'type': 'program',
                        'data': program,
                        'text': f"Program: {program.get('name', '')} {program.get('description', '')}"
                    })
                    
        except Exception as e:
            self.logger.error(f"Error loading programs: {str(e)}")
    
    def _create_fallback_knowledge_base(self):
        """Create fallback knowledge base if files don't exist"""
        fallback_data = [
            {
                'type': 'faq',
                'question': 'What are the admission requirements?',
                'answer': 'Admission requirements typically include high school diploma, minimum GPA, entrance exam scores, and application form.',
                'text': 'Question: What are the admission requirements? Answer: Admission requirements typically include high school diploma, minimum GPA, entrance exam scores, and application form.'
            },
            {
                'type': 'faq',
                'question': 'What are the application deadlines?',
                'answer': 'Application deadlines vary by program. Fall semester deadline is typically January 15, and Spring semester deadline is October 1.',
                'text': 'Question: What are the application deadlines? Answer: Application deadlines vary by program. Fall semester deadline is typically January 15, and Spring semester deadline is October 1.'
            },
            {
                'type': 'faq',
                'question': 'What programs do you offer?',
                'answer': 'We offer undergraduate and graduate programs in Business, Engineering, Computer Science, Liberal Arts, and more.',
                'text': 'Question: What programs do you offer? Answer: We offer undergraduate and graduate programs in Business, Engineering, Computer Science, Liberal Arts, and more.'
            },
            {
                'type': 'faq',
                'question': 'What are the tuition fees?',
                'answer': 'Tuition fees vary by program. Undergraduate fees range from $12,000-$25,000 per year, and graduate fees range from $15,000-$30,000 per year.',
                'text': 'Question: What are the tuition fees? Answer: Tuition fees vary by program. Undergraduate fees range from $12,000-$25,000 per year, and graduate fees range from $15,000-$30,000 per year.'
            },
            {
                'type': 'faq',
                'question': 'How can I contact the admission office?',
                'answer': 'You can contact us at admissions@college.edu or call (555) 123-4567. Office hours are Monday-Friday, 9 AM - 5 PM.',
                'text': 'Question: How can I contact the admission office? Answer: You can contact us at admissions@college.edu or call (555) 123-4567. Office hours are Monday-Friday, 9 AM - 5 PM.'
            }
        ]
        
        self.knowledge_data.extend(fallback_data)
        self.logger.info("Fallback knowledge base created")
    
    def _create_embeddings(self):
        """Create embeddings for all knowledge base entries"""
        if not self.model or not self.knowledge_data:
            return
            
        try:
            texts = [entry['text'] for entry in self.knowledge_data]
            self.embeddings = self.model.encode(texts)
            self.logger.info(f"Created embeddings for {len(texts)} entries")
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {str(e)}")
            self.embeddings = None
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar entries in the knowledge base"""
        if not self.knowledge_data:
            return []
        
        # If embeddings are available, use semantic search
        if self.model and self.embeddings is not None:
            return self._semantic_search(query, top_k)
        else:
            # Fall back to keyword search
            return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic search using embeddings"""
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k similar entries
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > self.config.SIMILARITY_THRESHOLD:
                    results.append({
                        'knowledge': self.knowledge_data[idx],
                        'similarity': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform keyword-based search"""
        query_lower = query.lower()
        results = []
        
        for entry in self.knowledge_data:
            score = 0
            text_lower = entry['text'].lower()
            
            # Simple keyword matching
            query_words = re.findall(r'\b\w+\b', query_lower)
            for word in query_words:
                if word in text_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    'knowledge': entry,
                    'similarity': score / len(query_words)
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge base statistics"""
        stats = {
            'total_entries': len(self.knowledge_data),
            'faqs': len([entry for entry in self.knowledge_data if entry['type'] == 'faq']),
            'programs': len([entry for entry in self.knowledge_data if entry['type'] == 'program']),
            'categories': len(set(entry['type'] for entry in self.knowledge_data))
        }
        return stats
    
    def get_entry_by_type(self, entry_type: str) -> List[Dict]:
        """Get all entries of a specific type"""
        return [entry for entry in self.knowledge_data if entry['type'] == entry_type]
    
    def add_entry(self, entry: Dict):
        """Add a new entry to the knowledge base"""
        self.knowledge_data.append(entry)
        
        # Recreate embeddings if model is available
        if self.model:
            self._create_embeddings()
    
    def update_entry(self, index: int, entry: Dict):
        """Update an existing entry"""
        if 0 <= index < len(self.knowledge_data):
            self.knowledge_data[index] = entry
            
            # Recreate embeddings if model is available
            if self.model:
                self._create_embeddings()
    
    def delete_entry(self, index: int):
        """Delete an entry from the knowledge base"""
        if 0 <= index < len(self.knowledge_data):
            del self.knowledge_data[index]
            
            # Recreate embeddings if model is available
            if self.model:
                self._create_embeddings()
    
    def search_by_intent(self, intent: str) -> List[Dict]:
        """Search for entries matching a specific intent"""
        intent_keywords = self.config.INTENTS.get(intent, [])
        results = []
        
        for entry in self.knowledge_data:
            text_lower = entry['text'].lower()
            score = 0
            
            for keyword in intent_keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    'knowledge': entry,
                    'similarity': score / len(intent_keywords)
                })
        
        # Sort by score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:3]         


        '''



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