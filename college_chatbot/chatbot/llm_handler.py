'''from openai import OpenAI
from config import Config
import json
import os

class LLMHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = Config.MODEL_NAME
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE
    
    def load_knowledge_base(self):
        """Load all knowledge base files"""
        knowledge = {}
        
        # Load college info
        with open('data/college_info.json', 'r') as f:
            knowledge['college_info'] = json.load(f)
        
        # Load FAQs
        with open('data/faqs.json', 'r') as f:
            knowledge['faqs'] = json.load(f)
        
        # Load programs
        with open('data/programs.json', 'r') as f:
            knowledge['programs'] = json.load(f)
        
        return knowledge
    
    def create_system_prompt(self, knowledge):
        """Create system prompt with knowledge base"""
        system_prompt = f"""
You are a helpful college admission assistant chatbot. Your role is to help prospective students with their admission questions.

Use the following knowledge base to answer questions:

COLLEGE INFORMATION:
{json.dumps(knowledge['college_info'], indent=2)}

FREQUENTLY ASKED QUESTIONS:
{json.dumps(knowledge['faqs'], indent=2)}

AVAILABLE PROGRAMS:
{json.dumps(knowledge['programs'], indent=2)}

Guidelines:
1. Be helpful, friendly, and professional
2. Provide accurate information based on the knowledge base
3. If you don't know something, admit it and suggest contacting the admission office
4. Ask follow-up questions to better understand student needs
5. Keep responses concise but informative
6. Always be encouraging and supportive
"""
        return system_prompt
    
    def get_response(self, user_message, conversation_history):
        """Generate response using OpenAI API"""
        try:
            knowledge = self.load_knowledge_base()
            system_prompt = self.create_system_prompt(knowledge)
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in conversation_history:
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # ✅ Call OpenAI using v1.0+ client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"ERROR: {str(e)}"    
        

            '''
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import Config
import json
import os
import warnings
warnings.filterwarnings("ignore")

class LLMHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model_name = Config.MODEL_NAME
        self.max_length = Config.MAX_LENGTH
        self.max_new_tokens = Config.MAX_NEW_TOKENS
        self.temperature = Config.TEMPERATURE
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=Config.CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=Config.CACHE_DIR,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            do_sample=Config.DO_SAMPLE,
            temperature=Config.TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def load_knowledge_base(self):
        """Load all knowledge base files"""
        knowledge = {}
        
        try:
            # Load college info
            with open('data/college_info.json', 'r', encoding='utf-8') as f:
                knowledge['college_info'] = json.load(f)
        except FileNotFoundError:
            knowledge['college_info'] = {}
            
        try:
            # Load FAQs
            with open('data/faqs.json', 'r', encoding='utf-8') as f:
                knowledge['faqs'] = json.load(f)
        except FileNotFoundError:
            knowledge['faqs'] = {'faqs': []}
            
        try:
            # Load programs
            with open('data/programs.json', 'r', encoding='utf-8') as f:
                knowledge['programs'] = json.load(f)
        except FileNotFoundError:
            knowledge['programs'] = {}
            
        return knowledge

    def create_context_prompt(self, knowledge, user_message, relevant_info=None):
        """Create context-aware prompt for the model"""
        
        # Basic college information
        college_info = knowledge.get('college_info', {}).get('general_info', {})
        college_name = college_info.get('name', 'Our College')
        
        # Create context
        context = f"""You are a helpful college admission assistant for {college_name}. 
        
Here's some key information:
- College: {college_name}
- Established: {college_info.get('established', 'N/A')}
- Location: {college_info.get('location', 'N/A')}

"""
        
        # Add relevant information from knowledge base if available
        if relevant_info:
            context += "Relevant Information:\n"
            for info in relevant_info:
                if info['knowledge']['type'] == 'faq':
                    context += f"Q: {info['knowledge']['question']}\nA: {info['knowledge']['answer']}\n\n"
                elif info['knowledge']['type'] == 'college_info':
                    context += f"{info['knowledge']['category']}: {str(info['knowledge']['data'])}\n\n"
        
        # Add guidelines
        context += """Guidelines:
- Be helpful, friendly, and professional
- Provide accurate information based on the college data
- If you don't know something, suggest contacting the admission office
- Keep responses concise but informative
- Be encouraging and supportive

Student Question: """
        
        return context + user_message

    def get_response(self, user_message, conversation_history=None, relevant_info=None):
        """Generate response using the transformer model"""
        try:
            knowledge = self.load_knowledge_base()
            
            # Create the input prompt
            input_prompt = self.create_context_prompt(knowledge, user_message, relevant_info)
            
            # Add conversation history if available
            if conversation_history:
                # Limit history to avoid token limit issues
                recent_history = conversation_history[-4:]  # Last 4 exchanges
                history_text = ""
                for msg in recent_history:
                    if msg["role"] == "user":
                        history_text += f"Student: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        history_text += f"Assistant: {msg['content']}\n"
                
                input_prompt = history_text + "\n" + input_prompt
            
            # Generate response
            response = self.generator(
                input_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract and clean the response
            generated_text = response[0]['generated_text'].strip()
            
            # Post-process the response
            generated_text = self.post_process_response(generated_text)
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try asking your question again, or contact our admission office directly for assistance."

    def post_process_response(self, response):
        """Clean and improve the generated response"""
        # Remove potential repetitions and clean up
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Student:') and not line.startswith('Assistant:'):
                cleaned_lines.append(line)
        
        cleaned_response = ' '.join(cleaned_lines)
        
        # Ensure reasonable length
        if len(cleaned_response) > 500:
            sentences = cleaned_response.split('.')
            cleaned_response = '. '.join(sentences[:3]) + '.'
        
        # Add friendly closing if response is too short
        if len(cleaned_response) < 50:
            cleaned_response += " Feel free to ask if you need more information!"
            
        return cleaned_response

    def get_quick_response(self, query_type):
        """Get quick responses for common queries"""
        knowledge = self.load_knowledge_base()
        
        quick_responses = {
            "admission_requirements": "Here are our admission requirements: ",
            "deadlines": "Important deadlines you should know: ",
            "programs": "We offer the following programs: ",
            "contact": "You can contact our admission office: "
        }
        
        if query_type in quick_responses:
            base_response = quick_responses[query_type]
            
            if query_type == "admission_requirements":
                req_info = knowledge.get('college_info', {}).get('admission_requirements', {})
                return base_response + str(req_info)
            elif query_type == "deadlines":
                dates_info = knowledge.get('college_info', {}).get('important_dates', {})
                return base_response + str(dates_info)
            elif query_type == "programs":
                programs_info = knowledge.get('programs', {})
                return base_response + str(programs_info)
            elif query_type == "contact":
                return base_response + "📧 admissions@college.edu | 📞 (555) 123-4567"
        
        return None
        '''


import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

class LLMHandler:
    """Handles LLM-based response generation for college admission queries"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.quick_responses = self.config.QUICK_RESPONSES
        
    def get_response(self, prompt: str, conversation_history: List[Dict], relevant_info: List[Dict]) -> str:
        """Generate a response using rule-based approach with context"""
        try:
            # Clean and prepare the prompt
            cleaned_prompt = self._clean_prompt(prompt)
            
            # Detect intent
            intent = self._detect_intent(cleaned_prompt)
            
            # Generate response based on intent and context
            if intent == 'greeting':
                return self._generate_greeting_response()
            elif intent in self.quick_responses:
                return self._generate_contextual_response(intent, cleaned_prompt, relevant_info)
            else:
                return self._generate_general_response(cleaned_prompt, relevant_info)
                
        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}")
            return self.config.ERROR_RESPONSE
    
    def get_quick_response(self, intent: str) -> Optional[str]:
        """Get a quick response for a specific intent"""
        return self.quick_responses.get(intent)
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize the input prompt"""
        # Remove extra spaces and normalize
        cleaned = re.sub(r'\s+', ' ', prompt.strip().lower())
        # Remove special characters except basic punctuation
        cleaned = re.sub(r'[^\w\s\?\!\.]', '', cleaned)
        return cleaned
    
    def _detect_intent(self, prompt: str) -> str:
        """Detect the intent of the user's message"""
        # Check for greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in prompt for pattern in greeting_patterns):
            return 'greeting'
        
        # Check for specific intents
        for intent, keywords in self.config.INTENTS.items():
            if any(keyword in prompt for keyword in keywords):
                return intent
        
        return 'general'
    
    def _generate_greeting_response(self) -> str:
        """Generate a personalized greeting response"""
        current_hour = datetime.now().hour
        
        if current_hour < 12:
            greeting = "Good morning! 🌅"
        elif current_hour < 17:
            greeting = "Good afternoon! ☀️"
        else:
            greeting = "Good evening! 🌙"
        
        return f"{greeting} I'm your College Admission Assistant. I can help you with admission requirements, program information, deadlines, fees, and more. What would you like to know today?"
    
    def _generate_contextual_response(self, intent: str, prompt: str, relevant_info: List[Dict]) -> str:
        """Generate a contextual response based on intent and relevant information"""
        base_response = self.quick_responses.get(intent, "")
        
        # Enhance response with relevant information
        if relevant_info:
            enhanced_response = self._enhance_with_context(base_response, relevant_info, intent)
            return enhanced_response
        
        # Add specific details based on intent
        if intent == 'admission_requirements':
            return self._generate_admission_requirements_response(prompt)
        elif intent == 'deadlines':
            return self._generate_deadlines_response(prompt)
        elif intent == 'programs':
            return self._generate_programs_response(prompt)
        elif intent == 'fees':
            return self._generate_fees_response(prompt)
        elif intent == 'contact':
            return self._generate_contact_response(prompt)
        
        return base_response
    
    def _generate_admission_requirements_response(self, prompt: str) -> str:
        """Generate detailed admission requirements response"""
        response = """📋 **Admission Requirements:**

**For Undergraduate Programs:**
• High school diploma or equivalent
• Minimum GPA of 3.0 (varies by program)
• Standardized test scores (SAT/ACT)
• Letters of recommendation (2-3)
• Personal statement/essay
• Application form and fee

**For Graduate Programs:**
• Bachelor's degree from accredited institution
• Minimum GPA of 3.0 in major field
• GRE/GMAT scores (program dependent)
• Letters of recommendation (3)
• Statement of purpose
• Resume/CV

**Additional Requirements:**
• Official transcripts
• English proficiency test (for international students)
• Portfolio (for specific programs)

Would you like more details about requirements for a specific program? 🎓"""
        
        return response
    
    def _generate_deadlines_response(self, prompt: str) -> str:
        """Generate deadlines response"""
        response = """📅 **Application Deadlines:**

**Fall Semester:**
• Early Decision: November 15
• Regular Decision: January 15
• International Students: December 1

**Spring Semester:**
• Regular Decision: October 1
• International Students: September 1

**Summer Session:**
• Regular Decision: March 15

**Graduate Programs:**
• Fall: February 1
• Spring: September 15

**Important Notes:**
• Some programs have earlier deadlines
• Rolling admissions for select programs
• All applications must be submitted by 11:59 PM on the deadline date

For program-specific deadlines, please contact our admission office! 📞"""
        
        return response
    
    def _generate_programs_response(self, prompt: str) -> str:
        """Generate programs response"""
        response = """🎓 **Academic Programs:**

**Undergraduate Programs:**
• Business Administration
• Computer Science
• Engineering
• Liberal Arts
• Sciences
• Education
• Nursing
• Psychology

**Graduate Programs:**
• Master's Programs (MA, MS, MBA)
• Doctoral Programs (PhD, EdD)
• Professional Programs
• Certificate Programs

**Popular Programs:**
• Business Administration (MBA)
• Computer Science (BS/MS)
• Engineering (Various specializations)
• Healthcare Programs
• Education

**Special Features:**
• Internship opportunities
• Research programs
• Study abroad options
• Online and hybrid programs

Would you like detailed information about a specific program? 📚"""
        
        return response
    
    def _generate_fees_response(self, prompt: str) -> str:
        """Generate fees response"""
        response = """💰 **Tuition and Fees:**

**Undergraduate (per year):**
• In-state: $12,000 - $15,000
• Out-of-state: $18,000 - $25,000
• International: $20,000 - $28,000

**Graduate (per year):**
• In-state: $15,000 - $20,000
• Out-of-state: $22,000 - $30,000
• International: $25,000 - $35,000

**Additional Costs:**
• Housing: $8,000 - $12,000
• Meal plans: $3,000 - $5,000
• Books & supplies: $1,200 - $1,800
• Personal expenses: $2,000 - $3,000

**Financial Aid:**
• Scholarships available
• Federal financial aid
• Work-study programs
• Payment plans available

Contact our Financial Aid Office for personalized assistance! 💳"""
        
        return response
    
    def _generate_contact_response(self, prompt: str) -> str:
        """Generate contact response"""
        response = """📞 **Contact Information:**

**Admission Office:**
• 📧 Email: admissions@college.edu
• 📞 Phone: (555) 123-4567
• 📠 Fax: (555) 123-4568

**Office Hours:**
• Monday - Friday: 9:00 AM - 5:00 PM
• Saturday: 10:00 AM - 2:00 PM (during peak season)
• Sunday: Closed

**Address:**
College Admission Office
123 University Drive
College Town, State 12345

**Other Contacts:**
• Financial Aid: (555) 123-4569
• International Students: (555) 123-4570
• Technical Support: (555) 123-4571

**Online:**
• Website: www.college.edu
• Virtual Tours: Available online
• Live Chat: Available during office hours

We're here to help! 🎓"""
        
        return response
    
    def _generate_general_response(self, prompt: str, relevant_info: List[Dict]) -> str:
        """Generate a general response with available information"""
        if relevant_info:
            response = "Based on your question, here's what I can tell you:\n\n"
            
            for info in relevant_info[:2]:  # Use top 2 relevant pieces
                if info['knowledge']['type'] == 'faq':
                    response += f"**Q:** {info['knowledge']['question']}\n"
                    response += f"**A:** {info['knowledge']['answer']}\n\n"
                else:
                    # Handle other types of information
                    data = info['knowledge']['data']
                    if isinstance(data, dict):
                        response += f"**Information:** {str(data)}\n\n"
            
            response += "Is there anything specific you'd like to know more about? 🤔"
            return response
        
        return self.config.FALLBACK_RESPONSE
    
    def _enhance_with_context(self, base_response: str, relevant_info: List[Dict], intent: str) -> str:
        """Enhance the base response with relevant context"""
        if not relevant_info:
            return base_response
        
        enhanced = base_response + "\n\n**Additional Information:**\n"
        
        for info in relevant_info[:2]:
            if info['knowledge']['type'] == 'faq':
                enhanced += f"• {info['knowledge']['answer']}\n"
            else:
                data = info['knowledge']['data']
                if isinstance(data, dict) and 'description' in data:
                    enhanced += f"• {data['description']}\n"
        
        return enhanced
    
    def validate_response(self, response: str) -> bool:
        """Validate if the response meets quality standards"""
        if not response or len(response.strip()) < self.config.MIN_RESPONSE_LENGTH:
            return False
        
        if len(response) > self.config.MAX_RESPONSE_LENGTH:
            return False
        
        # Check for generic responses
        generic_phrases = ['feel free to ask', 'let me know', 'anything else']
        if any(phrase in response.lower() for phrase in generic_phrases):
            return len(response) > 100  # Allow if it's a longer, more detailed response
        
        return True