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
