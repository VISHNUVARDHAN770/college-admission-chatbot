'''
import re
import json
from datetime import datetime

class ResponseHandler:
    def __init__(self):
        self.intent_patterns = {
            'admission_requirements': [
                r'admission.*requirement', r'what.*need.*apply', r'application.*requirement',
                r'eligibility', r'qualify', r'criteria.*admission'
            ],
            'deadlines': [
                r'deadline', r'when.*apply', r'application.*date', r'due.*date',
                r'last.*date', r'cutoff.*date'
            ],
            'programs': [
                r'program', r'course', r'major', r'degree', r'study.*option',
                r'what.*offer', r'available.*course'
            ],
            'fees': [
                r'fee', r'cost', r'tuition', r'expense', r'price', r'money',
                r'financial', r'scholarship'
            ],
            'contact': [
                r'contact', r'phone', r'email', r'address', r'location',
                r'reach.*you', r'how.*contact'
            ],
            'campus': [
                r'campus', r'facility', r'hostel', r'accommodation', r'library',
                r'infrastructure', r'lab'
            ]
        }
        
        self.greeting_patterns = [
            r'hello', r'hi', r'hey', r'good.*morning', r'good.*afternoon',
            r'good.*evening', r'greetings'
        ]
        
        self.goodbye_patterns = [
            r'bye', r'goodbye', r'see.*you', r'thank.*you', r'thanks'
        ]

    def detect_intent(self, user_message):
        """Detect user intent from the message"""
        message_lower = user_message.lower()
        
        # Check for greetings
        if any(re.search(pattern, message_lower) for pattern in self.greeting_patterns):
            return 'greeting'
        
        # Check for goodbyes
        if any(re.search(pattern, message_lower) for pattern in self.goodbye_patterns):
            return 'goodbye'
        
        # Check for specific intents
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, message_lower) for pattern in patterns):
                return intent
        
        return 'general'

    def format_response(self, response, intent=None, knowledge_data=None):
        """Format and enhance the response based on intent"""
        
        # Clean the response
        response = self.clean_response(response)
        
        # Add intent-specific enhancements
        if intent == 'greeting':
            response = self.enhance_greeting(response)
        elif intent == 'goodbye':
            response = self.enhance_goodbye(response)
        elif intent and knowledge_data:
            response = self.enhance_with_structured_data(response, intent, knowledge_data)
        
        return response

    def clean_response(self, response):
        """Clean and normalize the response"""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Remove any incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        return response

    def enhance_greeting(self, response):
        """Enhance greeting responses"""
        greetings = [
            "Hello! 👋 Welcome to our college admission assistance!",
            "Hi there! 🎓 I'm here to help you with all your admission questions.",
            "Greetings! 🌟 How can I assist you with your college application today?"
        ]
        
        if len(response) < 50:  # If response is too short, use a predefined greeting
            import random
            return random.choice(greetings)
        
        return response

    def enhance_goodbye(self, response):
        """Enhance goodbye responses"""
        goodbyes = [
            "Thank you for your interest in our college! 🎓 Best of luck with your application!",
            "Goodbye! 👋 Feel free to reach out if you have more questions. Good luck! 🍀",
            "Thanks for chatting! 😊 Don't hesitate to contact our admission office for further assistance."
        ]
        
        if len(response) < 50:
            import random
            return random.choice(goodbyes)
        
        return response

    def enhance_with_structured_data(self, response, intent, knowledge_data):
        """Enhance response with structured data based on intent"""
        
        if intent == 'admission_requirements' and knowledge_data:
            # Add structured admission requirements
            req_data = self.extract_admission_requirements(knowledge_data)
            if req_data:
                response += f"\n\n📋 **Key Requirements:**\n{req_data}"
        
        elif intent == 'deadlines' and knowledge_data:
            # Add structured deadline information
            deadline_data = self.extract_deadlines(knowledge_data)
            if deadline_data:
                response += f"\n\n📅 **Important Dates:**\n{deadline_data}"
        
        elif intent == 'programs' and knowledge_data:
            # Add structured program information
            program_data = self.extract_programs(knowledge_data)
            if program_data:
                response += f"\n\n🎓 **Available Programs:**\n{program_data}"
        
        elif intent == 'contact':
            response += "\n\n📞 **Contact Information:**\n"
            response += "📧 Email: admissions@college.edu\n"
            response += "📱 Phone: (555) 123-4567\n"
            response += "🏢 Visit our admission office for in-person assistance!"
        
        return response

    def extract_admission_requirements(self, knowledge_data):
        """Extract and format admission requirements"""
        req_text = ""
        
        for item in knowledge_data:
            if item['knowledge']['type'] == 'college_info' and 'admission' in item['knowledge']['category']:
                data = item['knowledge']['data']
                if isinstance(data, dict):
                    for level, requirements in data.items():
                        req_text += f"\n**{level.title()}:**\n"
                        if isinstance(requirements, dict):
                            for key, value in requirements.items():
                                if isinstance(value, list):
                                    req_text += f"• {key.replace('_', ' ').title()}: {', '.join(value)}\n"
                                else:
                                    req_text += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        return req_text.strip()

    def extract_deadlines(self, knowledge_data):
        """Extract and format deadline information"""
        deadline_text = ""
        
        for item in knowledge_data:
            if item['knowledge']['type'] == 'college_info' and 'date' in item['knowledge']['category']:
                data = item['knowledge']['data']
                if isinstance(data, dict):
                    for semester, dates in data.items():
                        deadline_text += f"\n**{semester.replace('_', ' ').title()}:**\n"
                        if isinstance(dates, dict):
                            for date_type, date_value in dates.items():
                                deadline_text += f"• {date_type.replace('_', ' ').title()}: {date_value}\n"
        
        return deadline_text.strip()

    def extract_programs(self, knowledge_data):
        """Extract and format program information"""
        program_text = ""
        programs_by_level = {}
        
        for item in knowledge_data:
            if item['knowledge']['type'] == 'program':
                level = item['knowledge']['category']
                if level not in programs_by_level:
                    programs_by_level[level] = []
                programs_by_level[level].append(item['knowledge']['data'])
        
        for level, programs in programs_by_level.items():
            program_text += f"\n**{level.title()} Programs:**\n"
            for program in programs[:3]:  # Limit to 3 programs per level
                name = program.get('name', 'Unknown Program')
                duration = program.get('duration', 'N/A')
                program_text += f"• {name} ({duration})\n"
        
        return program_text.strip()

    def add_helpful_suggestions(self, response, intent):
        """Add helpful suggestions based on intent"""
        suggestions = {
            'admission_requirements': [
                "Would you like to know about specific program requirements?",
                "Do you need information about application deadlines?",
                "Are you interested in our scholarship opportunities?"
            ],
            'deadlines': [
                "Would you like to know about admission requirements?",
                "Do you need help with the application process?",
                "Are you interested in scheduling a campus visit?"
            ],
            'programs': [
                "Would you like detailed information about any specific program?",
                "Do you want to know about career prospects for these programs?",
                "Are you interested in admission requirements for these programs?"
            ],
            'general': [
                "Is there anything specific about admissions you'd like to know?",
                "Would you like information about our programs or requirements?",
                "Do you have questions about deadlines or the application process?"
            ]
        }
        
        if intent in suggestions:
            response += f"\n\n💡 **You might also want to ask:**\n"
            for suggestion in suggestions[intent][:2]:  # Limit to 2 suggestions
                response += f"• {suggestion}\n"
        
        return response

    def validate_response_quality(self, response):
        """Validate and ensure response quality"""
        # Check minimum length
        if len(response.strip()) < 20:
            return False
        
        # Check for meaningful content (not just repeated characters)
        if len(set(response.lower().replace(' ', ''))) < 5:
            return False
        
        # Check for proper sentence structure
        if not re.search(r'[.!?]', response):
            return False
        
        return True

    def get_fallback_response(self, intent=None):
        """Get fallback response when main response fails"""
        fallback_responses = {
            'admission_requirements': "I'd be happy to help you with admission requirements! Our basic requirements include academic transcripts, test scores, and application materials. For detailed information, please contact our admission office.",
            'deadlines': "For application deadlines, I recommend checking our official website or contacting our admission office directly at admissions@college.edu for the most current dates.",
            'programs': "We offer various undergraduate and graduate programs across multiple disciplines. For a complete list and detailed information, please visit our programs page or contact the admission office.",
            'general': "Thank you for your interest in our college! I'm here to help with admission-related questions. Feel free to ask about requirements, deadlines, programs, or any other admission topics."
        }
        
        return fallback_responses.get(intent, fallback_responses['general'])
    












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

class ResponseHandler:
    """Handles response formatting and enhancement for college admission queries"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    def detect_intent(self, message: str) -> str:
        """Detect the intent of the user's message"""
        message_lower = message.lower()
        
        # Check for greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in message_lower for pattern in greeting_patterns):
            return 'greeting'
        
        # Check for specific intents
        intent_scores = {}
        for intent, keywords in self.config.INTENTS.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            # Return the intent with the highest score
            return max(intent_scores, key=intent_scores.get)
        
        return 'general'
    
    def format_response(self, response: str, intent: str, relevant_info: List[Dict]) -> str:
        """Format and enhance the response based on context"""
        try:
            # If response is too generic, enhance it with relevant information
            if self._is_generic_response(response) and relevant_info:
                enhanced_response = self._enhance_with_relevant_info(response, relevant_info, intent)
                return enhanced_response
            
            # Format the response based on intent
            formatted_response = self._format_by_intent(response, intent, relevant_info)
            
            # Add helpful suggestions
            final_response = self._add_helpful_suggestions(formatted_response, intent)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return response
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if the response is too generic"""
        generic_phrases = [
            'feel free to ask',
            'let me know',
            'anything else',
            'happy to help',
            'is there anything',
            'please let me know'
        ]
        
        response_lower = response.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        
        # Consider it generic if it's short and contains generic phrases
        return len(response) < 100 and generic_count > 0
    
    def _enhance_with_relevant_info(self, response: str, relevant_info: List[Dict], intent: str) -> str:
        """Enhance generic response with relevant information"""
        if not relevant_info:
            return response
        
        enhanced = f"Here's what I can tell you about {intent.replace('_', ' ')}:\n\n"
        
        for i, info in enumerate(relevant_info[:3]):
            knowledge = info['knowledge']
            
            if knowledge['type'] == 'faq':
                enhanced += f"**Q: {knowledge['question']}**\n"
                enhanced += f"{knowledge['answer']}\n\n"
            
            elif knowledge['type'] == 'program':
                data = knowledge['data']
                enhanced += f"**Program: {data.get('name', 'Unknown')}**\n"
                enhanced += f"{data.get('description', 'No description available')}\n\n"
            
            elif knowledge['type'] == 'general_info':
                data = knowledge['data']
                enhanced += f"**College Information:**\n"
                for key, value in data.items():
                    enhanced += f"• {key.replace('_', ' ').title()}: {value}\n"
                enhanced += "\n"
            
            elif knowledge['type'] == 'important_dates':
                data = knowledge['data']
                enhanced += f"**Important Dates:**\n"
                for semester, dates in data.items():
                    enhanced += f"**{semester.replace('_', ' ').title()}:**\n"
                    if isinstance(dates, dict):
                        for date_type, date_value in dates.items():
                            enhanced += f"• {date_type.replace('_', ' ').title()}: {date_value}\n"
                enhanced += "\n"
        
        enhanced += "Would you like more specific information about any of these topics? 🤔"
        return enhanced
    
    def _format_by_intent(self, response: str, intent: str, relevant_info: List[Dict]) -> str:
        """Format response based on the detected intent"""
        
        if intent == 'admission_requirements':
            return self._format_admission_requirements(response, relevant_info)
        elif intent == 'deadlines':
            return self._format_deadlines(response, relevant_info)
        elif intent == 'programs':
            return self._format_programs(response, relevant_info)
        elif intent == 'fees':
            return self._format_fees(response, relevant_info)
        elif intent == 'contact':
            return self._format_contact(response, relevant_info)
        else:
            return response
    
    def _format_admission_requirements(self, response: str, relevant_info: List[Dict]) -> str:
        """Format admission requirements response"""
        if not relevant_info:
            return response
        
        formatted = "📋 **Admission Requirements:**\n\n"
        
        # Look for specific requirements in relevant info
        for info in relevant_info:
            if info['knowledge']['type'] == 'faq' and 'requirement' in info['knowledge']['question'].lower():
                formatted += f"• {info['knowledge']['answer']}\n"
        
        # Add general requirements if no specific ones found
        if len(formatted) < 100:
            formatted += """**General Requirements:**
• High school diploma or equivalent
• Minimum GPA requirements
• Standardized test scores (SAT/ACT)
• Letters of recommendation
• Personal statement or essay
• Completed application form

**Additional Information:**
• Requirements may vary by program
• International students may need English proficiency tests
• Some programs require portfolios or interviews

"""
        
        formatted += "\n💡 **Need specific requirements for your program?** Contact our admission office!"
        return formatted
    
    def _format_deadlines(self, response: str, relevant_info: List[Dict]) -> str:
        """Format deadlines response"""
        formatted = "📅 **Application Deadlines:**\n\n"
        
        # Look for deadline information in relevant info
        deadline_found = False
        for info in relevant_info:
            if info['knowledge']['type'] == 'important_dates':
                data = info['knowledge']['data']
                for semester, dates in data.items():
                    formatted += f"**{semester.replace('_', ' ').title()}:**\n"
                    if isinstance(dates, dict):
                        for date_type, date_value in dates.items():
                            formatted += f"• {date_type.replace('_', ' ').title()}: {date_value}\n"
                deadline_found = True
                break
        
        if not deadline_found:
            formatted += """**General Deadlines:**
• Fall Semester: January 15
• Spring Semester: October 1
• Summer Session: March 15
• Graduate Programs: February 1 (Fall), September 15 (Spring)

**Important Notes:**
• Early decision deadlines are usually earlier
• International students may have different deadlines
• Some programs have rolling admissions

"""
        
        formatted += "\n⏰ **Always check program-specific deadlines!**"
        return formatted
    
    def _format_programs(self, response: str, relevant_info: List[Dict]) -> str:
        """Format programs response"""
        formatted = "🎓 **Academic Programs:**\n\n"
        
        # Look for program information
        programs_found = False
        for info in relevant_info:
            if info['knowledge']['type'] == 'program':
                data = info['knowledge']['data']
                formatted += f"**{data.get('name', 'Program')}**\n"
                formatted += f"{data.get('description', 'No description available')}\n"
                if 'duration' in data:
                    formatted += f"Duration: {data['duration']}\n"
                if 'degree' in data:
                    formatted += f"Degree: {data['degree']}\n"
                formatted += "\n"
                programs_found = True
        
        if not programs_found:
            formatted += """**Undergraduate Programs:**
• Business Administration
• Computer Science
• Engineering
• Liberal Arts
• Sciences
• Education

**Graduate Programs:**
• Master's Programs (MA, MS, MBA)
• Doctoral Programs (PhD)
• Professional Programs
• Certificate Programs

"""
        
        formatted += "\n📚 **Want detailed information about a specific program?** Just ask!"
        return formatted
    
    def _format_fees(self, response: str, relevant_info: List[Dict]) -> str:
        """Format fees response"""
        formatted = "💰 **Tuition and Fees:**\n\n"
        
        formatted += """**Undergraduate (Annual):**
• In-state: $12,000 - $15,000
• Out-of-state: $18,000 - $25,000
• International: $20,000 - $28,000

**Graduate (Annual):**
• In-state: $15,000 - $20,000
• Out-of-state: $22,000 - $30,000
• International: $25,000 - $35,000

**Additional Costs:**
• Housing: $8,000 - $12,000
• Meal Plans: $3,000 - $5,000
• Books & Supplies: $1,200 - $1,800

**Financial Aid Available:**
• Merit-based scholarships
• Need-based grants
• Work-study programs
• Payment plans

"""
        
        formatted += "\n💳 **Contact Financial Aid Office for personalized assistance!**"
        return formatted
    
    def _format_contact(self, response: str, relevant_info: List[Dict]) -> str:
        """Format contact response"""
        formatted = "📞 **Contact Information:**\n\n"
        
        formatted += """**Admission Office:**
• 📧 Email: admissions@college.edu
• 📞 Phone: (555) 123-4567
• 🌐 Website: www.college.edu

**Office Hours:**
• Monday - Friday: 9:00 AM - 5:00 PM
• Saturday: 10:00 AM - 2:00 PM (peak season)

**Address:**
Admission Office
123 University Drive
College Town, State 12345

**Other Departments:**
• Financial Aid: (555) 123-4569
• International Students: (555) 123-4570
• Academic Advising: (555) 123-4571

"""
        
        formatted += "\n🎓 **We're here to help you succeed!**"
        return formatted
    
    def _add_helpful_suggestions(self, response: str, intent: str) -> str:
        """Add helpful suggestions based on intent"""
        suggestions = {
            'admission_requirements': [
                "Would you like to know about specific program requirements?",
                "Need help with application documents?",
                "Want to know about deadlines?"
            ],
            'deadlines': [
                "Need help with the application process?",
                "Want to know about early decision options?",
                "Looking for program-specific deadlines?"
            ],
            'programs': [
                "Want details about a specific program?",
                "Interested in program requirements?",
                "Need information about career prospects?"
            ],
            'fees': [
                "Want to know about financial aid options?",
                "Need help with payment plans?",
                "Looking for scholarship opportunities?"
            ],
            'contact': [
                "Ready to schedule a campus visit?",
                "Want to speak with an advisor?",
                "Need help with your application?"
            ]
        }
        
        if intent in suggestions:
            response += f"\n\n**What else can I help you with?**\n"
            for suggestion in suggestions[intent][:2]:
                response += f"• {suggestion}\n"
        
        return response
    
    def validate_response_quality(self, response: str) -> bool:
        """Validate if the response meets quality standards"""
        if not response or len(response.strip()) < self.config.MIN_RESPONSE_LENGTH:
            return False
        
        if len(response) > self.config.MAX_RESPONSE_LENGTH:
            return False
        
        # Check for too many generic phrases
        generic_phrases = ['feel free to ask', 'let me know', 'anything else']
        generic_count = sum(1 for phrase in generic_phrases if phrase in response.lower())
        
        # If it's mostly generic phrases, it's low quality
        return generic_count < 2
    
    def get_fallback_response(self, intent: str) -> str:
        """Get a fallback response when the main response fails"""
        fallback_responses = {
            'admission_requirements': "I'd be happy to help you with admission requirements! Generally, you'll need your transcripts, test scores, and application materials. For specific requirements, please contact our admission office at admissions@college.edu or (555) 123-4567.",
            
            'deadlines': "Application deadlines are important! While they vary by program, our general deadlines are January 15 for Fall and October 1 for Spring. For exact dates, please check with our admission office.",
            
            'programs': "We offer many excellent programs! From undergraduate to graduate degrees across various fields. I'd recommend browsing our website or contacting our academic advisors for detailed program information.",
            
            'fees': "Tuition varies by program and residency status. For accurate fee information and financial aid options, please contact our Financial Aid Office at (555) 123-4569.",
            
            'contact': "You can reach our admission office at admissions@college.edu or (555) 123-4567. We're open Monday-Friday, 9 AM - 5 PM, and we're here to help!",
            
            'general': "I'm here to help with your college admission questions! I can provide information about requirements, deadlines, programs, fees, and contact details. What would you like to know?"
        }
        
        return fallback_responses.get(intent, fallback_responses['general'])
    
    def enhance_response_with_emojis(self, response: str, intent: str) -> str:
        """Add relevant emojis to make the response more engaging"""
        emoji_map = {
            'admission_requirements': '📋',
            'deadlines': '📅',
            'programs': '🎓',
            'fees': '💰',
            'contact': '📞',
            'general': '💭'
        }
        
        if intent in emoji_map and not response.startswith(emoji_map[intent]):
            response = f"{emoji_map[intent]} {response}"
        
        return response   '''










import re
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseHandler:
    """Handles response processing and formatting for the College Admission Chatbot"""
    
    def __init__(self):
        """Initialize the response handler"""
        self.intent_keywords = {
            'admission_requirements': [
                'requirements', 'eligibility', 'criteria', 'qualify', 'admit', 
                'application', 'documents', 'needed', 'required'
            ],
            'deadlines': [
                'deadline', 'last date', 'application date', 'closing date', 
                'when to apply', 'due date', 'submit by'
            ],
            'programs': [
                'courses', 'programs', 'departments', 'majors', 'degrees', 
                'studies', 'subjects', 'fields', 'disciplines'
            ],
            'fees': [
                'fees', 'cost', 'tuition', 'expenses', 'payment', 'scholarship',
                'financial aid', 'money', 'price', 'affordable'
            ],
            'contact': [
                'contact', 'phone', 'email', 'address', 'office', 'reach',
                'call', 'visit', 'location'
            ],
            'general': [
                'hello', 'hi', 'help', 'about', 'information', 'tell me',
                'what is', 'how', 'where'
            ]
        }
        
        # Response templates
        self.response_templates = {
            'admission_requirements': """
📋 **Admission Requirements**

For admission to our college, you typically need:
• ✅ Completed application form
• 📄 Academic transcripts from previous institutions
• 📊 Entrance exam scores (SAT/ACT or equivalent)
• 📝 Letters of recommendation (2-3)
• ✍️ Personal statement or essay
• 💳 Application fee payment

**Note:** Specific requirements may vary by program. Would you like information about requirements for a specific program?
            """,
            
            'deadlines': """
📅 **Application Deadlines**

**Fall Semester 2024:**
• Application Deadline: March 1, 2024
• Semester Start: August 15, 2024

**Spring Semester 2025:**
• Application Deadline: October 1, 2024
• Semester Start: January 15, 2025

**Summer Session 2024:**
• Application Deadline: March 15, 2024

⚠️ **Important:** Some programs may have earlier deadlines. Please check with specific departments for exact dates.
            """,
            
            'programs': """
🎓 **Academic Programs**

We offer a wide range of programs including:

**Undergraduate Programs:**
• 💻 Computer Science (4 years)
• 💼 Business Administration (4 years)
• 🔧 Engineering (4 years)
• 🎨 Arts & Sciences (4 years)

**Graduate Programs:**
• 🎓 Master's degrees in various fields
• 📚 PhD programs
• 🏆 Professional certifications

**Learning Options:**
• 🏫 On-campus classes
• 💻 Online programs
• 📱 Hybrid learning

Would you like detailed information about any specific program?
            """,
            
            'fees': """
💰 **Tuition & Fees**

**Undergraduate Programs:**
• In-state: $15,000 - $20,000 per year
• Out-of-state: $25,000 - $30,000 per year

**Graduate Programs:**
• Master's: $18,000 - $25,000 per year
• PhD: Varies by program

**Additional Costs:**
• 🏠 Housing: $8,000 - $12,000 per year
• 🍽️ Meal plans: $3,000 - $5,000 per year
• 📚 Books & supplies: $1,200 - $1,500 per year

**Financial Aid Available:**
• 🎓 Merit-based scholarships
• 💸 Need-based financial aid
• 💼 Work-study programs

Contact our financial aid office for personalized information!
            """,
            
            'contact': """
📞 **Contact Information**

**Admission Office:**
• 📧 Email: admissions@college.edu
• ☎️ Phone: (555) 123-4567
• 🕒 Office Hours: Monday-Friday, 9 AM - 5 PM

**Visit Us:**
• 🏢 Address: 123 College Street, Campus City, State 12345
• 🚗 Parking: Available on campus
• 🚌 Public Transport: Bus routes 15, 22, 45

**Online Resources:**
• 🌐 Website: www.college.edu
• 💬 Live Chat: Available on website
• 📱 Social Media: @CollegeOfficial

**Emergency Contact:**
• 🚨 24/7 Hotline: (555) 123-HELP

We're here to help you with your admission journey!
            """
        }
    
    def detect_intent(self, user_message: str) -> str:
        """
        Detect the intent of the user message
        
        Args:
            user_message: The user's input message
            
        Returns:
            Detected intent category
        """
        user_message_lower = user_message.lower()
        
        # Count keyword matches for each intent
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in user_message_lower:
                    score += 1
            intent_scores[intent] = score
        
        # Return the intent with highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return 'general'
    
    def format_response(self, response: str, intent: str, relevant_info: List[Dict] = None) -> str:
        """
        Format the response based on intent and relevant information
        
        Args:
            response: Raw response from LLM
            intent: Detected intent
            relevant_info: Relevant information from knowledge base
            
        Returns:
            Formatted response
        """
        try:
            # Use template if available and response is too generic
            if intent in self.response_templates and (
                len(response) < 50 or 
                "I don't know" in response or 
                "I'm not sure" in response or
                "Feel free to ask" in response
            ):
                base_response = self.response_templates[intent]
            else:
                base_response = response
            
            # Add relevant information from knowledge base
            if relevant_info:
                additional_info = self._extract_relevant_info(relevant_info, intent)
                if additional_info:
                    base_response += f"\n\n📌 **Additional Information:**\n{additional_info}"
            
            # Add helpful suggestions
            suggestions = self._get_helpful_suggestions(intent)
            if suggestions:
                base_response += f"\n\n💡 **You might also want to know:**\n{suggestions}"
            
            return base_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return self._get_fallback_response(intent)
    
    def _extract_relevant_info(self, relevant_info: List[Dict], intent: str) -> str:
        """Extract and format relevant information"""
        info_parts = []
        
        for info in relevant_info[:2]:  # Use top 2 relevant pieces
            knowledge = info.get('knowledge', {})
            
            if knowledge.get('type') == 'faq':
                info_parts.append(f"• **Q:** {knowledge.get('question', '')}")
                info_parts.append(f"  **A:** {knowledge.get('answer', '')}")
            
            elif knowledge.get('type') == 'program':
                program_data = knowledge.get('data', {})
                info_parts.append(f"• **{program_data.get('name', '')}** ({program_data.get('degree', '')})")
                info_parts.append(f"  {program_data.get('description', '')}")
        
        return '\n'.join(info_parts) if info_parts else ""
    
    def _get_helpful_suggestions(self, intent: str) -> str:
        """Get helpful suggestions based on intent"""
        suggestions = {
            'admission_requirements': '• Application deadlines\n• Available programs\n• Tuition fees',
            'deadlines': '• Admission requirements\n• Application process\n• Contact information',
            'programs': '• Admission requirements\n• Tuition fees\n• Application deadlines',
            'fees': '• Financial aid options\n• Payment plans\n• Scholarship opportunities',
            'contact': '• Campus visit scheduling\n• Virtual tour options\n• Application status check',
            'general': '• Admission requirements\n• Available programs\n• Application deadlines\n• Tuition fees'
        }
        
        return suggestions.get(intent, '')
    
    def validate_response_quality(self, response: str) -> bool:
        """
        Validate if the response meets quality standards
        
        Args:
            response: Response to validate
            
        Returns:
            True if response is good quality, False otherwise
        """
        if not response or len(response.strip()) < 20:
            return False
        
        # Check for generic/unhelpful responses
        unhelpful_phrases = [
            "I don't know",
            "I'm not sure",
            "I can't help",
            "I don't have information",
            "Sorry, I don't understand"
        ]
        
        response_lower = response.lower()
        for phrase in unhelpful_phrases:
            if phrase in response_lower:
                return False
        
        return True
    
    def get_fallback_response(self, intent: str) -> str:
        """
        Get a fallback response when main response generation fails
        
        Args:
            intent: The detected intent
            
        Returns:
            Fallback response
        """
        if intent in self.response_templates:
            return self.response_templates[intent]
        
        return """
🎓 **College Admission Assistant**

I'm here to help you with your college admission questions! I can provide information about:

• 📋 Admission requirements and eligibility criteria
• 📅 Application deadlines and important dates
• 🎓 Available programs and courses
• 💰 Tuition fees and financial aid
• 📞 Contact information and office hours

Please feel free to ask me about any of these topics, and I'll do my best to provide you with accurate and helpful information!

If you need immediate assistance, you can also contact our admission office directly at:
📧 admissions@college.edu
☎️ (555) 123-4567
        """
    
    def add_helpful_suggestions(self, response: str, intent: str) -> str:
        """Add helpful suggestions to the response"""
        suggestions_map = {
            'admission_requirements': [
                "Would you like to know about application deadlines?",
                "Are you interested in information about specific programs?",
                "Do you need help with the application process?"
            ],
            'deadlines': [
                "Would you like to know about admission requirements?",
                "Are you interested in information about tuition fees?",
                "Do you need help with the application process?"
            ],
            'programs': [
                "Would you like to know about admission requirements for specific programs?",
                "Are you interested in tuition fees for these programs?",
                "Do you need contact information for specific departments?"
            ],
            'fees': [
                "Would you like information about financial aid options?",
                "Are you interested in scholarship opportunities?",
                "Do you need help with payment plans?"
            ],
            'contact': [
                "Would you like to schedule a campus visit?",
                "Are you interested in virtual tour options?",
                "Do you need help with your application status?"
            ]
        }
        
        suggestions = suggestions_map.get(intent, [])
        if suggestions:
            response += f"\n\n💭 **Quick Questions:**\n"
            for suggestion in suggestions[:2]:  # Show max 2 suggestions
                response += f"• {suggestion}\n"
        
        return response