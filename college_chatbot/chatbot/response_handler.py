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
