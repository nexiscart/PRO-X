#!/usr/bin/env python3
"""
Lead Generation Agent for Pro Roofing AI
Specialized in lead qualification and customer outreach
"""

import asyncio
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json

from .base_agent import BaseRoofingAgent, AgentRequest

class RoofingLeadAgent(BaseRoofingAgent):
    """AI agent specialized in lead generation and qualification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "lead_agent")
        
        # Lead qualification criteria
        self.qualification_config = config.get('qualification', {})
        self.min_score = self.qualification_config.get('min_score', 70)
        self.scoring_factors = self.qualification_config.get('scoring_factors', {
            'budget_range': 30,
            'timeline_urgency': 25,
            'decision_authority': 25,
            'project_scope': 20
        })
        
        # Disqualifiers
        self.disqualifiers = self.qualification_config.get('disqualifiers', [
            "no budget", "just curious", "price shopping only"
        ])
        
        # Lead templates
        self.templates = config.get('templates', {})
        
        # CRM integration settings
        self.crm_config = config.get('crm_integration', {})
        
        self.logger.info("🎯 Lead Agent specialized configuration loaded")

    async def _initialize_resources(self):
        """Initialize lead agent resources"""
        # Load lead templates if configured
        await self._load_templates()
        
        # Initialize CRM connection if configured
        await self._initialize_crm()
        
        self.logger.info("✅ Lead agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup lead agent resources"""
        # Cleanup CRM connections
        self.logger.info("🧹 Lead agent resources cleaned up")

    async def _load_templates(self):
        """Load email and communication templates"""
        self.loaded_templates = {}
        
        for template_type, template_path in self.templates.items():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.loaded_templates[template_type] = f.read()
                self.logger.info(f"📄 Loaded template: {template_type}")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load template {template_type}: {e}")
                # Use default template
                self.loaded_templates[template_type] = self._get_default_template(template_type)

    async def _initialize_crm(self):
        """Initialize CRM integration"""
        # Placeholder for CRM initialization
        # Would integrate with HubSpot, Salesforce, etc.
        self.logger.info("🔗 CRM integration initialized")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for lead generation"""
        return """You are a professional roofing lead generation and qualification specialist. Your expertise includes:

**Lead Qualification:**
- Assessing customer needs and project requirements
- Determining budget ranges and timeline urgency
- Identifying decision makers and authority levels
- Evaluating project scope and complexity

**Customer Communication:**
- Professional, consultative communication style
- Building trust and rapport with potential customers
- Asking the right questions to understand needs
- Providing value before asking for commitment

**Industry Knowledge:**
- Commercial and industrial roofing systems
- Material options and their applications
- Typical project costs and timelines
- Common customer pain points and concerns

**Sales Process:**
- Lead scoring and prioritization
- Objection handling and concern resolution
- Setting appropriate expectations
- Guiding customers through the sales funnel

Always maintain a helpful, professional tone while gathering necessary information to qualify leads effectively. Focus on understanding the customer's specific needs and providing valuable guidance."""

    async def _process_request_impl(self, request: AgentRequest) -> Dict[str, Any]:
        """Process lead generation request"""
        request_type = request.type.lower()
        
        if request_type in ['qualify', 'qualification']:
            return await self._qualify_lead(request)
        elif request_type in ['outreach', 'contact']:
            return await self._generate_outreach(request)
        elif request_type in ['follow_up', 'followup']:
            return await self._generate_follow_up(request)
        else:
            # Default to lead qualification
            return await self._qualify_lead(request)

    async def _qualify_lead(self, request: AgentRequest) -> Dict[str, Any]:
        """Qualify a potential lead"""
        content = request.content
        metadata = request.metadata or {}
        
        # Extract lead information
        lead_info = self._extract_lead_info(content, metadata)
        
        # Calculate qualification score
        qualification_score = self._calculate_qualification_score(lead_info)
        
        # Check for disqualifiers
        is_disqualified, disqualify_reason = self._check_disqualifiers(content)
        
        # Generate qualification response
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Please help me qualify this potential roofing lead:

Lead Information: {content}

Additional Context: {json.dumps(metadata, indent=2) if metadata else 'None provided'}

Please provide:
1. Lead qualification assessment
2. Recommended next steps
3. Key information still needed
4. Priority level for follow-up"""}
        ]
        
        response = await self._generate_response(messages)
        
        # Determine if lead is qualified
        is_qualified = (qualification_score >= self.min_score and not is_disqualified)
        
        result = {
            'success': True,
            'content': response,
            'confidence': 0.85,
            'metadata': {
                'qualified': is_qualified,
                'qualification_score': qualification_score,
                'disqualified': is_disqualified,
                'disqualify_reason': disqualify_reason,
                'lead_info': lead_info,
                'recommended_action': self._get_recommended_action(is_qualified, qualification_score),
                'priority_level': self._get_priority_level(qualification_score)
            }
        }
        
        # Log qualification result
        self.logger.info(f"Lead qualified: {is_qualified}, Score: {qualification_score}")
        
        return result

    async def _generate_outreach(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate initial outreach communication"""
        content = request.content
        metadata = request.metadata or {}
        
        # Determine outreach type
        outreach_type = metadata.get('outreach_type', 'cold_email')
        
        # Get template
        template = self.loaded_templates.get(outreach_type, self.loaded_templates.get('cold_email'))
        
        # Generate personalized outreach
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Create a professional outreach message for this roofing lead:

Lead Information: {content}

Outreach Type: {outreach_type}
Template Style: Professional and consultative

Requirements:
- Personalize based on their specific needs
- Highlight relevant experience and expertise
- Include a clear call-to-action
- Maintain professional, helpful tone
- Focus on value proposition

Template to customize: {template}"""}
        ]
        
        response = await self._generate_response(messages)
        
        return {
            'success': True,
            'content': response,
            'confidence': 0.80,
            'metadata': {
                'outreach_type': outreach_type,
                'personalized': True,
                'template_used': outreach_type,
                'call_to_action': 'Schedule consultation'
            }
        }

    async def _generate_follow_up(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate follow-up communication"""
        content = request.content
        metadata = request.metadata or {}
        
        # Get previous interaction context
        previous_interactions = metadata.get('previous_interactions', [])
        follow_up_number = metadata.get('follow_up_number', 1)
        
        # Generate contextual follow-up
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Create a follow-up message for this roofing lead:

Current Situation: {content}

Follow-up Number: {follow_up_number}
Previous Interactions: {json.dumps(previous_interactions, indent=2) if previous_interactions else 'None'}

Requirements:
- Reference previous conversation appropriately
- Provide additional value or information
- Address any concerns that may have arisen
- Include a gentle but clear next step
- Maintain relationship-building focus"""}
        ]
        
        response = await self._generate_response(messages)
        
        return {
            'success': True,
            'content': response,
            'confidence': 0.75,
            'metadata': {
                'follow_up_number': follow_up_number,
                'context_used': bool(previous_interactions),
                'next_action': 'Schedule call or meeting'
            }
        }

    def _extract_lead_info(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from lead content"""
        lead_info = {
            'contact_info': self._extract_contact_info(content),
            'project_type': self._identify_project_type(content),
            'urgency_indicators': self._find_urgency_indicators(content),
            'budget_indicators': self._find_budget_indicators(content),
            'decision_authority': self._assess_decision_authority(content, metadata),
            'property_details': self._extract_property_details(content)
        }
        
        return lead_info

    def _extract_contact_info(self, content: str) -> Dict[str, str]:
        """Extract contact information from content"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone extraction
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, content)
        if phones:
            contact_info['phone'] = ''.join(phones[0])
        
        # Company name (basic extraction)
        company_keywords = ['company', 'corp', 'inc', 'llc', 'ltd']
        words = content.lower().split()
        for i, word in enumerate(words):
            if any(keyword in word for keyword in company_keywords):
                # Take surrounding words as company name
                start = max(0, i-2)
                end = min(len(words), i+3)
                contact_info['company'] = ' '.join(words[start:end])
                break
        
        return contact_info

    def _identify_project_type(self, content: str) -> str:
        """Identify the type of roofing project"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['commercial', 'office', 'warehouse', 'industrial']):
            return 'commercial'
        elif any(word in content_lower for word in ['residential', 'home', 'house']):
            return 'residential'
        elif any(word in content_lower for word in ['repair', 'fix', 'leak']):
            return 'repair'
        elif any(word in content_lower for word in ['new', 'construction', 'build']):
            return 'new_construction'
        elif any(word in content_lower for word in ['replace', 'replacement']):
            return 'replacement'
        else:
            return 'unknown'

    def _find_urgency_indicators(self, content: str) -> List[str]:
        """Find indicators of project urgency"""
        content_lower = content.lower()
        urgency_indicators = []
        
        high_urgency = ['emergency', 'urgent', 'asap', 'immediately', 'leak', 'damage']
        medium_urgency = ['soon', 'this month', 'next month', 'planning']
        low_urgency = ['sometime', 'future', 'considering', 'thinking about']
        
        for indicator in high_urgency:
            if indicator in content_lower:
                urgency_indicators.append(f"high:{indicator}")
        
        for indicator in medium_urgency:
            if indicator in content_lower:
                urgency_indicators.append(f"medium:{indicator}")
        
        for indicator in low_urgency:
            if indicator in content_lower:
                urgency_indicators.append(f"low:{indicator}")
        
        return urgency_indicators

    def _find_budget_indicators(self, content: str) -> Dict[str, Any]:
        """Find budget-related information"""
        budget_info = {'indicators': [], 'estimated_range': None}
        
        # Dollar amount extraction
        dollar_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(dollar_pattern, content)
        if amounts:
            budget_info['mentioned_amounts'] = amounts
        
        # Budget keywords
        budget_keywords = ['budget', 'cost', 'price', 'expensive', 'affordable', 'cheap']
        content_lower = content.lower()
        
        for keyword in budget_keywords:
            if keyword in content_lower:
                budget_info['indicators'].append(keyword)
        
        return budget_info

    def _assess_decision_authority(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess decision-making authority of the lead"""
        authority_info = {'level': 'unknown', 'indicators': []}
        
        content_lower = content.lower()
        
        # High authority indicators
        high_authority = ['owner', 'ceo', 'president', 'manager', 'director', 'decision', 'approve']
        medium_authority = ['supervisor', 'coordinator', 'assistant']
        low_authority = ['employee', 'worker', 'staff']
        
        for indicator in high_authority:
            if indicator in content_lower:
                authority_info['level'] = 'high'
                authority_info['indicators'].append(indicator)
                break
        
        if authority_info['level'] == 'unknown':
            for indicator in medium_authority:
                if indicator in content_lower:
                    authority_info['level'] = 'medium'
                    authority_info['indicators'].append(indicator)
                    break
        
        if authority_info['level'] == 'unknown':
            for indicator in low_authority:
                if indicator in content_lower:
                    authority_info['level'] = 'low'
                    authority_info['indicators'].append(indicator)
                    break
        
        return authority_info

    def _extract_property_details(self, content: str) -> Dict[str, Any]:
        """Extract property-related details"""
        property_info = {}
        
        # Square footage extraction
        sqft_pattern = r'(\d+(?:,\d+)?)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft)'
        sqft_matches = re.findall(sqft_pattern, content, re.IGNORECASE)
        if sqft_matches:
            property_info['square_footage'] = sqft_matches[0].replace(',', '')
        
        # Building type
        building_types = ['warehouse', 'office', 'retail', 'industrial', 'manufacturing']
        content_lower = content.lower()
        for building_type in building_types:
            if building_type in content_lower:
                property_info['building_type'] = building_type
                break
        
        return property_info

    def _calculate_qualification_score(self, lead_info: Dict[str, Any]) -> int:
        """Calculate lead qualification score"""
        score = 0
        
        # Budget range scoring
        budget_indicators = lead_info.get('budget_indicators', {})
        if budget_indicators.get('mentioned_amounts'):
            score += self.scoring_factors.get('budget_range', 30)
        elif budget_indicators.get('indicators'):
            score += self.scoring_factors.get('budget_range', 30) * 0.5
        
        # Timeline urgency scoring
        urgency_indicators = lead_info.get('urgency_indicators', [])
        high_urgency = len([i for i in urgency_indicators if i.startswith('high:')])
        medium_urgency = len([i for i in urgency_indicators if i.startswith('medium:')])
        
        if high_urgency > 0:
            score += self.scoring_factors.get('timeline_urgency', 25)
        elif medium_urgency > 0:
            score += self.scoring_factors.get('timeline_urgency', 25) * 0.7
        
        # Decision authority scoring
        authority_level = lead_info.get('decision_authority', {}).get('level', 'unknown')
        if authority_level == 'high':
            score += self.scoring_factors.get('decision_authority', 25)
        elif authority_level == 'medium':
            score += self.scoring_factors.get('decision_authority', 25) * 0.6
        
        # Project scope scoring
        property_details = lead_info.get('property_details', {})
        if property_details.get('square_footage'):
            sqft = int(property_details['square_footage'])
            if sqft > 10000:  # Large project
                score += self.scoring_factors.get('project_scope', 20)
            elif sqft > 5000:  # Medium project
                score += self.scoring_factors.get('project_scope', 20) * 0.7
            else:  # Small project
                score += self.scoring_factors.get('project_scope', 20) * 0.4
        
        return min(100, int(score))  # Cap at 100

    def _check_disqualifiers(self, content: str) -> Tuple[bool, str]:
        """Check if lead has disqualifying characteristics"""
        content_lower = content.lower()
        
        for disqualifier in self.disqualifiers:
            if disqualifier.lower() in content_lower:
                return True, disqualifier
        
        return False, None

    def _get_recommended_action(self, is_qualified: bool, score: int) -> str:
        """Get recommended action based on qualification"""
        if is_qualified:
            if score >= 85:
                return "immediate_contact"
            elif score >= 70:
                return "priority_follow_up"
            else:
                return "standard_follow_up"
        else:
            return "nurture_campaign"

    def _get_priority_level(self, score: int) -> str:
        """Get priority level based on score"""
        if score >= 85:
            return "high"
        elif score >= 70:
            return "medium"
        elif score >= 50:
            return "low"
        else:
            return "nurture"

    def _get_default_template(self, template_type: str) -> str:
        """Get default template if file not found"""
        templates = {
            'cold_email': """Subject: Professional Roofing Solutions for Your Property

Dear {name},

I hope this message finds you well. I'm reaching out because I understand you may be considering roofing solutions for your property.

As a trusted roofing contractor with extensive experience in commercial and industrial projects, I'd like to offer our expertise to help you find the best solution for your specific needs.

Our comprehensive services include:
• Free professional roof assessments
• Detailed written estimates with no hidden costs
• Premium materials from trusted suppliers
• Expert installation with quality warranties
• Ongoing maintenance and support

I'd be happy to schedule a convenient time to discuss your project requirements and provide a complimentary consultation.

Best regards,
Professional Roofing Team""",
            
            'follow_up': """Subject: Following Up on Your Roofing Project

Hello {name},

I wanted to follow up on our previous conversation about your roofing needs. I understand that choosing the right contractor is an important decision, and I'm here to provide any additional information you might need.

Since we last spoke, I've prepared some additional insights that might be valuable for your project. I'm also happy to arrange a site visit at your convenience to provide a more detailed assessment.

Please let me know if you have any questions or if there's anything specific I can help clarify about your roofing project.

Looking forward to hearing from you.

Best regards,
Professional Roofing Team"""
        }
        
        return templates.get(template_type, templates['cold_email'])