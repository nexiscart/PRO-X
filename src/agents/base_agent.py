#!/usr/bin/env python3
"""
Base Agent for Pro Roofing AI
Foundation class for all AI agents
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
import openai
from pathlib import Path

@dataclass
class AgentRequest:
    """Standard request format for agents"""
    id: str
    type: str
    content: str
    priority: int = 1
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class BaseRoofingAgent(ABC):
    """Base class for all roofing AI agents"""
    
    def __init__(self, config: Dict[str, Any], agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agents.{agent_name}")
        
        # Agent state
        self.is_active = False
        self.request_count = 0
        self.error_count = 0
        self.last_activity = None
        
        # Performance tracking
        self.response_times = []
        self.success_rate = 1.0
        
        # Model configuration
        self.model_config = self._setup_model_config()
        
        # Initialize OpenAI client if needed
        self.openai_client = None
        if self.config.get('use_openai_fallback', True):
            try:
                openai.api_key = self.config.get('openai_api_key', '')
                self.openai_client = openai
                self.logger.info("✅ OpenAI client initialized for fallback")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to initialize OpenAI client: {e}")
        
        self.logger.info(f"🤖 {self.agent_name} agent initialized")

    def _setup_model_config(self) -> Dict[str, Any]:
        """Setup model configuration for the agent"""
        # Get global model config
        global_config = self.config.get('global', {})
        
        # Get agent-specific config
        agent_config = self.config.get('model', {})
        
        # Merge configurations (agent-specific overrides global)
        model_config = {
            'name': global_config.get('model', {}).get('name', 'roofing-llama2-7b-lora'),
            'path': global_config.get('model', {}).get('path', './models/final'),
            'fallback_model': global_config.get('model', {}).get('fallback_model', 'gpt-4'),
            'temperature': agent_config.get('temperature', 0.7),
            'max_tokens': agent_config.get('max_tokens', 1000),
            'top_p': agent_config.get('top_p', 0.9),
            'frequency_penalty': agent_config.get('frequency_penalty', 0.1),
            'presence_penalty': agent_config.get('presence_penalty', 0.1)
        }
        
        # Update with global generation defaults
        generation_config = global_config.get('generation', {})
        for key, value in generation_config.items():
            if key not in model_config:
                model_config[key] = value
        
        return model_config

    async def start(self):
        """Start the agent"""
        self.is_active = True
        self.last_activity = datetime.now()
        await self._initialize_resources()
        self.logger.info(f"🚀 {self.agent_name} agent started")

    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        await self._cleanup_resources()
        self.logger.info(f"🛑 {self.agent_name} agent stopped")

    @abstractmethod
    async def _initialize_resources(self):
        """Initialize agent-specific resources"""
        pass

    @abstractmethod
    async def _cleanup_resources(self):
        """Cleanup agent-specific resources"""
        pass

    @abstractmethod
    async def _process_request_impl(self, request: AgentRequest) -> Dict[str, Any]:
        """Process request implementation (agent-specific)"""
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass

    async def process_request(self, request: AgentRequest) -> Dict[str, Any]:
        """Process a request (with common error handling and metrics)"""
        if not self.is_active:
            raise RuntimeError(f"Agent {self.agent_name} is not active")
        
        start_time = time.time()
        self.request_count += 1
        self.last_activity = datetime.now()
        
        try:
            self.logger.info(f"📥 Processing request {request.id}: {request.type}")
            
            # Validate request
            self._validate_request(request)
            
            # Process request
            result = await self._process_request_impl(request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.response_times.append(processing_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            # Update success rate
            self.success_rate = (self.success_rate * (self.request_count - 1) + 1.0) / self.request_count
            
            self.logger.info(f"✅ Request {request.id} processed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            
            # Update success rate
            self.success_rate = (self.success_rate * (self.request_count - 1) + 0.0) / self.request_count
            
            self.logger.error(f"❌ Error processing request {request.id}: {e}")
            
            # Return error response
            return {
                'success': False,
                'content': f"Error processing request: {str(e)}",
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'processing_time': processing_time
                }
            }

    def _validate_request(self, request: AgentRequest):
        """Validate request format and content"""
        if not request.id:
            raise ValueError("Request ID is required")
        
        if not request.content:
            raise ValueError("Request content is required")
        
        if len(request.content) > 10000:
            raise ValueError("Request content too long (max 10,000 characters)")

    async def _generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using the configured model or fallback"""
        try:
            # Try local model first
            response = await self._generate_with_local_model(messages, **kwargs)
            return response
        except Exception as e:
            self.logger.warning(f"⚠️  Local model failed: {e}")
            
            # Fallback to OpenAI if available
            if self.openai_client:
                return await self._generate_with_openai(messages, **kwargs)
            else:
                raise RuntimeError("No available model for response generation")

    async def _generate_with_local_model(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with local model (placeholder implementation)"""
        # This would integrate with the actual fine-tuned model
        # For now, return a placeholder response
        self.logger.info("🧠 Using local roofing model for response generation")
        
        # Extract the user's question
        user_message = messages[-1]['content'] if messages else "No input provided"
        
        # Simple rule-based response for demonstration
        if 'estimate' in user_message.lower() or 'quote' in user_message.lower():
            return self._generate_estimate_response(user_message)
        elif 'lead' in user_message.lower() or 'prospect' in user_message.lower():
            return self._generate_lead_response(user_message)
        else:
            return f"As a roofing expert, I understand you're asking about: {user_message[:100]}... Let me provide you with professional guidance based on industry best practices."

    async def _generate_with_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with OpenAI as fallback"""
        self.logger.info("🔄 Using OpenAI fallback for response generation")
        
        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model=self.model_config.get('fallback_model', 'gpt-4'),
                messages=messages,
                temperature=kwargs.get('temperature', self.model_config.get('temperature', 0.7)),
                max_tokens=kwargs.get('max_tokens', self.model_config.get('max_tokens', 1000)),
                top_p=kwargs.get('top_p', self.model_config.get('top_p', 0.9))
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI fallback failed: {e}")
            raise

    def _generate_estimate_response(self, user_message: str) -> str:
        """Generate estimate-related response"""
        return """I'd be happy to help you with a roofing estimate. To provide an accurate quote, I'll need some key information:

**Property Details:**
- Building size and type (square footage)
- Current roofing material and condition
- Building height and accessibility
- Location (for local code requirements)

**Project Scope:**
- Full replacement or repair?
- Preferred roofing material (TPO, EPDM, PVC, etc.)
- Any special requirements or upgrades?

**Timeline:**
- Desired completion date
- Any seasonal considerations

Once I have these details, I can provide a detailed estimate including materials, labor, permits, and disposal costs. Would you like to provide more specifics about your project?"""

    def _generate_lead_response(self, user_message: str) -> str:
        """Generate lead-related response"""
        return """Thank you for your interest in our roofing services! I'm here to help you find the perfect roofing solution.

**Let's start with understanding your needs:**

1. **Property Type:** Commercial, industrial, or residential?
2. **Current Situation:** New construction, replacement, or repair?
3. **Timeline:** When are you looking to start the project?
4. **Budget Range:** Do you have a preliminary budget in mind?

**What we offer:**
- Free comprehensive roof inspections
- Detailed written estimates
- Multiple material options
- Professional installation with warranty
- Maintenance and repair services

I'll personally ensure you receive expert guidance throughout the process. What specific roofing challenges are you facing?"""

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent"""
        health_status = {
            "agent_name": self.agent_name,
            "status": "healthy" if self.is_active else "inactive",
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "metrics": {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "success_rate": self.success_rate,
                "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
            }
        }
        
        # Check if agent is responsive
        if self.last_activity:
            inactive_minutes = (datetime.now() - self.last_activity).total_seconds() / 60
            if inactive_minutes > 60:  # Inactive for more than 1 hour
                health_status["status"] = "stale"
                health_status["warning"] = f"No activity for {inactive_minutes:.1f} minutes"
        
        # Check error rate
        if self.request_count > 0 and (self.error_count / self.request_count) > 0.1:
            health_status["status"] = "degraded"
            health_status["warning"] = f"High error rate: {self.error_count}/{self.request_count}"
        
        return health_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "agent_name": self.agent_name,
            "is_active": self.is_active,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "response_time_stats": {
                "average": sum(self.response_times) / len(self.response_times) if self.response_times else 0.0,
                "min": min(self.response_times) if self.response_times else 0.0,
                "max": max(self.response_times) if self.response_times else 0.0,
                "recent_count": len(self.response_times)
            },
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }

    def save_agent_state(self, output_path: str):
        """Save agent state to file"""
        state = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "is_active": self.is_active,
            "metrics": self.get_performance_metrics(),
            "config": self.config
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"💾 Agent state saved to: {output_path}")

    def load_agent_state(self, input_path: str) -> bool:
        """Load agent state from file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore metrics
            metrics = state.get('metrics', {})
            self.request_count = metrics.get('total_requests', 0)
            self.error_count = metrics.get('error_count', 0)
            self.success_rate = metrics.get('success_rate', 1.0)
            
            # Restore timestamps
            last_activity_str = metrics.get('last_activity')
            if last_activity_str:
                self.last_activity = datetime.fromisoformat(last_activity_str)
            
            self.logger.info(f"📂 Agent state loaded from: {input_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load agent state: {e}")
            return False