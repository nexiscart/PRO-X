#!/usr/bin/env python3
"""
Agent Orchestrator for Pro Roofing AI
Master controller that manages and coordinates all AI agents
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import uuid

from .lead_agent import RoofingLeadAgent
from .bidding_agent import RoofingBiddingAgent
from .email_agent import RoofingEmailAgent
from .supplier_agent import RoofingSupplierAgent
from .analytics_agent import RoofingAnalyticsAgent

@dataclass
class AgentRequest:
    """Represents a request to be processed by agents"""
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

@dataclass
class AgentResponse:
    """Represents a response from an agent"""
    request_id: str
    agent_name: str
    success: bool
    content: str
    confidence: float
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class RoofingAIOrchestrator:
    """Master orchestrator for all roofing AI agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Agent instances
        self.agents = {}
        self.agent_status = {}
        
        # Request processing
        self.request_queue = asyncio.Queue()
        self.response_history = []
        self.active_requests = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "agent_utilization": {}
        }
        
        # Initialize agents
        self._initialize_agents()
        
        self.logger.info("🎯 Roofing AI Orchestrator initialized")

    def _initialize_agents(self):
        """Initialize all AI agents"""
        self.logger.info("🤖 Initializing AI agents...")
        
        try:
            # Lead Generation Agent
            self.agents['lead'] = RoofingLeadAgent(self.config.get('lead_agent', {}))
            self.agent_status['lead'] = {'active': True, 'requests_processed': 0}
            
            # Bidding Agent
            self.agents['bidding'] = RoofingBiddingAgent(self.config.get('bidding_agent', {}))
            self.agent_status['bidding'] = {'active': True, 'requests_processed': 0}
            
            # Email Agent
            self.agents['email'] = RoofingEmailAgent(self.config.get('email_agent', {}))
            self.agent_status['email'] = {'active': True, 'requests_processed': 0}
            
            # Supplier Agent
            self.agents['supplier'] = RoofingSupplierAgent(self.config.get('supplier_agent', {}))
            self.agent_status['supplier'] = {'active': True, 'requests_processed': 0}
            
            # Analytics Agent
            self.agents['analytics'] = RoofingAnalyticsAgent(self.config.get('analytics_agent', {}))
            self.agent_status['analytics'] = {'active': True, 'requests_processed': 0}
            
            self.logger.info(f"✅ Initialized {len(self.agents)} agents successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agents: {e}")
            raise

    async def start(self):
        """Start the orchestrator and all agents"""
        self.logger.info("🚀 Starting Roofing AI Orchestrator...")
        
        # Start all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.start()
                self.logger.info(f"✅ Started {agent_name} agent")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {agent_name} agent: {e}")
                self.agent_status[agent_name]['active'] = False
        
        # Start request processing loop
        asyncio.create_task(self._process_request_queue())
        
        self.logger.info("🎉 Orchestrator started successfully")

    async def stop(self):
        """Stop the orchestrator and all agents"""
        self.logger.info("🛑 Stopping Roofing AI Orchestrator...")
        
        # Stop all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.stop()
                self.logger.info(f"✅ Stopped {agent_name} agent")
            except Exception as e:
                self.logger.error(f"❌ Error stopping {agent_name} agent: {e}")
        
        self.logger.info("✅ Orchestrator stopped")

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request by routing to appropriate agent"""
        self.logger.info(f"📥 Processing request {request.id}: {request.type}")
        
        start_time = datetime.now()
        
        try:
            # Route request to appropriate agent
            agent_name = self._route_request(request)
            
            if agent_name not in self.agents:
                raise ValueError(f"No agent found for request type: {request.type}")
            
            if not self.agent_status[agent_name]['active']:
                raise ValueError(f"Agent {agent_name} is not active")
            
            # Process request with selected agent
            agent = self.agents[agent_name]
            result = await agent.process_request(request)
            
            # Create response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = AgentResponse(
                request_id=request.id,
                agent_name=agent_name,
                success=True,
                content=result.get('content', ''),
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                processing_time=processing_time
            )
            
            # Update metrics
            self._update_metrics(agent_name, processing_time, True)
            
            # Store response
            self.response_history.append(response)
            
            # Trigger follow-up actions if needed
            await self._handle_follow_up_actions(request, response)
            
            self.logger.info(f"✅ Request {request.id} processed successfully by {agent_name}")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            error_response = AgentResponse(
                request_id=request.id,
                agent_name="orchestrator",
                success=False,
                content=f"Error processing request: {str(e)}",
                confidence=0.0,
                processing_time=processing_time
            )
            
            self._update_metrics("error", processing_time, False)
            self.response_history.append(error_response)
            
            self.logger.error(f"❌ Error processing request {request.id}: {e}")
            
            return error_response

    def _route_request(self, request: AgentRequest) -> str:
        """Route request to appropriate agent based on type and content"""
        request_type = request.type.lower()
        content = request.content.lower()
        
        # Explicit routing based on request type
        if request_type in ['lead', 'lead_generation', 'prospect']:
            return 'lead'
        elif request_type in ['bid', 'bidding', 'estimate', 'quote']:
            return 'bidding'
        elif request_type in ['email', 'communication', 'message']:
            return 'email'
        elif request_type in ['supplier', 'procurement', 'inventory']:
            return 'supplier'
        elif request_type in ['analytics', 'report', 'analysis']:
            return 'analytics'
        
        # Content-based routing using keywords
        routing_config = self.config.get('orchestrator', {}).get('routing', {})
        
        # Check lead generation keywords
        lead_keywords = routing_config.get('lead_generation_keywords', [])
        if any(keyword in content for keyword in lead_keywords):
            return 'lead'
        
        # Check bidding keywords
        bidding_keywords = routing_config.get('bidding_keywords', [])
        if any(keyword in content for keyword in bidding_keywords):
            return 'bidding'
        
        # Check technical keywords (route to bidding for technical questions)
        technical_keywords = routing_config.get('technical_keywords', [])
        if any(keyword in content for keyword in technical_keywords):
            return 'bidding'
        
        # Check supplier keywords
        supplier_keywords = routing_config.get('supplier_keywords', [])
        if any(keyword in content for keyword in supplier_keywords):
            return 'supplier'
        
        # Default to lead agent for general inquiries
        return 'lead'

    async def _handle_follow_up_actions(self, request: AgentRequest, response: AgentResponse):
        """Handle follow-up actions based on response"""
        if not response.success:
            return
        
        agent_name = response.agent_name
        
        # Lead agent follow-ups
        if agent_name == 'lead':
            metadata = response.metadata
            if metadata.get('qualified', False):
                # Create bidding request for qualified leads
                bidding_request = AgentRequest(
                    id=str(uuid.uuid4()),
                    type='estimate',
                    content=f"Create estimate for qualified lead: {request.content}",
                    metadata={
                        'original_request_id': request.id,
                        'lead_data': metadata
                    }
                )
                await self.request_queue.put(bidding_request)
                
                # Send welcome email
                email_request = AgentRequest(
                    id=str(uuid.uuid4()),
                    type='email',
                    content=f"Send welcome email to qualified lead",
                    metadata={
                        'template': 'welcome',
                        'lead_data': metadata,
                        'original_request_id': request.id
                    }
                )
                await self.request_queue.put(email_request)
        
        # Bidding agent follow-ups
        elif agent_name == 'bidding':
            metadata = response.metadata
            if metadata.get('estimate_ready', False):
                # Send estimate email
                email_request = AgentRequest(
                    id=str(uuid.uuid4()),
                    type='email',
                    content=f"Send estimate to customer",
                    metadata={
                        'template': 'estimate_ready',
                        'estimate_data': metadata,
                        'original_request_id': request.id
                    }
                )
                await self.request_queue.put(email_request)
        
        # Email agent follow-ups
        elif agent_name == 'email':
            # Log email activity for analytics
            analytics_request = AgentRequest(
                id=str(uuid.uuid4()),
                type='analytics',
                content=f"Log email activity",
                metadata={
                    'activity_type': 'email_sent',
                    'email_data': response.metadata,
                    'original_request_id': request.id
                }
            )
            await self.request_queue.put(analytics_request)

    async def _process_request_queue(self):
        """Background task to process request queue"""
        while True:
            try:
                # Get request from queue
                request = await self.request_queue.get()
                
                # Process request
                response = await self.process_request(request)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"❌ Error in request queue processing: {e}")
                await asyncio.sleep(1)  # Prevent rapid error loops

    def _update_metrics(self, agent_name: str, processing_time: float, success: bool):
        """Update performance metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        total_time = self.metrics['average_response_time'] * (self.metrics['total_requests'] - 1)
        self.metrics['average_response_time'] = (total_time + processing_time) / self.metrics['total_requests']
        
        # Update agent utilization
        if agent_name not in self.metrics['agent_utilization']:
            self.metrics['agent_utilization'][agent_name] = 0
        
        self.metrics['agent_utilization'][agent_name] += 1
        
        # Update agent status
        if agent_name in self.agent_status:
            self.agent_status[agent_name]['requests_processed'] += 1

    async def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "orchestrator": {
                "active": True,
                "queue_size": self.request_queue.qsize(),
                "active_requests": len(self.active_requests)
            },
            "agents": self.agent_status,
            "metrics": self.metrics,
            "recent_responses": len([r for r in self.response_history if 
                                   (datetime.now() - r.timestamp).seconds < 3600])  # Last hour
        }

    async def get_agent_performance(self) -> Dict[str, Any]:
        """Get detailed agent performance metrics"""
        performance = {}
        
        for agent_name, agent in self.agents.items():
            # Get recent responses for this agent
            recent_responses = [r for r in self.response_history if 
                             r.agent_name == agent_name and 
                             (datetime.now() - r.timestamp).seconds < 3600]
            
            if recent_responses:
                avg_response_time = sum(r.processing_time for r in recent_responses) / len(recent_responses)
                success_rate = sum(1 for r in recent_responses if r.success) / len(recent_responses)
                avg_confidence = sum(r.confidence for r in recent_responses) / len(recent_responses)
            else:
                avg_response_time = 0.0
                success_rate = 0.0
                avg_confidence = 0.0
            
            performance[agent_name] = {
                "requests_last_hour": len(recent_responses),
                "average_response_time": avg_response_time,
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "status": "active" if self.agent_status[agent_name]['active'] else "inactive"
            }
        
        return performance

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_status = {
            "orchestrator": "healthy",
            "agents": {},
            "overall_health": "healthy"
        }
        
        unhealthy_agents = 0
        
        for agent_name, agent in self.agents.items():
            try:
                # Perform agent health check
                agent_health = await agent.health_check()
                health_status["agents"][agent_name] = agent_health
                
                if agent_health.get("status") != "healthy":
                    unhealthy_agents += 1
                    
            except Exception as e:
                health_status["agents"][agent_name] = {
                    "status": "error",
                    "error": str(e)
                }
                unhealthy_agents += 1
        
        # Determine overall health
        if unhealthy_agents == 0:
            health_status["overall_health"] = "healthy"
        elif unhealthy_agents < len(self.agents) / 2:
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        return health_status

    def save_metrics_report(self, output_path: str):
        """Save comprehensive metrics report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "agent_status": self.agent_status,
            "recent_responses": [
                {
                    "agent": r.agent_name,
                    "success": r.success,
                    "processing_time": r.processing_time,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.response_history[-100:]  # Last 100 responses
            ]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Metrics report saved to: {output_path}")


async def main():
    """Test the orchestrator"""
    config = {
        "orchestrator": {
            "routing": {
                "lead_generation_keywords": ["lead", "prospect", "customer"],
                "bidding_keywords": ["bid", "estimate", "quote", "price"],
                "technical_keywords": ["material", "installation", "specification"],
                "supplier_keywords": ["supplier", "vendor", "material"]
            }
        }
    }
    
    # Create orchestrator
    orchestrator = RoofingAIOrchestrator(config)
    
    try:
        # Start orchestrator
        await orchestrator.start()
        
        # Test requests
        test_requests = [
            AgentRequest(id="1", type="lead", content="I need a new roof for my warehouse"),
            AgentRequest(id="2", type="bid", content="Create estimate for 10,000 sq ft TPO roof"),
            AgentRequest(id="3", type="email", content="Send follow-up to John Smith"),
        ]
        
        # Process test requests
        for request in test_requests:
            response = await orchestrator.process_request(request)
            print(f"Request {request.id}: {response.success} - {response.content[:100]}")
        
        # Get status
        status = await orchestrator.get_status()
        print("Orchestrator Status:", json.dumps(status, indent=2))
        
        # Health check
        health = await orchestrator.health_check()
        print("Health Check:", json.dumps(health, indent=2))
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())