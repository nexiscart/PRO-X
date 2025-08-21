#!/usr/bin/env python3
"""
Bidding Agent for Pro Roofing AI
Specialized in automated bidding, estimation, and proposal generation
"""

import asyncio
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from .base_agent import BaseRoofingAgent, AgentRequest

@dataclass
class MaterialCost:
    """Material cost information"""
    name: str
    unit_cost: float
    unit: str  # per_sqft, per_roll, per_piece, etc.
    coverage: float  # coverage per unit
    waste_factor: float = 0.1  # 10% default waste

@dataclass
class ProjectEstimate:
    """Complete project estimate"""
    project_id: str
    total_cost: float
    material_costs: Dict[str, float]
    labor_costs: Dict[str, float]
    equipment_costs: Dict[str, float]
    permit_costs: float
    disposal_costs: float
    overhead_percentage: float
    profit_margin: float
    estimated_duration: int  # days
    warranty_years: int

class RoofingBiddingAgent(BaseRoofingAgent):
    """AI agent specialized in bidding and estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "bidding_agent")
        
        # Pricing configuration
        self.pricing_config = config.get('pricing', {})
        self.material_rates = self.pricing_config.get('material_rates', {})
        self.labor_multipliers = self.pricing_config.get('labor_multipliers', {})
        self.regional_multipliers = self.pricing_config.get('regional_multipliers', {})
        
        # Blueprint analysis settings
        self.blueprint_config = config.get('blueprint_analysis', {})
        
        # Proposal generation settings
        self.proposal_config = config.get('proposal', {})
        
        # Material database
        self.materials_db = self._initialize_materials_database()
        
        # Labor rates
        self.labor_rates = self._initialize_labor_rates()
        
        # Equipment costs
        self.equipment_costs = self._initialize_equipment_costs()
        
        self.logger.info("💰 Bidding Agent specialized configuration loaded")

    async def _initialize_resources(self):
        """Initialize bidding agent resources"""
        # Load pricing databases
        await self._load_pricing_data()
        
        # Initialize blueprint analysis tools
        await self._initialize_blueprint_tools()
        
        self.logger.info("✅ Bidding agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup bidding agent resources"""
        self.logger.info("🧹 Bidding agent resources cleaned up")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for bidding and estimation"""
        return """You are an expert roofing estimator and bidding specialist with comprehensive knowledge of:

**Technical Expertise:**
- All roofing systems and materials (TPO, EPDM, PVC, Modified Bitumen, Built-up, Metal, etc.)
- Material specifications, performance characteristics, and costs
- Installation techniques and labor requirements
- Building codes and safety regulations
- Blueprint reading and interpretation

**Estimation Accuracy:**
- Precise material quantity calculations
- Labor hour estimation based on project complexity
- Equipment and tool requirements
- Permit and disposal costs
- Regional cost variations and market rates

**Proposal Development:**
- Professional proposal formatting and presentation
- Detailed cost breakdowns with transparency
- Warranty terms and service commitments
- Payment schedules and terms
- Project timeline and milestone planning

**Competitive Strategy:**
- Market-competitive pricing strategies
- Value proposition development
- Risk assessment and contingency planning
- Quality differentiation factors

Always provide detailed, accurate estimates based on industry standards and current market rates. Include comprehensive breakdowns that justify pricing and demonstrate value to customers."""

    async def _process_request_impl(self, request: AgentRequest) -> Dict[str, Any]:
        """Process bidding request"""
        request_type = request.type.lower()
        
        if request_type in ['estimate', 'quote', 'bid']:
            return await self._create_estimate(request)
        elif request_type in ['blueprint', 'blueprint_analysis']:
            return await self._analyze_blueprint(request)
        elif request_type in ['proposal', 'proposal_generation']:
            return await self._generate_proposal(request)
        elif request_type in ['cost_analysis', 'pricing']:
            return await self._analyze_costs(request)
        else:
            # Default to estimate creation
            return await self._create_estimate(request)

    async def _create_estimate(self, request: AgentRequest) -> Dict[str, Any]:
        """Create detailed roofing estimate"""
        content = request.content
        metadata = request.metadata or {}
        
        # Parse project requirements
        project_specs = self._parse_project_specifications(content, metadata)
        
        # Calculate material costs
        material_costs = self._calculate_material_costs(project_specs)
        
        # Calculate labor costs
        labor_costs = self._calculate_labor_costs(project_specs, material_costs)
        
        # Calculate additional costs
        equipment_costs = self._calculate_equipment_costs(project_specs)
        permit_costs = self._estimate_permit_costs(project_specs)
        disposal_costs = self._estimate_disposal_costs(project_specs)
        
        # Apply regional adjustments
        regional_multiplier = self._get_regional_multiplier(project_specs.get('location'))
        
        # Calculate totals
        subtotal = sum(material_costs.values()) + sum(labor_costs.values()) + equipment_costs + permit_costs + disposal_costs
        regional_adjusted = subtotal * regional_multiplier
        
        # Add overhead and profit
        overhead_rate = project_specs.get('overhead_rate', 0.15)  # 15% default
        profit_margin = project_specs.get('profit_margin', 0.20)  # 20% default
        
        overhead_amount = regional_adjusted * overhead_rate
        total_with_overhead = regional_adjusted + overhead_amount
        profit_amount = total_with_overhead * profit_margin
        final_total = total_with_overhead + profit_amount
        
        # Estimate project duration
        estimated_duration = self._estimate_project_duration(project_specs, sum(labor_costs.values()))
        
        # Create estimate object
        estimate = ProjectEstimate(
            project_id=f"EST_{int(datetime.now().timestamp())}",
            total_cost=round(final_total, 2),
            material_costs=material_costs,
            labor_costs=labor_costs,
            equipment_costs=equipment_costs,
            permit_costs=permit_costs,
            disposal_costs=disposal_costs,
            overhead_percentage=overhead_rate,
            profit_margin=profit_margin,
            estimated_duration=estimated_duration,
            warranty_years=self.proposal_config.get('warranty_years', 10)
        )
        
        # Generate estimate presentation
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Create a professional roofing estimate presentation based on these calculations:

Project Specifications: {json.dumps(project_specs, indent=2)}

Cost Breakdown:
- Materials: ${sum(material_costs.values()):,.2f}
- Labor: ${sum(labor_costs.values()):,.2f}
- Equipment: ${equipment_costs:,.2f}
- Permits: ${permit_costs:,.2f}
- Disposal: ${disposal_costs:,.2f}
- Regional Adjustment: {regional_multiplier:.2f}x
- Overhead ({overhead_rate:.1%}): ${overhead_amount:,.2f}
- Profit Margin ({profit_margin:.1%}): ${profit_amount:,.2f}

TOTAL ESTIMATE: ${final_total:,.2f}

Project Duration: {estimated_duration} days
Warranty: {estimate.warranty_years} years

Please format this as a professional estimate that explains the value and includes all necessary details."""}
        ]
        
        response = await self._generate_response(messages)
        
        return {
            'success': True,
            'content': response,
            'confidence': 0.90,
            'metadata': {
                'estimate_ready': True,
                'estimate_data': {
                    'project_id': estimate.project_id,
                    'total_cost': estimate.total_cost,
                    'detailed_breakdown': {
                        'materials': material_costs,
                        'labor': labor_costs,
                        'equipment': equipment_costs,
                        'permits': permit_costs,
                        'disposal': disposal_costs,
                        'overhead': overhead_amount,
                        'profit': profit_amount
                    },
                    'duration_days': estimated_duration,
                    'warranty_years': estimate.warranty_years,
                    'regional_multiplier': regional_multiplier
                },
                'project_specifications': project_specs
            }
        }

    async def _analyze_blueprint(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze blueprint and extract specifications"""
        content = request.content
        metadata = request.metadata or {}
        
        # Parse blueprint information
        blueprint_analysis = self._perform_blueprint_analysis(content, metadata)
        
        # Generate analysis response
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Analyze this roofing blueprint and provide detailed specifications:

Blueprint Information: {content}

Blueprint Analysis Results: {json.dumps(blueprint_analysis, indent=2)}

Please provide:
1. Extracted dimensions and square footage
2. Roofing system requirements
3. Special considerations or challenges
4. Material recommendations
5. Installation complexity assessment
6. Code compliance requirements"""}
        ]
        
        response = await self._generate_response(messages)
        
        confidence = blueprint_analysis.get('confidence_score', 0.8)
        manual_review = confidence < self.blueprint_config.get('confidence_threshold', 0.8)
        
        return {
            'success': True,
            'content': response,
            'confidence': confidence,
            'metadata': {
                'blueprint_analyzed': True,
                'analysis_data': blueprint_analysis,
                'manual_review_required': manual_review,
                'extracted_specs': blueprint_analysis.get('specifications', {}),
                'recommended_materials': blueprint_analysis.get('recommended_materials', [])
            }
        }

    async def _generate_proposal(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate comprehensive project proposal"""
        content = request.content
        metadata = request.metadata or {}
        
        # Get estimate data if provided
        estimate_data = metadata.get('estimate_data', {})
        
        # Generate professional proposal
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Create a comprehensive roofing project proposal:

Project Requirements: {content}

Estimate Data: {json.dumps(estimate_data, indent=2) if estimate_data else 'To be calculated'}

Include:
1. Executive Summary
2. Project Scope and Specifications
3. Materials and Installation Process
4. Timeline and Milestones
5. Pricing and Payment Terms
6. Warranty and Service Commitments
7. Company Qualifications
8. Next Steps

Make it professional, detailed, and compelling while maintaining transparency."""}
        ]
        
        response = await self._generate_response(messages)
        
        return {
            'success': True,
            'content': response,
            'confidence': 0.85,
            'metadata': {
                'proposal_ready': True,
                'includes_pricing': bool(estimate_data),
                'warranty_included': self.proposal_config.get('include_warranty', True),
                'maintenance_plan': self.proposal_config.get('include_maintenance_plan', True),
                'payment_terms': self.proposal_config.get('payment_terms', '30% down, 40% at material delivery, 30% on completion')
            }
        }

    async def _analyze_costs(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze and optimize project costs"""
        content = request.content
        metadata = request.metadata or {}
        
        # Perform cost analysis
        cost_analysis = self._perform_cost_analysis(content, metadata)
        
        # Generate cost optimization recommendations
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""Analyze the costs for this roofing project and provide optimization recommendations:

Project Details: {content}

Cost Analysis: {json.dumps(cost_analysis, indent=2)}

Provide:
1. Cost breakdown analysis
2. Market comparison
3. Cost optimization opportunities
4. Value engineering suggestions
5. Risk factors affecting costs
6. Alternative material options
7. Timing considerations for cost savings"""}
        ]
        
        response = await self._generate_response(messages)
        
        return {
            'success': True,
            'content': response,
            'confidence': 0.80,
            'metadata': {
                'cost_analysis': cost_analysis,
                'optimization_opportunities': cost_analysis.get('savings_potential', 0),
                'risk_factors': cost_analysis.get('risk_factors', []),
                'alternative_options': cost_analysis.get('alternatives', [])
            }
        }

    def _parse_project_specifications(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse project specifications from content"""
        specs = {
            'square_footage': self._extract_square_footage(content),
            'roofing_system': self._identify_roofing_system(content),
            'building_type': self._identify_building_type(content),
            'complexity': self._assess_complexity(content),
            'location': metadata.get('location', self._extract_location(content)),
            'timeline': self._extract_timeline(content),
            'special_requirements': self._extract_special_requirements(content)
        }
        
        return specs

    def _extract_square_footage(self, content: str) -> Optional[float]:
        """Extract square footage from content"""
        # Look for various formats of square footage
        patterns = [
            r'(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft)',
            r'(\d+(?:,\d+)?(?:\.\d+)?)(?:\s*x\s*|\s*by\s*)(\d+(?:,\d+)?(?:\.\d+)?)',  # dimensions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):  # dimensions
                    length = float(matches[0][0].replace(',', ''))
                    width = float(matches[0][1].replace(',', ''))
                    return length * width
                else:  # direct square footage
                    return float(matches[0].replace(',', ''))
        
        return None

    def _identify_roofing_system(self, content: str) -> str:
        """Identify the roofing system type"""
        content_lower = content.lower()
        
        systems = {
            'tpo': ['tpo', 'thermoplastic polyolefin'],
            'epdm': ['epdm', 'rubber', 'ethylene propylene'],
            'pvc': ['pvc', 'polyvinyl chloride'],
            'modified_bitumen': ['modified bitumen', 'mod bit', 'sbs', 'app'],
            'built_up': ['built-up', 'built up', 'bur', 'tar and gravel'],
            'metal': ['metal', 'steel', 'aluminum', 'standing seam'],
            'single_ply': ['single ply', 'membrane']
        }
        
        for system, keywords in systems.items():
            if any(keyword in content_lower for keyword in keywords):
                return system
        
        return 'unspecified'

    def _identify_building_type(self, content: str) -> str:
        """Identify building type"""
        content_lower = content.lower()
        
        building_types = {
            'warehouse': ['warehouse', 'distribution', 'storage'],
            'office': ['office', 'corporate', 'headquarters'],
            'retail': ['retail', 'store', 'shopping'],
            'industrial': ['industrial', 'manufacturing', 'factory', 'plant'],
            'educational': ['school', 'university', 'college', 'educational'],
            'healthcare': ['hospital', 'medical', 'clinic'],
            'residential': ['residential', 'apartment', 'condo', 'home']
        }
        
        for building_type, keywords in building_types.items():
            if any(keyword in content_lower for keyword in keywords):
                return building_type
        
        return 'commercial'

    def _assess_complexity(self, content: str) -> str:
        """Assess project complexity"""
        content_lower = content.lower()
        
        high_complexity_indicators = [
            'multi-level', 'complex', 'irregular', 'custom', 'challenging',
            'multiple roofs', 'difficult access', 'crane required'
        ]
        
        moderate_complexity_indicators = [
            'standard', 'typical', 'regular', 'straightforward'
        ]
        
        if any(indicator in content_lower for indicator in high_complexity_indicators):
            return 'high'
        elif any(indicator in content_lower for indicator in moderate_complexity_indicators):
            return 'moderate'
        else:
            return 'simple'

    def _extract_location(self, content: str) -> Optional[str]:
        """Extract location information"""
        # Simple state/city extraction
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, ST
            r'\b([A-Z]{2})\b',  # State abbreviation
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, content)
            if matches:
                if isinstance(matches[0], tuple):
                    return f"{matches[0][0]}, {matches[0][1]}"
                else:
                    return matches[0]
        
        return None

    def _extract_timeline(self, content: str) -> Optional[str]:
        """Extract project timeline"""
        timeline_patterns = [
            r'within\s+(\d+)\s+(?:days?|weeks?|months?)',
            r'by\s+([A-Z][a-z]+(?:\s+\d+)?)',  # by March, by March 15
            r'(\d+)[-\s](?:day|week|month)\s+(?:project|timeline|completion)'
        ]
        
        for pattern in timeline_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None

    def _extract_special_requirements(self, content: str) -> List[str]:
        """Extract special project requirements"""
        content_lower = content.lower()
        requirements = []
        
        special_reqs = [
            'fall protection', 'safety requirements', 'confined space',
            'after hours', 'occupied building', 'minimal disruption',
            'expedited', 'emergency', 'warranty extension',
            'green roof', 'solar ready', 'energy efficient'
        ]
        
        for req in special_reqs:
            if req in content_lower:
                requirements.append(req)
        
        return requirements

    def _calculate_material_costs(self, project_specs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate material costs based on specifications"""
        costs = {}
        square_footage = project_specs.get('square_footage', 0)
        roofing_system = project_specs.get('roofing_system', 'tpo')
        
        if square_footage <= 0:
            return {'materials_estimate': 5000.0}  # Default estimate
        
        # Primary roofing material
        material_rate = self.material_rates.get(roofing_system, 8.50)
        costs['primary_material'] = square_footage * material_rate
        
        # Insulation (if needed)
        if project_specs.get('complexity') != 'simple':
            costs['insulation'] = square_footage * 2.50
        
        # Fasteners and accessories
        costs['fasteners'] = square_footage * 0.75
        
        # Flashing and trim
        perimeter_estimate = math.sqrt(square_footage) * 4  # Rough perimeter estimate
        costs['flashing'] = perimeter_estimate * 12.00  # per linear foot
        
        # Sealants and adhesives
        costs['sealants'] = square_footage * 0.50
        
        # Add waste factor
        subtotal = sum(costs.values())
        waste_factor = 0.10  # 10% waste
        costs['waste_allowance'] = subtotal * waste_factor
        
        return costs

    def _calculate_labor_costs(self, project_specs: Dict[str, Any], material_costs: Dict[str, float]) -> Dict[str, float]:
        """Calculate labor costs"""
        costs = {}
        square_footage = project_specs.get('square_footage', 0)
        complexity = project_specs.get('complexity', 'simple')
        
        if square_footage <= 0:
            return {'labor_estimate': 3000.0}
        
        # Base labor rate per sq ft
        base_rate = 4.50  # per sq ft
        
        # Complexity multiplier
        multiplier = self.labor_multipliers.get(complexity, 1.0)
        
        # Installation labor
        costs['installation'] = square_footage * base_rate * multiplier
        
        # Tear-off (if replacement)
        if 'replacement' in project_specs.get('special_requirements', []):
            costs['tearoff'] = square_footage * 1.50
        
        # Setup and cleanup
        costs['setup_cleanup'] = costs['installation'] * 0.15
        
        # Supervision
        costs['supervision'] = sum(costs.values()) * 0.10
        
        return costs

    def _calculate_equipment_costs(self, project_specs: Dict[str, Any]) -> float:
        """Calculate equipment rental costs"""
        square_footage = project_specs.get('square_footage', 0)
        complexity = project_specs.get('complexity', 'simple')
        
        base_equipment_cost = 1500.0  # Base daily equipment cost
        
        # Complexity adjustments
        if complexity == 'high':
            base_equipment_cost *= 1.5
        elif complexity == 'moderate':
            base_equipment_cost *= 1.2
        
        # Size adjustments
        if square_footage > 20000:
            base_equipment_cost *= 1.3
        elif square_footage > 10000:
            base_equipment_cost *= 1.1
        
        return base_equipment_cost

    def _estimate_permit_costs(self, project_specs: Dict[str, Any]) -> float:
        """Estimate permit costs"""
        square_footage = project_specs.get('square_footage', 0)
        building_type = project_specs.get('building_type', 'commercial')
        
        # Base permit cost
        base_cost = 500.0
        
        # Size-based costs
        if square_footage > 10000:
            base_cost += (square_footage - 10000) * 0.05
        
        # Building type adjustments
        if building_type in ['industrial', 'healthcare']:
            base_cost *= 1.5
        elif building_type in ['educational', 'office']:
            base_cost *= 1.2
        
        return base_cost

    def _estimate_disposal_costs(self, project_specs: Dict[str, Any]) -> float:
        """Estimate disposal costs"""
        square_footage = project_specs.get('square_footage', 0)
        special_requirements = project_specs.get('special_requirements', [])
        
        # Base disposal cost per sq ft
        disposal_rate = 0.50
        
        # Special handling
        if any('hazmat' in req.lower() or 'asbestos' in req.lower() for req in special_requirements):
            disposal_rate *= 3.0
        
        return square_footage * disposal_rate

    def _get_regional_multiplier(self, location: Optional[str]) -> float:
        """Get regional cost multiplier"""
        if not location:
            return 1.0
        
        location_lower = location.lower()
        
        # Check for known regions
        for region, multiplier in self.regional_multipliers.items():
            if region.lower() in location_lower:
                return multiplier
        
        return 1.0  # Default multiplier

    def _estimate_project_duration(self, project_specs: Dict[str, Any], labor_cost: float) -> int:
        """Estimate project duration in days"""
        square_footage = project_specs.get('square_footage', 0)
        complexity = project_specs.get('complexity', 'simple')
        
        # Base production rate (sq ft per day)
        base_rate = 2000  # sq ft per day for simple projects
        
        if complexity == 'high':
            base_rate = 800
        elif complexity == 'moderate':
            base_rate = 1200
        
        # Calculate days
        days = max(1, math.ceil(square_footage / base_rate))
        
        # Add buffer for weather and contingencies
        days = int(days * 1.2)
        
        return days

    def _initialize_materials_database(self) -> Dict[str, MaterialCost]:
        """Initialize materials database"""
        return {
            'tpo_60mil': MaterialCost('TPO 60mil', 3.25, 'per_sqft', 1.0),
            'tpo_80mil': MaterialCost('TPO 80mil', 4.10, 'per_sqft', 1.0),
            'epdm_60mil': MaterialCost('EPDM 60mil', 2.85, 'per_sqft', 1.0),
            'pvc_60mil': MaterialCost('PVC 60mil', 4.50, 'per_sqft', 1.0),
            'polyiso_insulation': MaterialCost('Polyiso Insulation', 2.25, 'per_sqft', 1.0),
            'coverboard': MaterialCost('Cover Board', 1.15, 'per_sqft', 1.0),
            'fasteners': MaterialCost('Fasteners', 0.75, 'per_sqft', 1.0),
        }

    def _initialize_labor_rates(self) -> Dict[str, float]:
        """Initialize labor rates"""
        return {
            'foreman': 35.00,  # per hour
            'journeyman': 28.00,  # per hour
            'apprentice': 20.00,  # per hour
            'laborer': 18.00,  # per hour
        }

    def _initialize_equipment_costs(self) -> Dict[str, float]:
        """Initialize equipment costs"""
        return {
            'crane_daily': 800.00,
            'kettle_daily': 150.00,
            'compressor_daily': 100.00,
            'hot_air_welder': 75.00,
            'safety_equipment': 50.00,
        }

    async def _load_pricing_data(self):
        """Load pricing data from external sources"""
        # Placeholder for loading real-time pricing data
        self.logger.info("📊 Pricing data loaded")

    async def _initialize_blueprint_tools(self):
        """Initialize blueprint analysis tools"""
        # Placeholder for blueprint analysis tool initialization
        self.logger.info("📐 Blueprint analysis tools initialized")

    def _perform_blueprint_analysis(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform blueprint analysis (placeholder implementation)"""
        # This would integrate with actual blueprint analysis tools
        return {
            'confidence_score': 0.85,
            'specifications': {
                'estimated_sqft': self._extract_square_footage(content),
                'roofing_system': self._identify_roofing_system(content),
                'complexity': self._assess_complexity(content)
            },
            'recommended_materials': ['TPO 60mil', 'Polyiso Insulation', 'Cover Board'],
            'special_considerations': ['Access limitations', 'Code compliance required']
        }

    def _perform_cost_analysis(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed cost analysis"""
        # Parse project for cost analysis
        project_specs = self._parse_project_specifications(content, metadata)
        
        # Calculate costs
        material_costs = self._calculate_material_costs(project_specs)
        labor_costs = self._calculate_labor_costs(project_specs, material_costs)
        
        total_materials = sum(material_costs.values())
        total_labor = sum(labor_costs.values())
        total_cost = total_materials + total_labor
        
        # Analyze cost components
        analysis = {
            'cost_breakdown': {
                'materials_percentage': (total_materials / total_cost) * 100 if total_cost > 0 else 0,
                'labor_percentage': (total_labor / total_cost) * 100 if total_cost > 0 else 0,
                'total_estimated': total_cost
            },
            'savings_potential': total_cost * 0.10,  # 10% potential savings
            'risk_factors': [
                'Weather delays',
                'Material price volatility',
                'Labor availability',
                'Permit delays'
            ],
            'alternatives': [
                {'option': 'Alternative material', 'savings': total_cost * 0.08},
                {'option': 'Off-season timing', 'savings': total_cost * 0.05}
            ]
        }
        
        return analysis