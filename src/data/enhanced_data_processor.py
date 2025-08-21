#!/usr/bin/env python3
"""
Enhanced Data Processor for Pro Roofing AI
Handles the massive roofing dataset collection
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import random

class EnhancedRoofingDataProcessor:
    """Process the comprehensive roofing dataset collection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.training_data = []
        
        # Roofing expert system prompt
        self.roofing_system_prompt = """You are the ultimate commercial roofing AI expert with comprehensive knowledge of:

**Technical Expertise:**
- All roofing systems (TPO, EPDM, PVC, Modified Bitumen, Built-up, Metal, etc.)
- Installation techniques and best practices
- Material specifications and performance characteristics
- Building codes and safety protocols
- Weather resistance and energy efficiency

**Business Intelligence:**
- Supplier relationships and pricing strategies
- Commercial construction contracts and bidding
- Project management and scheduling
- Quality control and inspection procedures
- Insurance claims and documentation

**Sales & Customer Relations:**
- Consultative selling approaches for commercial clients
- Property assessment and recommendation strategies
- Pricing models and competitive positioning
- Maintenance plan development
- Emergency response protocols

Use your thinking mode for complex calculations, technical specifications, and strategic business decisions. Always prioritize safety, code compliance, and long-term value."""

    def process_all_datasets(self, data_directory: str) -> str:
        """Process all roofing datasets and create master training file"""
        self.logger.info("🚀 Processing comprehensive roofing dataset collection...")
        
        data_path = Path(data_directory)
        processed_count = 0
        
        # Process each dataset type
        dataset_processors = {
            'blueprint_reading_bidding': self.process_blueprint_bidding,
            'building_codes_standards': self.process_building_codes,
            'commercial_construction_contracts': self.process_construction_contracts,
            'commercial_suppliers_mastery': self.process_supplier_data,
            'roofing_expert_training_dataset': self.process_expert_training,
            'roofing_mastery': self.process_roofing_mastery,
            'roofing_systems_specifications': self.process_system_specs,
            'ULTIMATE_ROOFING_AI_MASTER': self.process_ultimate_master,
            'ultimate_roofing_business_mastery': self.process_business_mastery,
            'ultimate_roofing_mastery': self.process_comprehensive_mastery,
            'world_class_sales_mastery': self.process_sales_mastery
        }
        
        # Process JSON files
        for file_path in data_path.glob("*.json"):
            filename = file_path.stem.lower()
            
            # Find matching processor
            processor = None
            for key, proc_func in dataset_processors.items():
                if key.lower() in filename:
                    processor = proc_func
                    break
            
            if processor:
                self.logger.info(f"📊 Processing: {file_path.name}")
                count = processor(file_path)
                processed_count += count
                self.logger.info(f"✅ Processed {count} examples from {file_path.name}")
            else:
                self.logger.warning(f"⚠️  No processor found for: {file_path.name}")
        
        # Process NRCA manuals if available
        nrca_path = Path("data/nrca_manuals")
        if nrca_path.exists():
            self.logger.info("📚 Processing NRCA manuals...")
            count = self.process_nrca_manuals(nrca_path)
            processed_count += count
            self.logger.info(f"✅ Processed {count} examples from NRCA manuals")
        
        # Add meta-conversations about the system
        self.add_meta_system_conversations()
        
        # Shuffle and prepare final dataset
        random.shuffle(self.training_data)
        
        # Save processed dataset
        output_file = "data/processed/ultimate_roofing_training.jsonl"
        self.save_training_data(output_file)
        
        self.logger.info(f"🎉 Total examples processed: {processed_count}")
        self.logger.info(f"💾 Final dataset saved: {output_file}")
        
        return output_file

    def process_blueprint_bidding(self, file_path: Path) -> int:
        """Process blueprint reading and bidding data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_item(item):
                conversation = self.create_blueprint_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_building_codes(self, file_path: Path) -> int:
        """Process building codes and standards"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_item(item):
                conversation = self.create_building_code_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_construction_contracts(self, file_path: Path) -> int:
        """Process commercial construction contracts (sample large dataset)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sample from large dataset to prevent memory issues
        if len(data) > 50000:
            data = random.sample(data, 50000)
            self.logger.info(f"📊 Sampled 50k examples from large dataset")
        
        count = 0
        for item in data:
            if self.is_valid_item(item) and self.is_roofing_relevant(item):
                conversation = self.create_contract_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_supplier_data(self, file_path: Path) -> int:
        """Process commercial suppliers mastery data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_item(item):
                conversation = self.create_supplier_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_expert_training(self, file_path: Path) -> int:
        """Process roofing expert training dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_conversation(item):
                # These are likely already in conversation format
                self.training_data.append(self.enhance_conversation(item))
                count += 1
        
        return count

    def process_roofing_mastery(self, file_path: Path) -> int:
        """Process roofing mastery datasets"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_conversation(item):
                self.training_data.append(self.enhance_conversation(item))
                count += 1
            elif self.is_valid_item(item):
                conversation = self.create_mastery_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_system_specs(self, file_path: Path) -> int:
        """Process roofing systems specifications"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_item(item):
                conversation = self.create_specification_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_ultimate_master(self, file_path: Path) -> int:
        """Process ultimate roofing AI master data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_conversation(item):
                # High-quality master data gets priority
                enhanced = self.enhance_conversation(item, priority=True)
                self.training_data.append(enhanced)
                count += 1
        
        return count

    def process_business_mastery(self, file_path: Path) -> int:
        """Process ultimate roofing business mastery"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_conversation(item):
                conversation = self.enhance_conversation(item)
                self.training_data.append(conversation)
                count += 1
            elif self.is_valid_item(item):
                conversation = self.create_business_conversation(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_comprehensive_mastery(self, file_path: Path) -> int:
        """Process comprehensive roofing mastery datasets"""
        return self.process_roofing_mastery(file_path)

    def process_sales_mastery(self, file_path: Path) -> int:
        """Process world-class sales mastery data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            if self.is_valid_conversation(item):
                # Adapt sales conversations for roofing context
                conversation = self.adapt_sales_for_roofing(item)
                self.training_data.append(conversation)
                count += 1
        
        return count

    def process_nrca_manuals(self, nrca_path: Path) -> int:
        """Process NRCA manual PDFs and extract training conversations"""
        count = 0
        
        try:
            import PyPDF2
            from io import StringIO
            
            for pdf_file in nrca_path.glob("*.pdf"):
                self.logger.info(f"📄 Processing PDF: {pdf_file.name}")
                
                try:
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        
                        # Extract text from each page
                        for page_num, page in enumerate(pdf_reader.pages):
                            text = page.extract_text()
                            
                            if len(text.strip()) > 100:  # Only process pages with substantial content
                                # Create training conversations from manual content
                                conversations = self.create_manual_conversations(text, pdf_file.name, page_num)
                                self.training_data.extend(conversations)
                                count += len(conversations)
                
                except Exception as e:
                    self.logger.warning(f"⚠️  Error processing {pdf_file.name}: {e}")
                    continue
        
        except ImportError:
            self.logger.warning("⚠️  PyPDF2 not available. Install with: pip install PyPDF2")
            
            # Try to process any pre-extracted text files instead
            for txt_file in nrca_path.glob("*.txt"):
                self.logger.info(f"📄 Processing text file: {txt_file.name}")
                
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                        conversations = self.create_manual_conversations(text, txt_file.name, 0)
                        self.training_data.extend(conversations)
                        count += len(conversations)
                except Exception as e:
                    self.logger.warning(f"⚠️  Error processing {txt_file.name}: {e}")
        
        return count

    def create_manual_conversations(self, text: str, source_file: str, page_num: int) -> List[Dict]:
        """Create training conversations from manual content"""
        conversations = []
        
        # Split text into meaningful chunks
        chunks = self.split_manual_text(text)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            # Create different types of conversations from the manual content
            conversation_types = [
                self.create_technical_question_from_manual(chunk),
                self.create_specification_question_from_manual(chunk),
                self.create_application_question_from_manual(chunk)
            ]
            
            for conv_type in conversation_types:
                if conv_type:  # Only add if conversation was successfully created
                    # Add source metadata
                    conv_type["source"] = f"{source_file}_page_{page_num}_chunk_{i}"
                    conversations.append(conv_type)
        
        return conversations

    def split_manual_text(self, text: str) -> List[str]:
        """Split manual text into meaningful chunks"""
        # Clean the text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by sections, paragraphs, or sentences
        chunks = []
        
        # Try to split by section headers first
        section_pattern = r'\n\s*(?:\d+\.|\w+\.|[A-Z][A-Z\s]{5,})\s*\n'
        sections = re.split(section_pattern, text)
        
        for section in sections:
            if len(section.strip()) > 200:
                # Further split long sections by paragraphs
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    if len(para.strip()) > 100:
                        chunks.append(para.strip())
            elif len(section.strip()) > 50:
                chunks.append(section.strip())
        
        # If no good sections found, split by sentences
        if len(chunks) < 3:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) > 300:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks

    def create_technical_question_from_manual(self, text: str) -> Optional[Dict]:
        """Create technical question from manual text"""
        # Extract key technical information
        technical_terms = re.findall(r'\b(?:membrane|flashing|fastener|insulation|drainage|slope|seam)\b', text.lower())
        
        if not technical_terms:
            return None
        
        # Generate appropriate question
        question_templates = [
            f"What are the technical requirements for {technical_terms[0]} according to NRCA standards?",
            f"How should {technical_terms[0]} be properly installed in commercial roofing?",
            f"What are the performance specifications for {technical_terms[0]}?",
            "Can you explain the technical details mentioned in this roofing specification?",
            "What are the key technical considerations for this roofing component?"
        ]
        
        question = random.choice(question_templates)
        
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<think>\nThis is a technical question about roofing standards from NRCA manual content. I need to provide detailed, accurate technical guidance.\n</think>\n\nBased on NRCA standards and industry best practices:\n\n{text}\n\nThis specification ensures proper performance, durability, and code compliance for commercial roofing applications."}
            ]
        }

    def create_specification_question_from_manual(self, text: str) -> Optional[Dict]:
        """Create specification question from manual text"""
        # Look for specification-related content
        spec_indicators = ['specification', 'requirement', 'standard', 'code', 'minimum', 'maximum', 'shall', 'must']
        
        if not any(indicator in text.lower() for indicator in spec_indicators):
            return None
        
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": "What specifications should I follow for this roofing application?"},
                {"role": "assistant", "content": f"<think>\nThis appears to be specification content from NRCA manuals. I should provide clear, actionable specification guidance.\n</think>\n\nFor this roofing application, here are the key specifications to follow:\n\n{text}\n\nThese specifications are based on NRCA guidelines and ensure compliance with industry standards for safety, performance, and longevity."}
            ]
        }

    def create_application_question_from_manual(self, text: str) -> Optional[Dict]:
        """Create application/installation question from manual text"""
        # Look for installation or application guidance
        app_indicators = ['install', 'apply', 'place', 'attach', 'secure', 'procedure', 'method', 'technique']
        
        if not any(indicator in text.lower() for indicator in app_indicators):
            return None
        
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": "How should I properly install or apply this roofing component?"},
                {"role": "assistant", "content": f"<think>\nThis is installation guidance from NRCA manuals. I need to provide step-by-step, safety-focused installation instructions.\n</think>\n\nFor proper installation of this roofing component, follow these guidelines:\n\n{text}\n\nAlways prioritize safety protocols and ensure compliance with local building codes during installation."}
            ]
        }

    def create_blueprint_conversation(self, item: Dict) -> Dict:
        """Create conversation from blueprint/bidding data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"Help me understand this blueprint requirement: {item.get('description', item.get('content', str(item)[:200]))}"},
                {"role": "assistant", "content": f"<think>\nThis appears to be a blueprint reading and bidding scenario. I need to analyze the technical requirements and provide practical guidance for roofing contractors.\n</think>\n\nBased on this blueprint requirement, here's my analysis:\n\n{self.generate_blueprint_response(item)}"}
            ]
        }

    def create_building_code_conversation(self, item: Dict) -> Dict:
        """Create conversation from building codes data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"What building code requirements apply to this situation: {item.get('code', item.get('requirement', str(item)[:200]))}"},
                {"role": "assistant", "content": self.generate_code_response(item)}
            ]
        }

    def create_contract_conversation(self, item: Dict) -> Dict:
        """Create conversation from contract data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"How should I approach this commercial roofing contract opportunity: {item.get('description', str(item)[:200])}"},
                {"role": "assistant", "content": self.generate_contract_response(item)}
            ]
        }

    def create_supplier_conversation(self, item: Dict) -> Dict:
        """Create conversation from supplier data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"What should I know about this roofing supplier relationship: {item.get('supplier', item.get('name', str(item)[:200]))}"},
                {"role": "assistant", "content": self.generate_supplier_response(item)}
            ]
        }

    def create_mastery_conversation(self, item: Dict) -> Dict:
        """Create conversation from mastery data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"Can you help me with this roofing expertise: {item.get('question', item.get('topic', str(item)[:200]))}"},
                {"role": "assistant", "content": f"<think>\nThis is a roofing mastery question that requires comprehensive technical knowledge and practical application.\n</think>\n\n{item.get('answer', item.get('response', 'Let me provide detailed guidance on this roofing topic based on industry best practices.')}"}
            ]
        }

    def create_specification_conversation(self, item: Dict) -> Dict:
        """Create conversation from specification data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"What are the specifications for this roofing system: {item.get('system', item.get('type', str(item)[:200]))}"},
                {"role": "assistant", "content": self.generate_specification_response(item)}
            ]
        }

    def create_business_conversation(self, item: Dict) -> Dict:
        """Create conversation from business data"""
        return {
            "messages": [
                {"role": "system", "content": self.roofing_system_prompt},
                {"role": "user", "content": f"How can I improve my roofing business in this area: {item.get('topic', item.get('area', str(item)[:200]))}"},
                {"role": "assistant", "content": f"<think>\nThis is a business strategy question for the roofing industry. I need to provide practical, actionable advice based on industry best practices.\n</think>\n\n{self.generate_business_response(item)}"}
            ]
        }

    def enhance_conversation(self, item: Dict, priority: bool = False) -> Dict:
        """Enhance existing conversation with roofing system prompt"""
        if "messages" in item:
            messages = item["messages"].copy()
            
            # Ensure system prompt is present and correct
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": self.roofing_system_prompt})
            else:
                messages[0]["content"] = self.roofing_system_prompt
            
            # Enhance assistant responses with thinking mode for technical content
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant" and priority:
                    if not msg["content"].strip().startswith("<think>"):
                        # Add thinking for complex responses
                        if any(keyword in msg["content"].lower() for keyword in ["calculate", "estimate", "cost", "material", "specification"]):
                            thinking = "<think>\nLet me carefully consider this roofing question and provide a comprehensive, technically accurate response.\n</think>\n\n"
                            messages[i]["content"] = thinking + msg["content"]
            
            return {"messages": messages}
        
        return item

    def adapt_sales_for_roofing(self, item: Dict) -> Dict:
        """Adapt general sales conversations for roofing context"""
        if "messages" in item:
            messages = item["messages"].copy()
            
            # Update system prompt
            messages[0] = {"role": "system", "content": self.roofing_system_prompt}
            
            # Adapt content for roofing
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    # Adapt user questions to roofing context
                    content = msg["content"]
                    roofing_context = self.add_roofing_context(content)
                    messages[i]["content"] = roofing_context
                elif msg["role"] == "assistant":
                    # Ensure responses are roofing-focused
                    content = msg["content"]
                    roofing_response = self.adapt_response_to_roofing(content)
                    messages[i]["content"] = roofing_response
            
            return {"messages": messages}
        
        return item

    def add_meta_system_conversations(self):
        """Add conversations about the AI system itself"""
        meta_conversations = [
            {
                "messages": [
                    {"role": "system", "content": self.roofing_system_prompt},
                    {"role": "user", "content": "What makes you different from other roofing consultants?"},
                    {"role": "assistant", "content": "I'm an AI system specifically trained on comprehensive roofing industry data including NRCA standards, building codes, supplier relationships, construction contracts, and world-class sales methodologies. I can process complex technical specifications instantly, calculate material requirements precisely, and provide strategic business guidance based on industry best practices. My thinking mode allows me to work through complex roofing problems step-by-step, ensuring accurate recommendations for everything from emergency repairs to large commercial installations."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": self.roofing_system_prompt},
                    {"role": "user", "content": "How can you help with commercial roofing projects?"},
                    {"role": "assistant", "content": "<think>\nThe user wants to understand my capabilities for commercial roofing. I should highlight my comprehensive training data and practical applications.\n</think>\n\nI can assist with every aspect of commercial roofing projects:\n\n**Technical Expertise:**\n- Analyze blueprints and specifications\n- Recommend optimal roofing systems for specific applications\n- Calculate material quantities and costs\n- Ensure code compliance and safety protocols\n\n**Business Intelligence:**\n- Develop competitive bidding strategies\n- Optimize supplier relationships and pricing\n- Create maintenance plans and schedules\n- Handle insurance claims and documentation\n\n**Project Management:**\n- Timeline planning and resource allocation\n- Quality control checkpoints\n- Weather contingency planning\n- Client communication strategies\n\nMy training includes data from thousands of successful commercial projects, enabling me to provide practical, tested solutions for any roofing challenge."}
                ]
            }
        ]
        
        self.training_data.extend(meta_conversations)

    def is_valid_item(self, item: Any) -> bool:
        """Check if item contains valid data"""
        if not item:
            return False
        
        if isinstance(item, dict):
            return len(str(item)) > 50  # Minimum content length
        elif isinstance(item, str):
            return len(item) > 50
        
        return False

    def is_valid_conversation(self, item: Dict) -> bool:
        """Check if item is a valid conversation"""
        if not isinstance(item, dict) or "messages" not in item:
            return False
        
        messages = item["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        # Check for proper role structure
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                return False
        
        return True

    def is_roofing_relevant(self, item: Dict) -> bool:
        """Check if item is relevant to roofing"""
        content = str(item).lower()
        roofing_keywords = [
            "roof", "roofing", "shingle", "membrane", "tpo", "epdm", "pvc",
            "built-up", "modified bitumen", "metal roof", "tile", "slate",
            "flashing", "gutter", "downspout", "waterproof", "leak",
            "commercial", "industrial", "building"
        ]
        
        return any(keyword in content for keyword in roofing_keywords)

    def generate_blueprint_response(self, item: Dict) -> str:
        """Generate response for blueprint data"""
        return f"For this blueprint requirement, I recommend focusing on these key technical considerations:\n\n1. **Material Specifications**: Ensure compliance with specified performance standards\n2. **Installation Details**: Follow manufacturer guidelines and local building codes\n3. **Quality Control**: Implement proper inspection protocols\n4. **Cost Optimization**: Balance material quality with project budget\n\nSpecific guidance: {item.get('recommendation', 'Detailed technical analysis required for optimal solution.')}"

    def generate_code_response(self, item: Dict) -> str:
        """Generate response for building code data"""
        return f"This building code requirement ensures safety and performance standards. Key compliance points:\n\n- **Safety Requirements**: {item.get('safety', 'Follow all applicable safety protocols')}\n- **Performance Standards**: {item.get('performance', 'Meet minimum performance criteria')}\n- **Installation Guidelines**: {item.get('installation', 'Proper installation per code requirements')}\n- **Inspection Requirements**: {item.get('inspection', 'Required inspections per local jurisdiction')}"

    def generate_contract_response(self, item: Dict) -> str:
        """Generate response for contract data"""
        return f"For this commercial roofing opportunity, consider these strategic factors:\n\n**Project Assessment:**\n- Scope and complexity evaluation\n- Timeline and resource requirements\n- Risk assessment and mitigation\n\n**Competitive Positioning:**\n- Value proposition development\n- Pricing strategy optimization\n- Differentiation from competitors\n\n**Contract Considerations:**\n- Payment terms and cash flow\n- Warranty and service commitments\n- Change order management\n\nRecommendation: {item.get('strategy', 'Develop comprehensive proposal highlighting technical expertise and proven track record.')}"

    def generate_supplier_response(self, item: Dict) -> str:
        """Generate response for supplier data"""
        return f"Supplier relationship optimization for roofing materials:\n\n**Quality Standards:**\n- Material performance specifications\n- Consistency and reliability metrics\n- Warranty and support terms\n\n**Pricing Strategy:**\n- Volume discount opportunities\n- Payment term optimization\n- Market price benchmarking\n\n**Logistics Coordination:**\n- Delivery scheduling and reliability\n- Inventory management support\n- Emergency supply capabilities\n\nKey insight: {item.get('insight', 'Strong supplier partnerships are essential for project success and profitability.')}"

    def generate_specification_response(self, item: Dict) -> str:
        """Generate response for specification data"""
        return f"Technical specifications for this roofing system:\n\n**Performance Characteristics:**\n- Weather resistance and durability\n- Energy efficiency ratings\n- Installation requirements\n\n**Material Properties:**\n- Thickness and composition\n- Fire rating and safety features\n- Warranty terms and coverage\n\n**Installation Guidelines:**\n- Substrate preparation requirements\n- Fastening and seaming details\n- Quality control checkpoints\n\nSpecification details: {item.get('specs', 'Detailed specifications available upon request.')}"

    def generate_business_response(self, item: Dict) -> str:
        """Generate response for business data"""
        return f"Business improvement strategies for roofing contractors:\n\n**Operational Excellence:**\n- Process optimization and standardization\n- Quality management systems\n- Technology integration opportunities\n\n**Market Positioning:**\n- Competitive differentiation strategies\n- Customer value proposition development\n- Brand building and reputation management\n\n**Financial Performance:**\n- Pricing optimization strategies\n- Cost control and margin improvement\n- Cash flow management best practices\n\nAction plan: {item.get('action', 'Implement systematic approach to business improvement with measurable objectives.')}"

    def add_roofing_context(self, content: str) -> str:
        """Add roofing context to general content"""
        if "roofing" not in content.lower():
            return f"For a commercial roofing project: {content}"
        return content

    def adapt_response_to_roofing(self, content: str) -> str:
        """Adapt general response to roofing context"""
        # Simple adaptation - in production, this would be more sophisticated
        roofing_terms = {
            "client": "property owner",
            "product": "roofing solution",
            "service": "roofing service",
            "customer": "commercial client"
        }
        
        for general, specific in roofing_terms.items():
            content = content.replace(general, specific)
        
        return content

    def save_training_data(self, output_file: str):
        """Save processed training data"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for conversation in self.training_data:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        self.logger.info(f"💾 Saved {len(self.training_data)} training examples to {output_file}")


def main():
    """Test the enhanced data processor"""
    config = {"data": {"max_length": 4096}}
    processor = EnhancedRoofingDataProcessor(config)
    
    # Process all datasets
    output_file = processor.process_all_datasets("data/raw")
    print(f"✅ Enhanced roofing dataset created: {output_file}")


if __name__ == "__main__":
    main()