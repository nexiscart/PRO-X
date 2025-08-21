#!/usr/bin/env python3
"""
Data Validator for Pro Roofing AI
Validates and cleans training datasets
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
from collections import Counter
import statistics

class RoofingDataValidator:
    """Validates and cleans roofing training data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation criteria
        self.min_content_length = 10
        self.max_content_length = 8000
        self.required_roles = {"system", "user", "assistant"}
        
        # Roofing-specific keywords for relevance check
        self.roofing_keywords = {
            "materials": ["tpo", "epdm", "pvc", "membrane", "shingle", "tile", "metal", "slate", "tar", "gravel"],
            "systems": ["built-up", "modified bitumen", "single ply", "liquid applied", "green roof"],
            "components": ["flashing", "gutter", "downspout", "drain", "scupper", "parapet", "fascia"],
            "processes": ["installation", "repair", "maintenance", "inspection", "replacement", "waterproofing"],
            "business": ["estimate", "bid", "contract", "warranty", "insurance", "project", "commercial"],
            "technical": ["slope", "drainage", "insulation", "vapor barrier", "fastener", "seam", "penetration"]
        }
        
        # Statistics tracking
        self.validation_stats = {
            "total_processed": 0,
            "valid_conversations": 0,
            "invalid_conversations": 0,
            "filtered_out": 0,
            "issues": Counter()
        }

    def validate_dataset(self, dataset_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """Validate complete dataset and return cleaned data with statistics"""
        self.logger.info(f"🔍 Validating dataset: {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.json'):
                data = json.load(f)
            elif dataset_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Validate each item
        valid_conversations = []
        for i, item in enumerate(data):
            self.validation_stats["total_processed"] += 1
            
            is_valid, cleaned_item, issues = self.validate_conversation(item)
            
            if is_valid:
                valid_conversations.append(cleaned_item)
                self.validation_stats["valid_conversations"] += 1
            else:
                self.validation_stats["invalid_conversations"] += 1
                for issue in issues:
                    self.validation_stats["issues"][issue] += 1
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Invalid conversation {i}: {issues}")
        
        # Generate statistics
        stats = self.generate_statistics(valid_conversations)
        
        self.logger.info(f"✅ Validation complete: {len(valid_conversations)}/{len(data)} valid conversations")
        
        return valid_conversations, stats

    def validate_conversation(self, item: Any) -> Tuple[bool, Optional[Dict], List[str]]:
        """Validate a single conversation item"""
        issues = []
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            issues.append("not_dict")
            return False, None, issues
        
        # Check for messages key
        if "messages" not in item:
            issues.append("no_messages_key")
            return False, None, issues
        
        messages = item["messages"]
        
        # Check if messages is a list
        if not isinstance(messages, list):
            issues.append("messages_not_list")
            return False, None, issues
        
        # Check minimum number of messages
        if len(messages) < 2:
            issues.append("too_few_messages")
            return False, None, issues
        
        # Validate each message
        cleaned_messages = []
        for i, msg in enumerate(messages):
            is_valid, cleaned_msg, msg_issues = self.validate_message(msg)
            
            if not is_valid:
                issues.extend([f"message_{i}_{issue}" for issue in msg_issues])
                continue
            
            cleaned_messages.append(cleaned_msg)
        
        # Check if we have enough valid messages
        if len(cleaned_messages) < 2:
            issues.append("insufficient_valid_messages")
            return False, None, issues
        
        # Check conversation structure
        structure_valid, structure_issues = self.validate_conversation_structure(cleaned_messages)
        if not structure_valid:
            issues.extend(structure_issues)
            return False, None, issues
        
        # Check roofing relevance
        if not self.is_roofing_relevant(cleaned_messages):
            issues.append("not_roofing_relevant")
            return False, None, issues
        
        # Create cleaned conversation
        cleaned_conversation = {
            "messages": cleaned_messages
        }
        
        # Add metadata if present
        for key in ["id", "source", "quality_score"]:
            if key in item:
                cleaned_conversation[key] = item[key]
        
        return True, cleaned_conversation, issues

    def validate_message(self, message: Any) -> Tuple[bool, Optional[Dict], List[str]]:
        """Validate a single message"""
        issues = []
        
        # Check if message is a dictionary
        if not isinstance(message, dict):
            issues.append("not_dict")
            return False, None, issues
        
        # Check required keys
        required_keys = ["role", "content"]
        for key in required_keys:
            if key not in message:
                issues.append(f"missing_{key}")
        
        if issues:
            return False, None, issues
        
        # Validate role
        role = message["role"]
        if role not in self.required_roles:
            issues.append(f"invalid_role_{role}")
        
        # Validate content
        content = message["content"]
        if not isinstance(content, str):
            issues.append("content_not_string")
            return False, None, issues
        
        # Clean and validate content
        cleaned_content = self.clean_content(content)
        
        # Check content length
        if len(cleaned_content) < self.min_content_length:
            issues.append("content_too_short")
        elif len(cleaned_content) > self.max_content_length:
            issues.append("content_too_long")
        
        if issues:
            return False, None, issues
        
        return True, {"role": role, "content": cleaned_content}, issues

    def validate_conversation_structure(self, messages: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate the structure of a conversation"""
        issues = []
        
        # Extract roles
        roles = [msg["role"] for msg in messages]
        
        # Check if first message is system (preferred)
        if roles[0] != "system":
            # Insert a generic system message if missing
            messages.insert(0, {
                "role": "system",
                "content": "You are a helpful AI assistant specialized in roofing industry expertise."
            })
            roles.insert(0, "system")
        
        # Check alternating user/assistant pattern after system
        for i in range(1, len(roles) - 1):
            if roles[i] == "user" and roles[i + 1] != "assistant":
                issues.append(f"no_assistant_after_user_at_{i}")
            elif roles[i] == "assistant" and roles[i + 1] not in ["user", "assistant"]:
                issues.append(f"invalid_role_after_assistant_at_{i}")
        
        # Check that conversation ends with assistant
        if roles[-1] != "assistant":
            issues.append("not_ending_with_assistant")
        
        return len(issues) == 0, issues

    def clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', '', content)
        
        # Fix common encoding issues
        content = content.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        
        # Remove control characters
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        
        return content

    def is_roofing_relevant(self, messages: List[Dict]) -> bool:
        """Check if conversation is relevant to roofing"""
        # Combine all content
        all_content = " ".join([msg["content"].lower() for msg in messages])
        
        # Count roofing keyword matches
        keyword_matches = 0
        total_keywords = 0
        
        for category, keywords in self.roofing_keywords.items():
            total_keywords += len(keywords)
            for keyword in keywords:
                if keyword in all_content:
                    keyword_matches += 1
        
        # General roofing terms
        general_terms = ["roof", "roofing", "rooftop", "building", "construction", "contractor"]
        for term in general_terms:
            if term in all_content:
                keyword_matches += 2  # Weight general terms higher
        
        # Check for relevance threshold
        relevance_score = keyword_matches / max(total_keywords, 1)
        return relevance_score > 0.1 or any(term in all_content for term in general_terms)

    def generate_statistics(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Generate statistics about the validated dataset"""
        if not conversations:
            return {"error": "No valid conversations to analyze"}
        
        # Message statistics
        message_counts = [len(conv["messages"]) for conv in conversations]
        content_lengths = []
        role_distribution = Counter()
        
        for conv in conversations:
            for msg in conv["messages"]:
                content_lengths.append(len(msg["content"]))
                role_distribution[msg["role"]] += 1
        
        # Calculate statistics
        stats = {
            "total_conversations": len(conversations),
            "message_statistics": {
                "total_messages": sum(message_counts),
                "avg_messages_per_conversation": statistics.mean(message_counts),
                "min_messages": min(message_counts),
                "max_messages": max(message_counts),
                "median_messages": statistics.median(message_counts)
            },
            "content_statistics": {
                "avg_content_length": statistics.mean(content_lengths),
                "min_content_length": min(content_lengths),
                "max_content_length": max(content_lengths),
                "median_content_length": statistics.median(content_lengths)
            },
            "role_distribution": dict(role_distribution),
            "validation_issues": dict(self.validation_stats["issues"]),
            "validation_summary": {
                "total_processed": self.validation_stats["total_processed"],
                "valid_rate": self.validation_stats["valid_conversations"] / max(self.validation_stats["total_processed"], 1),
                "most_common_issues": self.validation_stats["issues"].most_common(5)
            }
        }
        
        return stats

    def save_validation_report(self, stats: Dict[str, Any], output_path: str):
        """Save validation report to file"""
        report = {
            "validation_timestamp": str(Path().cwd()),
            "statistics": stats,
            "configuration": {
                "min_content_length": self.min_content_length,
                "max_content_length": self.max_content_length,
                "required_roles": list(self.required_roles)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Validation report saved to: {output_path}")

    def filter_by_quality(self, conversations: List[Dict], min_quality_score: float = 0.7) -> List[Dict]:
        """Filter conversations by quality score"""
        if not conversations:
            return []
        
        # Simple quality scoring based on various factors
        scored_conversations = []
        
        for conv in conversations:
            score = self.calculate_quality_score(conv)
            if score >= min_quality_score:
                conv["quality_score"] = score
                scored_conversations.append(conv)
        
        self.logger.info(f"🎯 Quality filtering: {len(scored_conversations)}/{len(conversations)} conversations retained")
        
        return scored_conversations

    def calculate_quality_score(self, conversation: Dict) -> float:
        """Calculate quality score for a conversation"""
        messages = conversation["messages"]
        score = 0.0
        factors = 0
        
        # Factor 1: Message balance (good user/assistant interaction)
        user_msgs = sum(1 for msg in messages if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in messages if msg["role"] == "assistant")
        
        if user_msgs > 0 and assistant_msgs > 0:
            balance = min(user_msgs, assistant_msgs) / max(user_msgs, assistant_msgs)
            score += balance * 0.3
            factors += 0.3
        
        # Factor 2: Content richness
        avg_length = sum(len(msg["content"]) for msg in messages) / len(messages)
        length_score = min(avg_length / 200, 1.0)  # Normalize to 200 chars
        score += length_score * 0.3
        factors += 0.3
        
        # Factor 3: Roofing relevance
        all_content = " ".join([msg["content"].lower() for msg in messages])
        keyword_matches = sum(
            1 for keywords in self.roofing_keywords.values()
            for keyword in keywords
            if keyword in all_content
        )
        relevance_score = min(keyword_matches / 10, 1.0)  # Normalize to 10 keywords
        score += relevance_score * 0.4
        factors += 0.4
        
        return score / factors if factors > 0 else 0.0


def main():
    """Test the data validator"""
    config = {"validation": {"min_quality_score": 0.7}}
    validator = RoofingDataValidator(config)
    
    # Test with sample data file
    if Path("data/raw").exists():
        for json_file in Path("data/raw").glob("*.json"):
            print(f"Validating {json_file.name}...")
            try:
                valid_data, stats = validator.validate_dataset(str(json_file))
                print(f"✅ {len(valid_data)} valid conversations")
                
                # Save validation report
                report_path = f"data/validation/{json_file.stem}_validation_report.json"
                Path("data/validation").mkdir(exist_ok=True)
                validator.save_validation_report(stats, report_path)
                
            except Exception as e:
                print(f"❌ Error validating {json_file.name}: {e}")
            
            break  # Just test one file for demo


if __name__ == "__main__":
    main()