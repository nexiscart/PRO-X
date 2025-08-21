#!/usr/bin/env python3
"""
Data Statistics Generator for Pro Roofing AI
Analyzes and generates comprehensive statistics for datasets
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import statistics
from datetime import datetime
import re

class RoofingDataStatistics:
    """Generate comprehensive statistics for roofing datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize data containers
        self.dataset_stats = {}
        self.combined_stats = {}
        
        # Roofing domain categories
        self.roofing_categories = {
            "materials": {
                "single_ply": ["tpo", "epdm", "pvc", "tpo membrane", "epdm membrane", "pvc membrane"],
                "built_up": ["built-up", "tar and gravel", "bur", "built up roofing"],
                "modified_bitumen": ["modified bitumen", "mod bit", "sbs", "app"],
                "metal": ["metal roofing", "steel", "aluminum", "copper", "standing seam"],
                "tile_slate": ["clay tile", "concrete tile", "slate", "ceramic"],
                "shingle": ["asphalt shingle", "architectural shingle", "3-tab"]
            },
            "components": {
                "drainage": ["drain", "scupper", "gutter", "downspout", "overflow"],
                "flashing": ["flashing", "counterflashing", "step flashing", "valley"],
                "insulation": ["insulation", "polyiso", "eps", "xps", "rigid board"],
                "membrane": ["membrane", "underlayment", "vapor barrier", "air barrier"]
            },
            "processes": {
                "installation": ["install", "application", "placement", "mounting"],
                "maintenance": ["maintenance", "repair", "inspection", "cleaning"],
                "waterproofing": ["waterproof", "seal", "caulk", "weatherproof"],
                "safety": ["safety", "fall protection", "harness", "osha"]
            },
            "business": {
                "estimation": ["estimate", "bid", "quote", "pricing", "cost"],
                "contracts": ["contract", "agreement", "proposal", "specification"],
                "project_management": ["schedule", "timeline", "milestone", "deadline"],
                "quality": ["quality", "inspection", "warranty", "guarantee"]
            }
        }

    def analyze_datasets(self, data_directory: str) -> Dict[str, Any]:
        """Analyze all datasets in the directory"""
        self.logger.info(f"📊 Analyzing datasets in: {data_directory}")
        
        data_path = Path(data_directory)
        individual_stats = {}
        
        # Analyze each dataset file
        for file_path in data_path.glob("*.json"):
            self.logger.info(f"Analyzing {file_path.name}...")
            
            try:
                stats = self.analyze_single_dataset(file_path)
                individual_stats[file_path.stem] = stats
                self.logger.info(f"✅ Completed analysis of {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {file_path.name}: {e}")
                continue
        
        # Generate combined statistics
        self.combined_stats = self.generate_combined_statistics(individual_stats)
        
        # Generate comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_datasets": len(individual_stats),
            "individual_datasets": individual_stats,
            "combined_statistics": self.combined_stats,
            "recommendations": self.generate_recommendations()
        }
        
        return report

    def analyze_single_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset file"""
        # Load dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        # Basic statistics
        stats = {
            "file_info": {
                "filename": file_path.name,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "total_items": len(data)
            },
            "content_analysis": self.analyze_content(data),
            "conversation_analysis": self.analyze_conversations(data),
            "roofing_domain_analysis": self.analyze_roofing_domain(data),
            "quality_metrics": self.calculate_quality_metrics(data)
        }
        
        return stats

    def analyze_content(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze content characteristics"""
        content_lengths = []
        word_counts = []
        char_counts = []
        
        for item in data:
            if isinstance(item, dict) and "messages" in item:
                # Conversation format
                for msg in item.get("messages", []):
                    content = msg.get("content", "")
                    content_lengths.append(len(content))
                    word_counts.append(len(content.split()))
                    char_counts.append(len(content))
            elif isinstance(item, dict):
                # Single item format
                content = str(item)
                content_lengths.append(len(content))
                word_counts.append(len(content.split()))
                char_counts.append(len(content))
            else:
                content = str(item)
                content_lengths.append(len(content))
                word_counts.append(len(content.split()))
                char_counts.append(len(content))
        
        if not content_lengths:
            return {"error": "No content found"}
        
        return {
            "total_content_pieces": len(content_lengths),
            "content_length_stats": {
                "mean": statistics.mean(content_lengths),
                "median": statistics.median(content_lengths),
                "std_dev": statistics.stdev(content_lengths) if len(content_lengths) > 1 else 0,
                "min": min(content_lengths),
                "max": max(content_lengths),
                "percentiles": {
                    "25th": statistics.quantiles(content_lengths, n=4)[0] if len(content_lengths) > 3 else min(content_lengths),
                    "75th": statistics.quantiles(content_lengths, n=4)[2] if len(content_lengths) > 3 else max(content_lengths),
                    "90th": statistics.quantiles(content_lengths, n=10)[8] if len(content_lengths) > 9 else max(content_lengths)
                }
            },
            "word_count_stats": {
                "mean": statistics.mean(word_counts),
                "median": statistics.median(word_counts),
                "total_words": sum(word_counts)
            }
        }

    def analyze_conversations(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze conversation-specific characteristics"""
        conversations = [item for item in data if isinstance(item, dict) and "messages" in item]
        
        if not conversations:
            return {"conversation_format": False, "note": "No conversation format detected"}
        
        message_counts = []
        role_distribution = Counter()
        system_prompt_analysis = {}
        
        for conv in conversations:
            messages = conv.get("messages", [])
            message_counts.append(len(messages))
            
            for msg in messages:
                role = msg.get("role", "unknown")
                role_distribution[role] += 1
                
                # Analyze system prompts
                if role == "system":
                    content = msg.get("content", "")
                    if "system_prompts" not in system_prompt_analysis:
                        system_prompt_analysis["system_prompts"] = []
                        system_prompt_analysis["avg_length"] = 0
                        system_prompt_analysis["unique_count"] = 0
                    
                    system_prompt_analysis["system_prompts"].append(content[:200])  # First 200 chars
        
        # Calculate system prompt statistics
        if "system_prompts" in system_prompt_analysis:
            unique_prompts = set(system_prompt_analysis["system_prompts"])
            system_prompt_analysis["unique_count"] = len(unique_prompts)
            system_prompt_analysis["avg_length"] = statistics.mean([len(p) for p in system_prompt_analysis["system_prompts"]])
        
        return {
            "conversation_format": True,
            "total_conversations": len(conversations),
            "message_statistics": {
                "avg_messages_per_conversation": statistics.mean(message_counts) if message_counts else 0,
                "median_messages": statistics.median(message_counts) if message_counts else 0,
                "min_messages": min(message_counts) if message_counts else 0,
                "max_messages": max(message_counts) if message_counts else 0
            },
            "role_distribution": dict(role_distribution),
            "role_percentages": {
                role: (count / sum(role_distribution.values())) * 100 
                for role, count in role_distribution.items()
            } if role_distribution else {},
            "system_prompt_analysis": system_prompt_analysis
        }

    def analyze_roofing_domain(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze roofing domain-specific content"""
        # Extract all text content
        all_text = self.extract_all_text(data).lower()
        
        # Analyze by category
        category_analysis = {}
        total_matches = 0
        
        for category, subcategories in self.roofing_categories.items():
            category_matches = {}
            category_total = 0
            
            for subcategory, keywords in subcategories.items():
                matches = sum(1 for keyword in keywords if keyword.lower() in all_text)
                category_matches[subcategory] = matches
                category_total += matches
            
            category_analysis[category] = {
                "subcategory_matches": category_matches,
                "total_matches": category_total,
                "coverage_percentage": (category_total / len(keywords)) * 100 if keywords else 0
            }
            total_matches += category_total
        
        # Overall domain relevance
        domain_relevance = self.calculate_domain_relevance(all_text)
        
        return {
            "category_analysis": category_analysis,
            "total_roofing_matches": total_matches,
            "domain_relevance_score": domain_relevance,
            "most_covered_categories": sorted(
                [(cat, info["total_matches"]) for cat, info in category_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "keyword_density": total_matches / len(all_text.split()) if all_text else 0
        }

    def calculate_domain_relevance(self, text: str) -> float:
        """Calculate overall domain relevance score"""
        # Core roofing terms
        core_terms = ["roof", "roofing", "rooftop", "commercial roofing", "industrial roofing"]
        core_matches = sum(1 for term in core_terms if term in text)
        
        # Technical terms
        technical_terms = ["membrane", "flashing", "drainage", "insulation", "fastener"]
        tech_matches = sum(1 for term in technical_terms if term in text)
        
        # Business terms
        business_terms = ["estimate", "contract", "warranty", "installation", "maintenance"]
        business_matches = sum(1 for term in business_terms if term in text)
        
        # Calculate weighted score
        total_possible = len(core_terms) + len(technical_terms) + len(business_terms)
        actual_matches = (core_matches * 2) + tech_matches + business_matches  # Weight core terms more
        max_possible = (len(core_terms) * 2) + len(technical_terms) + len(business_terms)
        
        return (actual_matches / max_possible) if max_possible > 0 else 0

    def calculate_quality_metrics(self, data: List[Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset"""
        # Extract conversations
        conversations = [item for item in data if isinstance(item, dict) and "messages" in item]
        
        if not conversations:
            # Non-conversation format
            quality_scores = [self.score_individual_item(item) for item in data]
        else:
            # Conversation format
            quality_scores = [self.score_conversation(conv) for conv in conversations]
        
        if not quality_scores:
            return {"error": "No items to score"}
        
        return {
            "average_quality_score": statistics.mean(quality_scores),
            "median_quality_score": statistics.median(quality_scores),
            "quality_distribution": {
                "excellent": sum(1 for score in quality_scores if score >= 0.8),
                "good": sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                "fair": sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                "poor": sum(1 for score in quality_scores if score < 0.4)
            },
            "total_items_scored": len(quality_scores)
        }

    def score_conversation(self, conversation: Dict) -> float:
        """Score a conversation for quality"""
        messages = conversation.get("messages", [])
        if not messages:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Factor 1: Structure quality (0.3)
        has_system = any(msg.get("role") == "system" for msg in messages)
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        
        structure_score = (has_system + has_user + has_assistant) / 3
        score += structure_score * 0.3
        factors += 0.3
        
        # Factor 2: Content quality (0.4)
        avg_length = statistics.mean([len(msg.get("content", "")) for msg in messages])
        length_score = min(avg_length / 100, 1.0)  # Normalize to 100 chars
        
        content_diversity = len(set(msg.get("content", "")[:50] for msg in messages)) / len(messages)
        
        content_score = (length_score + content_diversity) / 2
        score += content_score * 0.4
        factors += 0.4
        
        # Factor 3: Roofing relevance (0.3)
        all_content = " ".join([msg.get("content", "") for msg in messages])
        relevance_score = self.calculate_domain_relevance(all_content.lower())
        
        score += relevance_score * 0.3
        factors += 0.3
        
        return score / factors if factors > 0 else 0.0

    def score_individual_item(self, item: Any) -> float:
        """Score an individual item for quality"""
        content = str(item)
        
        # Length score
        length_score = min(len(content) / 200, 1.0)
        
        # Relevance score
        relevance_score = self.calculate_domain_relevance(content.lower())
        
        # Structure score (simple heuristic)
        has_structure = any(keyword in content.lower() for keyword in ["description", "example", "question", "answer"])
        structure_score = 1.0 if has_structure else 0.5
        
        return (length_score + relevance_score + structure_score) / 3

    def extract_all_text(self, data: List[Any]) -> str:
        """Extract all text content from data"""
        all_text = []
        
        for item in data:
            if isinstance(item, dict) and "messages" in item:
                for msg in item.get("messages", []):
                    all_text.append(msg.get("content", ""))
            elif isinstance(item, dict):
                all_text.append(str(item))
            else:
                all_text.append(str(item))
        
        return " ".join(all_text)

    def generate_combined_statistics(self, individual_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate combined statistics across all datasets"""
        if not individual_stats:
            return {"error": "No individual statistics to combine"}
        
        # Aggregate file information
        total_items = sum(stats["file_info"]["total_items"] for stats in individual_stats.values())
        total_size_mb = sum(stats["file_info"]["file_size_mb"] for stats in individual_stats.values())
        
        # Aggregate conversation statistics
        total_conversations = sum(
            stats.get("conversation_analysis", {}).get("total_conversations", 0)
            for stats in individual_stats.values()
        )
        
        # Aggregate quality scores
        all_quality_scores = []
        for stats in individual_stats.values():
            quality_metrics = stats.get("quality_metrics", {})
            if "average_quality_score" in quality_metrics:
                all_quality_scores.append(quality_metrics["average_quality_score"])
        
        # Aggregate roofing domain coverage
        combined_categories = defaultdict(int)
        for stats in individual_stats.values():
            domain_analysis = stats.get("roofing_domain_analysis", {})
            category_analysis = domain_analysis.get("category_analysis", {})
            
            for category, info in category_analysis.items():
                combined_categories[category] += info.get("total_matches", 0)
        
        return {
            "dataset_summary": {
                "total_datasets": len(individual_stats),
                "total_items": total_items,
                "total_size_mb": round(total_size_mb, 2),
                "total_conversations": total_conversations
            },
            "quality_overview": {
                "datasets_with_quality_scores": len(all_quality_scores),
                "average_quality_across_datasets": statistics.mean(all_quality_scores) if all_quality_scores else 0,
                "quality_range": {
                    "min": min(all_quality_scores) if all_quality_scores else 0,
                    "max": max(all_quality_scores) if all_quality_scores else 0
                }
            },
            "roofing_domain_coverage": {
                "category_totals": dict(combined_categories),
                "most_covered": sorted(combined_categories.items(), key=lambda x: x[1], reverse=True)[:5],
                "coverage_balance": self.calculate_coverage_balance(combined_categories)
            },
            "recommendations": self.generate_dataset_recommendations(individual_stats)
        }

    def calculate_coverage_balance(self, category_totals: Dict[str, int]) -> Dict[str, Any]:
        """Calculate how balanced the coverage is across categories"""
        if not category_totals:
            return {"balance_score": 0, "note": "No categories found"}
        
        values = list(category_totals.values())
        if not values:
            return {"balance_score": 0, "note": "No category values"}
        
        # Calculate coefficient of variation
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        balance_score = 1 - (std_val / mean_val) if mean_val > 0 else 0
        balance_score = max(0, balance_score)  # Ensure non-negative
        
        return {
            "balance_score": balance_score,
            "interpretation": "High" if balance_score > 0.7 else "Medium" if balance_score > 0.4 else "Low",
            "most_overrepresented": max(category_totals.items(), key=lambda x: x[1])[0],
            "least_represented": min(category_totals.items(), key=lambda x: x[1])[0]
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not self.combined_stats:
            return ["Run analysis first to generate recommendations"]
        
        # Quality recommendations
        quality_overview = self.combined_stats.get("quality_overview", {})
        avg_quality = quality_overview.get("average_quality_across_datasets", 0)
        
        if avg_quality < 0.6:
            recommendations.append("Overall dataset quality is below optimal. Consider data cleaning and validation.")
        elif avg_quality > 0.8:
            recommendations.append("Excellent dataset quality detected. Ready for training.")
        
        # Coverage recommendations
        coverage = self.combined_stats.get("roofing_domain_coverage", {})
        balance = coverage.get("coverage_balance", {})
        balance_score = balance.get("balance_score", 0)
        
        if balance_score < 0.4:
            recommendations.append(f"Unbalanced domain coverage. Consider adding more examples for {balance.get('least_represented', 'underrepresented areas')}.")
        
        # Size recommendations
        dataset_summary = self.combined_stats.get("dataset_summary", {})
        total_items = dataset_summary.get("total_items", 0)
        
        if total_items < 1000:
            recommendations.append("Dataset size is small. Consider augmentation or additional data collection.")
        elif total_items > 100000:
            recommendations.append("Large dataset detected. Consider sampling strategies for efficient training.")
        
        return recommendations

    def generate_dataset_recommendations(self, individual_stats: Dict[str, Dict]) -> List[str]:
        """Generate specific recommendations for individual datasets"""
        recommendations = []
        
        for dataset_name, stats in individual_stats.items():
            quality = stats.get("quality_metrics", {}).get("average_quality_score", 0)
            
            if quality < 0.5:
                recommendations.append(f"{dataset_name}: Low quality score ({quality:.2f}). Review and clean data.")
            
            # Check conversation format
            conv_analysis = stats.get("conversation_analysis", {})
            if not conv_analysis.get("conversation_format", False):
                recommendations.append(f"{dataset_name}: Convert to conversation format for consistent training.")
        
        return recommendations

    def save_statistics_report(self, report: Dict[str, Any], output_path: str):
        """Save comprehensive statistics report"""
        # Save JSON report
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate and save markdown report
        md_path = Path(output_path).with_suffix('.md')
        self.generate_markdown_report(report, md_path)
        
        self.logger.info(f"📊 Statistics report saved to: {json_path} and {md_path}")

    def generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Generate a markdown report"""
        md_content = []
        md_content.append("# Pro Roofing AI Dataset Statistics Report")
        md_content.append(f"\n**Generated on:** {report.get('analysis_timestamp', 'Unknown')}")
        md_content.append(f"**Total Datasets Analyzed:** {report.get('total_datasets', 0)}")
        
        # Combined statistics
        combined = report.get('combined_statistics', {})
        if combined:
            md_content.append("\n## Overall Dataset Summary")
            summary = combined.get('dataset_summary', {})
            md_content.append(f"- **Total Items:** {summary.get('total_items', 0):,}")
            md_content.append(f"- **Total Size:** {summary.get('total_size_mb', 0):.2f} MB")
            md_content.append(f"- **Total Conversations:** {summary.get('total_conversations', 0):,}")
        
        # Quality overview
        quality = combined.get('quality_overview', {})
        if quality:
            md_content.append("\n## Quality Assessment")
            md_content.append(f"- **Average Quality Score:** {quality.get('average_quality_across_datasets', 0):.3f}")
            quality_range = quality.get('quality_range', {})
            md_content.append(f"- **Quality Range:** {quality_range.get('min', 0):.3f} - {quality_range.get('max', 0):.3f}")
        
        # Domain coverage
        coverage = combined.get('roofing_domain_coverage', {})
        if coverage:
            md_content.append("\n## Roofing Domain Coverage")
            most_covered = coverage.get('most_covered', [])[:5]
            for i, (category, count) in enumerate(most_covered, 1):
                md_content.append(f"{i}. **{category.title()}:** {count} matches")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            md_content.append("\n## Recommendations")
            for i, rec in enumerate(recommendations, 1):
                md_content.append(f"{i}. {rec}")
        
        # Save markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))


def main():
    """Test the statistics generator"""
    config = {}
    stats_generator = RoofingDataStatistics(config)
    
    # Analyze datasets
    if Path("data/raw").exists():
        report = stats_generator.analyze_datasets("data/raw")
        
        # Save report
        output_path = "data/processed/dataset_statistics_report"
        Path("data/processed").mkdir(exist_ok=True)
        stats_generator.save_statistics_report(report, output_path)
        
        print("✅ Dataset statistics analysis completed")
        print(f"📊 Report saved to: {output_path}.json and {output_path}.md")
    else:
        print("❌ No data/raw directory found")


if __name__ == "__main__":
    main()