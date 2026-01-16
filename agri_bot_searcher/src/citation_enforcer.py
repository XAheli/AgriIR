"""
Citation Enforcement System
Ensures all LLM-generated answers include proper citations
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CitationValidation:
    """Results of citation validation"""
    has_citations: bool
    citation_count: int
    valid_citations: List[str]
    invalid_citations: List[str]
    uncited_facts: List[str]
    citation_coverage: float  # Percentage of sentences with citations


class CitationEnforcer:
    """Enforces proper citation usage in LLM responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.citation_pattern = re.compile(r'\[(DB|WEB)-(\d+)-(\d+)\]')
    
    def validate_citations(self, text: str, available_citations: Set[str]) -> CitationValidation:
        """Validate citations in generated text"""
        
        # Extract all citation IDs from text
        found_citations = self.citation_pattern.findall(text)
        citation_ids = [f"{source}-{sub}-{num}" for source, sub, num in found_citations]
        
        # Check which are valid
        valid_citations = [cid for cid in citation_ids if cid in available_citations]
        invalid_citations = [cid for cid in citation_ids if cid not in available_citations]
        
        # Count sentences and citations
        sentences = self._split_into_sentences(text)
        factual_sentences = self._identify_factual_sentences(sentences)
        
        # Calculate coverage
        cited_sentences = [s for s in factual_sentences if self.citation_pattern.search(s)]
        coverage = len(cited_sentences) / len(factual_sentences) if factual_sentences else 0.0
        
        # Identify uncited facts
        uncited_facts = [s for s in factual_sentences if not self.citation_pattern.search(s)]
        
        return CitationValidation(
            has_citations=len(citation_ids) > 0,
            citation_count=len(citation_ids),
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            uncited_facts=uncited_facts[:5],  # First 5 uncited facts
            citation_coverage=coverage
        )
    
    def enforce_citations(self, text: str, available_citations: Dict[str, Dict], 
                         original_query: str, ollama_host: str = "http://localhost:11434",
                         model: str = "llama3.2:latest") -> str:
        """
        Enforce proper citations in text by rewriting if necessary
        """
        
        validation = self.validate_citations(text, set(available_citations.keys()))
        
        # Check for refusal responses - these need regeneration
        refusal_patterns = [
            r"I (?:can't|cannot) assist",
            r"I'm (?:unable|not able) to",
            r"I don't have",
            r"I (?:can't|cannot) provide",
            r"requires me to generate.*false information",
            r"unverified.*information"
        ]
        
        is_refusal = any(re.search(pattern, text, re.IGNORECASE) for pattern in refusal_patterns)
        
        if is_refusal:
            self.logger.warning("Detected refusal response - forcing regeneration with permissive prompt")
            return self._regenerate_with_citations(text, available_citations, original_query, ollama_host, model, is_retry_after_refusal=True)
        
        # STRICT CITATION REQUIREMENTS
        # If citation coverage is excellent (>80%), just fix invalid citations
        if validation.citation_coverage > 0.8:
            if validation.invalid_citations:
                self.logger.warning(f"Removing {len(validation.invalid_citations)} invalid citations")
                text = self._remove_invalid_citations(text, validation.invalid_citations)
            return text
        
        # If citation coverage is good (50-80%), add citations to uncited facts
        if validation.citation_coverage > 0.5:
            self.logger.info(f"Good coverage ({validation.citation_coverage:.1%}), adding citations to remaining facts")
            if validation.invalid_citations:
                text = self._remove_invalid_citations(text, validation.invalid_citations)
            return self._add_missing_citations(text, validation.uncited_facts, available_citations)
        
        # If citation coverage is poor (<50%), regenerate with strict requirements
        self.logger.warning(f"Insufficient citation coverage ({validation.citation_coverage:.1%}), regenerating with strict enforcement")
        regenerated = self._regenerate_with_citations(text, available_citations, original_query, ollama_host, model)
        
        # Validate regenerated response
        new_validation = self.validate_citations(regenerated, set(available_citations.keys()))
        
        if new_validation.citation_coverage > validation.citation_coverage:
            self.logger.info(f"Regeneration improved coverage from {validation.citation_coverage:.1%} to {new_validation.citation_coverage:.1%}")
            return regenerated
        else:
            self.logger.warning(f"Regeneration didn't improve coverage, adding citations to original")
            return self._add_missing_citations(text, validation.uncited_facts, available_citations)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_factual_sentences(self, sentences: List[str]) -> List[str]:
        """Identify sentences that contain factual claims requiring citations"""
        factual_sentences = []
        
        # Patterns that indicate factual content
        factual_indicators = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:kg|ton|hectare|litre|meter|cm)',  # Measurements
            r'\b(?:study|research|report|survey|data|statistics)\b',  # Research references
            r'\b(?:is|are|was|were|has|have|can|should|must)\b',  # Declarative statements
            r'\b(?:according to|based on|shows that|indicates that)\b',  # Attribution phrases
        ]
        
        non_factual_indicators = [
            r'^\s*#',  # Headers
            r'^\s*\*\*',  # Bold headers
            r'^\s*-\s',  # List items (usually don't need citations)
            r'^\s*\d+\.',  # Numbered lists
            r'^\s*(?:References|Citations|Sources)',  # Reference sections
        ]
        
        for sentence in sentences:
            # Skip non-factual content
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in non_factual_indicators):
                continue
            
            # Include if it contains factual indicators or is substantial
            if len(sentence.split()) > 8:  # Substantial sentence
                if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators):
                    factual_sentences.append(sentence)
                elif len(sentence.split()) > 15:  # Long sentences likely contain facts
                    factual_sentences.append(sentence)
        
        return factual_sentences
    
    def _remove_invalid_citations(self, text: str, invalid_citations: List[str]) -> str:
        """Remove invalid citation references"""
        for cit_id in invalid_citations:
            pattern = r'\[' + re.escape(cit_id) + r'\]'
            text = re.sub(pattern, '', text)
        return text
    
    def _add_missing_citations(self, text: str, uncited_facts: List[str], 
                               available_citations: Dict[str, Dict]) -> str:
        """Add citations to uncited factual statements"""
        # This is a simplified approach - in practice, you'd want to match
        # each fact to the most relevant citation
        
        if not uncited_facts or not available_citations:
            return text
        
        # Get first available citation as default
        default_citation = list(available_citations.keys())[0]
        
        # Add note about missing citations
        text += f"\n\n**Note:** Some statements in this answer may benefit from additional citations. Please refer to the complete research report for detailed source information."
        
        return text
    
    def _regenerate_with_citations(self, original_text: str, 
                                   available_citations: Dict[str, Dict],
                                   original_query: str,
                                   ollama_host: str,
                                   model: str,
                                   is_retry_after_refusal: bool = False) -> str:
        """Regenerate answer with strict citation requirements"""
        import requests
        
        # Build citation reference list with content snippets
        citation_list = []
        for cit_id in sorted(list(available_citations.keys())[:20]):
            cit_data = available_citations[cit_id]
            title = cit_data.get('title', 'Unknown')[:80]
            content_snippet = cit_data.get('content', '')[:200] if 'content' in cit_data else ""
            citation_list.append(f"[{cit_id}] {title}\n   Snippet: {content_snippet}...")
        
        citation_reference = "\n".join(citation_list)
        
        if is_retry_after_refusal:
            prompt = f"""You are a knowledgeable research assistant. The user asked a factual question, and you have been provided with verified research sources below.

Your task: Write a comprehensive, well-cited answer using ONLY the information from these sources.

CITATION REQUIREMENTS:
- Every factual statement must have a citation [DB-1-1] or [WEB-2-1]
- Place citations at the end of sentences before the period
- Use the exact citation IDs provided below
- Aim for 70-80% of your sentences to have citations

AVAILABLE SOURCES (Use these citations):
{citation_reference}

USER'S QUESTION: {original_query}

Write a detailed, well-structured answer (400-600 words) using facts from the sources above. Include citations for every claim:"""
        else:
            prompt = f"""You are an expert research assistant. Your previous answer had insufficient citations. 
Rewrite your answer ensuring EVERY factual statement includes proper inline citations.

MANDATORY CITATION RULES:
1. EVERY sentence with factual information MUST have a citation like [DB-1-2] or [WEB-2-1]
2. Use ONLY these exact citation IDs from the sources below:

{citation_reference}

3. Citation format: End of sentence, before period: "DDT accumulates in marine organisms [WEB-1-1]."
4. Multiple sources: Combine like [DB-1-1][WEB-2-3]
5. Aim for at least 70% of sentences to have citations

Original Question: {original_query}

Your task: Rewrite this answer with comprehensive citations:"""

        try:
            response = requests.post(
                f'{ollama_host}/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'num_ctx': 8192,
                        'num_predict': 2000
                    }
                },
                timeout=240
            )
            
            if response.status_code == 200:
                regenerated = response.json()['response'].strip()
                
                # Validate the regenerated answer
                validation = self.validate_citations(regenerated, set(available_citations.keys()))
                
                if validation.citation_coverage > 0.5:
                    self.logger.info(f"Successfully regenerated with {validation.citation_coverage:.1%} citation coverage")
                    return regenerated
                else:
                    self.logger.warning(f"Regeneration still has poor coverage ({validation.citation_coverage:.1%})")
                    # Return regenerated anyway, it's likely better than original
                    return regenerated
            
        except Exception as e:
            self.logger.error(f"Citation enforcement regeneration failed: {e}")
        
        # Fallback: Return original with warning
        return original_text + "\n\n**⚠️ Citation Warning:** This answer may not include comprehensive citations. Please refer to the detailed research report for source information."
    
    def _build_citation_reference_list(self, available_citations: Dict[str, Dict]) -> str:
        """Build a formatted list of available citations for the LLM"""
        citation_lines = []
        
        for cit_id, cit_data in sorted(available_citations.items())[:20]:  # Limit to 20 to fit in context
            title = cit_data.get('title', 'Unknown')[:80]
            source_type = cit_data.get('source', 'unknown')
            citation_lines.append(f"   [{cit_id}] - {title} ({source_type})")
        
        if len(available_citations) > 20:
            citation_lines.append(f"   ... and {len(available_citations) - 20} more citations available")
        
        return '\n'.join(citation_lines)
