#!/usr/bin/env python3
"""
Agriculture Multi-Agent Chatbot System

A specialized chatbot that deploys multiple Ollama agents to search the internet
and provide comprehensive answers to agricultural queries with inline citations.

Features:
- Multi-agent collaboration using Ollama models
- Web search integration for real-time information
- Inline citation system for credible sources
- Agriculture-specific query enhancement
- Robust error handling and fallback mechanisms
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import requests
from ddgs import DDGS
import re
from urllib.parse import urlparse


class AgentRole(Enum):
    """Specialized agent roles for agricultural queries"""
    CROP_SPECIALIST = "crop_specialist"
    DISEASE_EXPERT = "disease_expert"
    ECONOMICS_ANALYST = "economics_analyst"
    CLIMATE_RESEARCHER = "climate_researcher"
    TECHNOLOGY_ADVISOR = "technology_advisor"
    POLICY_ANALYST = "policy_analyst"


@dataclass
class SearchResult:
    """Search result with metadata for citations"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: float = 0.0
    domain: str = ""
    timestamp: float = 0.0


@dataclass
class AgentResponse:
    """Response from an individual agent"""
    agent_id: int
    role: AgentRole
    port: int
    content: str
    search_results: List[SearchResult] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class AgricultureSearchEngine:
    """Specialized search engine for agricultural content"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.agriculture_domains = [
            'extension.org', 'agric.gov', 'fao.org', 'usda.gov', 
            'icrisat.org', 'cgiar.org', 'cimmyt.org', 'irri.org',
            'croplife.org', 'agprofessional.com', 'agriculture.com'
        ]
        
    def enhance_agriculture_query(self, query: str) -> List[str]:
        """Generate enhanced search queries for agricultural topics"""
        base_queries = [query]
        
        # Add agriculture-specific terms
        agriculture_terms = [
            "agriculture", "farming", "crop", "livestock", 
            "agricultural practices", "farming techniques"
        ]
        
        # Create enhanced queries
        enhanced_queries = []
        for term in agriculture_terms[:2]:  # Limit to avoid too many queries
            enhanced_queries.append(f"{query} {term}")
        
        # Add specific queries based on query content
        if any(word in query.lower() for word in ['disease', 'pest', 'infection']):
            enhanced_queries.append(f"{query} plant pathology treatment")
        elif any(word in query.lower() for word in ['price', 'cost', 'market', 'economic']):
            enhanced_queries.append(f"{query} agricultural economics market analysis")
        elif any(word in query.lower() for word in ['climate', 'weather', 'rain']):
            enhanced_queries.append(f"{query} climate agriculture impact")
        
        return base_queries + enhanced_queries[:3]  # Limit total queries
    
    def search(self, query: str) -> List[SearchResult]:
        """Perform web search with agriculture focus"""
        try:
            ddgs = DDGS()
            raw_results = ddgs.text(query, max_results=self.max_results)
            
            search_results = []
            for result in raw_results:
                url = result.get('href', '')
                domain = urlparse(url).netloc.lower()
                
                # Calculate relevance score (higher for agriculture domains)
                relevance_score = 1.0
                if any(ag_domain in domain for ag_domain in self.agriculture_domains):
                    relevance_score = 2.0
                
                search_results.append(SearchResult(
                    title=result.get('title', ''),
                    url=url,
                    snippet=result.get('body', ''),
                    domain=domain,
                    relevance_score=relevance_score,
                    timestamp=time.time()
                ))
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return search_results
            
        except Exception as e:
            logging.warning(f"Search failed for query '{query}': {e}")
            return []


class AgricultureAgent:
    """Individual agent specialized for agricultural analysis"""
    
    def __init__(self, agent_id: int, role: AgentRole, port: int, model: str = "gemma3:1b"):
        self.agent_id = agent_id
        self.role = role
        self.port = port
        self.model = model
        # Use environment variable for host or default to localhost
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        if ':' in ollama_host:
            host, _ = ollama_host.split(':', 1)
        else:
            host = ollama_host
        self.base_url = f"http://{host}:{port}"
        self.search_engine = AgricultureSearchEngine()
        
        # Role-specific system prompts
        self.system_prompts = {
            AgentRole.CROP_SPECIALIST: "You are an agricultural crop specialist. Focus on crop varieties, cultivation practices, yield optimization, and crop management techniques.",
            AgentRole.DISEASE_EXPERT: "You are a plant pathology expert. Focus on plant diseases, pest management, diagnosis, and treatment options.",
            AgentRole.ECONOMICS_ANALYST: "You are an agricultural economics analyst. Focus on market trends, pricing, economic impacts, and financial aspects of agriculture.",
            AgentRole.CLIMATE_RESEARCHER: "You are a climate and agriculture researcher. Focus on climate impacts, weather patterns, adaptation strategies, and environmental factors.",
            AgentRole.TECHNOLOGY_ADVISOR: "You are an agricultural technology advisor. Focus on modern farming technologies, precision agriculture, and innovative solutions.",
            AgentRole.POLICY_ANALYST: "You are an agricultural policy analyst. Focus on policies, regulations, government programs, and institutional factors."
        }
    
    def search_and_analyze(self, query: str, num_searches: int = 2) -> AgentResponse:
        """Search the web and analyze findings from agent's perspective"""
        start_time = time.time()
        
        try:
            # Generate enhanced queries
            search_queries = self.search_engine.enhance_agriculture_query(query)[:num_searches]
            
            # Perform searches
            all_search_results = []
            for search_query in search_queries:
                results = self.search_engine.search(search_query)
                all_search_results.extend(results)
            
            # Remove duplicates and limit results
            unique_results = {}
            for result in all_search_results:
                if result.url not in unique_results:
                    unique_results[result.url] = result
            
            search_results = list(unique_results.values())[:5]  # Limit to top 5
            
            # Create search context for LLM
            search_context = self._format_search_context(search_results)
            
            # Generate analysis using LLM
            analysis = self._generate_analysis(query, search_context)
            
            # Extract citations
            citations = self._extract_citations(search_results)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                port=self.port,
                content=analysis,
                search_results=search_results,
                citations=citations,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                port=self.port,
                content=f"Analysis failed: {str(e)}",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _format_search_context(self, search_results: List[SearchResult]) -> str:
        """Format search results as context for LLM"""
        if not search_results:
            return "No search results available."
        
        context = "Web Search Results:\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"{i}. **{result.title}**\n"
            context += f"   Source: {result.url}\n"
            context += f"   Domain: {result.domain}\n"
            context += f"   Content: {result.snippet}\n\n"
        
        return context
    
    def _generate_analysis(self, query: str, search_context: str) -> str:
        """Generate analysis using Ollama LLM"""
        system_prompt = self.system_prompts.get(self.role, "You are an agricultural expert.")
        
        prompt = f"""System: {system_prompt}

Query: {query}

{search_context}

Please provide a comprehensive analysis from your {self.role.value.replace('_', ' ')} perspective. 
Include specific insights, recommendations, and reference the search results with inline citations [1], [2], etc.
Focus on practical, actionable information for farmers and agricultural professionals.

Your response should be well-structured and informative."""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def _extract_citations(self, search_results: List[SearchResult]) -> List[str]:
        """Extract citations from search results"""
        citations = []
        for i, result in enumerate(search_results, 1):
            citation = f"[{i}] {result.title}. {result.domain}. {result.url}"
            citations.append(citation)
        return citations


class AgricultureChatbot:
    """Main chatbot orchestrator for agriculture queries"""
    
    def __init__(self, base_port: int = 11434, num_agents: int = 3):
        self.base_port = base_port
        self.num_agents = num_agents
        self.available_roles = list(AgentRole)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AgricultureChatbot')
    
    def check_ollama_instances(self) -> List[int]:
        """Check which Ollama instances are available"""
        available_ports = []
        
        # Get Ollama host from environment variable
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        if ':' in ollama_host:
            host, _ = ollama_host.split(':', 1)
        else:
            host = ollama_host
        
        for i in range(self.num_agents):
            port = self.base_port + i
            try:
                response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
                if response.status_code == 200:
                    available_ports.append(port)
                    self.logger.info(f"Ollama instance available on port {port}")
                else:
                    self.logger.warning(f"Ollama instance on port {port} returned {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Ollama instance on port {port} not available: {e}")
        
        return available_ports
    
    def answer_query(self, query: str, num_searches: int = 2, exact_answer: bool = False) -> Dict[str, Any]:
        """Answer agricultural query using multiple agents"""
        self.logger.info(f"Processing query: {query}")
        
        # Check available Ollama instances
        available_ports = self.check_ollama_instances()
        
        if not available_ports:
            return {
                "success": False,
                "error": "No Ollama instances available",
                "answer": "Sorry, the chatbot service is currently unavailable.",
                "citations": []
            }
        
        # Create agents
        agents = []
        num_active_agents = min(len(available_ports), len(self.available_roles))
        
        for i in range(num_active_agents):
            role = self.available_roles[i % len(self.available_roles)]
            port = available_ports[i]
            agent = AgricultureAgent(i + 1, role, port)
            agents.append(agent)
        
        self.logger.info(f"Deploying {len(agents)} agents")
        
        # Deploy agents in parallel
        responses = []
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            future_to_agent = {
                executor.submit(agent.search_and_analyze, query, num_searches): agent
                for agent in agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    response = future.result(timeout=120)  # 2 minute timeout
                    responses.append(response)
                    self.logger.info(f"Agent {agent.agent_id} ({agent.role.value}) completed in {response.execution_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"Agent {agent.agent_id} ({agent.role.value}) failed: {e}")
                    responses.append(AgentResponse(
                        agent_id=agent.agent_id,
                        role=agent.role,
                        port=agent.port,
                        content=f"Agent failed: {str(e)}",
                        success=False,
                        error_message=str(e)
                    ))
        
        # Synthesize responses
        return self._synthesize_responses(query, responses, exact_answer)
    
    def _synthesize_responses(self, query: str, responses: List[AgentResponse], exact_answer: bool = False) -> Dict[str, Any]:
        """Synthesize agent responses into final answer"""
        successful_responses = [r for r in responses if r.success]
        failed_responses = [r for r in responses if not r.success]
        
        if not successful_responses:
            return {
                "success": False,
                "error": "All agents failed",
                "answer": "Sorry, I couldn't process your query due to technical issues.",
                "citations": []
            }
        
        # Combine all citations
        all_citations = []
        citation_map = {}
        citation_counter = 1
        
        for response in successful_responses:
            for citation in response.citations:
                if citation not in citation_map:
                    citation_map[citation] = citation_counter
                    all_citations.append(f"[{citation_counter}] {citation[4:]}")  # Remove [X] prefix
                    citation_counter += 1
        
        # Calculate total time
        total_time = sum(r.execution_time for r in responses if r.success)
        
        # Create synthesized answer
        if exact_answer:
            # Generate a concise, focused answer using LLM synthesis
            answer = self._generate_exact_answer(query, successful_responses, all_citations)
        else:
            # Use the original verbose format
            answer = f"# Agricultural Analysis: {query}\n\n"
            
            for response in successful_responses:
                if response.success and response.content:
                    role_name = response.role.value.replace('_', ' ').title()
                    answer += f"## {role_name} Perspective\n\n"
                    answer += f"{response.content}\n\n"
            
            # Add statistics
            if failed_responses:
                answer += f"---\n\n*Note: {len(failed_responses)} agent(s) encountered issues during analysis.*\n\n"
            
            answer += f"---\n\n**Sources and Citations:**\n\n"
            for citation in all_citations:
                answer += f"{citation}\n"
            
            answer += f"\n*Analysis completed using {len(successful_responses)} specialized agricultural agents in {total_time:.1f} seconds.*"
        
        return {
            "success": True,
            "answer": answer,
            "citations": all_citations,
            "agent_count": len(successful_responses),
            "total_time": total_time,
            "failed_agents": len(failed_responses)
        }
    
    def _generate_exact_answer(self, query: str, successful_responses: List[AgentResponse], all_citations: List[str]) -> str:
        """Generate a concise, focused answer using LLM synthesis"""
        try:
            # Combine all agent insights
            combined_insights = "\n\n".join([
                f"**{response.role.value.replace('_', ' ').title()} Analysis:**\n{response.content}"
                for response in successful_responses if response.success and response.content
            ])
            
            # Create synthesis prompt
            synthesis_prompt = f"""You are an expert agricultural advisor. Based on the following comprehensive research from multiple specialists, provide a CONCISE, PRACTICAL, and DIRECT answer to the farmer's question.

QUESTION: {query}

RESEARCH FINDINGS:
{combined_insights}

INSTRUCTIONS:
1. Answer the question directly and concisely
2. Focus only on the most important, actionable information
3. Use clear, simple language that farmers can understand
4. Include key recommendations or steps
5. Reference sources using [1], [2], etc. format
6. Keep the answer under 300 words
7. Stay strictly on topic - no unnecessary elaboration

ANSWER:"""

            # Use the first available port to generate synthesis
            available_ports = self.check_ollama_instances()
            if not available_ports:
                return "Error: No Ollama instances available for synthesis"
            
            synthesis_port = available_ports[0]
            
            # Get Ollama host from environment variable
            ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
            if ':' in ollama_host:
                host, _ = ollama_host.split(':', 1)
            else:
                host = ollama_host
            
            response = requests.post(
                f"http://{host}:{synthesis_port}/api/generate",
                json={
                    "model": "gemma3:1b",
                    "prompt": synthesis_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                exact_answer = result.get('response', 'Failed to generate exact answer')
                
                # Add citations
                exact_answer += "\n\n**Sources:**\n"
                for citation in all_citations:
                    exact_answer += f"{citation}\n"
                
                return exact_answer
            else:
                return f"Error generating exact answer: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error generating exact answer: {str(e)}"


def main():
    """Main function for testing the chatbot"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agriculture Multi-Agent Chatbot")
    parser.add_argument("--query", "-q", type=str, required=True, help="Agricultural query to answer")
    parser.add_argument("--agents", "-a", type=int, default=3, help="Number of agents to deploy")
    parser.add_argument("--searches", "-s", type=int, default=2, help="Number of searches per agent")
    parser.add_argument("--port", "-p", type=int, default=11434, help="Base Ollama port")
    parser.add_argument("--exact", "-e", action="store_true", help="Generate concise, exact answer instead of detailed analysis")
    
    args = parser.parse_args()
    
    # Create chatbot
    chatbot = AgricultureChatbot(base_port=args.port, num_agents=args.agents)
    
    # Process query
    print(f"\n🌾 Agriculture Chatbot - Processing Query: {args.query}\n")
    print("=" * 80)
    
    start_time = time.time()
    result = chatbot.answer_query(args.query, args.searches, args.exact)
    total_time = time.time() - start_time
    
    if result["success"]:
        print(result["answer"])
        print(f"\n{'=' * 80}")
        print(f"✅ Query processed successfully in {total_time:.1f} seconds")
        print(f"📊 Agents used: {result['agent_count']}, Citations: {len(result['citations'])}")
        if result.get("failed_agents", 0) > 0:
            print(f"⚠️  {result['failed_agents']} agent(s) failed")
    else:
        print(f"❌ Error: {result['error']}")
        print(result["answer"])
    
    print()


if __name__ == "__main__":
    main()
