#!/usr/bin/env python3
"""
AgriIR Benchmark Runner

This script runs comprehensive benchmarks on the agricultural question answering system.
It tests the system with various question types and generates detailed performance metrics.
"""

import os
import sys
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR / "agri_bot_searcher" / "src"))

from enhanced_rag_system import EnhancedRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BenchmarkRunner:
    """Run benchmarks on the AgriIR system"""
    
    def __init__(self, 
                 input_csv: str,
                 output_csv: str,
                 web_search_results: int = 8,
                 db_search_results: int = 3,
                 ollama_model: str = "gemma3:27b"):
        """
        Initialize benchmark runner
        
        Args:
            input_csv: Path to input CSV with questions
            output_csv: Path to output CSV for results
            web_search_results: Number of web search results (default 8)
            db_search_results: Number of database search results (default 3)
            ollama_model: Ollama model to use
        """
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        self.web_search_results = web_search_results
        self.db_search_results = db_search_results
        self.ollama_model = ollama_model
        self.ollama_host = "http://localhost:11434"
        
        # Pre-load required models to avoid 404 errors during benchmark
        self._preload_models()
        
        # Initialize RAG system
        logging.info("🚀 Initializing Enhanced RAG System...")
        self.rag_system = EnhancedRAGSystem(
            embeddings_dir="agriculture_embeddings",
            ollama_host=self.ollama_host
        )
        
        # Create output file with headers immediately
        self._initialize_output_csv()
        
        logging.info(f"✅ Benchmark runner initialized")
        logging.info(f"📊 Configuration:")
        logging.info(f"   - Web search results: {web_search_results}")
        logging.info(f"   - DB search results: {db_search_results}")
        logging.info(f"   - Ollama model: {ollama_model}")
    
    def _preload_models(self):
        """Pre-load required Ollama models to avoid 404 errors during benchmark"""
        import requests
        
        # Models that will be used during benchmark
        required_models = [
            self.ollama_model,      # Main synthesis model (e.g., gemma3:27b)
            "llama3.2:latest",      # Used by web scraper for article selection
            "gemma3:1b"             # Used for sub-query generation
        ]
        
        logging.info("🔄 Pre-loading required Ollama models...")
        
        for model in required_models:
            try:
                logging.info(f"   Loading {model}...")
                # Make a small test request to load the model into memory
                response = requests.post(
                    f'{self.ollama_host}/api/generate',
                    json={
                        'model': model,
                        'prompt': 'test',
                        'stream': False,
                        'options': {'num_predict': 1}
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    logging.info(f"   ✅ {model} loaded successfully")
                elif response.status_code == 404:
                    logging.warning(f"   ⚠️  {model} not found - may cause errors during benchmark")
                else:
                    logging.warning(f"   ⚠️  {model} returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logging.warning(f"   ⏱️  {model} loading timed out - continuing anyway")
            except Exception as e:
                logging.warning(f"   ⚠️  Error pre-loading {model}: {e}")
        
        logging.info("✅ Model pre-loading complete")
    
    def _initialize_output_csv(self):
        """Initialize output CSV with headers"""
        headers = [
            'id',
            'question',
            'answer',
            'citations',
            'citation_count',
            'web_sources_used',
            'db_sources_used',
            'time_taken_seconds',
            'tokens_generated',
            'model_used',
            'timestamp',
            'status',
            'error_message'
        ]
        
        # Create parent directory if needed
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Write headers
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        logging.info(f"📝 Output CSV initialized: {self.output_csv}")
    
    def _write_result_immediately(self, result: Dict[str, Any]):
        """Write a single result to CSV immediately"""
        headers = [
            'id', 'question', 'answer', 'citations', 'citation_count',
            'web_sources_used', 'db_sources_used', 'time_taken_seconds',
            'tokens_generated', 'model_used', 'timestamp', 'status', 'error_message'
        ]
        
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(result)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Ensure OS writes to disk
    
    def _extract_citations(self, response: Dict[str, Any]) -> List[str]:
        """Extract citations from response"""
        citations = []
        
        # Get used citations from the response
        used_citations = response.get('citations', [])
        
        for citation in used_citations:
            if isinstance(citation, dict):
                # Format: "Title - URL" or just URL
                title = citation.get('title', '')
                url = citation.get('url', citation.get('link', ''))
                if url:
                    if title:
                        citations.append(f"{title} - {url}")
                    else:
                        citations.append(url)
            elif isinstance(citation, str):
                citations.append(citation)
        
        return citations
    
    def process_question(self, question_id: int, question: str) -> Dict[str, Any]:
        """
        Process a single question and return results
        
        Args:
            question_id: Question ID
            question: Question text
            
        Returns:
            Dictionary with results
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing Question {question_id}: {question}")
        logging.info(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Get answer from RAG system using process_query
            response = self.rag_system.process_query(
                user_query=question,
                num_sub_queries=3,
                db_chunks_per_query=self.db_search_results,
                web_results_per_query=self.web_search_results,
                synthesis_model=self.ollama_model,
                enable_database_search=True,
                enable_web_search=True
            )
            
            time_taken = time.time() - start_time
            
            # Extract information from response
            answer = response.get('answer', '')
            citations = self._extract_citations(response)
            
            # Get stats from pipeline_info
            pipeline_info = response.get('pipeline_info', {})
            web_sources = pipeline_info.get('total_web_results', 0)
            db_sources = pipeline_info.get('total_db_chunks', 0)
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            tokens_generated = len(answer) // 4
            
            result = {
                'id': question_id,
                'question': question,
                'answer': answer,
                'citations': json.dumps(citations, ensure_ascii=False),
                'citation_count': len(citations),
                'web_sources_used': web_sources,
                'db_sources_used': db_sources,
                'time_taken_seconds': round(time_taken, 2),
                'tokens_generated': tokens_generated,
                'model_used': self.ollama_model,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'error_message': ''
            }
            
            logging.info(f"✅ Question {question_id} completed in {time_taken:.2f}s")
            logging.info(f"   - Answer length: {len(answer)} chars")
            logging.info(f"   - Citations: {len(citations)}")
            logging.info(f"   - Web sources: {web_sources}, DB sources: {db_sources}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            logging.error(f"❌ Error processing question {question_id}: {e}")
            
            result = {
                'id': question_id,
                'question': question,
                'answer': '',
                'citations': '[]',
                'citation_count': 0,
                'web_sources_used': 0,
                'db_sources_used': 0,
                'time_taken_seconds': round(time_taken, 2),
                'tokens_generated': 0,
                'model_used': self.ollama_model,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error_message': str(e)
            }
        
        return result
    
    def run_benchmark(self):
        """Run benchmark on all questions from input CSV"""
        if not self.input_csv.exists():
            logging.error(f"❌ Input CSV not found: {self.input_csv}")
            return
        
        # Read questions
        questions = []
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    'id': row.get('id', row.get('ID', '')),
                    'question': row.get('query_question', row.get('question', ''))
                })
        
        total_questions = len(questions)
        logging.info(f"\n{'='*80}")
        logging.info(f"🚀 STARTING BENCHMARK")
        logging.info(f"{'='*80}")
        logging.info(f"📊 Total questions: {total_questions}")
        logging.info(f"📁 Input: {self.input_csv}")
        logging.info(f"📁 Output: {self.output_csv}")
        logging.info(f"{'='*80}\n")
        
        benchmark_start = time.time()
        successful = 0
        failed = 0
        
        # Process each question
        for idx, question_data in enumerate(questions, 1):
            question_id = question_data['id']
            question = question_data['question']
            
            logging.info(f"\n[{idx}/{total_questions}] Processing question {question_id}...")
            
            # Process question
            result = self.process_question(question_id, question)
            
            # Write result immediately
            self._write_result_immediately(result)
            logging.info(f"💾 Result written to CSV immediately")
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            # Progress update
            logging.info(f"📈 Progress: {idx}/{total_questions} ({(idx/total_questions)*100:.1f}%)")
            logging.info(f"✅ Successful: {successful}, ❌ Failed: {failed}")
        
        # Final summary
        total_time = time.time() - benchmark_start
        
        logging.info(f"\n{'='*80}")
        logging.info(f"🎉 BENCHMARK COMPLETED")
        logging.info(f"{'='*80}")
        logging.info(f"📊 Statistics:")
        logging.info(f"   - Total questions: {total_questions}")
        logging.info(f"   - Successful: {successful}")
        logging.info(f"   - Failed: {failed}")
        logging.info(f"   - Total time: {total_time/60:.2f} minutes")
        logging.info(f"   - Average time per question: {total_time/total_questions:.2f} seconds")
        logging.info(f"📁 Results saved to: {self.output_csv}")
        logging.info(f"{'='*80}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AgriIR Benchmark Runner')
    parser.add_argument('--input', '-i', 
                       default='benchmark/final_agri_query.csv',
                       help='Input CSV file with questions')
    parser.add_argument('--output', '-o',
                       default='benchmark/benchmark_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--web-results', '-w',
                       type=int, default=8,
                       help='Number of web search results (default: 8)')
    parser.add_argument('--db-results', '-d',
                       type=int, default=3,
                       help='Number of database search results (default: 3)')
    parser.add_argument('--model', '-m',
                       default='gemma3:27b',
                       help='Ollama model to use (default: gemma3:27b)')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        input_csv=args.input,
        output_csv=args.output,
        web_search_results=args.web_results,
        db_search_results=args.db_results,
        ollama_model=args.model
    )
    
    # Run benchmark
    try:
        runner.run_benchmark()
    except KeyboardInterrupt:
        logging.info("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        logging.error(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
