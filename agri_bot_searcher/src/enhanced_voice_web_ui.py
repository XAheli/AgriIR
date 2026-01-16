#!/usr/bin/env python3
"""
Enhanced AgriIR - Web UI with Integrated Voice Transcription
A comprehensive Flask web interface for the agriculture chatbot with voice transcription capabilities
"""

try:
    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import sys
import os
import json
import logging
import time
import tempfile
import base64
import subprocess
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

if HAS_FLASK:
    from enhanced_rag_system import EnhancedRAGSystem
    from agriir_voice_integration import AgriIRVoiceTranscriber, get_supported_languages
    
    # Initialize voice transcriber
    voice_transcriber = AgriIRVoiceTranscriber()
    
    app = Flask(__name__, template_folder='static')
    CORS(app)  # Enable CORS for all domains
    
    # Initialize RAG system
    enhanced_rag_system = None
    
    # Language mappings for AgriIR
    LANGUAGE_MAPPINGS = get_supported_languages()
    
    def get_enhanced_rag_system():
        """Get or create Enhanced RAG system instance"""
        global enhanced_rag_system
        if enhanced_rag_system is None:
            print("🔄 Initializing Enhanced RAG System...")
            
            # Only use relative path - embeddings should be in project root
            embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agriculture_embeddings')
            
            if os.path.exists(embeddings_dir):
                print(f"📁 Loading embeddings from: {embeddings_dir}")
            else:
                print("⚠️ No embeddings directory found - using web search only mode")
                print("📖 To set up embeddings, see: EMBEDDINGS_SETUP.md")
                embeddings_dir = None
            
            try:
                enhanced_rag_system = EnhancedRAGSystem(embeddings_dir=embeddings_dir)
                if embeddings_dir:
                    print("✅ Enhanced RAG System initialized with database and web search!")
                else:
                    print("✅ Enhanced RAG System initialized with web search only!")
                    print("💡 For better results, set up local embeddings (see EMBEDDINGS_SETUP.md)")
                return enhanced_rag_system
            except Exception as e:
                print(f"❌ Failed to initialize Enhanced RAG System: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return enhanced_rag_system
    
    def process_audio_file(audio_path, language_code, use_local_model=True, api_key=None, hf_token=None):
        """Process audio file using AgriIR voice transcription"""
        try:
            if voice_transcriber.is_available() or api_key:
                # Use the integrated AgriIR voice transcription
                original_text, english_text = voice_transcriber.transcribe_audio(
                    audio_path=audio_path,
                    language_code=language_code,
                    use_local_model=use_local_model,
                    api_key=api_key,
                    hf_token=hf_token
                )
                return original_text, english_text
            else:
                error_msg = "Voice transcription requires SarvamAI API key. Please enter your API key in the settings."
                return error_msg, error_msg
        except Exception as e:
            logging.error(f"Audio processing error: {e}")
            error_msg = f"Error processing audio: {str(e)}"
            return error_msg, error_msg

    @app.route('/')
    def index():
        """Render the main interface"""
        return render_template('index.html', languages=LANGUAGE_MAPPINGS)

    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon"""
        return '', 204  # No content

    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        """Handle voice transcription requests"""
        try:
            if 'audio' not in request.files:
                return jsonify({'success': False, 'error': 'No audio file provided'})
            
            audio_file = request.files['audio']
            language_code = request.form.get('language', 'hin_Deva')
            use_local_model = request.form.get('use_local_model', 'true').lower() == 'true'
            api_key = request.form.get('api_key', '')
            hf_token = request.form.get('hf_token', '')
            
            # Save uploaded audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_audio_path = tmp_file.name
            
            try:
                # Process audio using AgriIR voice transcription
                original_text, english_text = process_audio_file(
                    audio_path=temp_audio_path,
                    language_code=language_code,
                    use_local_model=use_local_model,
                    api_key=api_key if api_key else None,
                    hf_token=hf_token if hf_token else None
                )
                
                return jsonify({
                    'success': True,
                    'original': original_text,
                    'english': english_text
                })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/chat', methods=['POST'])
    def chat():
        """Handle chat requests with enhanced RAG pipeline"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No JSON data provided'})
                
            query = data.get('query', '').strip()
            
            # Enhanced RAG parameters
            num_sub_queries = data.get('num_sub_queries', 3)
            db_chunks_per_query = data.get('db_chunks_per_query', 5)
            web_results_per_query = data.get('web_results_per_query', 3)
            enable_database_search = data.get('enable_database_search', True)
            enable_web_search = data.get('enable_web_search', True)
            synthesis_model = data.get('synthesis_model', 'gemma3:27b')
            
            # Legacy parameters for backward compatibility
            num_agents = data.get('num_agents', 2)
            base_port = data.get('base_port', 11434)
            
            if not query:
                return jsonify({'success': False, 'error': 'No query provided'})
            
            start_time = time.time()
            
            # Get enhanced RAG system
            rag_system = get_enhanced_rag_system()
            
            # Process the query with enhanced RAG
            if rag_system:
                print(f"🔍 Processing query with Enhanced RAG: {query}")
                print(f"📊 Parameters: sub_queries={num_sub_queries}, db_chunks={db_chunks_per_query}, web_results={web_results_per_query}")
                print(f"🔧 Database search: {enable_database_search}, Web search: {enable_web_search}")
                
                result = rag_system.process_query(
                    user_query=query,
                    num_sub_queries=num_sub_queries,
                    db_chunks_per_query=db_chunks_per_query,
                    web_results_per_query=web_results_per_query,
                    enable_database_search=enable_database_search,
                    enable_web_search=enable_web_search,
                    synthesis_model=synthesis_model
                )
                
                processing_time = time.time() - start_time
                
                # Log detailed pipeline information
                print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
                if 'pipeline_info' in result:
                    pipeline = result['pipeline_info']
                    print(f"📝 Refined query: {pipeline.get('refined_query', 'N/A')}")
                    print(f"🔗 Sub-queries generated: {len(pipeline.get('sub_queries', []))}")
                    print(f"📚 Database chunks retrieved: {pipeline.get('total_db_chunks', 0)}")
                    print(f"🌐 Web results retrieved: {pipeline.get('total_web_results', 0)}")
                
                # Enhanced response format for frontend
                enhanced_result = {
                    'success': True,
                    'response': result.get('answer', 'No answer generated'),  # Changed from 'answer' to 'response'
                    'enhanced_rag': True,  # Flag to indicate enhanced RAG was used
                    'processing_time': round(processing_time, 2),
                    'pipeline_info': result.get('pipeline_info', {}),
                    'markdown_content': result.get('markdown_content', ''),
                    'citations': result.get('citations', []),  # Citations used in the answer
                    'all_citations': result.get('all_citations', []),  # All available citations
                    'citations_count': len(result.get('citations', [])),
                    'sub_queries_count': len(result.get('pipeline_info', {}).get('sub_queries', [])),
                    'db_results_count': result.get('pipeline_info', {}).get('total_db_chunks', 0),
                    'web_results_count': result.get('pipeline_info', {}).get('total_web_results', 0),
                    'search_stats': {
                        'total_db_chunks': result.get('pipeline_info', {}).get('total_db_chunks', 0),
                        'total_web_results': result.get('pipeline_info', {}).get('total_web_results', 0),
                        'sub_queries_processed': len(result.get('pipeline_info', {}).get('sub_queries', [])),
                        'citations_used': len(result.get('citations', [])),
                        'total_citations_available': len(result.get('all_citations', []))
                    },
                    'sub_query_results': result.get('pipeline_info', {}).get('sub_query_results', [])
                }
                
                # Log citation information for debugging
                citations = result.get('citations', [])
                all_citations = result.get('all_citations', [])
                print(f"📖 Citations in response: {len(citations)} used out of {len(all_citations)} available")
                if citations:
                    citation_ids = [c.get('id', 'Unknown') for c in citations]
                    print(f"📋 Citation IDs used: {citation_ids}")
                else:
                    print("⚠️ No citations found in the response!")
                
                return jsonify(enhanced_result)
                
            else:
                # Enhanced RAG system not available
                print(f"⚠️ Enhanced RAG system not available, using fallback")
                processing_time = time.time() - start_time
                
                response = f"Enhanced RAG system is not available. Please ensure the embeddings database is properly set up. Query: {query}"
                
                return jsonify({
                    'success': False,
                    'error': 'Enhanced RAG system not available',
                    'answer': response,
                    'processing_time': round(processing_time, 2),
                    'pipeline_info': {'message': 'Enhanced RAG system not available'},
                    'markdown_content': response,
                    'citations': [],
                    'sub_queries_count': 0,
                    'db_results_count': 0,
                    'web_results_count': 0
                })
            
        except Exception as e:
            logging.error(f"Chat error: {e}")
            print(f"❌ Chat processing error: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint with comprehensive status"""
        try:
            # Check embeddings directory
            embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agriculture_embeddings')
            embeddings_available = os.path.exists(embeddings_dir)
            embeddings_path = embeddings_dir if embeddings_available else None
            
            # Check RAG system
            rag_available = False
            try:
                rag_system = get_enhanced_rag_system()
                rag_available = rag_system is not None
            except Exception:
                rag_available = False
            
            # Check voice system
            voice_available = voice_transcriber.is_available()
            
            # Check Ollama models
            ollama_models = 0
            try:
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    ollama_models = len([line for line in lines if line.strip()])
            except Exception:
                ollama_models = 0
            
            return jsonify({
                'status': 'healthy',
                'voice_transcription_available': voice_available,
                'embeddings_available': embeddings_available,
                'embeddings_path': embeddings_path,
                'mode': 'enhanced' if embeddings_available else 'web-only',
                'components': {
                    'enhanced_rag': rag_available,
                    'voice_transcriber': voice_available,
                    'ollama_models': ollama_models,
                    'embeddings_database': embeddings_available
                },
                'setup_instructions': 'EMBEDDINGS_SETUP.md' if not embeddings_available else None,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'embeddings_available': False,
                'mode': 'web-only',
                'components': {
                    'enhanced_rag': False,
                    'voice_transcriber': False,
                    'ollama_models': 0,
                    'embeddings_database': False
                },
                'timestamp': datetime.now().isoformat()
            })

    @app.route('/models', methods=['GET'])
    def get_models():
        """Get available Ollama models"""
        try:
            # Try to get models from Ollama
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(model_name)
                
                # Define recommended models based on what's available
                recommended_models = []
                preferred_order = ['llama3.2:latest', 'llama3.2:3b', 'llama3.1:latest', 'gemma2:latest', 'mistral:latest']
                
                for preferred in preferred_order:
                    if preferred in models:
                        recommended_models.append(preferred)
                
                # If no preferred models found, recommend the first few available
                if not recommended_models and models:
                    recommended_models = models[:3]
                
                return jsonify({
                    'success': True,
                    'models': models,
                    'recommended': recommended_models,
                    'count': len(models)
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Ollama not available',
                    'models': [],
                    'recommended': [],
                    'count': 0
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'models': [],
                'recommended': [],
                'count': 0
            })

    @app.route('/status', methods=['GET'])
    def get_status():
        """Get system status for RAG, voice, and models"""
        try:
            status = {
                'rag_system': 'unknown',
                'voice_system': 'available' if voice_transcriber.is_available() else 'unavailable',
                'models_available': False
            }
            
            # Check RAG system
            try:
                rag_system = get_enhanced_rag_system()
                if rag_system:
                    status['rag_system'] = 'enhanced'
                else:
                    status['rag_system'] = 'unavailable'
            except Exception:
                status['rag_system'] = 'error'
            
            # Check models
            try:
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                status['models_available'] = result.returncode == 0
            except Exception:
                status['models_available'] = False
            
            return jsonify({'success': True, 'status': status})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        if not HAS_FLASK:
            print("Flask not available. Please install with: pip install flask flask-cors")
            return
            
        print(f"🌾 Starting AgriIR Web Interface on http://{host}:{port}")
        print(f"Voice transcription: {'✓ Available' if voice_transcriber.is_available() else '✗ Not available'}")
        
        # Initialize RAG system during startup
        print("🔄 Pre-loading Enhanced RAG System...")
        rag_system = get_enhanced_rag_system()
        if rag_system:
            print("✅ Enhanced RAG System loaded successfully!")
        else:
            print("⚠️ Enhanced RAG System failed to load - will show as unavailable")
        
        app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AgriIR - Enhanced Voice & Text Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if HAS_FLASK:
        run_server(host=args.host, port=args.port, debug=args.debug)
    else:
        print("Flask not available. Please install requirements first.")
