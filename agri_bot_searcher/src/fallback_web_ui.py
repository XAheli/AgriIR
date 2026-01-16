#!/usr/bin/env python3
"""
Enhanced Web UI with Fallback Support
Automatically falls back to legacy mode if enhanced RAG is not available
"""

try:
    from flask import Flask, request, jsonify, render_template_string, send_file
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import sys
import os
import json
import logging
import time
import requests
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Try to import enhanced RAG system
try:
    from enhanced_rag_system import EnhancedRAGSystem
    HAS_ENHANCED_RAG = True
except ImportError:
    HAS_ENHANCED_RAG = False

# Try to import legacy chatbot
try:
    from agriculture_chatbot import AgricultureChatbot
    HAS_LEGACY_CHATBOT = True
except ImportError:
    HAS_LEGACY_CHATBOT = False

if HAS_FLASK:
    app = Flask(__name__)
    CORS(app)

    # Configuration
    EMBEDDINGS_DIR = "/store/Answering_Agriculture/agriculture_embeddings"
    OLLAMA_HOST = "http://localhost:11434"
    
    # Global instances
    rag_system = None
    legacy_chatbot = None

    def get_rag_system():
        """Get or create RAG system instance with error handling"""
        global rag_system
        if rag_system is None and HAS_ENHANCED_RAG:
            try:
                if os.path.exists(EMBEDDINGS_DIR):
                    rag_system = EnhancedRAGSystem(EMBEDDINGS_DIR, OLLAMA_HOST)
                    logging.info("Enhanced RAG system initialized successfully")
                else:
                    logging.warning(f"Embeddings directory not found: {EMBEDDINGS_DIR}")
            except Exception as e:
                logging.error(f"Failed to initialize RAG system: {e}")
        return rag_system

    def get_legacy_chatbot(base_port=11434, num_agents=2):
        """Get or create legacy chatbot instance with error handling"""
        global legacy_chatbot
        if legacy_chatbot is None and HAS_LEGACY_CHATBOT:
            try:
                legacy_chatbot = AgricultureChatbot(base_port=base_port, num_agents=num_agents)
                logging.info("Legacy chatbot initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize legacy chatbot: {e}")
        return legacy_chatbot

    @app.route('/')
    def index():
        """Main page - redirect to appropriate UI"""
        rag_sys = get_rag_system()
        
        if rag_sys is not None:
            # Enhanced RAG is available, use enhanced UI
            try:
                from enhanced_web_ui import HTML_TEMPLATE
                return render_template_string(HTML_TEMPLATE)
            except ImportError:
                # Fallback to basic interface
                return render_template_string(FALLBACK_TEMPLATE)
        else:
            # Use legacy interface
            try:
                from web_ui import HTML_TEMPLATE as LEGACY_TEMPLATE
                return render_template_string(LEGACY_TEMPLATE)
            except ImportError:
                return render_template_string(FALLBACK_TEMPLATE)

    @app.route('/api/system-status')
    def system_status():
        """Get system component status"""
        rag_sys = get_rag_system()
        legacy_bot = get_legacy_chatbot()
        
        return jsonify({
            'enhanced_rag_available': rag_sys is not None,
            'legacy_chatbot_available': legacy_bot is not None,
            'embeddings_available': os.path.exists(EMBEDDINGS_DIR),
            'has_enhanced_dependencies': HAS_ENHANCED_RAG,
            'has_legacy_dependencies': HAS_LEGACY_CHATBOT,
            'mode': 'enhanced' if rag_sys is not None else 'legacy'
        })

    @app.route('/api/enhanced-query', methods=['POST'])
    def enhanced_query():
        """Process query using enhanced RAG system"""
        try:
            rag_sys = get_rag_system()
            if not rag_sys:
                return jsonify({'error': 'Enhanced RAG system not available'}), 503
            
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            # Extract parameters with defaults
            num_sub_queries = data.get('num_sub_queries', 3)
            db_chunks_per_query = data.get('db_chunks_per_query', 3)
            web_results_per_query = data.get('web_results_per_query', 3)
            synthesis_model = data.get('synthesis_model', 'gemma3:27b')
            enable_database_search = data.get('enable_database_search', True)
            enable_web_search = data.get('enable_web_search', True)
            
            # Process the query
            result = rag_sys.process_query(
                user_query=query,
                num_sub_queries=num_sub_queries,
                db_chunks_per_query=db_chunks_per_query,
                web_results_per_query=web_results_per_query,
                synthesis_model=synthesis_model,
                enable_database_search=enable_database_search,
                enable_web_search=enable_web_search
            )
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error in enhanced query processing: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/legacy-query', methods=['POST'])
    def legacy_query():
        """Process query using legacy chatbot system"""
        try:
            chatbot = get_legacy_chatbot()
            if not chatbot:
                return jsonify({'error': 'Legacy chatbot system not available'}), 503
            
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            # For now, return a placeholder - would need to implement legacy integration
            result = {
                'query': query,
                'answer': 'Legacy chatbot processing is available but needs specific integration.',
                'processing_time': 0.1,
                'sources': []
            }
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error in legacy query processing: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/available-models')
    def available_models():
        """Get available Ollama models"""
        try:
            import requests
            response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                # Filter for Gemma3 models
                gemma_models = [name for name in model_names if 'gemma3' in name.lower()]
                return jsonify(gemma_models if gemma_models else model_names)
            return jsonify([])
        except Exception as e:
            logging.error(f"Error getting available models: {e}")
            return jsonify(['gemma3:27b', 'gemma3:8b', 'gemma3:1b'])

    @app.route('/api/download-markdown')
    def download_markdown():
        """Download markdown report file"""
        try:
            file_path = request.args.get('file')
            if not file_path or not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            return send_file(
                file_path,
                as_attachment=True,
                download_name='agriculture_research_report.md',
                mimetype='text/markdown'
            )
        except Exception as e:
            logging.error(f"Error downloading markdown file: {e}")
            return jsonify({'error': str(e)}), 500

    # Fallback template for when enhanced UI is not available
    FALLBACK_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌾 Agriculture Bot - Fallback Mode</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
            .header { text-align: center; color: #2e7d32; margin-bottom: 30px; }
            .status { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .query-box { width: 100%; height: 100px; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: #2e7d32; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #1b5e20; }
            .result { background: #e8f5e8; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #4caf50; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 Agriculture Bot - Fallback Mode</h1>
                <p>Basic interface while enhanced features are loading</p>
            </div>
            
            <div class="status">
                <h3>📊 System Status</h3>
                <p id="status-info">Checking system status...</p>
            </div>
            
            <div>
                <h3>💭 Ask Your Agriculture Question</h3>
                <textarea id="query" class="query-box" placeholder="Enter your agriculture question here..."></textarea>
                <br>
                
                <!-- Quick Controls for Enhanced Mode -->
                <div id="enhanced-controls" style="margin: 10px 0; padding: 15px; background: #f0f8ff; border-radius: 5px; display: none;">
                    <h4>🎛️ Search Settings</h4>
                    <label><input type="checkbox" id="enable-db" checked> Database Search</label>
                    <label style="margin-left: 20px;"><input type="checkbox" id="enable-web" checked> Web Search</label>
                    <br><br>
                    <label>Sub-queries: <input type="range" id="sub-queries" min="1" max="5" value="3"> <span id="sub-q-val">3</span></label>
                    <br>
                    <label>DB Chunks: <input type="range" id="db-chunks" min="1" max="10" value="3"> <span id="db-val">3</span></label>
                    <label style="margin-left: 20px;">Web Results: <input type="range" id="web-results" min="1" max="10" value="3"> <span id="web-val">3</span></label>
                </div>
                
                <button class="btn" onclick="askQuestion()">🚀 Ask Question</button>
                <button class="btn" onclick="toggleControls()" style="margin-left: 10px; background: #2196f3;">⚙️ Settings</button>
            </div>
            
            <div id="result" style="display: none;"></div>
        </div>
        
        <script>
            // Check system status
            fetch('/api/system-status')
                .then(response => response.json())
                .then(data => {
                    let status = 'System Status:<br>';
                    status += `Enhanced RAG: ${data.enhanced_rag_available ? '✅ Available' : '❌ Not Available'}<br>`;
                    status += `Legacy Chatbot: ${data.legacy_chatbot_available ? '✅ Available' : '❌ Not Available'}<br>`;
                    status += `Mode: ${data.mode}<br>`;
                    document.getElementById('status-info').innerHTML = status;
                    
                    // Show enhanced controls if RAG is available
                    if (data.enhanced_rag_available) {
                        document.getElementById('enhanced-controls').style.display = 'block';
                    }
                })
                .catch(error => {
                    document.getElementById('status-info').innerHTML = '❌ Could not check system status';
                });
            
            // Update range values
            document.getElementById('sub-queries').oninput = function() {
                document.getElementById('sub-q-val').textContent = this.value;
            };
            document.getElementById('db-chunks').oninput = function() {
                document.getElementById('db-val').textContent = this.value;
            };
            document.getElementById('web-results').oninput = function() {
                document.getElementById('web-val').textContent = this.value;
            };
            
            function toggleControls() {
                const controls = document.getElementById('enhanced-controls');
                controls.style.display = controls.style.display === 'none' ? 'block' : 'none';
            }
            
            function askQuestion() {
                const query = document.getElementById('query').value.trim();
                if (!query) {
                    alert('Please enter a question');
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>🤔 Processing your question...</p>';
                resultDiv.style.display = 'block';
                
                // Check which mode is available and use appropriate endpoint
                fetch('/api/system-status')
                    .then(response => response.json())
                    .then(statusData => {
                        let endpoint = '/api/enhanced-query';
                        let requestBody = { 
                            query: query, 
                            num_sub_queries: parseInt(document.getElementById('sub-queries').value),
                            db_chunks_per_query: parseInt(document.getElementById('db-chunks').value),
                            web_results_per_query: parseInt(document.getElementById('web-results').value),
                            enable_database_search: document.getElementById('enable-db').checked,
                            enable_web_search: document.getElementById('enable-web').checked,
                            synthesis_model: 'gemma3:27b'
                        };
                        
                        if (!statusData.enhanced_rag_available && statusData.legacy_chatbot_available) {
                            endpoint = '/api/legacy-query';
                            requestBody = { query: query };
                        }
                        
                        return fetch(endpoint, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(requestBody)
                        });
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        let answerContent = '';
                        if (data.final_answer) {
                            // Enhanced RAG response
                            answerContent = `
                                <h3>🤖 Enhanced RAG Answer</h3>
                                <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                    ${formatAnswer(data.final_answer)}
                                </div>
                                <p><strong>Query Processing:</strong></p>
                                <p>• Original: ${data.original_query}</p>
                                <p>• Refined: ${data.refined_query}</p>
                                <p>• Sub-queries: ${data.sub_queries ? data.sub_queries.length : 0}</p>
                                <p>• Processing time: ${data.processing_time?.toFixed(2)}s</p>
                                
                                <p><strong>Search Results:</strong></p>
                                <p>• Database chunks: ${data.stats?.total_db_chunks || 0}</p>
                                <p>• Web results: ${data.stats?.total_web_results || 0}</p>
                                
                                ${data.markdown_file_path ? `
                                    <button onclick="downloadReport('${data.markdown_file_path}')" class="btn" style="margin-top: 10px; background: #2196f3;">
                                        📥 Download Full Report
                                    </button>
                                ` : ''}
                            `;
                        } else if (data.answer) {
                            // Legacy response
                            answerContent = `
                                <h3>🔍 Legacy Search Answer</h3>
                                <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                    ${formatAnswer(data.answer)}
                                </div>
                                <p><small>Processing time: ${data.processing_time?.toFixed(2)}s</small></p>
                            `;
                        } else {
                            answerContent = '<h3>❓ No Answer</h3><p>No response generated</p>';
                        }
                        
                        resultDiv.innerHTML = `<div class="result">${answerContent}</div>`;
                    })
                    .catch(error => {
                        resultDiv.innerHTML = `
                            <div class="result">
                                <h3>❌ Error</h3>
                                <p>Could not process your question: ${error.message}</p>
                                <p>Please make sure the system is properly configured and try a simpler question.</p>
                            </div>
                        `;
                    });
            }
            
            function formatAnswer(text) {
                return text
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`(.*?)`/g, '<code>$1</code>');
            }
            
            function downloadReport(filePath) {
                fetch(`/api/download-markdown?file=${encodeURIComponent(filePath)}`)
                    .then(response => response.blob())
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'agriculture_research_report.md';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    })
                    .catch(error => alert('Error downloading report'));
            }
        </script>
        </script>
    </body>
    </html>
    """

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server with fallback support"""
        if not HAS_FLASK:
            print("Flask is not installed. Please install Flask to run the web UI.")
            return
        
        logging.basicConfig(level=logging.INFO)
        
        print(f"Starting Agriculture Bot Web UI (Fallback Mode)...")
        print(f"Enhanced RAG Available: {HAS_ENHANCED_RAG}")
        print(f"Legacy Chatbot Available: {HAS_LEGACY_CHATBOT}")
        print(f"Server: http://{host}:{port}")
        
        app.run(host=host, port=port, debug=debug)

else:
    def run_server(host='0.0.0.0', port=5000, debug=False):
        print("Flask is not installed. Please install Flask to run the web UI.")

if __name__ == '__main__':
    run_server(debug=True)
