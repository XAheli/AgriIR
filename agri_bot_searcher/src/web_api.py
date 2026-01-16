#!/usr/bin/env python3
"""
Agriculture Bot Searcher - Advanced Web UI
A comprehensive Flask web interface for the agriculture chatbot with configurable parameters
"""

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import sys
import os
import json
import logging
import yaml
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

if HAS_FLASK:
    from agriculture_chatbot import AgricultureChatbot

    app = Flask(__name__)
    CORS(app)  # Enable CORS for all domains

    # Initialize chatbot with default settings
    chatbot = None
    
    def get_chatbot_instance(base_port=11434, num_agents=2):
        """Get or create chatbot instance with specified parameters"""
        global chatbot
        if chatbot is None or chatbot.base_port != base_port or chatbot.num_agents != num_agents:
            chatbot = AgricultureChatbot(base_port=base_port, num_agents=num_agents)
        return chatbot

    # Advanced HTML template with configurable parameters
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌾 Agriculture Bot Searcher - Web Interface</title>
        <style>
            :root {
                --primary-color: #2e7d32;
                --secondary-color: #4caf50;
                --accent-color: #81c784;
                --background-color: #f1f8e9;
                --card-background: #ffffff;
                --text-color: #2c3e50;
                --border-color: #e0e0e0;
                --error-color: #f44336;
                --success-color: #4caf50;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, var(--background-color), #e8f5e8);
                color: var(--text-color);
                line-height: 1.6;
                min-height: 100vh;
            }

            .header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 2rem 0;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }

            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                font-weight: 300;
            }

            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 2rem;
            }

            .config-panel {
                background: var(--card-background);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.1);
                height: fit-content;
                position: sticky;
                top: 2rem;
            }

            .main-panel {
                background: var(--card-background);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.1);
            }

            .section {
                margin-bottom: 2rem;
                padding-bottom: 1.5rem;
                border-bottom: 1px solid var(--border-color);
            }

            .section:last-child {
                border-bottom: none;
                margin-bottom: 0;
            }

            .section h3 {
                color: var(--primary-color);
                margin-bottom: 1rem;
                font-size: 1.3rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .form-group {
                margin-bottom: 1rem;
            }

            .form-group label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
                color: var(--text-color);
            }

            .form-control {
                width: 100%;
                padding: 0.75rem;
                border: 2px solid var(--border-color);
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.3s ease;
            }

            .form-control:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1);
            }

            .form-control-inline {
                display: flex;
                gap: 1rem;
                align-items: center;
            }

            .form-control-inline input[type="range"] {
                flex: 1;
            }

            .range-value {
                min-width: 3rem;
                text-align: center;
                font-weight: bold;
                color: var(--primary-color);
            }

            .checkbox-group {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .btn {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 8px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
            }

            .btn:active {
                transform: translateY(0);
            }

            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            .query-input {
                width: 100%;
                min-height: 120px;
                resize: vertical;
                font-family: inherit;
            }

            .result-container {
                margin-top: 2rem;
                padding: 2rem;
                background: #f8f9fa;
                border-radius: 12px;
                border-left: 4px solid var(--primary-color);
            }

            .result-content {
                white-space: pre-wrap;
                line-height: 1.8;
                font-size: 1rem;
            }

            .loading {
                display: flex;
                align-items: center;
                gap: 1rem;
                color: var(--primary-color);
                font-style: italic;
                padding: 2rem;
                text-align: center;
                justify-content: center;
            }

            .spinner {
                width: 24px;
                height: 24px;
                border: 3px solid var(--border-color);
                border-top: 3px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
                padding: 1rem;
                background: var(--background-color);
                border-radius: 8px;
            }

            .stat-item {
                text-align: center;
                padding: 0.5rem;
            }

            .stat-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--primary-color);
            }

            .stat-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .error {
                background: #ffebee;
                color: var(--error-color);
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid var(--error-color);
                margin-top: 1rem;
            }

            .success {
                background: #e8f5e9;
                color: var(--success-color);
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid var(--success-color);
                margin-top: 1rem;
            }

            .model-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-top: 0.5rem;
                font-size: 0.9rem;
            }

            .status-indicator {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--success-color);
                animation: pulse 2s infinite;
            }

            .status-offline {
                background: var(--error-color);
                animation: none;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                    padding: 1rem;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .config-panel {
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 Agriculture Bot Searcher</h1>
            <p>Multi-Agent AI Assistant for Agricultural Intelligence</p>
        </div>

        <div class="container">
            <!-- Configuration Panel -->
            <div class="config-panel">
                <div class="section">
                    <h3>🔧 System Configuration</h3>
                    
                    <div class="form-group">
                        <label for="base-port">Ollama Base Port:</label>
                        <input type="number" id="base-port" class="form-control" value="11434" min="1024" max="65535">
                    </div>
                    
                    <div class="form-group">
                        <label for="num-agents">Number of Agents:</label>
                        <div class="form-control-inline">
                            <input type="range" id="num-agents" min="1" max="6" value="2" class="form-control">
                            <span class="range-value" id="num-agents-value">2</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="num-searches">Web Searches per Query:</label>
                        <div class="form-control-inline">
                            <input type="range" id="num-searches" min="1" max="5" value="2" class="form-control">
                            <span class="range-value" id="num-searches-value">2</span>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h3>🎯 Answer Preferences</h3>
                    
                    <div class="form-group">
                        <label>Answer Mode:</label>
                        <div class="checkbox-group">
                            <div class="checkbox-item">
                                <input type="radio" id="detailed-mode" name="answer-mode" value="detailed" checked>
                                <label for="detailed-mode">Detailed Analysis</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="radio" id="exact-mode" name="answer-mode" value="exact">
                                <label for="exact-mode">Concise Answer</label>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h3>📊 System Status</h3>
                    <div id="system-status">
                        <div class="model-status">
                            <span class="status-indicator" id="status-indicator"></span>
                            <span id="status-text">Checking status...</span>
                        </div>
                        <div id="available-ports" style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;"></div>
                    </div>
                </div>
            </div>

            <!-- Main Query Panel -->
            <div class="main-panel">
                <div class="section">
                    <h3>🌱 Ask Your Agriculture Question</h3>
                    
                    <div class="form-group">
                        <label for="query-input">Enter your agricultural query:</label>
                        <textarea 
                            id="query-input" 
                            class="form-control query-input" 
                            placeholder="Example: What are the best practices for organic pest control in tomato cultivation?"
                        ></textarea>
                    </div>
                    
                    <button onclick="submitQuery()" class="btn" id="submit-btn">
                        🔍 Get Agricultural Insights
                    </button>
                </div>

                <div id="result-section" style="display: none;">
                    <div class="result-container">
                        <div id="result-content" class="result-content"></div>
                        <div id="result-stats" class="stats" style="display: none;"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Update range value displays
            document.getElementById('num-agents').addEventListener('input', function(e) {
                document.getElementById('num-agents-value').textContent = e.target.value;
            });

            document.getElementById('num-searches').addEventListener('input', function(e) {
                document.getElementById('num-searches-value').textContent = e.target.value;
            });

            // Check system status on load
            window.addEventListener('load', function() {
                checkSystemStatus();
                // Auto-refresh status every 30 seconds
                setInterval(checkSystemStatus, 30000);
            });

            function checkSystemStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        const indicator = document.getElementById('status-indicator');
                        const statusText = document.getElementById('status-text');
                        const portsDiv = document.getElementById('available-ports');
                        
                        if (data.available_ports && data.available_ports.length > 0) {
                            indicator.className = 'status-indicator';
                            statusText.textContent = `${data.available_ports.length} agent(s) ready`;
                            portsDiv.textContent = `Ports: ${data.available_ports.join(', ')}`;
                        } else {
                            indicator.className = 'status-indicator status-offline';
                            statusText.textContent = 'No agents available';
                            portsDiv.textContent = 'Please start Ollama instances';
                        }
                    })
                    .catch(error => {
                        console.error('Status check failed:', error);
                        const indicator = document.getElementById('status-indicator');
                        const statusText = document.getElementById('status-text');
                        indicator.className = 'status-indicator status-offline';
                        statusText.textContent = 'Connection error';
                    });
            }

            function submitQuery() {
                const query = document.getElementById('query-input').value.trim();
                if (!query) {
                    alert('Please enter a question first!');
                    return;
                }

                const basePort = parseInt(document.getElementById('base-port').value);
                const numAgents = parseInt(document.getElementById('num-agents').value);
                const numSearches = parseInt(document.getElementById('num-searches').value);
                const exactAnswer = document.getElementById('exact-mode').checked;

                // Show loading state
                const submitBtn = document.getElementById('submit-btn');
                const resultSection = document.getElementById('result-section');
                const resultContent = document.getElementById('result-content');
                const resultStats = document.getElementById('result-stats');

                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner"></span> Processing...';
                
                resultSection.style.display = 'block';
                resultContent.innerHTML = '<div class="loading"><span class="spinner"></span>Consulting agricultural experts...</div>';
                resultStats.style.display = 'none';

                // Submit query
                fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        base_port: basePort,
                        num_agents: numAgents,
                        num_searches: numSearches,
                        exact_answer: exactAnswer
                    })
                })
                .then(response => response.json())
                .then(data => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '🔍 Get Agricultural Insights';

                    if (data.success) {
                        resultContent.innerHTML = data.answer;
                        
                        // Show stats
                        if (data.stats) {
                            const statsHtml = `
                                <div class="stat-item">
                                    <div class="stat-value">${data.stats.execution_time}s</div>
                                    <div class="stat-label">Response Time</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.stats.agents_used}</div>
                                    <div class="stat-label">Agents Used</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.stats.citations_count}</div>
                                    <div class="stat-label">Citations</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.stats.search_results}</div>
                                    <div class="stat-label">Sources Found</div>
                                </div>
                            `;
                            resultStats.innerHTML = statsHtml;
                            resultStats.style.display = 'grid';
                        }
                    } else {
                        resultContent.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    console.error('Query failed:', error);
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '🔍 Get Agricultural Insights';
                    resultContent.innerHTML = '<div class="error">Network error. Please try again.</div>';
                });
            }

            // Allow Enter key to submit (Ctrl+Enter for newline in textarea)
            document.getElementById('query-input').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                    e.preventDefault();
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        """Serve the main web interface"""
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/status')
    def status():
        """Get system status and available Ollama instances"""
        try:
            base_port = int(request.args.get('base_port', 11434))
            num_agents = int(request.args.get('num_agents', 2))
            
            bot = get_chatbot_instance(base_port, num_agents)
            available_ports = bot.check_ollama_instances()
            agent_roles = [role.value for role in bot.available_roles]
            
            return jsonify({
                "available_ports": available_ports,
                "agent_roles": agent_roles,
                "base_port": bot.base_port,
                "max_agents": bot.num_agents
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/query', methods=['POST'])
    def query():
        """Process agriculture query"""
        try:
            data = request.get_json()
            
            query_text = data.get('query', '').strip()
            if not query_text:
                return jsonify({"success": False, "error": "Query is required"}), 400
            
            base_port = int(data.get('base_port', 11434))
            num_agents = int(data.get('num_agents', 2))
            num_searches = int(data.get('num_searches', 2))
            exact_answer = bool(data.get('exact_answer', False))
            
            # Get chatbot instance with specified parameters
            bot = get_chatbot_instance(base_port, num_agents)
            
            # Process query
            start_time = time.time()
            result = bot.answer_query(
                query=query_text,
                num_searches=num_searches,
                exact_answer=exact_answer
            )
            execution_time = round(time.time() - start_time, 1)
            
            if result["success"]:
                # Prepare response with stats
                response = {
                    "success": True,
                    "answer": result["answer"],
                    "citations": result.get("citations", []),
                    "stats": {
                        "execution_time": execution_time,
                        "agents_used": result.get("agents_used", 1),
                        "citations_count": len(result.get("citations", [])),
                        "search_results": result.get("search_results_count", 0)
                    }
                }
                return jsonify(response)
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Unknown error occurred"),
                    "answer": result.get("answer", "Sorry, I couldn't process your query.")
                })
                
        except Exception as e:
            logging.error(f"Query processing error: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Server error: {str(e)}"
            }), 500

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run the Flask development server"""
        if not HAS_FLASK:
            print("Flask is not installed. Please install it with: pip install flask flask-cors")
            return
        
        print(f"🌾 Agriculture Bot Searcher Web Interface")
        print(f"🚀 Starting server on http://{host}:{port}")
        print(f"📝 Make sure Ollama is running on the configured ports")
        print(f"🔧 Default configuration: Base port 11434, 2 agents")
        
        app.run(host=host, port=port, debug=debug)

else:
    def run_server(*args, **kwargs):
        print("Flask is not installed. Please install it with: pip install flask flask-cors")

if __name__ == '__main__':
    run_server(debug=True)
                border: 3px solid var(--border-color);
                border-top: 3px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
                padding: 1rem;
                background: var(--background-color);
                border-radius: 8px;
            }

            .stat-item {
                text-align: center;
                padding: 0.5rem;
            }

            .stat-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--primary-color);
            }

            .stat-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .error {
                background: #ffebee;
                color: var(--error-color);
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid var(--error-color);
                margin-top: 1rem;
            }

            .success {
                background: #e8f5e9;
                color: var(--success-color);
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid var(--success-color);
                margin-top: 1rem;
            }

            .model-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-top: 0.5rem;
                font-size: 0.9rem;
            }

            .status-indicator {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--success-color);
                animation: pulse 2s infinite;
            }

            .status-offline {
                background: var(--error-color);
                animation: none;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                    padding: 1rem;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .config-panel {
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 Agriculture Bot Searcher</h1>
            <p>Multi-Agent AI Assistant for Agricultural Intelligence</p>
        </div>

        <div class="container">
            <h1>🌾 Agriculture Bot Searcher</h1>
            <p>Ask agricultural questions and get expert answers with citations!</p>
            
            <div class="query-form">
                <input type="text" id="query" placeholder="Enter your agricultural question..." />
                <button onclick="submitQuery()">Search</button>
                <label><input type="checkbox" id="exact"> Exact Answer Mode</label>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            async function submitQuery() {
                const query = document.getElementById('query').value;
                const exact = document.getElementById('exact').checked;
                const resultDiv = document.getElementById('result');
                
                if (!query.trim()) {
                    alert('Please enter a question!');
                    return;
                }
                
                resultDiv.innerHTML = '<div class="loading">🤖 Processing your query...</div>';
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, exact_answer: exact })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultDiv.innerHTML = `
                            <div class="result">
                                <h3>Answer:</h3>
                                <pre style="white-space: pre-wrap;">${data.answer}</pre>
                                <p><strong>Processing time:</strong> ${data.total_time.toFixed(1)}s</p>
                                <p><strong>Citations:</strong> ${data.citations.length}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<div class="result" style="background: #ffe6e6;"><strong>Error:</strong> ${data.error}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result" style="background: #ffe6e6;"><strong>Error:</strong> ${error.message}</div>`;
                }
            }
            
            // Submit on Enter key
            document.getElementById('query').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') submitQuery();
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        """Serve the web interface"""
        return render_template_string(HTML_TEMPLATE)

    @app.route('/health')
    def health():
        """Health check endpoint"""
        try:
            # Check if Ollama instances are available
            available_ports = chatbot.check_ollama_instances()
            return jsonify({
                "status": "healthy",
                "ollama_instances": len(available_ports),
                "available_ports": available_ports,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500

    @app.route('/api/query', methods=['POST'])
    def query():
        """Process agriculture query"""
        try:
            data = request.get_json()
            
            if not data or 'query' not in data:
                return jsonify({"success": False, "error": "Query is required"}), 400
            
            query_text = data['query']
            exact_answer = data.get('exact_answer', False)
            num_searches = data.get('num_searches', 2)
            
            if not query_text.strip():
                return jsonify({"success": False, "error": "Query cannot be empty"}), 400
            
            # Process the query
            result = chatbot.answer_query(
                query_text, 
                num_searches=num_searches, 
                exact_answer=exact_answer
            )
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return jsonify({
                "success": False, 
                "error": f"Internal server error: {str(e)}"
            }), 500

    @app.route('/api/agents')
    def agents():
        """Get information about available agents"""
        try:
            available_ports = chatbot.check_ollama_instances()
            agent_roles = [role.value for role in chatbot.available_roles]
            
            return jsonify({
                "available_ports": available_ports,
                "agent_roles": agent_roles,
                "max_agents": chatbot.num_agents
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run_server(host='0.0.0.0', port=8000, debug=False):
        """Run the Flask server"""
        logging.basicConfig(level=logging.INFO)
        app.logger.info(f"Starting Agriculture Bot Searcher API on {host}:{port}")
        app.run(host=host, port=port, debug=debug)

    if __name__ == "__main__":
        run_server()

else:
    print("Flask not installed. Web API not available.")
    print("Install Flask with: pip install flask flask-cors")
    sys.exit(1)
