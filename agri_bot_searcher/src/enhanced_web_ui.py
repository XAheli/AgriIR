#!/usr/bin/env python3
"""
AgriIR - Web UI with RAG Integration
A comprehensive Flask web interface combining embeddings-based retrieval with web search
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
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

if HAS_FLASK:
    try:
        from enhanced_rag_system import EnhancedRAGSystem
        HAS_RAG_SYSTEM = True
    except ImportError:
        HAS_RAG_SYSTEM = False
        
    try:
        from agriculture_chatbot import AgricultureChatbot
        HAS_LEGACY_CHATBOT = True
    except ImportError:
        HAS_LEGACY_CHATBOT = False

    app = Flask(__name__)
    CORS(app)  # Enable CORS for all domains

    # Global variables
    rag_system = None
    legacy_chatbot = None
    
    # Configuration - Only use relative path
    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agriculture_embeddings')
    
    if os.path.exists(embeddings_dir):
        EMBEDDINGS_DIR = embeddings_dir
        logging.info(f"📁 Loading embeddings from: {EMBEDDINGS_DIR}")
    else:
        EMBEDDINGS_DIR = None
        logging.warning("⚠️ No embeddings directory found - using web search only mode")
        logging.info("📖 To set up embeddings, see: EMBEDDINGS_SETUP.md")
    
    OLLAMA_HOST = "http://localhost:11434"
    
    def get_rag_system():
        """Get or create RAG system instance"""
        global rag_system
        if rag_system is None and HAS_RAG_SYSTEM:
            try:
                rag_system = EnhancedRAGSystem(EMBEDDINGS_DIR, OLLAMA_HOST)
                if EMBEDDINGS_DIR:
                    logging.info(f"Enhanced RAG system initialized with embeddings from {EMBEDDINGS_DIR}")
                else:
                    logging.info("Enhanced RAG system initialized with web search only")
                    logging.info("💡 For better results, set up local embeddings (see EMBEDDINGS_SETUP.md)")
            except Exception as e:
                logging.error(f"Failed to initialize RAG system: {e}")
        return rag_system
    
    def get_legacy_chatbot(base_port=11434, num_agents=2):
        """Get or create legacy chatbot instance"""
        global legacy_chatbot
        if legacy_chatbot is None and HAS_LEGACY_CHATBOT:
            try:
                legacy_chatbot = AgricultureChatbot(base_port=base_port, num_agents=num_agents)
                logging.info("Legacy chatbot initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize legacy chatbot: {e}")
        return legacy_chatbot

    # Enhanced HTML template with RAG integration
    HTML_TEMPLATE = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌾 AgriIR - RAG + Web Search</title>
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
                --info-color: #2196f3;
                --warning-color: #ff9800;
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

            .mode-toggle {
                background: rgba(255,255,255,0.1);
                padding: 1rem;
                margin-top: 1rem;
                border-radius: 8px;
                display: inline-block;
            }

            .toggle-button {
                background: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                padding: 0.5rem 1rem;
                margin: 0 0.25rem;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .toggle-button.active {
                background: rgba(255,255,255,0.9);
                color: var(--primary-color);
                border-color: rgba(255,255,255,0.9);
            }

            .container {
                max-width: 1400px;
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
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(46, 125, 50, 0.3);
            }

            .btn:disabled {
                background: #bbb;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            .btn-secondary {
                background: linear-gradient(135deg, var(--info-color), #64b5f6);
            }

            .btn-warning {
                background: linear-gradient(135deg, var(--warning-color), #ffb74d);
            }

            .query-input {
                width: 100%;
                min-height: 120px;
                resize: vertical;
                font-family: inherit;
            }

            .response-container {
                margin-top: 2rem;
                padding: 2rem;
                background: #f8f9fa;
                border-radius: 12px;
                border-left: 5px solid var(--primary-color);
            }

            .response-container p {
                line-height: 1.6;
                margin-bottom: 1rem;
            }

            .response-container br {
                margin: 0.5rem 0;
            }

            .answer-text {
                line-height: 1.8;
                font-size: 1rem;
                color: #333;
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 2rem;
                color: var(--primary-color);
            }

            .loading.show {
                display: block;
            }

            .spinner {
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .response-section {
                margin-bottom: 2rem;
                padding: 1.5rem;
                background: white;
                border-radius: 8px;
                border-left: 4px solid var(--accent-color);
            }

            .response-section h4 {
                color: var(--primary-color);
                margin-bottom: 1rem;
                font-size: 1.2rem;
            }

            .sub-query {
                background: #e8f5e8;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 6px;
                border-left: 3px solid var(--secondary-color);
            }

            .source-item {
                background: #f0f7ff;
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 6px;
                border-left: 3px solid var(--info-color);
            }

            .source-item .source-title {
                font-weight: bold;
                color: var(--info-color);
                margin-bottom: 0.5rem;
            }

            .source-item .source-url {
                color: #666;
                font-size: 0.9rem;
                word-break: break-all;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }

            .stat-card {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 2px solid var(--border-color);
            }

            .stat-number {
                font-size: 2rem;
                font-weight: bold;
                color: var(--primary-color);
            }

            .stat-label {
                color: #666;
                font-size: 0.9rem;
            }

            .markdown-preview {
                background: #f8f9fa;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1.5rem;
                max-height: 400px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
                white-space: pre-wrap;
            }

            .download-button {
                margin-top: 1rem;
                background: var(--info-color);
            }

            .mode-section {
                display: none;
            }

            .mode-section.active {
                display: block;
            }

            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 0.5rem;
            }

            .status-available {
                background-color: var(--success-color);
            }

            .status-unavailable {
                background-color: var(--error-color);
            }

            .system-status {
                background: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            }

            .system-status h4 {
                color: var(--info-color);
                margin-bottom: 0.5rem;
            }

            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                    padding: 1rem;
                }
                
                .config-panel {
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 AgriIR</h1>
            <p>Advanced RAG System with Database + Web Search Integration</p>
            
            <div class="mode-toggle">
                <button class="toggle-button active" onclick="switchMode('enhanced')" id="enhanced-toggle">
                    🚀 Enhanced RAG Mode
                </button>
                <button class="toggle-button" onclick="switchMode('legacy')" id="legacy-toggle">
                    🔍 Legacy Search Mode
                </button>
            </div>
        </div>

        <div class="container">
            <div class="config-panel">
                <!-- System Status -->
                <div class="system-status">
                    <h4>🔧 System Status</h4>
                    <div id="system-status-content">
                        <p>Loading system status...</p>
                    </div>
                </div>

                <!-- Enhanced RAG Configuration -->
                <div id="enhanced-config" class="mode-section active">
                    <div class="section">
                        <h3>🎯 Query Configuration</h3>
                        <div class="form-group">
                            <label for="num-sub-queries">Number of Sub-queries:</label>
                            <div class="form-control-inline">
                                <input type="range" id="num-sub-queries" min="1" max="5" value="3" 
                                       oninput="updateRangeValue('num-sub-queries', 'sub-queries-value')">
                                <span class="range-value" id="sub-queries-value">3</span>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>� Search Settings</h3>
                        <div class="checkbox-group">
                            <div class="checkbox-item">
                                <input type="checkbox" id="enable-database" checked>
                                <label for="enable-database">Enable Database Search</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="enable-web-search" checked>
                                <label for="enable-web-search">Enable Web Search</label>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>�📚 Database Retrieval</h3>
                        <div class="form-group">
                            <label for="db-chunks">Database Chunks per Sub-query:</label>
                            <div class="form-control-inline">
                                <input type="range" id="db-chunks" min="1" max="10" value="3"
                                       oninput="updateRangeValue('db-chunks', 'db-chunks-value')">
                                <span class="range-value" id="db-chunks-value">3</span>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>🌐 Web Search</h3>
                        <div class="form-group">
                            <label for="web-results">Web Results per Sub-query:</label>
                            <div class="form-control-inline">
                                <input type="range" id="web-results" min="1" max="10" value="3"
                                       oninput="updateRangeValue('web-results', 'web-results-value')">
                                <span class="range-value" id="web-results-value">3</span>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>🤖 LLM Configuration</h3>
                        <div class="form-group">
                            <label for="synthesis-model">Synthesis Model:</label>
                            <select id="synthesis-model" class="form-control">
                                <option value="gemma3:27b">Gemma3 27B (Recommended)</option>
                                <option value="gemma3:8b">Gemma3 8B</option>
                                <option value="gemma3:2b">Gemma3 2B</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Legacy Configuration -->
                <div id="legacy-config" class="mode-section">
                    <div class="section">
                        <h3>🔧 Agent Configuration</h3>
                        <div class="form-group">
                            <label for="base-port">Base Port:</label>
                            <input type="number" id="base-port" class="form-control" value="11434" min="1024" max="65535">
                        </div>
                        <div class="form-group">
                            <label for="num-agents">Number of Agents:</label>
                            <div class="form-control-inline">
                                <input type="range" id="num-agents" min="1" max="5" value="2"
                                       oninput="updateRangeValue('num-agents', 'agents-value')">
                                <span class="range-value" id="agents-value">2</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="main-panel">
                <div class="section">
                    <h3>💭 Query Input</h3>
                    <div class="form-group">
                        <textarea id="query-input" class="form-control query-input" 
                                  placeholder="Enter your agriculture-related question here...&#10;&#10;Examples:&#10;• What are the best practices for wheat cultivation in semi-arid regions?&#10;• How can I prevent fungal diseases in tomato crops?&#10;• What are the economic benefits of crop rotation?"></textarea>
                    </div>
                    <button onclick="processQuery()" class="btn" id="process-btn">
                        🚀 Process Query
                    </button>
                    <button onclick="clearResults()" class="btn btn-secondary" style="margin-left: 1rem;">
                        🗑️ Clear Results
                    </button>
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your query... This may take a few moments.</p>
                    <p id="loading-stage">Initializing...</p>
                </div>

                <div id="response-container" class="response-container" style="display: none;">
                    <!-- Response content will be populated by JavaScript -->
                </div>
            </div>
        </div>

        <script>
            let currentMode = 'enhanced';
            let isProcessing = false;

            // Initialize page
            document.addEventListener('DOMContentLoaded', function() {
                loadSystemStatus();
                loadAvailableModels();
            });

            function switchMode(mode) {
                currentMode = mode;
                
                // Update toggle buttons
                document.querySelectorAll('.toggle-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(mode + '-toggle').classList.add('active');
                
                // Update config sections
                document.querySelectorAll('.mode-section').forEach(section => section.classList.remove('active'));
                document.getElementById(mode + '-config').classList.add('active');
                
                // Clear results when switching modes
                clearResults();
            }

            function updateRangeValue(sliderId, valueId) {
                const slider = document.getElementById(sliderId);
                const valueSpan = document.getElementById(valueId);
                valueSpan.textContent = slider.value;
            }

            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/system-status');
                    const data = await response.json();
                    
                    let statusHtml = '';
                    
                    // RAG System Status
                    const ragStatus = data.rag_system_available ? 'available' : 'unavailable';
                    statusHtml += `<p><span class="status-indicator status-${ragStatus}"></span>Enhanced RAG System: ${data.rag_system_available ? 'Available' : 'Unavailable'}</p>`;
                    
                    // Legacy Chatbot Status
                    const legacyStatus = data.legacy_chatbot_available ? 'available' : 'unavailable';
                    statusHtml += `<p><span class="status-indicator status-${legacyStatus}"></span>Legacy Search: ${data.legacy_chatbot_available ? 'Available' : 'Unavailable'}</p>`;
                    
                    // Embeddings Status
                    const embeddingsStatus = data.embeddings_available ? 'available' : 'unavailable';
                    statusHtml += `<p><span class="status-indicator status-${embeddingsStatus}"></span>Embeddings Database: ${data.embeddings_available ? 'Available' : 'Unavailable'}</p>`;
                    
                    document.getElementById('system-status-content').innerHTML = statusHtml;
                    
                } catch (error) {
                    document.getElementById('system-status-content').innerHTML = 
                        '<p><span class="status-indicator status-unavailable"></span>Error loading system status</p>';
                }
            }

            async function loadAvailableModels() {
                try {
                    const response = await fetch('/api/available-models');
                    const models = await response.json();
                    
                    const select = document.getElementById('synthesis-model');
                    select.innerHTML = '';
                    
                    if (models.length > 0) {
                        models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            if (model.includes('gemma3:27b')) {
                                option.selected = true;
                            }
                            select.appendChild(option);
                        });
                    } else {
                        const option = document.createElement('option');
                        option.value = 'gemma3:27b';
                        option.textContent = 'gemma3:27b (Default)';
                        select.appendChild(option);
                    }
                } catch (error) {
                    console.error('Error loading available models:', error);
                }
            }

            async function processQuery() {
                if (isProcessing) return;
                
                const query = document.getElementById('query-input').value.trim();
                if (!query) {
                    alert('Please enter a question.');
                    return;
                }

                isProcessing = true;
                document.getElementById('process-btn').disabled = true;
                document.getElementById('loading').classList.add('show');
                document.getElementById('response-container').style.display = 'none';

                try {
                    let apiEndpoint, requestData;

                    if (currentMode === 'enhanced') {
                        apiEndpoint = '/api/enhanced-query';
                        requestData = {
                            query: query,
                            num_sub_queries: parseInt(document.getElementById('num-sub-queries').value),
                            db_chunks_per_query: parseInt(document.getElementById('db-chunks').value),
                            web_results_per_query: parseInt(document.getElementById('web-results').value),
                            synthesis_model: document.getElementById('synthesis-model').value,
                            enable_database_search: document.getElementById('enable-database').checked,
                            enable_web_search: document.getElementById('enable-web-search').checked
                        };
                    } else {
                        apiEndpoint = '/api/legacy-query';
                        requestData = {
                            query: query,
                            base_port: parseInt(document.getElementById('base-port').value),
                            num_agents: parseInt(document.getElementById('num-agents').value)
                        };
                    }

                    updateLoadingStage('Sending request...');

                    const response = await fetch(apiEndpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestData)
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    updateLoadingStage('Processing response...');
                    const data = await response.json();

                    displayResponse(data);

                } catch (error) {
                    console.error('Error:', error);
                    displayError('An error occurred while processing your query. Please try again.');
                } finally {
                    isProcessing = false;
                    document.getElementById('process-btn').disabled = false;
                    document.getElementById('loading').classList.remove('show');
                }
            }

            function updateLoadingStage(stage) {
                document.getElementById('loading-stage').textContent = stage;
            }

            function displayResponse(data) {
                const container = document.getElementById('response-container');
                
                if (currentMode === 'enhanced') {
                    displayEnhancedResponse(data);
                } else {
                    displayLegacyResponse(data);
                }
                
                container.style.display = 'block';
                container.scrollIntoView({ behavior: 'smooth' });
            }

            function displayEnhancedResponse(data) {
                const container = document.getElementById('response-container');
                
                let html = `
                    <div class="response-section">
                        <h4>🎯 Query Processing Summary</h4>
                        <p><strong>Original Query:</strong> ${data.original_query}</p>
                        <p><strong>Refined Query:</strong> ${data.refined_query}</p>
                        <p><strong>Processing Time:</strong> ${data.processing_time?.toFixed(2)}s</p>
                        
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-number">${data.stats?.num_sub_queries || 0}</div>
                                <div class="stat-label">Sub-queries</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${data.stats?.total_db_chunks || 0}</div>
                                <div class="stat-label">Database Chunks</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${data.stats?.total_web_results || 0}</div>
                                <div class="stat-label">Web Results</div>
                            </div>
                        </div>
                    </div>

                    <div class="response-section">
                        <h4>🤖 Final Answer</h4>
                        <div class="answer-text" style="background: white; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--success-color);">
                            ${formatText(data.final_answer)}
                        </div>
                    </div>

                    <div class="response-section">
                        <h4>🔍 Sub-query Results</h4>
                `;

                // Display sub-queries and their results
                if (data.sub_queries && data.sub_queries.length > 0) {
                    data.sub_queries.forEach((subQuery, index) => {
                        html += `
                            <div class="sub-query">
                                <strong>Sub-query ${index + 1}:</strong> ${subQuery}
                            </div>
                        `;
                    });
                }

                html += '</div>';

                // Markdown report section
                if (data.markdown_content) {
                    html += `
                        <div class="response-section">
                            <h4>📄 Research Report</h4>
                            <div class="markdown-preview">${data.markdown_content}</div>
                            ${data.markdown_file_path ? `
                                <button class="btn download-button" onclick="downloadMarkdown('${data.markdown_file_path}')">
                                    📥 Download Full Report
                                </button>
                            ` : ''}
                        </div>
                    `;
                }

                container.innerHTML = html;
            }

            function displayLegacyResponse(data) {
                const container = document.getElementById('response-container');
                
                let html = `
                    <div class="response-section">
                        <h4>🔍 Legacy Search Results</h4>
                        <p><strong>Query:</strong> ${data.query}</p>
                        <p><strong>Processing Time:</strong> ${data.processing_time?.toFixed(2)}s</p>
                    </div>

                    <div class="response-section">
                        <h4>📝 Answer</h4>
                        <div class="answer-text" style="background: white; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--info-color);">
                            ${formatText(data.answer)}
                        </div>
                    </div>
                `;

                if (data.sources && data.sources.length > 0) {
                    html += `
                        <div class="response-section">
                            <h4>📚 Sources</h4>
                    `;
                    
                    data.sources.forEach((source, index) => {
                        html += `
                            <div class="source-item">
                                <div class="source-title">${index + 1}. ${source.title}</div>
                                <div class="source-url">${source.url}</div>
                                <p>${source.snippet}</p>
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                }

                container.innerHTML = html;
            }

            function displayError(message) {
                const container = document.getElementById('response-container');
                container.innerHTML = `
                    <div class="response-section" style="border-left-color: var(--error-color);">
                        <h4>❌ Error</h4>
                        <p style="color: var(--error-color);">${message}</p>
                    </div>
                `;
                container.style.display = 'block';
            }

            function formatText(text) {
                if (!text) return '';
                return text
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`(.*?)`/g, '<code>$1</code>')
                    .replace(/\r\n/g, '<br>')
                    .replace(/\r/g, '<br>');
            }

            function clearResults() {
                document.getElementById('response-container').style.display = 'none';
                document.getElementById('query-input').value = '';
            }

            async function downloadMarkdown(filePath) {
                try {
                    const response = await fetch(`/api/download-markdown?file=${encodeURIComponent(filePath)}`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'agriculture_research_report.md';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    } else {
                        alert('Error downloading file');
                    }
                } catch (error) {
                    console.error('Error downloading markdown:', error);
                    alert('Error downloading file');
                }
            }

            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    processQuery();
                }
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        """Main page route"""
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/system-status')
    def system_status():
        """Get system component status"""
        rag_sys = get_rag_system()
        legacy_bot = get_legacy_chatbot()
        
        return jsonify({
            'rag_system_available': rag_sys is not None,
            'legacy_chatbot_available': legacy_bot is not None,
            'embeddings_available': os.path.exists(EMBEDDINGS_DIR),
            'has_rag_dependencies': HAS_RAG_SYSTEM,
            'has_legacy_dependencies': HAS_LEGACY_CHATBOT
        })

    @app.route('/api/available-models')
    def available_models():
        """Get available Ollama models"""
        try:
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
            return jsonify(['gemma3:27b', 'gemma3:8b', 'gemma3:2b'])

    @app.route('/api/enhanced-query', methods=['POST'])
    def enhanced_query():
        """Process query using enhanced RAG system"""
        try:
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            rag_sys = get_rag_system()
            if not rag_sys:
                return jsonify({'error': 'Enhanced RAG system not available'}), 503
            
            # Extract parameters
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
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            base_port = data.get('base_port', 11434)
            num_agents = data.get('num_agents', 2)
            
            chatbot = get_legacy_chatbot(base_port, num_agents)
            if not chatbot:
                return jsonify({'error': 'Legacy chatbot system not available'}), 503
            
            # Process the query (this would need to be implemented based on the legacy system)
            # For now, return a placeholder response
            result = {
                'query': query,
                'answer': 'Legacy chatbot processing not yet implemented in this integration.',
                'processing_time': 0.1,
                'sources': []
            }
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error in legacy query processing: {e}")
            return jsonify({'error': str(e)}), 500

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

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        if not HAS_FLASK:
            print("Flask is not installed. Please install Flask to run the web UI.")
            return
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print(f"Starting AgriIR Web UI...")
        print(f"Server will be available at: http://{host}:{port}")
        print(f"RAG System Available: {HAS_RAG_SYSTEM}")
        print(f"Legacy Chatbot Available: {HAS_LEGACY_CHATBOT}")
        
        app.run(host=host, port=port, debug=debug)

else:
    def run_server(host='0.0.0.0', port=5000, debug=False):
        print("Flask is not installed. Please install Flask to run the web UI.")

if __name__ == '__main__':
    run_server(debug=True)
