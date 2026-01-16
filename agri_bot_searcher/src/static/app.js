// AgriIR Bot - Enhanced Voice & Text Interface with Pipeline Visualization
// Global variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isProcessing = false;
let currentResponseData = null; // Store current response data for tab switching

// DOM elements - Core functionality
const recordBtn = document.getElementById('record-btn');
const recordingStatus = document.getElementById('recording-status');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const responseContent = document.getElementById('response-content');

// DOM elements - Enhanced RAG controls
const numSubQueriesSlider = document.getElementById('num-sub-queries');
const numSubQueriesValue = document.getElementById('num-sub-queries-value');
const dbChunksSlider = document.getElementById('db-chunks-per-query');
const dbChunksValue = document.getElementById('db-chunks-per-query-value');
const webResultsSlider = document.getElementById('web-results-per-query');
const webResultsValue = document.getElementById('web-results-per-query-value');
const enableDatabaseSearch = document.getElementById('enable-database-search');
const enableWebSearch = document.getElementById('enable-web-search');
const synthesisModelSelect = document.getElementById('synthesis-model');

// DOM elements - Status indicators
const ragStatus = document.getElementById('rag-status');
const voiceStatus = document.getElementById('voice-status');
const modelsStatus = document.getElementById('models-status');
const refreshStatusBtn = document.getElementById('refresh-status');
const refreshModelsBtn = document.getElementById('refresh-models');

// DOM elements - Processing stats
const processingTimeSpan = document.getElementById('processing-time');
const subQueriesCountSpan = document.getElementById('sub-queries-count');
const dbResultsCountSpan = document.getElementById('db-results-count');
const webResultsCountSpan = document.getElementById('web-results-count');

// DOM elements - Tab system
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const loadingIndicator = document.getElementById('loading-indicator');

// DOM elements - Response tabs
const markdownContent = document.getElementById('markdown-content');
const citationsContent = document.getElementById('citations-content');
const downloadMarkdownBtn = document.getElementById('download-markdown');
const pipelineDetails = document.getElementById('pipeline-details');

// DOM elements - Voice settings
const languageSelect = document.getElementById('language-select');
const apiKeyInput = document.getElementById('api-key');
const hfTokenInput = document.getElementById('hf-token');
const useLocalModelCheck = document.getElementById('use-local-model');

// Language mappings for AgriIR
const LANGUAGE_MAPPINGS = {
    'asm_Beng': { 'name': 'Assamese (Bengali script)', 'code': 'asm_Beng' },
    'ben_Beng': { 'name': 'Bengali', 'code': 'ben_Beng' },
    'brx_Deva': { 'name': 'Bodo', 'code': 'brx_Deva' },
    'doi_Deva': { 'name': 'Dogri', 'code': 'doi_Deva' },
    'guj_Gujr': { 'name': 'Gujarati', 'code': 'guj_Gujr' },
    'hin_Deva': { 'name': 'Hindi', 'code': 'hin_Deva' },
    'kan_Knda': { 'name': 'Kannada', 'code': 'kan_Knda' },
    'gom_Deva': { 'name': 'Konkani', 'code': 'gom_Deva' },
    'kas_Arab': { 'name': 'Kashmiri (Arabic script)', 'code': 'kas_Arab' },
    'kas_Deva': { 'name': 'Kashmiri (Devanagari script)', 'code': 'kas_Deva' },
    'mai_Deva': { 'name': 'Maithili', 'code': 'mai_Deva' },
    'mal_Mlym': { 'name': 'Malayalam', 'code': 'mal_Mlym' },
    'mni_Beng': { 'name': 'Manipuri (Bengali script)', 'code': 'mni_Beng' },
    'mni_Mtei': { 'name': 'Manipuri (Meitei script)', 'code': 'mni_Mtei' },
    'mar_Deva': { 'name': 'Marathi', 'code': 'mar_Deva' },
    'npi_Deva': { 'name': 'Nepali', 'code': 'npi_Deva' },
    'ory_Orya': { 'name': 'Odia', 'code': 'ory_Orya' },
    'pan_Guru': { 'name': 'Punjabi', 'code': 'pan_Guru' },
    'san_Deva': { 'name': 'Sanskrit', 'code': 'san_Deva' },
    'sat_Olck': { 'name': 'Santali (Ol Chiki script)', 'code': 'sat_Olck' },
    'snd_Arab': { 'name': 'Sindhi (Arabic script)', 'code': 'snd_Arab' },
    'snd_Deva': { 'name': 'Sindhi (Devanagari script)', 'code': 'snd_Deva' },
    'urd_Arab': { 'name': 'Urdu', 'code': 'urd_Arab' },
    'eng_Latn': { 'name': 'English (Latin script)', 'code': 'eng_Latn' }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌾 AgriIR Bot - Initializing...');
    initializeSliders();
    initializeEventListeners();
    initializeTabs();
    
    // Load system status and models
    checkSystemStatus();
    loadAvailableModels();
    
    // Initialize microphone on user interaction (not immediately)
    addMicrophoneInitializer();
    
    console.log('✅ AgriIR Bot - Initialization complete');
});

// Add microphone initializer that triggers on first user interaction
function addMicrophoneInitializer() {
    if (recordBtn) {
        // Add a one-time click handler to initialize microphone
        const initializeOnFirstClick = async () => {
            console.log('🎤 First interaction - requesting microphone access...');
            await initializeMicrophone();
            
            // Remove this handler after first use
            recordBtn.removeEventListener('click', initializeOnFirstClick);
        };
        
        // Add the initializer before the main toggle function
        recordBtn.addEventListener('click', initializeOnFirstClick, { once: true });
    }
}

// Initialize slider controls
function initializeSliders() {
    // Enhanced RAG sliders
    if (numSubQueriesSlider && numSubQueriesValue) {
        numSubQueriesSlider.addEventListener('input', function() {
            numSubQueriesValue.textContent = this.value;
        });
    }
    
    if (dbChunksSlider && dbChunksValue) {
        dbChunksSlider.addEventListener('input', function() {
            dbChunksValue.textContent = this.value;
        });
    }
    
    if (webResultsSlider && webResultsValue) {
        webResultsSlider.addEventListener('input', function() {
            webResultsValue.textContent = this.value;
        });
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Voice recording
    if (recordBtn) {
        recordBtn.addEventListener('click', toggleRecording);
    }
    
    // Text input
    if (sendBtn) {
        sendBtn.addEventListener('click', handleTextInput);
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleTextInput();
            }
        });
    }
    
    // Download markdown report
    if (downloadMarkdownBtn) {
        downloadMarkdownBtn.addEventListener('click', downloadMarkdownReport);
    }
    
    // Status refresh buttons
    if (refreshModelsBtn) {
        refreshModelsBtn.addEventListener('click', loadAvailableModels);
    }
    
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', checkSystemStatus);
    }
    
    // Search toggles validation
    if (enableDatabaseSearch) {
        enableDatabaseSearch.addEventListener('change', validateSearchSettings);
    }
    
    if (enableWebSearch) {
        enableWebSearch.addEventListener('change', validateSearchSettings);
    }
}

// Initialize tab system
function initializeTabs() {
    if (tabButtons) {
        tabButtons.forEach(tab => {
            tab.addEventListener('click', function() {
                switchTab(this.dataset.tab);
            });
        });
    }
}

// Validate that at least one search method is enabled
function validateSearchSettings() {
    if (!enableDatabaseSearch || !enableWebSearch) return;
    
    const dbEnabled = enableDatabaseSearch.checked;
    const webEnabled = enableWebSearch.checked;
    
    if (!dbEnabled && !webEnabled) {
        // Force enable web search if both are disabled
        enableWebSearch.checked = true;
        showNotification('At least one search method must be enabled. Web search has been enabled.', 'warning');
    }
}

// Switch tabs in the response panel
function switchTab(tabName) {
    console.log(`Switching to tab: ${tabName}`);
    
    // Update tab buttons
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });
    
    // Update tab panes - handle specific tab name mappings
    tabPanes.forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Map tab names to actual element IDs
    let targetElementId;
    switch (tabName) {
        case 'response':
            targetElementId = 'response-content';
            break;
        case 'pipeline':
            targetElementId = 'pipeline-info';
            break;
        case 'markdown':
            targetElementId = 'markdown-content';
            break;
        case 'citations':
            targetElementId = 'citations-content';
            break;
        default:
            targetElementId = `${tabName}-content`;
    }
    
    const targetPane = document.getElementById(targetElementId);
    if (targetPane) {
        targetPane.classList.add('active');
        console.log(`Activated tab pane: ${targetElementId}`);
        
        // If switching to markdown tab and we have response data, render it
        if (tabName === 'markdown' && currentResponseData && currentResponseData.markdown_content) {
            displayMarkdownContent(currentResponseData.markdown_content);
        }
        
        // If switching to citations tab and we have response data, render it
        if (tabName === 'citations' && currentResponseData && currentResponseData.response) {
            extractAndDisplayCitations(currentResponseData.response, currentResponseData.markdown_content);
        }
        
        // If switching to pipeline tab and we have response data, render it
        if (tabName === 'pipeline' && currentResponseData && currentResponseData.pipeline_info) {
            displayPipelineInfo(currentResponseData.pipeline_info, currentResponseData.search_stats, currentResponseData.sub_query_results);
        }
    } else {
        console.error(`Target pane not found: ${targetElementId}`);
    }
}

// Show notification to user
function showNotification(message, type = 'info') {
    console.log(`${type.toUpperCase()}: ${message}`);
    
    // Create a simple notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#f44336' : type === 'warning' ? '#ff9800' : '#4caf50'};
        color: white;
        padding: 15px;
        border-radius: 5px;
        z-index: 1000;
        max-width: 300px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Toggle recording functionality
async function toggleRecording() {
    if (!recordBtn) {
        console.error('Record button not found');
        return;
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        console.log('🔄 Starting recording...');
        
        // Initialize microphone if not already done
        if (!mediaRecorder) {
            const hasPermission = await initializeMicrophone();
            if (!hasPermission) {
                return;
            }
        }
        
        // Request fresh microphone access for recording
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            } 
        });
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/wav'
        });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            console.log('🔄 Processing recorded audio...');
            
            // Stop the stream
            stream.getTracks().forEach(track => track.stop());
            
            // Process the audio
            const audioBlob = new Blob(audioChunks, { 
                type: mediaRecorder.mimeType || 'audio/wav' 
            });
            
            await processAudioBlob(audioBlob);
        };
        
        mediaRecorder.onerror = event => {
            console.error('MediaRecorder error:', event.error);
            showStatus('Recording error: ' + event.error.message, 'error');
            stopRecording();
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        updateRecordingUI(true);
        showStatus('Recording... Click again to stop', 'recording');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        
        let errorMessage = 'Recording failed: ';
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Microphone permission denied. Please allow access and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found.';
        } else {
            errorMessage += error.message;
        }
        
        showStatus(errorMessage, 'error');
        updateRecordingUI(false);
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        console.log('⏹️ Stopping recording...');
        mediaRecorder.stop();
        isRecording = false;
        updateRecordingUI(false);
        showStatus('Processing audio...', 'processing');
    }
}

// Process audio blob
async function processAudioBlob(audioBlob) {
    try {
        console.log('🔄 Sending audio for transcription...');
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('language', languageSelect ? languageSelect.value : 'hin_Deva');
        formData.append('use_local_model', 'false');
        formData.append('api_key', apiKeyInput ? apiKeyInput.value : '');
        formData.append('hf_token', hfTokenInput ? hfTokenInput.value : '');
        
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('✅ Transcription successful');
            showStatus('Transcription completed', 'ready');
            
            // Update the input field with the transcribed text
            if (userInput) {
                userInput.value = result.english || result.original || '';
            }
            
            // Show transcription results
            console.log('Original:', result.original);
            console.log('English:', result.english);
            
        } else {
            console.error('❌ Transcription failed:', result.error);
            showStatus('Transcription failed: ' + (result.error || 'Unknown error'), 'error');
        }
        
    } catch (error) {
        console.error('❌ Error processing audio:', error);
        showStatus('Error processing audio: ' + error.message, 'error');
    }
}

// Show status message
function showStatus(message, type = 'info') {
    console.log(`${type.toUpperCase()}: ${message}`);
    
    if (recordingStatus) {
        recordingStatus.textContent = message;
        recordingStatus.className = `status-indicator status-${type}`;
    }
}

// Update recording UI
function updateRecordingUI(recording) {
    if (!recordBtn) {
        console.error('Record button not found');
        return;
    }
    
    const recordLabel = document.querySelector('.record-btn-label');
    
    if (recording) {
        recordBtn.classList.add('recording');
        recordBtn.textContent = '⏹️';
        if (recordLabel) recordLabel.textContent = 'Stop Recording';
    } else {
        recordBtn.classList.remove('recording');
        recordBtn.textContent = '🎤';
        if (recordLabel) recordLabel.textContent = 'Start Recording';
    }
}

// Initialize microphone access
async function initializeMicrophone() {
    try {
        console.log('🎤 Requesting microphone access...');
        
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia not supported by this browser');
        }
        
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        // Stop the stream immediately - we just needed permission
        stream.getTracks().forEach(track => track.stop());
        
        console.log('✅ Microphone access granted');
        showStatus('Microphone ready', 'ready');
        return true;
        
    } catch (error) {
        console.error('❌ Microphone access failed:', error);
        
        let errorMessage = 'Microphone access failed: ';
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Permission denied. Please allow microphone access and refresh the page.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found. Please connect a microphone.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage += 'Microphone not supported by your browser.';
        } else {
            errorMessage += error.message;
        }
        
        showStatus(errorMessage, 'error');
        return false;
    }
}

// Process audio through voice transcription
async function processAudio(audioBlob) {
    isProcessing = true;
    
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('language_code', languageSelect.value);
        formData.append('use_local_model', useLocalModelCheck.checked);
        
        const apiKey = apiKeyInput.value.trim();
        const hfToken = hfTokenInput.value.trim();
        
        if (apiKey) formData.append('api_key', apiKey);
        if (hfToken) formData.append('hf_token', hfToken);
        
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showStatus('Audio transcribed successfully', 'success');
            
            // Use the English translated text for the query
            const query = result.english_text || result.original_text;
            userInput.value = query;
            
            // Automatically process the query through RAG
            await processQuery(query);
        } else {
            showStatus(`Transcription failed: ${result.error}`, 'error');
        }
        
    } catch (error) {
        console.error('Error processing audio:', error);
        showStatus('Error processing audio', 'error');
    } finally {
        isProcessing = false;
    }
}

// Handle text input
async function handleTextInput() {
    const query = userInput.value.trim();
    if (!query) {
        showStatus('Please enter a question', 'warning');
        return;
    }
    
    await processQuery(query);
}

// Process query through Enhanced RAG System
async function processQuery(query) {
    if (isProcessing) {
        showStatus('Please wait for the current query to complete', 'warning');
        return;
    }
    
    isProcessing = true;
    showLoading(true);
    showStatus('Processing your agriculture question...', 'processing');
    
    // Start progress monitoring for long requests
    let progressTimer = null;
    let timeoutWarningShown = false;
    
    try {
        // Get request parameters
        const requestData = {
            query: query,
            num_sub_queries: numSubQueriesSlider ? parseInt(numSubQueriesSlider.value) : 3,
            db_chunks_per_query: dbChunksSlider ? parseInt(dbChunksSlider.value) : 5,
            web_results_per_query: webResultsSlider ? parseInt(webResultsSlider.value) : 3,
            enable_database_search: enableDatabaseSearch ? enableDatabaseSearch.checked : true,
            enable_web_search: enableWebSearch ? enableWebSearch.checked : true,
            synthesis_model: synthesisModelSelect ? synthesisModelSelect.value : 'llama3.2:latest',
            // Legacy parameters for backward compatibility
            num_agents: 2,
            base_port: 11434
        };
        
        console.log('Processing query with parameters:', requestData);
        
        // Determine expected timeout based on model
        const model = requestData.synthesis_model.toLowerCase();
        let expectedTimeout;
        if (model.includes('70b') || model.includes('72b')) {
            expectedTimeout = 900000; // 15 minutes
        } else if (model.includes('27b') || model.includes('30b')) {
            expectedTimeout = 720000; // 12 minutes (increased from 8)
        } else if (model.includes('13b') || model.includes('14b')) {
            expectedTimeout = 480000; // 8 minutes (increased from 5)
        } else if (model.includes('7b') || model.includes('8b')) {
            expectedTimeout = 300000; // 5 minutes (increased from 3)
        } else {
            expectedTimeout = 180000; // 3 minutes (increased from 2)
        }
        
        // Start progress monitoring
        let elapsedTime = 0;
        progressTimer = setInterval(() => {
            elapsedTime += 10000; // 10 seconds
            
            const progressMessage = document.getElementById('progress-message');
            if (progressMessage) {
                if (elapsedTime >= 30000 && elapsedTime < 60000) {
                    progressMessage.textContent = 'Analyzing database content and generating sub-queries...';
                    showStatus('Processing... This may take a few minutes for comprehensive analysis', 'processing');
                } else if (elapsedTime >= 60000 && elapsedTime < 120000) {
                    progressMessage.textContent = 'Performing web search and retrieving relevant information...';
                    showStatus('Still processing... Large models require more time for quality responses', 'processing');
                } else if (elapsedTime >= 120000 && elapsedTime < expectedTimeout * 0.8) {
                    progressMessage.textContent = `AI synthesis in progress using ${requestData.synthesis_model}...`;
                    showStatus('Processing continues... Please be patient for comprehensive AI analysis', 'processing');
                } else if (elapsedTime >= expectedTimeout * 0.8 && !timeoutWarningShown) {
                    progressMessage.textContent = `Large model processing - this may take up to ${Math.round(expectedTimeout/60000)} minutes...`;
                    showStatus(`Processing is taking longer than expected. Large model (${requestData.synthesis_model}) requires significant computation time.`, 'warning');
                    timeoutWarningShown = true;
                    
                    // Show timeout info
                    const timeoutInfo = document.getElementById('timeout-info');
                    if (timeoutInfo) timeoutInfo.style.display = 'block';
                }
            }
        }, 10000);
        
        // Make the request with appropriate timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), expectedTimeout + 30000); // Add 30s buffer
        
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        clearInterval(progressTimer);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentResponseData = result;
            displayResponse(result);
            showStatus('Response generated successfully', 'success');
        } else {
            showStatus(`Error: ${result.error}`, 'error');
            responseContent.innerHTML = `<div class="error-message">Error: ${result.error}</div>`;
        }
        
    } catch (error) {
        clearInterval(progressTimer);
        console.error('Error processing query:', error);
        
        if (error.name === 'AbortError') {
            showStatus(`Request timed out after ${Math.round(expectedTimeout/60000)} minutes. Try a smaller model for faster responses.`, 'error');
            responseContent.innerHTML = `
                <div class="error-message">
                    <h3>⏱️ Request Timeout</h3>
                    <p>The request timed out after ${Math.round(expectedTimeout/60000)} minutes.</p>
                    <p><strong>Suggestions:</strong></p>
                    <ul>
                        <li>Try a smaller model (e.g., llama3.2:3b, gemma2:9b) for faster responses</li>
                        <li>Reduce the number of sub-queries or search results</li>
                        <li>Check if your system has sufficient resources</li>
                    </ul>
                </div>
            `;
        } else {
            showStatus('Error processing query', 'error');
            responseContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        }
    } finally {
        isProcessing = false;
        showLoading(false);
        if (progressTimer) clearInterval(progressTimer);
    }
}// Display comprehensive response
function displayResponse(result) {
    console.log('Displaying response:', result);
    
    // Store response data globally for tab switching
    currentResponseData = result;
    
    // Main answer (always displayed in response tab)
    if (responseContent) {
        responseContent.innerHTML = formatAnswer(result.response);
        console.log('Main response displayed');
    }
    
    // Pipeline information (populate immediately)
    if (result.enhanced_rag && result.pipeline_info) {
        displayPipelineInfo(result.pipeline_info, result.search_stats, result.sub_query_results);
        console.log('Pipeline info displayed');
    }
    
    // Markdown content (populate immediately)
    if (result.markdown_content) {
        displayMarkdownContent(result.markdown_content);
        console.log('Markdown content displayed');
    }
    
    // Citations (populate immediately)
    if (result.response) {
        extractAndDisplayCitations(result.response, result.markdown_content);
        console.log('Citations displayed');
    }
    
    // Update processing stats in header
    if (processingTimeSpan) {
        processingTimeSpan.textContent = result.processing_time || '--';
    }
    if (subQueriesCountSpan) {
        subQueriesCountSpan.textContent = result.sub_queries_count || '--';
    }
    if (dbResultsCountSpan) {
        dbResultsCountSpan.textContent = result.db_results_count || '--';
    }
    if (webResultsCountSpan) {
        webResultsCountSpan.textContent = result.web_results_count || '--';
    }
    
    // Show download button if markdown is available
    if (downloadMarkdownBtn && result.markdown_content) {
        downloadMarkdownBtn.style.display = 'block';
    }
    
    // Switch back to response tab to show the main answer
    switchTab('response');
}

// Format the main answer with citation highlighting
function formatAnswer(answer) {
    // Handle null or undefined answer
    if (!answer || typeof answer !== 'string') {
        return '<div class="error-message">No answer generated. This may be due to a timeout or processing error.</div>';
    }
    
    // Highlight citations in the answer
    const citationRegex = /\[([A-Z]+-\d+-\d+)\]/g;
    const formattedAnswer = answer.replace(citationRegex, '<span class="citation-link" data-citation="$1">[$1]</span>');
    
    // Convert markdown-style formatting
    let formatted = formattedAnswer
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Convert line breaks to paragraphs
    formatted = formatted.split('\n\n').map(para => para.trim() ? `<p>${para}</p>` : '').join('');
    
    return formatted;
}

// Display pipeline information
function displayPipelineInfo(pipelineInfo, searchStats, subQueryResults) {
    // Update stats
    if (processingTimeSpan) processingTimeSpan.textContent = `${pipelineInfo.processing_time || 0}s`;
    if (subQueriesCountSpan) subQueriesCountSpan.textContent = pipelineInfo.sub_queries ? pipelineInfo.sub_queries.length : 0;
    if (dbResultsCountSpan) dbResultsCountSpan.textContent = pipelineInfo.total_db_chunks || 0;
    if (webResultsCountSpan) webResultsCountSpan.textContent = pipelineInfo.total_web_results || 0;
    
    // Display debug-style pipeline output
    let detailsHTML = `
        <div class="terminal-output">
            <h4>🔍 Enhanced RAG Pipeline Debug Output</h4>
            <div class="debug-section">
                <div class="debug-line">📝 Original Query: ${pipelineInfo.original_query || 'N/A'}</div>
                <div class="debug-line">✨ Refined Query: ${pipelineInfo.refined_query || 'N/A'}</div>
                <div class="debug-line">🔗 Sub-queries Generated: ${pipelineInfo.sub_queries ? pipelineInfo.sub_queries.length : 0}</div>
                <div class="debug-line">🤖 Synthesis Model: ${pipelineInfo.synthesis_model || 'N/A'}</div>
                <div class="debug-line">📚 Database Chunks Retrieved: ${pipelineInfo.total_db_chunks || 0}</div>
                <div class="debug-line">🌐 Web Results Retrieved: ${pipelineInfo.total_web_results || 0}</div>
                <div class="debug-line">⏱️ Processing Time: ${pipelineInfo.processing_time || 0}s</div>
            </div>
        </div>
        
        <div class="terminal-output">
            <h4>📋 Generated Sub-queries</h4>
            <div class="debug-section">`;
    
    if (pipelineInfo.sub_queries && pipelineInfo.sub_queries.length > 0) {
        pipelineInfo.sub_queries.forEach((query, index) => {
            detailsHTML += `<div class="debug-line">${index + 1}. ${query}</div>`;
        });
    } else {
        detailsHTML += `<div class="debug-line">No sub-queries generated</div>`;
    }
    
    detailsHTML += `</div></div>`;
    
    // Show sub-query results in debug format
    if (pipelineInfo.sub_query_results && pipelineInfo.sub_query_results.length > 0) {
        detailsHTML += `
            <div class="terminal-output">
                <h4>🔍 Sub-query Processing Results</h4>
                <div class="debug-section">`;
        
        pipelineInfo.sub_query_results.forEach((result, index) => {
            detailsHTML += `
                <div class="debug-line">⚡ Sub-query ${index + 1}:</div>
                <div class="debug-line">   📚 DB chunks: ${result.db_chunks || 0}</div>
                <div class="debug-line">   🌐 Web results: ${result.web_results || 0}</div>`;
            
            if (result.agent_info) {
                detailsHTML += `<div class="debug-line">   🤖 Agent: ${result.agent_info.agent || 'N/A'}</div>`;
            }
        });
        
        detailsHTML += `</div></div>`;
    }
    
    if (pipelineDetails) {
        pipelineDetails.innerHTML = detailsHTML;
    }
}

// Display markdown content
function displayMarkdownContent(markdownText) {
    const markdownDisplay = document.getElementById('markdown-display');
    if (!markdownDisplay) {
        console.warn('markdown-display element not found');
        return;
    }
    
    if (!markdownText || markdownText.trim() === '') {
        markdownDisplay.innerHTML = '<p class="placeholder">No markdown content available.</p>';
        return;
    }
    
    // Enhanced markdown to HTML conversion
    let html = markdownText
        // Headers
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
        
        // Bold and italic
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Code blocks
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        
        // Lists
        .replace(/^\* (.*$)/gim, '<li>$1</li>')
        .replace(/^- (.*$)/gim, '<li>$1</li>')
        .replace(/^\d+\. (.*$)/gim, '<li>$1</li>')
        
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        
        // Line breaks and paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    // Wrap in paragraphs and clean up
    html = '<p>' + html + '</p>';
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>(<h[1-6]>)/g, '$1');
    html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)<\/p>/g, '$1');
    html = html.replace(/<p>(<li>)/g, '<ul>$1');
    html = html.replace(/(<\/li>)<\/p>/g, '$1</ul>');
    
    // Handle consecutive list items
    html = html.replace(/<\/ul><br><ul>/g, '');
    
    markdownDisplay.innerHTML = `<div class="markdown-preview">${html}</div>`;
    console.log('Markdown content rendered');
}

// Extract and display citations
function extractAndDisplayCitations(answer, markdownText) {
    const citationsDisplay = document.getElementById('citations-display');
    if (!citationsDisplay) {
        console.warn('citations-display element not found');
        return;
    }
    
    if (!answer || typeof answer !== 'string') {
        citationsDisplay.innerHTML = '<p class="placeholder">No answer text available for citation extraction.</p>';
        return;
    }
    
    const citationRegex = /\[([A-Z]+-\d+-\d+)\]/g;
    const citations = [];
    let match;
    
    while ((match = citationRegex.exec(answer)) !== null) {
        citations.push(match[1]);
    }
    
    if (citations.length === 0) {
        citationsDisplay.innerHTML = '<p class="placeholder">No citations found in the response.</p>';
        return;
    }
    
    // Extract citation details from markdown content
    let citationsHTML = '<div class="citations-list">';
    
    const uniqueCitations = [...new Set(citations)];
    uniqueCitations.forEach(citation => {
        // Try to find the citation in the markdown content
        const citationPattern = new RegExp(`\\[${citation}\\].*?([^\\n]+)`, 'i');
        const citationMatch = markdownText ? markdownText.match(citationPattern) : null;
        
        let citationText = 'Citation details not found';
        if (citationMatch) {
            citationText = citationMatch[1].trim();
        }
        
        citationsHTML += `
            <div class="citation-item" data-citation="${citation}">
                <div class="citation-id">${citation}</div>
                <div class="citation-content">${citationText}</div>
            </div>
        `;
    });
    
    citationsHTML += '</div>';
    citationsDisplay.innerHTML = citationsHTML;
    
    // Add click handlers for citation highlighting
    document.querySelectorAll('.citation-link').forEach(link => {
        link.addEventListener('click', function() {
            const citationId = this.dataset.citation;
            highlightCitation(citationId);
        });
    });
    
    console.log(`Citations displayed: ${uniqueCitations.length} unique citations`);
}

// Highlight specific citation
function highlightCitation(citationId) {
    // Switch to citations tab
    switchTab('citations');
    
    // Highlight the citation
    document.querySelectorAll('.citation-item').forEach(item => {
        item.classList.remove('highlighted');
    });
    
    const targetCitation = document.querySelector(`[data-citation="${citationId}"]`);
    if (targetCitation) {
        targetCitation.classList.add('highlighted');
        targetCitation.scrollIntoView({ behavior: 'smooth' });
    }
}

// Download markdown report
async function downloadMarkdownReport() {
    if (!currentResponseData || !currentResponseData.markdown_content) {
        showStatus('No markdown content available for download', 'warning');
        return;
    }
    
    try {
        // Create a blob with the markdown content
        const markdownBlob = new Blob([currentResponseData.markdown_content], { type: 'text/markdown' });
        
        // Create a download link
        const url = URL.createObjectURL(markdownBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `agriir_response_${new Date().toISOString().split('T')[0]}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showStatus('Markdown report downloaded successfully', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showStatus('Error downloading markdown report', 'error');
    }
}

// Load available synthesis models
async function loadAvailableModels() {
    console.log('🔄 Loading available models...');
    
    if (!modelsStatus) {
        console.warn('Models status element not found');
        return;
    }
    
    // Show loading state
    updateStatusIndicator(modelsStatus, null, 'Loading...');
    
    try {
        const response = await fetch('/models');
        const result = await response.json();
        
        if (result.success && result.models.length > 0) {
            // Update synthesis model dropdown if it exists
            if (synthesisModelSelect) {
                // Clear existing options
                synthesisModelSelect.innerHTML = '';
                
                // Add available models with timeout information
                result.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    
                    // Add timeout information to model names
                    let timeoutInfo = '';
                    const modelLower = model.toLowerCase();
                    if (modelLower.includes('70b') || modelLower.includes('72b')) {
                        timeoutInfo = ' (⏰ ~15 min)';
                    } else if (modelLower.includes('27b') || modelLower.includes('30b')) {
                        timeoutInfo = ' (⏰ ~12 min)';
                    } else if (modelLower.includes('13b') || modelLower.includes('14b')) {
                        timeoutInfo = ' (⏰ ~8 min)';
                    } else if (modelLower.includes('7b') || modelLower.includes('8b')) {
                        timeoutInfo = ' (⏰ ~5 min)';
                    } else if (modelLower.includes('3b') || modelLower.includes('1b')) {
                        timeoutInfo = ' (⚡ ~3 min)';
                    }
                    
                    option.textContent = model + timeoutInfo;
                    
                    // Mark recommended models
                    if (result.recommended && result.recommended.includes(model)) {
                        option.textContent += ' (Recommended)';
                    }
                    
                    synthesisModelSelect.appendChild(option);
                });
                
                // Set default to first recommended model, preferring smaller ones
                const preferredModels = ['llama3.2:3b', 'llama3.2:latest', 'gemma2:9b', 'gemma2:latest'];
                let defaultSet = false;
                for (const preferred of preferredModels) {
                    if (result.models.includes(preferred)) {
                        synthesisModelSelect.value = preferred;
                        defaultSet = true;
                        break;
                    }
                }
                
                // Fallback to first recommended model
                if (!defaultSet && result.recommended && result.recommended.length > 0) {
                    synthesisModelSelect.value = result.recommended[0];
                }
            }
            
            updateStatusIndicator(modelsStatus, true, `${result.count} Models Available`);
            console.log(`✅ Loaded ${result.count} models`);
        } else {
            updateStatusIndicator(modelsStatus, false, result.error || 'No models found');
            console.warn('⚠️ No models available');
        }
    } catch (error) {
        console.error('❌ Error loading models:', error);
        updateStatusIndicator(modelsStatus, false, 'Error loading models');
    }
}

// Check system status
async function checkSystemStatus() {
    console.log('🔄 Checking system status...');
    
    // Show loading state for all status indicators
    if (ragStatus) updateStatusIndicator(ragStatus, null, 'Checking...');
    if (voiceStatus) updateStatusIndicator(voiceStatus, null, 'Checking...');
    if (modelsStatus) updateStatusIndicator(modelsStatus, null, 'Checking...');
    
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        if (status.components) {
            // Update RAG system status with embeddings info
            if (ragStatus) {
                const ragAvailable = status.components.enhanced_rag;
                const embeddingsAvailable = status.embeddings_available;
                const mode = status.mode || 'unknown';
                
                let statusText = '';
                if (ragAvailable) {
                    if (embeddingsAvailable) {
                        statusText = 'Enhanced Mode (DB + Web)';
                    } else {
                        statusText = 'Web-Only Mode';
                    }
                } else {
                    statusText = 'RAG Unavailable';
                }
                
                updateStatusIndicator(ragStatus, ragAvailable, statusText);
                
                // Show setup link if embeddings not available
                if (ragAvailable && !embeddingsAvailable) {
                    console.log('💡 Running in web-only mode. For better results, see EMBEDDINGS_SETUP.md');
                }
            }
            
            // Update voice system status
            if (voiceStatus) {
                updateStatusIndicator(voiceStatus, status.components.voice_transcriber, 
                    status.components.voice_transcriber ? 'Voice Available' : 'Voice Unavailable');
            }
            
            // Update models status
            if (modelsStatus) {
                const modelCount = status.components.ollama_models || 0;
                updateStatusIndicator(modelsStatus, modelCount > 0, 
                    modelCount > 0 ? `${modelCount} Models` : 'No Models');
            }
            
            console.log('✅ System status updated');
        } else {
            throw new Error('Invalid status response format');
        }
        
    } catch (error) {
        console.error('❌ Error checking status:', error);
        
        // Set error state for all indicators
        if (ragStatus) updateStatusIndicator(ragStatus, false, 'Error');
        if (voiceStatus) updateStatusIndicator(voiceStatus, false, 'Error');
        if (modelsStatus) updateStatusIndicator(modelsStatus, false, 'Error');
    }
}

// Update status indicator
function updateStatusIndicator(element, isAvailable, text) {
    if (!element) return;
    
    element.textContent = text;
    
    if (isAvailable === null) {
        // Loading state
        element.className = 'status-indicator status-loading';
    } else if (isAvailable) {
        // Available/success state
        element.className = 'status-indicator status-available';
    } else {
        // Unavailable/error state
        element.className = 'status-indicator status-unavailable';
    }
}

// Show loading indicator
function showLoading(show) {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = show ? 'flex' : 'none';
        
        if (show) {
            // Reset progress message
            const progressMessage = document.getElementById('progress-message');
            if (progressMessage) {
                progressMessage.textContent = 'Initializing Enhanced RAG Pipeline...';
            }
            
            // Hide timeout info initially
            const timeoutInfo = document.getElementById('timeout-info');
            if (timeoutInfo) {
                timeoutInfo.style.display = 'none';
            }
        }
    }
}

// Show status message
function showStatus(message, type = 'info') {
    recordingStatus.textContent = message;
    recordingStatus.className = `recording-status status-${type}`;
    
    // Auto-clear status after 5 seconds for non-error messages
    if (type !== 'error') {
        setTimeout(() => {
            recordingStatus.textContent = '';
            recordingStatus.className = 'recording-status';
        }, 5000);
    }
}

// Utility function to format time
function formatTime(seconds) {
    return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
}
