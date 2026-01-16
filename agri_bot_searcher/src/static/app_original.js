// Global variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isProcessing = false;

// DOM elements
const recordButton = document.getElementById('record-button');
const recordIcon = document.getElementById('record-icon');
const recordText = document.getElementById('record-text');
const recordingStatus = document.getElementById('recording-status');
const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-button');
const responseContent = document.getElementById('response-content');
const numAgentsSlider = document.getElementById('num-agents');
const numAgentsValue = document.getElementById('num-agents-value');
const originalContent = document.getElementById('original-content');
const translatedContent = document.getElementById('translated-content');

// Language mappings
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

// Update agent count display
numAgentsSlider.addEventListener('input', function() {
    numAgentsValue.textContent = this.value;
});

// Initialize voice recording with better browser compatibility
async function initializeVoiceRecording() {
    try {
        // Check browser support using multiple methods
        let getUserMedia = null;
        
        // Modern browsers
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        } 
        // Older browsers with prefixes
        else if (navigator.getUserMedia) {
            getUserMedia = navigator.getUserMedia.bind(navigator);
        } 
        else if (navigator.webkitGetUserMedia) {
            getUserMedia = navigator.webkitGetUserMedia.bind(navigator);
        } 
        else if (navigator.mozGetUserMedia) {
            getUserMedia = navigator.mozGetUserMedia.bind(navigator);
        }
        
        if (!getUserMedia) {
            throw new Error('Browser does not support microphone access. Please use Chrome 53+, Firefox 36+, or Safari 11+.');
        }

        // Check secure context
        const isSecureContext = window.isSecureContext || location.protocol === 'https:' || 
                              location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        
        if (!isSecureContext) {
            throw new Error('Microphone access requires HTTPS or localhost. Please access via https:// or localhost.');
        }

        // Show permission request status
        recordingStatus.innerHTML = '<span>⚠️</span> Requesting microphone access...';
        recordingStatus.className = 'status-indicator status-warning';

        // Request microphone access with fallback for older browsers
        let stream;
        const constraints = { 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,
                channelCount: 1
            } 
        };

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Modern promise-based approach
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        } else {
            // Legacy callback-based approach
            stream = await new Promise((resolve, reject) => {
                getUserMedia(constraints, resolve, reject);
            });
        }
        
        // Check MediaRecorder support
        if (!window.MediaRecorder) {
            throw new Error('MediaRecorder not supported. Please use Chrome 47+, Firefox 25+, or Safari 14+.');
        }

        // Determine best MIME type
        let mimeType = '';
        const supportedTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/wav'
        ];
        
        for (const type of supportedTypes) {
            if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type)) {
                mimeType = type;
                break;
            }
        }
        
        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
            
            const audioBlob = new Blob(audioChunks, { type: mimeType || 'audio/wav' });
            audioChunks = [];
            await processAudio(audioBlob);
        };

        mediaRecorder.onerror = function(event) {
            console.error('MediaRecorder error:', event.error);
            showError('Recording error: ' + event.error.message);
            resetRecordingState();
        };

        // Success - update status
        recordingStatus.innerHTML = '<span>🟢</span> Ready to Record';
        recordingStatus.className = 'status-indicator status-ready';
        showSuccess('Microphone access granted. Ready to record!');
        
    } catch (error) {
        console.error('Error initializing voice recording:', error);
        let errorMessage = 'Microphone access failed: ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Permission denied. Please allow microphone access in your browser and refresh the page.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found. Please connect a microphone and try again.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage += 'Microphone not supported by your browser. Please use Chrome 53+, Firefox 36+, or Safari 11+.';
        } else if (error.name === 'SecurityError') {
            errorMessage += 'Security error. Please access via HTTPS or localhost.';
        } else if (error.message.includes('HTTPS') || error.message.includes('localhost')) {
            errorMessage += error.message;
        } else {
            errorMessage += error.message || 'Unknown error occurred.';
        }
        
        // Add helpful suggestions without problematic escaping
        errorMessage += ' Troubleshooting: Make sure you are using HTTPS or localhost, check browser permissions for microphone, try refreshing the page, use Chrome, Firefox, or Safari latest version.';
        
        showError(errorMessage);
        recordingStatus.innerHTML = '<span>❌</span> Microphone unavailable';
        recordingStatus.className = 'status-indicator status-error';
    }
}

// Record button click handler
recordButton.addEventListener('click', async function() {
    if (!mediaRecorder) {
        await initializeVoiceRecording();
        // If initialization still failed, don't proceed
        if (!mediaRecorder) {
            showError('Voice recording initialization failed. Please check the troubleshooting steps above.');
            return;
        }
    }
    
    if (!isRecording && !isProcessing) {
        startRecording();
    } else if (isRecording) {
        stopRecording();
    }
});

function startRecording() {
    if (!mediaRecorder) {
        showError('Microphone not initialized. Please check browser compatibility and try again.');
        return;
    }
    
    isRecording = true;
    recordButton.classList.add('recording');
    recordIcon.textContent = '⏹️';
    recordText.textContent = 'Stop Recording';
    recordingStatus.innerHTML = '<span>🔴</span> Recording...';
    recordingStatus.className = 'status-indicator status-recording';
    
    audioChunks = [];
    mediaRecorder.start();
}

function stopRecording() {
    if (!mediaRecorder) {
        showError('Microphone not initialized. Cannot stop recording.');
        resetRecordingState();
        return;
    }
    
    isRecording = false;
    isProcessing = true;
    recordButton.classList.remove('recording');
    recordButton.classList.add('processing');
    recordIcon.textContent = '⏳';
    recordText.textContent = 'Processing...';
    recordingStatus.innerHTML = '<span>🟡</span> Processing Audio...';
    recordingStatus.className = 'status-indicator status-processing';
    
    mediaRecorder.stop();
}

async function processAudio(audioBlob) {
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('language', document.getElementById('language-select').value);
        formData.append('use_local_model', document.getElementById('use-local-model').checked);
        formData.append('api_key', document.getElementById('api-key').value);
        formData.append('hf_token', document.getElementById('hf-token').value);

        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            originalContent.textContent = result.original || 'No original text';
            translatedContent.textContent = result.english || 'No translation';
            
            // Auto-fill the query input with English translation
            if (result.english) {
                queryInput.value = result.english;
            }
            
            showSuccess('Voice transcription completed successfully!');
        } else {
            showError('Transcription failed: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error processing audio:', error);
        showError('Error processing audio: ' + error.message);
    } finally {
        resetRecordingState();
    }
}

function resetRecordingState() {
    isRecording = false;
    isProcessing = false;
    recordButton.classList.remove('recording', 'processing');
    recordIcon.textContent = '🎤';
    recordText.textContent = 'Start Recording';
    recordingStatus.innerHTML = '<span>🟢</span> Ready to Record';
    recordingStatus.className = 'status-indicator status-ready';
}

// Send query
sendButton.addEventListener('click', async function() {
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a question or record voice input');
        return;
    }

    sendButton.disabled = true;
    sendButton.textContent = 'Processing...';
    responseContent.innerHTML = '<div class="loading"><div class="spinner"></div>Getting response from IndicAgri Bot...</div>';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                num_agents: parseInt(numAgentsSlider.value),
                base_port: parseInt(document.getElementById('ollama-port').value)
            })
        });

        const result = await response.json();
        
        if (result.success) {
            responseContent.innerHTML = result.response;
        } else {
            responseContent.innerHTML = '<div style="color: var(--voice-recording);">Error: ' + (result.error || 'Unknown error occurred') + '</div>';
        }
    } catch (error) {
        console.error('Error sending query:', error);
        responseContent.innerHTML = '<div style="color: var(--voice-recording);">Error: ' + error.message + '</div>';
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = 'Send Query';
    }
});

// Helper functions
function showSuccess(message) {
    console.log('Success:', message);
    // You can implement a toast notification here
}

function showError(message) {
    console.error('Error:', message);
    alert(message);
}

// Allow Enter key to send query (Ctrl+Enter for new line)
queryInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.ctrlKey) {
        e.preventDefault();
        sendButton.click();
    }
});

// API Key Help Functions
function showApiKeyHelp(service) {
    let title, content;
    
    if (service === 'sarvam') {
        title = "How to get SarvamAI API Key";
        content = `
            <p><strong>Steps to get SarvamAI API Key:</strong></p>
            <ol>
                <li>Visit <a href="https://www.sarvam.ai/" target="_blank">SarvamAI website</a></li>
                <li>Sign up for an account or log in</li>
                <li>Navigate to the API section</li>
                <li>Generate your API key</li>
                <li>Copy and paste it in the field above</li>
            </ol>
            <p><em>Note: You may need to verify your email and complete the registration process.</em></p>
        `;
    } else if (service === 'huggingface') {
        title = "How to get Hugging Face Token";
        content = `
            <p><strong>Steps to get Hugging Face Token:</strong></p>
            <ol>
                <li>Visit <a href="https://huggingface.co/" target="_blank">Hugging Face website</a></li>
                <li>Sign up for an account or log in</li>
                <li>Go to Settings → Access Tokens</li>
                <li>Create a new token with 'Read' permissions</li>
                <li>Copy and paste it in the field above</li>
            </ol>
            <p><em>Note: This token is used for accessing gated models and may not be required for all features.</em></p>
        `;
    }
    
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    `;
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white;
        padding: 30px;
        border-radius: 15px;
        max-width: 600px;
        max-height: 80%;
        overflow-y: auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
    `;
    
    modalContent.innerHTML = `
        <button onclick="this.closest('[style*=position]').remove()" 
                style="position: absolute; top: 10px; right: 15px; background: none; 
                       border: none; font-size: 24px; cursor: pointer; color: #666;">×</button>
        <h3 style="color: var(--primary-color); margin-bottom: 20px;">${title}</h3>
        ${content}
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="this.closest('[style*=position]').remove()" 
                    style="background: var(--primary-color); color: white; border: none; 
                           padding: 10px 20px; border-radius: 5px; cursor: pointer;">Close</button>
        </div>
    `;
    
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
    
    // Close on background click
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function checkBrowserCompatibility() {
    const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    
    if (!isSecureContext) {
        showWarning('⚠️ For voice features, please access via HTTPS or localhost. Current URL: ' + location.href);
        recordingStatus.innerHTML = '<span>⚠️</span> HTTPS required for voice';
        recordingStatus.className = 'status-indicator status-warning';
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showWarning('⚠️ Your browser does not support microphone access. Please use Chrome 53+, Firefox 36+, or Safari 11+.');
        recordingStatus.innerHTML = '<span>❌</span> Browser not supported';
        recordingStatus.className = 'status-indicator status-error';
        return;
    }

    if (!window.MediaRecorder) {
        showWarning('⚠️ Your browser does not support audio recording. Please use Chrome 47+, Firefox 25+, or Safari 14+.');
        recordingStatus.innerHTML = '<span>❌</span> Recording not supported';
        recordingStatus.className = 'status-indicator status-error';
        return;
    }

    // Browser is compatible
    recordingStatus.innerHTML = '<span>🟢</span> Click to start recording';
    recordingStatus.className = 'status-indicator status-ready';
}

function showWarning(message) {
    console.warn(message);
    // Create a subtle warning banner
    const warningDiv = document.createElement('div');
    warningDiv.style.cssText = `
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 14px;
    `;
    warningDiv.innerHTML = message;
    
    // Add to top of container
    const container = document.querySelector('.container');
    container.insertBefore(warningDiv, container.firstChild);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('IndicAgri Bot initialized');
    
    // Check browser compatibility first
    checkBrowserCompatibility();
    
    // Load saved API keys from localStorage
    const savedSarvamKey = localStorage.getItem('sarvam_api_key');
    const savedHfToken = localStorage.getItem('hf_token');
    
    if (savedSarvamKey) {
        document.getElementById('api-key').value = savedSarvamKey;
    }
    if (savedHfToken) {
        document.getElementById('hf-token').value = savedHfToken;
    }
    
    // Initialize voice recording (request microphone access on first click)
    // No automatic initialization to avoid permission prompt on page load
    
    // Save API keys when they change
    document.getElementById('api-key').addEventListener('change', function() {
        if (this.value) {
            localStorage.setItem('sarvam_api_key', this.value);
        } else {
            localStorage.removeItem('sarvam_api_key');
        }
    });
    
    document.getElementById('hf-token').addEventListener('change', function() {
        if (this.value) {
            localStorage.setItem('hf_token', this.value);
        } else {
            localStorage.removeItem('hf_token');
        }
    });
});
