// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE_URL: window.location.origin.includes('localhost') 
        ? 'http://localhost:8000' 
        : window.location.origin,
    API_ENDPOINT: '/api/ask',
    HEALTH_ENDPOINT: '/health',
    TIMEOUT: 60000, // 60 seconds
    MAX_HISTORY: 10
};

// ============================================
// State Management
// ============================================
const state = {
    isLoading: false,
    history: loadHistory(),
    currentQuestion: null,
    currentAnswer: null
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    questionForm: document.getElementById('questionForm'),
    questionInput: document.getElementById('questionInput'),
    submitBtn: document.getElementById('submitBtn'),
    answerSection: document.getElementById('answerSection'),
    answerContent: document.getElementById('answerContent'),
    answerTimestamp: document.getElementById('answerTimestamp'),
    loadingSection: document.getElementById('loadingSection'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),
    copyBtn: document.getElementById('copyBtn'),
    examplesGrid: document.getElementById('examplesGrid'),
    historySection: document.getElementById('historySection'),
    historyList: document.getElementById('historyList'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.querySelector('.status-text'),
    indexStatus: document.getElementById('indexStatus')
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Check backend health
    checkHealth();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load and display history
    updateHistoryDisplay();
    
    // Focus on input
    elements.questionInput.focus();
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Form submission
    elements.questionForm.addEventListener('submit', handleSubmit);
    
    // Example question chips
    const exampleChips = document.querySelectorAll('.example-chip');
    exampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const question = chip.getAttribute('data-question');
            elements.questionInput.value = question;
            elements.questionInput.focus();
            handleSubmit(new Event('submit'));
        });
    });
    
    // Copy button
    elements.copyBtn.addEventListener('click', handleCopy);
    
    // Retry button
    elements.retryBtn.addEventListener('click', () => {
        hideError();
        if (state.currentQuestion) {
            askQuestion(state.currentQuestion);
        }
    });
    
    // Clear history
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
    
    // History item clicks
    elements.historyList.addEventListener('click', (e) => {
        const historyItem = e.target.closest('.history-item');
        if (historyItem) {
            const question = historyItem.getAttribute('data-question');
            elements.questionInput.value = question;
            elements.questionInput.focus();
            handleSubmit(new Event('submit'));
        }
    });
}

// ============================================
// Health Check
// ============================================
async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator('connected', data.index_ready);
            if (elements.indexStatus) {
                elements.indexStatus.textContent = data.index_ready 
                    ? 'Index ready' 
                    : 'Index building...';
            }
        } else {
            updateStatusIndicator('error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator('error');
    }
}

function updateStatusIndicator(status, indexReady = false) {
    elements.statusIndicator.className = `status-indicator ${status}`;
    if (status === 'connected') {
        elements.statusText.textContent = indexReady ? 'Connected' : 'Connecting...';
    } else if (status === 'error') {
        elements.statusText.textContent = 'Disconnected';
    } else {
        elements.statusText.textContent = 'Checking...';
    }
}

// ============================================
// Form Handling
// ============================================
function handleSubmit(e) {
    e.preventDefault();
    
    const question = elements.questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question');
        return;
    }
    
    askQuestion(question);
}

async function askQuestion(question) {
    // Update state
    state.isLoading = true;
    state.currentQuestion = question;
    
    // Hide previous sections
    hideAnswer();
    hideError();
    
    // Show loading
    showLoading();
    
    // Disable form
    setFormDisabled(true);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question }),
            signal: AbortSignal.timeout(CONFIG.TIMEOUT)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        const answer = data.answer;
        
        // Update state
        state.currentAnswer = answer;
        
        // Add to history
        addToHistory(question, answer);
        
        // Show answer
        showAnswer(answer);
        
        // Update history display
        updateHistoryDisplay();
        
    } catch (error) {
        console.error('Error asking question:', error);
        
        let errorMsg = 'An error occurred while processing your question.';
        
        if (error.name === 'AbortError' || error.name === 'TimeoutError') {
            errorMsg = 'Request timed out. Please try again.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMsg = 'Unable to connect to the server. Please check your connection.';
        } else if (error.message) {
            errorMsg = error.message;
        }
        
        showError(errorMsg);
        
    } finally {
        // Update state
        state.isLoading = false;
        
        // Hide loading
        hideLoading();
        
        // Enable form
        setFormDisabled(false);
    }
}

// ============================================
// UI State Management
// ============================================
function showLoading() {
    elements.loadingSection.style.display = 'block';
    elements.loadingSection.classList.add('fade-in');
}

function hideLoading() {
    elements.loadingSection.style.display = 'none';
}

function showAnswer(answer) {
    elements.answerContent.textContent = answer;
    elements.answerTimestamp.textContent = `Answered at ${new Date().toLocaleTimeString()}`;
    elements.answerSection.style.display = 'block';
    elements.answerSection.classList.add('slide-up');
    
    // Scroll to answer
    setTimeout(() => {
        elements.answerSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

function hideAnswer() {
    elements.answerSection.style.display = 'none';
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorSection.style.display = 'block';
    elements.errorSection.classList.add('fade-in');
    
    // Scroll to error
    setTimeout(() => {
        elements.errorSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

function hideError() {
    elements.errorSection.style.display = 'none';
}

function setFormDisabled(disabled) {
    elements.questionInput.disabled = disabled;
    elements.submitBtn.disabled = disabled;
    
    if (disabled) {
        elements.submitBtn.style.opacity = '0.6';
        elements.submitBtn.style.cursor = 'not-allowed';
    } else {
        elements.submitBtn.style.opacity = '1';
        elements.submitBtn.style.cursor = 'pointer';
    }
}

// ============================================
// Copy to Clipboard
// ============================================
async function handleCopy() {
    if (!state.currentAnswer) return;
    
    try {
        await navigator.clipboard.writeText(state.currentAnswer);
        
        // Visual feedback
        const originalText = elements.copyBtn.querySelector('.copy-text').textContent;
        elements.copyBtn.querySelector('.copy-text').textContent = 'Copied!';
        elements.copyBtn.classList.add('copied');
        
        setTimeout(() => {
            elements.copyBtn.querySelector('.copy-text').textContent = originalText;
            elements.copyBtn.classList.remove('copied');
        }, 2000);
        
    } catch (error) {
        console.error('Failed to copy:', error);
        // Fallback: select text
        const range = document.createRange();
        range.selectNode(elements.answerContent);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
    }
}

// ============================================
// History Management
// ============================================
function loadHistory() {
    try {
        const stored = localStorage.getItem('auroraQAHistory');
        return stored ? JSON.parse(stored) : [];
    } catch (error) {
        console.error('Failed to load history:', error);
        return [];
    }
}

function saveHistory() {
    try {
        localStorage.setItem('auroraQAHistory', JSON.stringify(state.history));
    } catch (error) {
        console.error('Failed to save history:', error);
    }
}

function addToHistory(question, answer) {
    // Remove if already exists
    state.history = state.history.filter(item => item.question !== question);
    
    // Add to beginning
    state.history.unshift({
        question,
        answer,
        timestamp: new Date().toISOString()
    });
    
    // Limit history size
    if (state.history.length > CONFIG.MAX_HISTORY) {
        state.history = state.history.slice(0, CONFIG.MAX_HISTORY);
    }
    
    saveHistory();
}

function updateHistoryDisplay() {
    if (state.history.length === 0) {
        elements.historySection.style.display = 'none';
        return;
    }
    
    elements.historySection.style.display = 'block';
    elements.historyList.innerHTML = '';
    
    state.history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.setAttribute('data-question', item.question);
        
        historyItem.innerHTML = `
            <div class="history-question">${escapeHtml(item.question)}</div>
            <div class="history-answer">${escapeHtml(item.answer)}</div>
        `;
        
        elements.historyList.appendChild(historyItem);
    });
}

function clearHistory() {
    if (confirm('Are you sure you want to clear your question history?')) {
        state.history = [];
        saveHistory();
        updateHistoryDisplay();
    }
}

// ============================================
// Keyboard Shortcuts
// ============================================
function handleKeyboard(e) {
    // Ctrl+Enter or Cmd+Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!state.isLoading) {
            handleSubmit(e);
        }
    }
    
    // Escape to clear input
    if (e.key === 'Escape' && document.activeElement === elements.questionInput) {
        elements.questionInput.value = '';
        elements.questionInput.blur();
    }
}

// ============================================
// Utility Functions
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Periodic health check (every 30 seconds)
setInterval(checkHealth, 30000);

