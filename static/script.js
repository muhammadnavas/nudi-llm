// ─── Loader ────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    setTimeout(() => {
        const loader = document.getElementById('loader-wrapper');
        if (loader) loader.classList.add('hidden');
    }, 1800);
});

// ─── Main Chat Logic ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const chatHistory  = document.getElementById('chat-history');
    const promptInput  = document.getElementById('prompt-input');
    const sendBtn      = document.getElementById('send-btn');
    const chatForm     = document.getElementById('chat-form');
    const newChatBtn   = document.getElementById('new-chat-btn');
    const welcomeState = document.getElementById('welcome-state');

    // ── Enable/disable send button based on input ────────────────────────
    promptInput.addEventListener('input', function () {
        const hasText = this.value.trim().length > 0;
        sendBtn.classList.toggle('active', hasText);
        sendBtn.disabled = !hasText;

        // Auto-grow textarea
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });

    // ── Enter to send (Shift+Enter for newline) ──────────────────────────
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (promptInput.value.trim()) submitPrompt();
        }
    });

    // ── Submit via button ────────────────────────────────────────────────
    sendBtn.addEventListener('click', () => {
        if (promptInput.value.trim()) submitPrompt();
    });

    // ── Suggestion chips ─────────────────────────────────────────────────
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            promptInput.value = chip.dataset.prompt;
            promptInput.dispatchEvent(new Event('input'));
            promptInput.focus();
            submitPrompt();
        });
    });

    // ── New chat ─────────────────────────────────────────────────────────
    newChatBtn && newChatBtn.addEventListener('click', () => {
        // Remove all messages (keep welcome state wrapper)
        const rows = chatHistory.querySelectorAll('.message-row');
        rows.forEach(r => r.remove());
        if (welcomeState) welcomeState.style.display = '';
        promptInput.value = '';
        promptInput.style.height = 'auto';
        sendBtn.classList.remove('active');
        sendBtn.disabled = true;
    });

    // ─────────────────────────────────────────────────────────────────────
    async function submitPrompt() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Hide welcome state on first message
        if (welcomeState) welcomeState.style.display = 'none';

        // Append user message
        appendUserMessage(prompt);

        // Clear input
        promptInput.value = '';
        promptInput.style.height = 'auto';
        sendBtn.classList.remove('active');
        sendBtn.disabled = true;

        // Append AI loading indicator
        const loadingId = appendAILoading();

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            const data = await res.json();

            document.getElementById(loadingId).remove();

            if (res.ok) {
                appendAIMessage(data.result);
            } else {
                appendAIMessage('⚠️ Error: ' + (data.detail || data.error || 'Something went wrong.'));
            }
        } catch (err) {
            document.getElementById(loadingId).remove();
            appendAIMessage('⚠️ Could not connect to the server.');
        } finally {
            promptInput.focus();
        }
    }

    // ── Render helpers ───────────────────────────────────────────────────
    function appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'message-row user';
        row.innerHTML = `<div class="bubble">${escapeHTML(text)}</div>`;
        chatHistory.appendChild(row);
        scrollBottom();
    }

    function appendAIMessage(text) {
        const row = document.createElement('div');
        row.className = 'message-row ai';
        row.innerHTML = `
            <div class="ai-avatar">✦</div>
            <div class="bubble">${escapeHTML(text)}</div>
        `;
        chatHistory.appendChild(row);
        scrollBottom();
    }

    function appendAILoading() {
        const id = 'loading-' + Date.now();
        const row = document.createElement('div');
        row.className = 'message-row ai';
        row.id = id;
        row.innerHTML = `
            <div class="ai-avatar">✦</div>
            <div class="bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatHistory.appendChild(row);
        scrollBottom();
        return id;
    }

    function scrollBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
        }[c]));
    }
});
