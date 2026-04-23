/**
 * Intent-Recognition Chat UI — Frontend Logic
 */

const API_BASE = '';
let sessionId = null;
let pendingFiles = [];

// ─── DOM Elements ───
const chatArea = document.getElementById('chat-area');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const roleSelect = document.getElementById('role-select');
const newChatBtn = document.getElementById('new-chat-btn');
const welcomeContainer = document.getElementById('welcome-container');

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    loadRoles();
    chatInput.focus();
});

// ─── Load Roles ───
async function loadRoles() {
    try {
        const res = await fetch(`${API_BASE}/roles`);
        const data = await res.json();
        roleSelect.innerHTML = '';
        if (data.roles) {
            data.roles.forEach(role => {
                const opt = document.createElement('option');
                opt.value = role.id;
                opt.textContent = role.name;
                if (role.id === data.default_role) opt.selected = true;
                roleSelect.appendChild(opt);
            });
        }
        if (roleSelect.options.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'No roles';
            roleSelect.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load roles:', e);
    }
}

// ─── Send Message ───
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query && pendingFiles.length === 0) return;

    const text = query || '请分析上传的文件';

    // Hide welcome
    if (welcomeContainer) {
        welcomeContainer.style.display = 'none';
    }

    // Show user message
    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show file names if any
    if (pendingFiles.length > 0) {
        const fileNames = pendingFiles.map(f => f.name).join(', ');
        appendMessage('user', `📎 ${fileNames}`);
    }

    // Show thinking
    const thinkingEl = showThinking();

    // Disable input
    sendBtn.disabled = true;

    try {
        let result;
        if (pendingFiles.length > 0) {
            result = await sendWithFiles(text);
        } else {
            result = await sendText(text);
        }

        // Remove thinking
        thinkingEl.remove();

        // Show agent response
        if (result.answer) {
            appendAgentMessage(result.answer, result);
        } else if (result.error) {
            appendMessage('agent', `❌ 错误: ${result.error}`);
        }

        // Update session
        if (result.session_id) {
            sessionId = result.session_id;
        }

    } catch (e) {
        thinkingEl.remove();
        appendMessage('agent', `❌ 网络错误: ${e.message}`);
    }

    // Clear files
    pendingFiles = [];
    filePreview.innerHTML = '';

    // Re-enable
    sendBtn.disabled = false;
    chatInput.focus();
}

async function sendText(query) {
    const role = roleSelect.value || null;
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            role,
            session_id: sessionId,
        }),
    });
    return await res.json();
}

async function sendWithFiles(query) {
    const role = roleSelect.value || null;
    const formData = new FormData();
    formData.append('query', query);
    if (role) formData.append('role', role);
    if (sessionId) formData.append('session_id', sessionId);
    pendingFiles.forEach(f => formData.append('files', f));

    const res = await fetch(`${API_BASE}/chat-with-files`, {
        method: 'POST',
        body: formData,
    });
    return await res.json();
}

// ─── UI Helpers ───
function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = formatMarkdown(text);

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatArea.appendChild(msg);
    scrollToBottom();
}

function appendAgentMessage(text, result) {
    const msg = document.createElement('div');
    msg.className = 'message agent';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = formatMarkdown(text);

    // Process details
    if (result.thinking_chain && result.thinking_chain.length > 0) {
        const details = document.createElement('div');
        details.className = 'process-details';

        const toggle = document.createElement('button');
        toggle.className = 'process-toggle';
        toggle.innerHTML = `<span class="arrow">▶</span> 查看完整 Agent 思考过程 (${result.thinking_chain.length} 轮迭代)`;
        toggle.onclick = () => {
            toggle.classList.toggle('open');
            content.classList.toggle('open');
        };

        const content = document.createElement('div');
        content.className = 'process-content';

        let html = '';
        result.thinking_chain.forEach((item, index) => {
            html += `<div class="iteration"><h4>第 ${item.iteration} 轮迭代</h4>`;
            if (item.plan_rationale) {
                html += `<div class="process-item"><div class="process-label">🧠 规划思路</div>${escapeHtml(item.plan_rationale)}</div>`;
            }
            if (item.eval_action) {
                const emoji = { PASS: '✅', PARTIAL_ACCEPT: '⚠️', NEEDS_REVISION: '🔄' }[item.eval_action] || '❓';
                html += `<div class="process-item"><div class="process-label">🎯 评估决策</div>${emoji} ${escapeHtml(item.eval_action)}</div>`;
            }
            if (item.eval_thought) {
                html += `<div class="process-item"><div class="process-label">🧐 评估分析</div>${escapeHtml(item.eval_thought)}</div>`;
            }
            if (item.agent_results && Object.keys(item.agent_results).length > 0) {
                html += `<div class="process-item"><div class="process-label">📊 Agent 结果</div>`;
                for (const [k, v] of Object.entries(item.agent_results)) {
                    html += `<div><strong>${escapeHtml(k)}</strong>: ${escapeHtml(String(v).substring(0, 300))}${String(v).length > 300 ? '...' : ''}</div>`;
                }
                html += `</div>`;
            }
            html += `</div>`;
        });

        content.innerHTML = html;
        details.appendChild(toggle);
        details.appendChild(content);
        bubble.appendChild(details);
    }

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatArea.appendChild(msg);
    scrollToBottom();
}

function showThinking() {
    const container = document.createElement('div');
    container.className = 'thinking-container';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.style.background = 'var(--bg-tertiary)';
    avatar.style.border = '1px solid var(--border-subtle)';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'thinking-bubble';
    bubble.innerHTML = `
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <div class="thinking-text">正在思考...</div>
    `;

    container.appendChild(avatar);
    container.appendChild(bubble);
    chatArea.appendChild(container);
    scrollToBottom();
    return container;
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ─── Markdown (simple) ───
function formatMarkdown(text) {
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    // Lists
    html = html.replace(/^- (.+)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── File Upload ───
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    files.forEach(f => {
        if (!pendingFiles.find(p => p.name === f.name)) {
            pendingFiles.push(f);
            addFileTag(f);
        }
    });
    fileInput.value = '';
});

function addFileTag(file) {
    const tag = document.createElement('div');
    tag.className = 'file-tag';
    const ext = file.name.split('.').pop().toLowerCase();
    const icon = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext) ? '🖼️' : '📄';
    tag.innerHTML = `${icon} ${escapeHtml(file.name)} <span class="remove" onclick="removeFile('${file.name}', this)">✕</span>`;
    filePreview.appendChild(tag);
}

function removeFile(name, el) {
    pendingFiles = pendingFiles.filter(f => f.name !== name);
    el.parentElement.remove();
}

// ─── Input Events ───
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

sendBtn.addEventListener('click', sendMessage);

// ─── New Chat ───
newChatBtn.addEventListener('click', () => {
    sessionId = null;
    chatArea.innerHTML = '';
    if (welcomeContainer) {
        welcomeContainer.style.display = 'flex';
        if (!chatArea.contains(welcomeContainer)) {
            chatArea.appendChild(welcomeContainer);
        }
    }
    pendingFiles = [];
    filePreview.innerHTML = '';
    chatInput.value = '';
    chatInput.focus();
});

// ─── Quick Actions ───
function quickAction(text) {
    chatInput.value = text;
    sendMessage();
}
