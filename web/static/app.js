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
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
const closeSidebarBtn = document.getElementById('close-sidebar-btn');
const conversationList = document.getElementById('conversation-list');
const sidebarNewChatBtn = document.getElementById('sidebar-new-chat-btn');

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    loadRoles();
    loadConversations();
    chatInput.focus();

    // Sidebar
    if (toggleSidebarBtn) toggleSidebarBtn.onclick = () => sidebar.classList.toggle('hidden');
    if (closeSidebarBtn) closeSidebarBtn.onclick = () => sidebar.classList.add('hidden');
    if (sidebarNewChatBtn) sidebarNewChatBtn.onclick = startNewChat;
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

    // Create streaming response container
    const messageEl = createStreamingMessageElement();
    const thinkingChain = [];

    // Disable input
    sendBtn.disabled = true;

    try {
        if (pendingFiles.length > 0) {
            await sendWithFilesStream(text, messageEl, thinkingChain);
        } else {
            await sendTextStream(text, messageEl, thinkingChain);
        }

        // Update session
        const sessionIdHeader = messageEl.dataset.sessionId;
        if (sessionIdHeader) {
            const isNew = !sessionId;
            sessionId = sessionIdHeader;
            if (isNew) loadConversations();
        }

    } catch (e) {
        appendMessage('agent', `❌ 网络错误: ${e.message}`);
    }

    // Clear files
    pendingFiles = [];
    filePreview.innerHTML = '';

    // Re-enable
    sendBtn.disabled = false;
    chatInput.focus();
}

function createStreamingMessageElement() {
    const msg = document.createElement('div');
    msg.className = 'message agent';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'bubble streaming';
    bubble.innerHTML = `
        <div class="streaming-content"></div>
        <div class="process-details"></div>
    `;

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatArea.appendChild(msg);
    scrollToBottom();

    return msg;
}

async function sendTextStream(query) {
    const role = roleSelect.value || null;
    const url = `${API_BASE}/chat-stream`;
    
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            role,
            session_id: sessionId,
        }),
    });

    return processStreamResponse(response, arguments[1], arguments[2]);
}

async function sendWithFilesStream(query) {
    const role = roleSelect.value || null;
    const formData = new FormData();
    formData.append('query', query);
    if (role) formData.append('role', role);
    if (sessionId) formData.append('session_id', sessionId);
    pendingFiles.forEach(f => formData.append('files', f));

    const response = await fetch(`${API_BASE}/chat-with-files-stream`, {
        method: 'POST',
        body: formData,
    });

    return processStreamResponse(response, arguments[1], arguments[2]);
}

async function processStreamResponse(response, messageEl, thinkingChain) {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get session ID from response header
    const sessionIdHeader = response.headers.get('X-Session-Id');
    if (sessionIdHeader) {
        messageEl.dataset.sessionId = sessionIdHeader;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalAnswer = '';
    let planRationale = '';
    let evalAction = '';
    let evalThought = '';
    let agentResults = {};
    let totalIterations = 0;

    const contentEl = messageEl.querySelector('.streaming-content');
    const detailsEl = messageEl.querySelector('.process-details');
    let plannerCount = 0;

    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            
            // Keep the last incomplete line in the buffer
            buffer = lines[lines.length - 1];

            for (let i = 0; i < lines.length - 1; i++) {
                const line = lines[i].trim();
                
                if (line.startsWith('event: ')) {
                    const eventType = line.substring(7);
                    const dataLine = lines[i + 1];
                    
                    if (dataLine && dataLine.startsWith('data: ')) {
                        try {
                            const eventData = JSON.parse(dataLine.substring(6));
                            
                            switch (eventType) {
                                case 'start':
                                    contentEl.innerHTML = '<div class="thinking-dots"><span></span><span></span><span></span></div><div class="thinking-text">开始执行思维链...</div>';
                                    break;

                                case 'planner':
                                    plannerCount++;
                                    planRationale = eventData.rationale;
                                    const tasksHtml = eventData.tasks.map((t, idx) => 
                                        `<div class="task-item"><strong>Task ${idx + 1}:</strong> ${escapeHtml(t.target)}<br/><em>${escapeHtml(t.instruction)}</em></div>`
                                    ).join('');
                                    
                                    contentEl.innerHTML = `
                                        <div class="planner-result">
                                            <div class="process-label">🧠 规划阶段 (第 ${eventData.iteration} 轮)</div>
                                            <div class="rationale">${escapeHtml(eventData.rationale)}</div>
                                            <div class="tasks">${tasksHtml}</div>
                                        </div>
                                    `;
                                    
                                    // Add to thinking chain
                                    if (!thinkingChain[plannerCount - 1]) {
                                        thinkingChain[plannerCount - 1] = {};
                                    }
                                    thinkingChain[plannerCount - 1].iteration = eventData.iteration;
                                    thinkingChain[plannerCount - 1].plan_rationale = planRationale;
                                    
                                    break;

                                case 'agent_result':
                                    agentResults[eventData.agent_id] = eventData.result_preview;
                                    contentEl.innerHTML = `
                                        <div class="dispatcher-result">
                                            <div class="thinking-dots"><span></span><span></span><span></span></div>
                                            <div class="agent-executing">正在调用 Agent: <strong>${escapeHtml(eventData.agent_id)}</strong></div>
                                        </div>
                                    `;
                                    break;

                                case 'dispatcher':
                                    contentEl.innerHTML = `
                                        <div class="dispatcher-result">
                                            <div class="process-label">📊 Agent 执行阶段</div>
                                            <div>已完成 ${eventData.agents_count} 个 Agent 的调用</div>
                                        </div>
                                    `;
                                    break;

                                case 'evaluator':
                                    evalAction = eventData.action;
                                    evalThought = eventData.thought;
                                    const actionEmoji = {
                                        'PASS': '✅',
                                        'PARTIAL_ACCEPT': '⚠️',
                                        'NEEDS_REVISION': '🔄'
                                    }[eventData.action] || '❓';
                                    
                                    contentEl.innerHTML = `
                                        <div class="evaluator-result">
                                            <div class="process-label">🎯 评估阶段</div>
                                            <div class="eval-action">${actionEmoji} ${escapeHtml(eventData.action)}</div>
                                            <div class="eval-thought">${escapeHtml(eventData.thought.substring(0, 200))}</div>
                                            <div class="iteration-info">迭代: ${eventData.iteration}/${eventData.max_iterations}</div>
                                        </div>
                                    `;
                                    
                                    // Add to thinking chain
                                    if (thinkingChain[plannerCount - 1]) {
                                        thinkingChain[plannerCount - 1].eval_action = evalAction;
                                        thinkingChain[plannerCount - 1].eval_thought = evalThought;
                                    }
                                    
                                    break;

                                case 'final_reply':
                                    finalAnswer = eventData.answer;
                                    planRationale = eventData.plan_rationale;
                                    evalAction = eventData.eval_action;
                                    totalIterations = eventData.total_iterations;
                                    agentResults = eventData.agent_results;
                                    
                                    // 使用后端返回的完整思维链替换前端累积的数据
                                    if (eventData.thinking_chain && eventData.thinking_chain.length > 0) {
                                        // 清除旧数据，使用后端发送的完整thinking_chain
                                        thinkingChain = eventData.thinking_chain;
                                    }
                                    
                                    // Clear streaming indicator
                                    contentEl.innerHTML = formatMarkdown(finalAnswer);
                                    
                                    // Build process details
                                    if (Object.keys(agentResults).length > 0 || planRationale || evalAction) {
                                        buildProcessDetails(detailsEl, thinkingChain, totalIterations);
                                    }
                                    break;

                                case 'done':
                                    // Final message already shown
                                    break;

                                case 'error':
                                    contentEl.innerHTML = `<div class="error-msg">❌ 错误: ${escapeHtml(eventData.message)}</div>`;
                                    break;
                            }

                            scrollToBottom();
                        } catch (e) {
                            console.error('Error parsing event data:', e);
                        }
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

function buildProcessDetails(detailsEl, thinkingChain, totalIterations) {
    detailsEl.innerHTML = '';

    const toggle = document.createElement('button');
    toggle.className = 'process-toggle';
    // 计算实际有内容的迭代项数
    const validIterations = thinkingChain.filter(item => item && (item.plan_rationale || item.eval_action)).length;
    toggle.innerHTML = `<span class="arrow">▶</span> 查看完整 Agent 思考过程 (${validIterations} 轮迭代)`;

    const content = document.createElement('div');
    content.className = 'process-content';

    toggle.onclick = () => {
        toggle.classList.toggle('open');
        content.classList.toggle('open');
    };

    let html = '';
    thinkingChain.forEach((item, index) => {
        if (item && (item.plan_rationale || item.eval_action)) {
            html += `<div class="iteration"><h4>第 ${item.iteration || index + 1} 轮迭代</h4>`;
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
            html += `</div>`;
        }
    });

    content.innerHTML = html || '<div>无可用的思考过程信息</div>';
    detailsEl.appendChild(toggle);
    detailsEl.appendChild(content);
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
        const validItems = result.thinking_chain.filter(item => item && (item.plan_rationale || item.eval_action)).length;
        toggle.innerHTML = `<span class="arrow">▶</span> 查看完整 Agent 思考过程 (${validItems} 轮迭代)`;
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
                    html += `<div><strong>${escapeHtml(k)}</strong>: ${escapeHtml(String(v))}</div>`;
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

// ─── Persistence Helpers ───

async function loadConversations() {
    try {
        const res = await fetch(`${API_BASE}/conversations`);
        const data = await res.json();
        renderConversationList(data.conversations || []);
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

function renderConversationList(conversations) {
    if (!conversationList) return;
    conversationList.innerHTML = '';
    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = `conversation-item ${conv.session_id === sessionId ? 'active' : ''}`;
        item.dataset.id = conv.session_id;
        
        const info = document.createElement('div');
        info.className = 'conv-info';
        info.onclick = () => loadConversation(conv.session_id);
        
        const title = document.createElement('div');
        title.className = 'conv-title';
        title.textContent = conv.title || 'Untitled Chat';
        
        const meta = document.createElement('div');
        meta.className = 'conv-meta';
        const date = new Date(conv.updated_at).toLocaleDateString();
        meta.textContent = `${date} · ${conv.message_count} msgs`;
        
        const delBtn = document.createElement('button');
        delBtn.className = 'btn-delete-conv';
        delBtn.innerHTML = '✕';
        delBtn.onclick = (e) => {
            e.stopPropagation();
            if (confirm('Delete this conversation?')) {
                deleteConversation(conv.session_id);
            }
        };
        
        info.appendChild(title);
        info.appendChild(meta);
        item.appendChild(info);
        item.appendChild(delBtn);
        conversationList.appendChild(item);
    });
}

async function loadConversation(id) {
    if (id === sessionId) return;
    
    sessionId = id;
    chatArea.innerHTML = '';
    if (welcomeContainer) welcomeContainer.style.display = 'none';
    
    document.querySelectorAll('.conversation-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === id);
    });
    
    try {
        const res = await fetch(`${API_BASE}/conversations/${id}/messages`);
        const data = await res.json();
        
        // Restore messages
        data.messages.forEach(msg => {
            appendMessage(msg.role, msg.content);
        });
        
        // Restore thinking chain for the last agent message
        if (data.thinking_chain && data.thinking_chain.length > 0) {
            const agentBubbles = chatArea.querySelectorAll('.message.agent .bubble');
            if (agentBubbles.length > 0) {
                const lastBubble = agentBubbles[agentBubbles.length - 1];
                appendThinkingToBubble(lastBubble, data.thinking_chain);
            }
        }
        
    } catch (e) {
        console.error('Failed to load conversation details:', e);
        appendMessage('agent', `❌ Failed to load history: ${e.message}`);
    }
}

async function deleteConversation(id) {
    try {
        await fetch(`${API_BASE}/conversations/${id}`, { method: 'DELETE' });
        if (id === sessionId) startNewChat();
        loadConversations();
    } catch (e) {
        console.error('Failed to delete conversation:', e);
    }
}

function startNewChat() {
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
    
    document.querySelectorAll('.conversation-item').forEach(el => {
        el.classList.remove('active');
    });
}

function appendThinkingToBubble(bubble, thinking_chain) {
    const details = document.createElement('div');
    details.className = 'process-details';

    const toggle = document.createElement('button');
    toggle.className = 'process-toggle';
    const validItems = thinking_chain.filter(item => item && (item.plan_rationale || item.eval_action)).length;
    toggle.innerHTML = `<span class="arrow">▶</span> View Thought Process (${validItems} steps)`;
    
    const content = document.createElement('div');
    content.className = 'process-content';

    toggle.onclick = () => {
        toggle.classList.toggle('open');
        content.classList.toggle('open');
    };

    let html = '';
    thinking_chain.forEach((item) => {
        html += `<div class="iteration"><h4>Step ${item.iteration}</h4>`;
        if (item.plan_rationale) {
            html += `<div class="process-item"><div class="process-label">🧠 Rationale</div>${escapeHtml(item.plan_rationale)}</div>`;
        }
        if (item.eval_action) {
            const emoji = { PASS: '✅', PARTIAL_ACCEPT: '⚠️', NEEDS_REVISION: '🔄' }[item.eval_action] || '❓';
            html += `<div class="process-item"><div class="process-label">🎯 Action</div>${emoji} ${escapeHtml(item.eval_action)}</div>`;
        }
        if (item.eval_thought) {
            html += `<div class="process-item"><div class="process-label">🧐 Analysis</div>${escapeHtml(item.eval_thought)}</div>`;
        }
        if (item.agent_results && Object.keys(item.agent_results).length > 0) {
            html += `<div class="process-item"><div class="process-label">📊 Results</div>`;
            for (const [k, v] of Object.entries(item.agent_results)) {
                html += `<div><strong>${escapeHtml(k)}</strong>: ${escapeHtml(String(v))}</div>`;
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

// Update existing newChatBtn click listener
newChatBtn.onclick = startNewChat;

// ─── Quick Actions ───
function quickAction(text) {
    chatInput.value = text;
    sendMessage();
}
