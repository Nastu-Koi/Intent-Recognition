/**
 * Intent-Recognition Chat UI — Frontend Logic
 */

const API_BASE = '';
let sessionId = null;
let pendingFiles = [];
let currentStreamingController = null; // For stopping stream generation
let isGenerating = false; // Track generation state
const AGENT_DISPLAY_NAMES = {
    expense_assistant: '报销助手',
    general_chat: '通用对话助手',
};

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
    // 恢复保存的 sessionId（用于相同页面中的多轮对话）
    const savedSessionId = localStorage.getItem('intentRecognitionSessionId');
    if (savedSessionId) {
        sessionId = savedSessionId;
        console.log(`[Init] Restored sessionId from localStorage: ${sessionId}`);
    }
    
    loadRoles();
    loadConversations();
    chatInput.focus();

    // Initialize send button state
    updateSendButtonState();

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
    chatInput.placeholder = '输入问题或拖入文件...'; // Reset placeholder

    // Show file names if any
    if (pendingFiles.length > 0) {
        const fileNames = pendingFiles.map(f => f.name).join(', ');
        appendMessage('user', `📎 ${fileNames}`);
    }

    // Create streaming response container
    const messageEl = createStreamingMessageElement();
    const thinkingChain = [];

    // Create AbortController for this stream
    currentStreamingController = new AbortController();
    isGenerating = true;
    updateSendButtonState();

    // Disable input & file upload
    chatInput.disabled = true;
    uploadBtn.disabled = true;

    try {
        if (pendingFiles.length > 0) {
            await sendWithFilesStream(text, messageEl, thinkingChain);
        } else {
            await sendTextStream(text, messageEl, thinkingChain);
        }

        // 不需要再调用一次 adoptSessionId，因为 processStreamResponse 中已经调用过了

    } catch (e) {
        if (e.name !== 'AbortError') {
            appendMessage('agent', `❌ 网络错误: ${e.message}`);
        }
    }

    // Clear files
    pendingFiles = [];
    filePreview.innerHTML = '';

    // Reset generation state
    isGenerating = false;
    currentStreamingController = null;
    updateSendButtonState();

    // Re-enable input
    chatInput.disabled = false;
    uploadBtn.disabled = false;
    chatInput.placeholder = '输入问题或拖入文件...'; // Ensure placeholder is reset
    chatInput.focus();
}

// ─── Stop Generation ───
function stopGeneration() {
    if (currentStreamingController) {
        currentStreamingController.abort();
        isGenerating = false;
        updateSendButtonState();
    }
}

function updateSendButtonState() {
    if (isGenerating) {
        sendBtn.innerHTML = '■'; 
        sendBtn.title = '停止生成';
        sendBtn.classList.add('generating');
        sendBtn.onclick = stopGeneration;
    } else {
        sendBtn.innerHTML = '➤'; // Send arrow icon
        sendBtn.title = '发送';
        sendBtn.classList.remove('generating');
        sendBtn.onclick = sendMessage;
    }
}

function adoptSessionId(nextSessionId) {
    if (!nextSessionId) return;

    const isNew = !sessionId || sessionId !== nextSessionId;
    sessionId = nextSessionId;
    
    // 保存 sessionId 到 localStorage（用于相同页面中的多轮对话）
    localStorage.setItem('intentRecognitionSessionId', sessionId);
    console.log(`[adoptSessionId] Updated sessionId: ${sessionId}`);
    
    if (isNew) {
        console.log(`[adoptSessionId] New session detected, loading conversations`);
        loadConversations();
    }
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
    
    console.log(`[sendTextStream] Sending query with sessionId: ${sessionId}`);
    
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            role,
            session_id: sessionId,
        }),
        signal: currentStreamingController.signal,
    });

    return processStreamResponse(response, arguments[1], arguments[2]);
}

async function sendWithFilesStream(query) {
    const role = roleSelect.value || null;
    const formData = new FormData();
    formData.append('query', query);
    if (role) formData.append('role', role);
    // Always append session_id (even if null) to maintain consistency with sendTextStream
    if (sessionId) formData.append('session_id', sessionId);
    pendingFiles.forEach(f => formData.append('files', f));

    console.log(`[sendWithFilesStream] Sending query with sessionId: ${sessionId}`);

    const response = await fetch(`${API_BASE}/chat-with-files-stream`, {
        method: 'POST',
        body: formData,
        signal: currentStreamingController.signal,
    });

    return processStreamResponse(response, arguments[1], arguments[2]);
}

function ensureDispatcherProgress(contentEl) {
    let dispatcherResult = contentEl.querySelector('.dispatcher-result');
    if (!dispatcherResult) {
        contentEl.innerHTML = `
            <div class="dispatcher-result">
                <div class="process-label">📊 Agent 执行阶段</div>
                <div class="agent-list"></div>
            </div>
        `;
        dispatcherResult = contentEl.querySelector('.dispatcher-result');
    }
    return dispatcherResult;
}

function upsertAgentProgress(contentEl, eventData) {
    const dispatcherResult = ensureDispatcherProgress(contentEl);
    const agentList = dispatcherResult.querySelector('.agent-list');
    const agentId = eventData.agent_id || '';
    const agentName = eventData.agent_name || agentId;
    let agentItem = Array.from(agentList.children).find(el => el.dataset.agentId === agentId);

    if (!agentItem) {
        agentItem = document.createElement('div');
        agentItem.className = 'agent-item';
        agentItem.dataset.agentId = agentId;
        agentItem.innerHTML = `
            <div class="agent-executing"></div>
            <div class="agent-result-preview"></div>
        `;
        agentList.appendChild(agentItem);
    }

    const statusEl = agentItem.querySelector('.agent-executing');
    const previewEl = agentItem.querySelector('.agent-result-preview');
    const statusLabel = {
        started: '⏳ 正在调用 Agent',
        completed: '✓ 已完成 Agent',
        skipped: '⚠️ 已跳过 Agent',
    }[eventData.status] || '• Agent 状态更新';

    statusEl.innerHTML = `${statusLabel}: <strong>${escapeHtml(agentName)}</strong>`;
    agentItem.classList.toggle('running', eventData.status === 'started');
    agentItem.classList.toggle('completed', eventData.status === 'completed');

    const preview = eventData.result_preview || eventData.message || eventData.instruction || '';
    if (preview) {
        previewEl.textContent = preview;
    }
}


function renderAgentResultsHtml(agentResults, labels = {}) {
    if (!agentResults || typeof agentResults !== 'object') return '';

    const entries = Object.entries(agentResults).filter(([agentId]) => !agentId.startsWith('_'));
    if (entries.length === 0) return '';

    const completedLabel = labels.completed || '✓ 已完成 Agent';
    const agentNames = labels.agentNames || {};
    return `
        <div class="agent-list">
            ${entries.map(([agentId, result]) => {
                const preview = String(result || '');
                const agentName = agentNames[agentId] || AGENT_DISPLAY_NAMES[agentId] || agentId;
                return `
                    <div class="agent-item completed">
                        <div class="agent-executing">${completedLabel}: <strong>${escapeHtml(agentName)}</strong></div>
                        <div class="agent-result-preview">${escapeHtml(preview)}</div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}


async function processStreamResponse(response, messageEl, thinkingChain) {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get session ID from response header
    const sessionIdHeader = response.headers.get('X-Session-Id');
    if (sessionIdHeader) {
        messageEl.dataset.sessionId = sessionIdHeader;
        adoptSessionId(sessionIdHeader);
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
    let currentState = null; // For paused context
    let accumulatedContent = ''; // 累积所有已显示过的内容
    let streamingTokens = false; // 是否正在通过 token 流式输出
    let streamedRawText = ''; // 流式期间累积的纯文本
    const agentNameMap = {};

    const contentEl = messageEl.querySelector('.streaming-content');
    const detailsEl = messageEl.querySelector('.process-details');

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

                                case 'conversation_router':
                                    if (eventData.relation === 'ambiguous') {
                                        contentEl.innerHTML = `
                                            <div class="feedback-eval">
                                                <div class="eval-header">需要更多信息</div>
                                                <div class="eval-details">
                                                    <p>${escapeHtml(eventData.clarification_question || '可以再补充一点背景吗？')}</p>
                                                </div>
                                            </div>
                                        `;
                                    } else {
                                        const routeText = eventData.relation === 'related'
                                            ? `识别为关联输入：${eventData.related_type || 'related'}`
                                            : '识别为新对话';
                                        contentEl.innerHTML = `<div class="thinking-text">${escapeHtml(routeText)}</div>`;
                                    }
                                    break;

                                case 'planner':
                                    plannerCount++;
                                    planRationale = eventData.rationale;
                                    accumulatedContent = `规划阶段 (第 ${eventData.iteration} 轮)\n${eventData.rationale}`; // 累积规划思考过程
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

                                case 'dispatcher_progress':
                                    const dispatcherProgress = ensureDispatcherProgress(contentEl);
                                    let progressSummary = dispatcherProgress.querySelector('.dispatcher-progress-summary');
                                    if (!progressSummary) {
                                        progressSummary = document.createElement('div');
                                        progressSummary.className = 'dispatcher-progress-summary';
                                        dispatcherProgress.insertBefore(progressSummary, dispatcherProgress.querySelector('.agent-list'));
                                    }
                                    if (eventData.status === 'started') {
                                        progressSummary.textContent = `正在调度 ${eventData.tasks_count || 0} 个 Agent...`;
                                    } else if (eventData.status === 'completed') {
                                        progressSummary.textContent = `Agent 调度完成，共 ${eventData.agents_count || 0} 个结果`;
                                    }
                                    break;

                                case 'agent_progress':
                                    if (eventData.agent_id && eventData.agent_name) {
                                        agentNameMap[eventData.agent_id] = eventData.agent_name;
                                    }
                                    upsertAgentProgress(contentEl, eventData);
                                    if (eventData.status === 'completed' && eventData.result_preview) {
                                        const progressAgentName = eventData.agent_name || eventData.agent_id;
                                        accumulatedContent += `\n\nAgent: ${progressAgentName}\n${eventData.result_preview}`;
                                    }
                                    break;

                                case 'agent_result':
                                    agentResults[eventData.agent_id] = eventData.result_preview;
                                    const agentName = eventData.agent_name || eventData.agent_id;
                                    if (eventData.agent_id && agentName) {
                                        agentNameMap[eventData.agent_id] = agentName;
                                    }
                                    upsertAgentProgress(contentEl, {
                                        agent_id: eventData.agent_id,
                                        agent_name: agentName,
                                        status: 'completed',
                                        result_preview: eventData.result_preview,
                                    });
                                    break;

                                case 'dispatcher':
                                    // 更新 dispatcher 摘要信息（追加而不覆盖）
                                    const dispatcherResult = contentEl.querySelector('.dispatcher-result');
                                    if (dispatcherResult) {
                                        const summaryDiv = dispatcherResult.querySelector('.dispatcher-summary') || 
                                                          (() => {
                                                              const div = document.createElement('div');
                                                              div.className = 'dispatcher-summary';
                                                              dispatcherResult.appendChild(div);
                                                              return div;
                                                          })();
                                        summaryDiv.innerHTML = `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color);">已完成 ${eventData.agents_count} 个 Agent 的调用</div>`;
                                    }
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
                                    
                                    // Save state for potential pause
                                    currentState = {
                                        thinking_chain: thinkingChain,
                                        current_answer: finalAnswer
                                    };
                                    
                                    break;

                                case 'final_reply_token':
                                    // ─── 逐 token 流式输出 ───
                                    if (!streamingTokens) {
                                        streamingTokens = true;
                                        streamedRawText = '';
                                        contentEl.innerHTML = '<span class="streaming-text"></span><span class="streaming-cursor"></span>';
                                    }
                                    streamedRawText += eventData.token;
                                    {
                                        const textSpan = contentEl.querySelector('.streaming-text');
                                        if (textSpan) {
                                            textSpan.textContent = streamedRawText;
                                        }
                                    }
                                    scrollToBottom();
                                    break;

                                case 'final_reply_done':
                                    // ─── 流式完成：格式化 Markdown + 构建 process-details ───
                                    if (streamingTokens && streamedRawText) {
                                        finalAnswer = streamedRawText;
                                        accumulatedContent = finalAnswer;
                                        contentEl.innerHTML = formatMarkdown(finalAnswer);
                                    }
                                    planRationale = eventData.plan_rationale || planRationale;
                                    evalAction = eventData.eval_action || evalAction;
                                    totalIterations = eventData.total_iterations || totalIterations;
                                    agentResults = eventData.agent_results || agentResults;
                                    if (eventData.thinking_chain && eventData.thinking_chain.length > 0) {
                                        thinkingChain = eventData.thinking_chain;
                                    }
                                    if (Object.keys(agentResults).length > 0 || planRationale || evalAction) {
                                        buildProcessDetails(detailsEl, thinkingChain, totalIterations, agentNameMap);
                                    }
                                    streamingTokens = false;
                                    break;

                                case 'final_reply':
                                    // ─── 图级别兜底事件 ───
                                    finalAnswer = eventData.answer || finalAnswer;
                                    accumulatedContent = finalAnswer;
                                    planRationale = eventData.plan_rationale || planRationale;
                                    evalAction = eventData.eval_action || evalAction;
                                    totalIterations = eventData.total_iterations || totalIterations;
                                    agentResults = eventData.agent_results || agentResults;

                                    if (eventData.thinking_chain && eventData.thinking_chain.length > 0) {
                                        thinkingChain = eventData.thinking_chain;
                                    }

                                    // 如果内容已通过 token 流式推送，不重复覆盖 DOM
                                    if (!eventData.streamed) {
                                        contentEl.innerHTML = formatMarkdown(finalAnswer);
                                        if (Object.keys(agentResults).length > 0 || planRationale || evalAction) {
                                            buildProcessDetails(detailsEl, thinkingChain, totalIterations, agentNameMap);
                                        }
                                    }
                                    break;

                                case 'done':
                                    // Final message already shown
                                    break;

                                case 'feedback_eval':
                                    // 处理反馈评估结果
                                    const isRelevant = eventData.is_relevant;
                                    const relevanceScore = eventData.relevance_score;
                                    const reason = eventData.reason;
                                    const newSessionId = eventData.new_session_id;
                                    
                                    if (isRelevant) {
                                        contentEl.innerHTML = `
                                            <div class="feedback-eval">
                                                <div class="eval-header">✅ 反馈已评估为相关内容</div>
                                                <div class="eval-details">
                                                    <p><strong>相关度分数:</strong> ${(relevanceScore * 100).toFixed(1)}%</p>
                                                    <p><strong>分析:</strong> ${escapeHtml(reason)}</p>
                                                    <p style="color: var(--text-secondary);">正在基于您的补充信息继续处理...</p>
                                                </div>
                                            </div>
                                        `;
                                    } else {
                                        contentEl.innerHTML = `
                                            <div class="feedback-eval">
                                                <div class="eval-header">🔄 启动新会话</div>
                                                <div class="eval-details">
                                                    <p><strong>相关度分数:</strong> ${(relevanceScore * 100).toFixed(1)}%</p>
                                                    <p><strong>分析:</strong> ${escapeHtml(reason)}</p>
                                                    <p style="color: var(--text-secondary);">您的问题与之前的内容关联度低，正在以新会话处理您的输入...</p>
                                                </div>
                                            </div>
                                        `;
                                    }
                                    
                                    // 如果是新会话，更新会话ID
                                    if (newSessionId) {
                                        sessionId = newSessionId;
                                    }
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
    } catch (e) {
        throw e;
    } finally {
        reader.releaseLock();
        
        // 保存前端收集的内容到后端（即使被中止也能保存）
        if (messageEl.dataset.sessionId) {
            try {
                const sessionId = messageEl.dataset.sessionId;
                
                // 优先级：finalAnswer > accumulatedContent > DOM 文本
                let partialReply = finalAnswer || accumulatedContent;
                
                if (!partialReply) {
                    // 从 contentEl 中提取所有文本（包括嵌套的文本）
                    partialReply = contentEl.innerText || contentEl.textContent || '';
                }
                
                // 清理和处理提取的文本
                partialReply = partialReply.trim();
                
                if (partialReply.length > 0) {
                    console.log('[SavePartialReply] Saving content:', { 
                        length: partialReply.length, 
                        source: finalAnswer ? 'finalAnswer' : (accumulatedContent ? 'accumulatedContent' : 'DOM'),
                        sessionId 
                    });
                    
                    fetch(`${API_BASE}/conversations/${sessionId}/save-partial-reply`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ partial_reply: partialReply })
                    }).then(r => {
                        if (r.ok) console.log('[SavePartialReply] ✅ Success');
                        else console.warn('[SavePartialReply] ❌ Server error:', r.status);
                    }).catch(err => console.warn('[SavePartialReply] ❌ Network failed:', err));
                } else {
                    console.log('[SavePartialReply] ⚠️ No content to save (finalAnswer, accumulatedContent, DOM all empty)');
                }
            } catch (err) {
                console.warn('[SavePartialReply] ❌ Exception:', err);
            }
        }
    }
}

function buildProcessDetails(detailsEl, thinkingChain, totalIterations, agentNameMap = {}) {
    detailsEl.innerHTML = '';

    const toggle = document.createElement('button');
    toggle.className = 'process-toggle';
    // 计算实际有内容的迭代项数（包括有agent_results的项）
    const validIterations = thinkingChain.filter(item => item && (item.plan_rationale || item.agent_results || item.eval_action)).length;
    toggle.innerHTML = `<span class="arrow">▶</span> 查看完整 Agent 思考过程 (${validIterations} 轮迭代)`;

    const content = document.createElement('div');
    content.className = 'process-content';

    toggle.onclick = () => {
        toggle.classList.toggle('open');
        content.classList.toggle('open');
    };

    let html = '';
    thinkingChain.forEach((item, index) => {
        // 只要有任何内容就显示，不要求必须有plan_rationale或eval_action
        const hasContent = item && (item.plan_rationale || item.agent_results || item.eval_action || item.eval_thought);
        if (hasContent) {
            html += `<div class="iteration"><h4>第 ${item.iteration || index + 1} 轮迭代</h4>`;
            if (item.plan_rationale) {
                html += `<div class="process-item"><div class="process-label">🧠 规划思路</div>${escapeHtml(item.plan_rationale)}</div>`;
            }
            // 添加 Agent 执行阶段的结果 (在评估决策之前)
            if (item.agent_results && typeof item.agent_results === 'object' && Object.keys(item.agent_results).length > 0) {
                const agentResultsHtml = renderAgentResultsHtml(item.agent_results, {
                    agentNames: { ...(item.agent_names || {}), ...agentNameMap },
                });
                if (agentResultsHtml) {
                    html += `<div class="process-item"><div class="process-label">📊 Agent 执行结果</div>${agentResultsHtml}</div>`;
                }
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
        const validItems = result.thinking_chain.filter(item => item && (item.plan_rationale || item.agent_results || item.eval_action)).length;
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
                const agentResultsHtml = renderAgentResultsHtml(item.agent_results, {
                    agentNames: item.agent_names || {},
                });
                if (agentResultsHtml) {
                    html += `<div class="process-item"><div class="process-label">📊 Agent 执行结果</div>${agentResultsHtml}</div>`;
                }
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
        
        // 显示保存的回答预览（在 meta 之后）
        if (conv.last_reply) {
            const preview = document.createElement('div');
            preview.className = 'conv-preview';
            preview.textContent = conv.last_reply.substring(0, 100) + (conv.last_reply.length > 100 ? '...' : '');
            info.appendChild(preview);
        }
        
        item.appendChild(info);
        item.appendChild(delBtn);
        conversationList.appendChild(item);
    });
}

async function loadConversation(id) {
    if (id === sessionId) return;
    
    adoptSessionId(id);
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
    localStorage.removeItem('intentRecognitionSessionId');
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
    const validItems = thinking_chain.filter(item => item && (item.plan_rationale || item.agent_results || item.eval_action)).length;
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
        if (item.agent_results && Object.keys(item.agent_results).length > 0) {
            const agentResultsHtml = renderAgentResultsHtml(item.agent_results, {
                completed: '✓ Completed Agent',
                agentNames: item.agent_names || {},
            });
            if (agentResultsHtml) {
                html += `<div class="process-item"><div class="process-label">📊 Results</div>${agentResultsHtml}</div>`;
            }
        }
        if (item.eval_action) {
            const emoji = { PASS: '✅', PARTIAL_ACCEPT: '⚠️', NEEDS_REVISION: '🔄' }[item.eval_action] || '❓';
            html += `<div class="process-item"><div class="process-label">🎯 Action</div>${emoji} ${escapeHtml(item.eval_action)}</div>`;
        }
        if (item.eval_thought) {
            html += `<div class="process-item"><div class="process-label">🧐 Analysis</div>${escapeHtml(item.eval_thought)}</div>`;
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
