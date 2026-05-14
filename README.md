# Intent-Recognition

基于 LangGraph 的多智能体编排系统，采用 **Planner-Evaluator 反馈循环架构** + **A2A (Agent-to-Agent) 协议** 实现智能体的动态编排、远程调用与结果质量自检。

![architecture](./docs/architecture.svg)

---

## 目录

- [Intent-Recognition](#intent-recognition)
  - [目录](#目录)
  - [项目简介](#项目简介)
  - [核心特性](#核心特性)
  - [快速开始](#快速开始)
    - [前置要求](#前置要求)
    - [步骤](#步骤)
      - [1. 克隆并配置环境变量](#1-克隆并配置环境变量)
      - [2. 安装依赖](#2-安装依赖)
      - [3. 初始化数据库](#3-初始化数据库)
      - [4. 启动 A2A Agent 服务](#4-启动-a2a-agent-服务)
      - [5. 启动主编排服务](#5-启动主编排服务)
      - [6. 验证系统](#6-验证系统)
      - [7. 访问 Web UI](#7-访问-web-ui)
  - [项目结构](#项目结构)
  - [运行流程详解](#运行流程详解)
  - [配置说明](#配置说明)
    - [环境变量](#环境变量)
      - [LLM 配置](#llm-配置)
      - [Dify 配置](#dify-配置)
      - [数据库配置](#数据库配置)
      - [日志配置](#日志配置)
    - [LLM 供应商](#llm-供应商)
    - [A2A 发现配置](#a2a-发现配置)
    - [RBAC 角色权限](#rbac-角色权限)
  - [API 文档](#api-文档)
    - [端点一览](#端点一览)
    - [`/chat`](#chat)
    - [`/chat-with-files`](#chat-with-files)
  - [Agent 系统](#agent-系统)
    - [内置 Agent](#内置-agent)
    - [Agent Card 声明](#agent-card-声明)
    - [本地 Agent 实现](#本地-agent-实现)
    - [添加新 Agent](#添加新-agent)
      - [添加新本地 Agent](#添加新本地-agent)
      - [添加新 Dify Agent](#添加新-dify-agent)
  - [LangGraph 编排引擎](#langgraph-编排引擎)
    - [State 设计](#state-设计)
    - [Planner 节点](#planner-节点)
    - [Dispatcher 节点](#dispatcher-节点)
    - [Evaluator 节点](#evaluator-节点)
    - [Final Reply 节点](#final-reply-节点)
  - [Web 前端](#web-前端)
  - [日志系统](#日志系统)
  - [常见问题](#常见问题)

---

## 项目简介

Intent-Recognition 是一个生产级的多智能体编排框架，核心思路是用 **LLM 驱动的工作流** 替代传统的硬编码业务逻辑路由。

**运作方式**：用户请求到达后，系统通过 RBAC 验证角色权限，通过 A2A 协议动态发现可用 Agent 的能力，由 Planner (LLM) 分析需求并拆解为任务，Dispatcher 并发调用远程 Sub Agent 执行，Evaluator (LLM) 评估结果质量并决定是否需重新规划，最终由 Final Reply 综合输出。

**典型场景**：
- 企业智能助手（报销查询、知识库问答、文件识别）
- 多 Agent 协作系统（不同 Agent 分属不同业务域）
- 需要 LLM 自主规划与质量自检的复杂任务管线

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **Planner-Evaluator 循环** | LangGraph 有状态图驱动，LLM 自主规划→执行→评估反馈闭环 |
| **A2A 协议通信** | Agent-to-Agent JSON-RPC，每个 Agent 独立进程，语言无关 |
| **动态 Agent 发现** | 通过 `./well-known/agent-card.json` 自动发现远程 Agent 的能力描述 |
| **RBAC 权限控制** | 基于 YAML 配置的角色- Agent 访问矩阵，支持管理员通配 |
| **多轮对话记忆** | PostgreSQL 持久化 LangGraph Checkpoint，会话可中断恢复 |
| **思维链可追溯** | 完整记录每轮迭代的规划思路、评估决策、Agent 结果，前端可展开查看 |
| **防死锁机制** | 三重保护：查重熔断 + 动态宽容度 + 5 轮硬性熔断 |
| **多模态输入** | 图片识别 (OCR/发票/场景)、文档总结，通过 LLM Tool Calling 自主决策 |
| **通用 LLM 接口** | 统一 OpenAI-Compatible API，支持 Qwen / DeepSeek / OpenAI / Ollama 等 |
| **Web UI 开箱即用** | 响应式聊天界面，支持角色切换、文件拖传、历史会话管理 |

---

## 快速开始

### 前置要求

- **Python 3.11+**
- **PostgreSQL 14+**（用于会话持久化，可降级运行）
- **LLM API Key**（支持所有 OpenAI-compatible 供应商）

### 步骤

#### 1. 克隆并配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 LLM 配置：

```ini
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
DATABASE_URL=postgresql://postgres:password@localhost:5432/intent_recognition
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 初始化数据库

确保 PostgreSQL 运行中，并创建数据库：

```bash
createdb intent_recognition
```

系统首次启动时会自动创建所需表（`conversation_metadata` + LangGraph Checkpoint 表）。

#### 4. 启动 A2A Agent 服务

每个 Agent 运行在独立进程中：

```bash
# 终端 1：通用对话助手（建议优先启动）
A2A_AGENT_ID=general_chat A2A_PORT=8101 python agent_a2a_service.py

# 终端 2：报销助手（如已配置 Dify）
A2A_AGENT_ID=expense_assistant A2A_PORT=8102 python agent_a2a_service.py
```

每个 Agent 启动后会自动加载对应 `agents/<agent_id>/agent_card.yaml`，并通过 HTTP 暴露 A2A Agent Card 和 JSON-RPC 端点。验证启动成功：

```bash
curl http://127.0.0.1:8101/health
# {"status":"ok","agent_id":"general_chat","card_loaded":true,...}
```

#### 5. 启动主编排服务

```bash
python main.py
```

预期输出：
```
✅ 已加载环境变量文件: .env
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     LangGraph orchestrator ready.
INFO:     Discovered X remote A2A agents
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 6. 验证系统

```bash
# 健康检查
curl http://localhost:8000/health

# 查看已发现 Agent
curl http://localhost:8000/agents

# 发送对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "你好，介绍一下自己"}'
```

#### 7. 访问 Web UI

打开浏览器访问 `http://localhost:8000`

---

## 项目结构

```
Intent-Recognition/
│
├── main.py                          # FastAPI 应用入口 — HTTP 端点、RBAC 验证、A2A 发现、LangGraph 调度
├── agent_a2a_service.py             # A2A Agent 独立服务进程 — 加载本地 Agent Card + 处理 JSON-RPC 请求
├── requirements.txt                 # Python 依赖清单
├── .env.example                     # 环境变量模板（含注释说明）
├── .gitignore                       # Git 忽略规则
│
├── .config/                         # 全局配置目录
│   ├── a2a_agents.yaml              #   A2A Agent 端点注册表（card_url 列表）
│   ├── role_permissions.yaml        #   RBAC 角色权限矩阵
│   └── llm_config.yml               #   LLM 模型参数（环境变量可覆盖）
│
├── engine/                          # 引擎层 — 基础设施、协议、客户端
│   ├── llm_factory.py               #   LLM 工厂：加载 .env + YAML → 初始化 ChatOpenAI
│   ├── agent_card.py                #   Agent Card 数据模型：YAML ↔ Python Dataclass
│   ├── a2a.py                       #   A2A 协议：Agent Card 发现 + JSON-RPC 客户端
│   ├── agent_cards.py               #   Agent Card 管理器：缓存、刷新、索引
│   ├── agent_card_loader.py         #   Agent Card 加载器：本地 YAML 发现 + SubAgent 动态导入
│   ├── dify_client.py               #   Dify API 客户端：Chat / Workflow / Retrieval
│   ├── subagent.py             #   SubAgent 基类（异常处理封装）
│   ├── rbac.py                      #   RBAC：YAML 配置加载 → 角色验证 → Agent 过滤
│   └── logging_config.py            #   日志系统：文件 + 控制台双输出，支持 LOG_LEVEL 控制
│
├── orchestrator/                    # 编排层 — LangGraph 有状态工作流
│   ├── graph.py                     #   LangGraph 图定义：节点注册 + 条件边路由
│   ├── state.py                     #   OrchestratorState：全局状态类型定义 + Pydantic 结构化输出
│   └── nodes/
│       ├── planner.py               #     Planner 节点：LLM 分析需求 → 生成 PlanOutput
│       ├── dispatcher.py            #     Dispatcher 节点：A2A JSON-RPC 并发调用远程 Agent
│       ├── evaluator.py             #     Evaluator 节点：LLM 评估质量 → 决策路由
│       ├── final_reply.py           #     Final Reply 节点：综合多源结果 → 自然语言回复
│       └── planner_modules/
│           ├── agent_router.py      #     关键词/意图匹配路由工具
│           └── task_builder.py      #     任务项构建与校验工具
│
├── agents/                          # Agent 本地实现
│   ├── general_chat/                #   通用对话助手（内置工具调用）
│   │   ├── agent_card.yaml          #     能力声明：技能、关键词、意图模式
│   │   ├── subagent.py              #     GeneralChatAgent：LLM ReAct 循环 + Tool Calling
│   │   └── tools.py                 #     工具函数：image_recognition, document_summary
│   ├── expense_assistant/      #   报销助手（Dify Workflow 封装）
│   │   ├── agent_card.yaml
│   │   └── subagent.py              #     DifyExpenseAssistantAgent
│   └── __init__.py
│
├── db/
│   └── store.py                     # 会话元数据持久化：ConversationStore（PostgreSQL）
│
├── web/                             # Web 前端
│   ├── static/
│   │   ├── app.js                   #   前端逻辑：聊天、上传、会话管理、思维链展示
│   │   └── style.css                #   样式
│   └── templates/
│       └── index.html               #   Jinja2 模板：单页聊天界面
│
└── uploads/                         # 文件上传存储目录（.gitignore 中排除）
```

---

## 运行流程详解

```
┌────────── 用户输入 ──────────┐
                              │
                              ▼
┌── Step 1: FastAPI 入口 ──────────────────────────────────┐
│  main.py 接收 HTTP 请求 (POST /chat 或 POST /chat-with-files) │
│  生成/复用 session_id                                     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌── Step 2: RBAC 验证 ─────────────────────────────────────┐
│  RoleBasedAccessControl.get_accessible_agents(role)       │
│  ✓ 有效角色 → 返回可访问的 agent_id 集合                    │
│  ✗ 无效角色 → 返回 403                                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌── Step 3: A2A Agent 发现 ─────────────────────────────────┐
│  discover_a2a_agent_cards() → 从 .config/a2a_agents.yaml   │
│  注册的端点获取 Agent Card (JSON)                          │
│  RBAC.filter_cards() → 过滤出角色可访问的 Agent             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌── Step 4: LangGraph 图执行 ──────────────────────────────────┐
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  4a. Planner Node                                    │    │
│  │  LLM(with_structured_output=PlanOutput)              │    │
│  │  分析: query + file_ctx + feedback_history           │    │
│  │  输出: rationale + tasks[{target, instruction}]      │    │
│  └──────────┬───────────────────────────────────────────┘    │
│             │                                                │
│             ▼                                                │
│    ┌───────────────┐                                        │
│    │  tasks 为空?   │                                        │
│    │  (直接回复)    │──── 是 ──────┐                           │
│    └───────┬───────┘              │                           │
│            │ 否                   │                           │
│            ▼                      │                           │
│  ┌───────────────────────┐        │                         │
│  │  4b. Dispatcher Node  │        │                         │
│  │  asyncio 并发调用      │        │                         │
│  │  A2A JSON-RPC         │        │                         │
│  │  message/send         │        │                         │
│  │  累积 result          │        │                         │
│  └────────┬──────────────┘        │                        │
│           ▼                       │                        │
│  ┌────────────────────┐           │                        │
│  │  4c. Evaluator Node│           │                        │
│  │  LLM 评估累积结果   │           │                        │
│  │  输出: action      │           │                        │
│  │  PASS / PARTIAL    │───────── 是 ─┐                     │
│  │  ACCEPT / NEEDS_   │              │                    │
│  │  REVISION          │              │                    │
│  └────────┬───────────┘              │                    │
│     否 ↑  │ action=NEEDS_REVISION    │                    │
│     (迭代)│ && iter < 5             │                    │
│           ▼                          ▼                    │
│  ┌────────────────────┐  ┌────────────────────┐           │
│  │ 返回 Planner       │  │  4d. Final Reply   │           │
│  │ (反馈循环)         │  │  LLM 综合结果      │           │
│  │ iter + 1           │  │  生成自然语言回复   │           │
│  └────────────────────┘  └────────┬───────────┘           │
│                                   │                       │
└───────────────────────────────────┼───────────────────────┘
                                    │
                                    ▼
┌── Step 5: HTTP 响应 ─────────────────────────────────────┐
│  返回 ChatResponse: answer + session_id + thinking_chain   │
│  异步持久化: upsert_conversation_metadata                  │
└──────────────────────────────────────────────────────────┘
```

**关键路由逻辑**（定义在 `orchestrator/graph.py`）：

| 条件 | 路由 | 说明 |
|------|------|------|
| Planner 生成了任务 | → Dispatcher | 需要调度 Agent 执行 |
| Planner 未生成任务 | → Final Reply | 直接对话，无需 Agent |
| Evaluator PASS/PARTIAL | → Final Reply | 结果可接受，综合输出 |
| Evaluator NEEDS_REVISION & iter < 5 | → Planner | 需要修改，重新规划 |
| iter >= 5 | → Final Reply | 硬性熔断，强制输出 |

---

## 配置说明

### 环境变量

#### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key | — |
| `LLM_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名 | `gpt-4o` |
| `LLM_TEMPERATURE` | 温度 (0-1) | `0.0` |

#### Dify 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DIFY_API_BASE_URL` | Dify API 地址 | `https://api.dify.ai/v1` |
| `DIFY_API_KEY` | 全局 Dify API Key (所有 Agent 共用) | — |
| `DIFY_APP_TYPE` | Dify App 类型 | `chat` |
| `DIFY_TIMEOUT` | 请求超时(秒) | `90` |

每个 Dify Agent 支持独立 API Key，格式为 `DIFY_{AGENT_ID}_API_KEY`（全大写，`-` 替换为 `_`）。例如 `DIFY_VISION_API_KEY`。不配置时降级使用全局 `DIFY_API_KEY`。

#### 数据库配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql://postgres:password@localhost:5432/intent_recognition` |

#### 日志配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LOG_LEVEL` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `LOG_FILE` | 日志文件路径 | `log.txt` |
| `LOG_ENABLE_CONSOLE` | 是否输出到控制台 | `true` |
| `LOG_ENABLE_FILE` | 是否输出到日志文件 | `true` |

### LLM 供应商

系统通过 `engine/llm_factory.py` 统一使用 OpenAI-Compatible API 格式。只需修改 `.env` 中的三个变量即可切换供应商：

```ini
# 阿里云 Qwen
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max

# DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# 本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

配置优先级：**环境变量 > `.config/llm_config.yml` > 代码默认值**。

### A2A 发现配置

`.config/a2a_agents.yaml` 定义了主服务从哪些端点发现远程 Agent：

```yaml
agents:
  - id: general_chat
    card_url: "http://127.0.0.1:8101/.well-known/agent-card.json"
  - id: expense_assistant
    card_url: "http://127.0.0.1:8102/.well-known/agent-card.json"
```

每个端点返回 A2A Agent Card (JSON)，包含 Agent 的名称、描述、技能、关键词、意图模式等。主服务缓存这些信息用于 Planner 的决策。

### RBAC 角色权限

`.config/role_permissions.yaml` 定义了角色与 Agent 的访问关系：

```yaml
roles:
  role_a:
    name: "普通员工"
    description: "基础员工，仅可访问知识库问答"
    accessible_agents:
      - general_chat           # 只允许访问 general_chat

  role_b:
    name: "高级员工"
    description: "高级员工，可使用知识库问答和报销助手"
    accessible_agents:
      - general_chat
      - expense_assistant

  role_admin:
    name: "管理员"
    description: "完整访问所有 Agent"
    accessible_agents: []      # 空列表 = 通配所有 Agent

default_role: role_admin
```

角色在 Web UI 右上角下拉菜单中切换，或在 API 请求中通过 `role` 字段指定。

---

## API 文档

### 端点一览

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/` | Web UI 入口页面 |
| GET | `/health` | 健康检查（含 Agent 发现数和角色数） |
| GET | `/agents` | 列出所有已发现的 A2A Agent |
| GET | `/roles` | 列出所有 RBAC 角色 |
| POST | `/refresh-agents` | 手动刷新 Agent 发现缓存 |
| POST | `/chat` | 聊天 API (JSON) |
| POST | `/chat-with-files` | 聊天 API (multipart，含文件上传) |
| POST | `/upload` | 文件上传（返回文件元信息） |
| GET | `/conversations` | 历史会话列表 |
| GET | `/conversations/{session_id}/messages` | 会话消息 + 思维链 |
| DELETE | `/conversations/{session_id}` | 删除会话（含 Checkpoint 数据） |

### `/chat`

**请求体**：

```json
{
  "query": "差旅费怎么报销？",
  "role": "role_b",
  "session_id": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 用户输入文本 |
| `role` | string | 否 | 角色 ID（不传则使用 default_role） |
| `session_id` | string | 否 | 会话 ID（不传则新建会话） |

**响应体**：

```json
{
  "answer": "差旅费报销流程如下：...",
  "session_id": "a1b2c3d4-...",
  "iterations": 2,
  "plan_rationale": "用户询问报销流程，需要调度报销助手...",
  "eval_action": "PASS",
  "eval_thought": "累积结果已完整解答了用户问题...",
  "agent_results": {
    "expense_assistant": "差旅费报销标准为..."
  },
  "thinking_chain": [
    {
      "iteration": 1,
      "plan_rationale": "第一轮规划...",
      "eval_action": "NEEDS_REVISION",
      "eval_thought": "需要补充...",
      "agent_results": {}
    },
    {
      "iteration": 2,
      "plan_rationale": "根据反馈修正...",
      "eval_action": "PASS",
      "eval_thought": "已完整回答...",
      "agent_results": {
        "expense_assistant": "..."
      }
    }
  ]
}
```

### `/chat-with-files`

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 用户输入 |
| `role` | string | 否 | 角色 ID |
| `session_id` | string | 否 | 会话 ID |
| `files` | File[] | 否 | 多个文件（自动判断图片/文档） |

**响应**：与 `/chat` 同结构。

---

## Agent 系统

### 内置 Agent

| Agent ID | 名称 | 类型 | 端口 | 说明 |
|----------|------|------|------|------|
| `general_chat` | 通用对话助手 | 本地 Agent | 8101 | 对话 + 图片识别 (OCR/发票/场景) + 文档总结 |
| `expense_assistant` | 报销助手 | Dify Agent | 8102 | 财务报销政策与流程咨询 |

**两种 Agent 类型**：

- **本地 Agent**：Python 进程直接加载，在 `agent_a2a_service.py` 中运行。如 `general_chat` 使用 LLM Tool Calling 实现内部 ReAct 循环。
- **Dify Agent**：通过 Dify API 调用 Dify App，支持 Chat 和 Workflow 两种模式。

### Agent Card 声明

每个 Agent 通过 `agents/<agent_id>/agent_card.yaml` 声明其身份与能力。Planner 据此决策任务分配。

以 `general_chat` 为例：

```yaml
# agents/general_chat/agent_card.yaml

metadata:
  agent_id: general_chat           # Agent 唯一标识（与目录名一致）
  name: 通用对话助手                # 显示名称
  description: "通用对话助手，支持日常聊天、问答、图像识别和文档总结..."  # Planner 决策依据
  version: "1.0.0"
  category: "general"
  author: "System"

capabilities:
  skills:                          # 技能标签
    - general_conversation
    - image_recognition
    - document_summary
  keywords:                        # 关键词匹配
    - 聊天
    - 图片
    - 总结
    - OCR
  intent_patterns:                 # 意图模式
    - 日常对话
    - 识别图片
    - 总结文档
  confidence_threshold: 0.5
  priority: 50                     # 优先级（越高越优先，A2A 发现排序用）

execution:
  module: "agents.general_chat.subagent"    # Python 模块路径
  class_name: "GeneralChatAgent"            # 执行器类
  mode: "sync"

configuration:
  max_iterations: 5
  timeout: 180

scope:                             # 业务范围（用于展示）
  - 日常聊天与通用问答
  - 图片/截图内容识别 (OCR、场景分析)
  - 文档总结与要点提炼

examples:                          # 示例问题（用于展示）
  - 你好
  - 帮我识别这张发票上的金额
  - 帮我总结这份合同的关键条款
```

### 本地 Agent 实现

以 `general_chat` 为例，其内部实现了一个 **LLM ReAct 循环**：

```
用户输入 + 文件上下文
         │
         ▼
┌──────────────────────┐
│  LLM (绑定工具)       │
│  with_tools:          │
│    - image_recognition│
│    - document_summary │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │ 调用工具?    │
    ├──否─────────┤
    │ 直接文本回复  │
    ├──是─────────┤
    │ ▼
    │ ┌──────────────────────┐
    │ │  执行工具 (上传 →     │
    │ │  Dify API → 返回结果) │
    │ └──────────┬───────────┘
    │            │
    └────────────┘
       (最多 5 轮)
```

工具函数 (`engine/tools.py`):

| 工具 | 功能 | Upload → API 链路 |
|------|------|-------------------|
| `image_recognition` | 图片 OCR / 场景分析 / 发票识别 | 上传到 Dify → Dify Vision API |
| `document_summary` | 文档总结 / 要点提炼 | 上传到 Dify → Dify Doc Summary API |

工具内部自动封装了「上传文件到 Dify → 获取 file_id → 调用 Dify App」的完整流程，Planner 无需关心文件上传细节。

### 添加新 Agent

#### 添加新本地 Agent

1. **创建目录与能力声明**

```bash
mkdir -p agents/my_agent
touch agents/my_agent/__init__.py
```

创建 `agents/my_agent/agent_card.yaml`，填写元数据、能力描述和执行模块路径。

2. **创建执行器**

```python
# agents/my_agent/subagent.py
from engine.subagent import SubAgent

class MyAgent(SubAgent):
    def __init__(self):
        super().__init__(agent_id="my_agent")

    def execute(self, input_data):
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        # 业务逻辑...
        return {"status": "success", "result": "处理结果", "agent": self.agent_id}
```

3. **注册到 A2A 发现**

编辑 `.config/a2a_agents.yaml`，添加端点：

```yaml
agents:
  - id: my_agent
    card_url: "http://127.0.0.1:8103/.well-known/agent-card.json"
```

4. **配置 RBAC 权限**

编辑 `.config/role_permissions.yaml`，将 `my_agent` 添加到对应角色。

5. **启动**

```bash
A2A_AGENT_ID=my_agent A2A_PORT=8103 python agent_a2a_service.py
```

#### 添加新 Dify Agent

与本地 Agent 步骤类似，执行器继承 `SubAgent` 并调用 `query_dify_app()`：

```python
from engine.subagent import SubAgent
from engine.dify_client import query_dify_app

class MyDifyAgent(SubAgent):
    def __init__(self):
        super().__init__(agent_id="my_dify_agent")

    def execute(self, input_data):
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        result = query_dify_app(
            agent_id=self.agent_id,
            query=query,
            inputs=context.get("prior_structured"),
        )
        return {"status": "success", "result": result, "agent": self.agent_id}
```

---

## LangGraph 编排引擎

### State 设计

`OrchestratorState`（定义在 `orchestrator/state.py`）继承自 LangGraph 的 `MessagesState`，包含：

**不可变输入上下文**（由 FastAPI 层注入，图中不修改）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | `str` | 用户当前输入 |
| `file_ctx` | `dict` | 文件上下文 (images / documents) |
| `role` | `str` | 用户角色 |
| `available_agents` | `list[dict]` | RBAC 过滤后的 Agent 能力描述 |

**动态数据**（图中各节点读写）：

| 字段 | 类型 | Reducer | 说明 |
|------|------|---------|------|
| `plan` | `dict` | 覆盖 | Planner 输出的任务计划 (rationale + tasks) |
| `results` | `dict` | `operator.ior` | Dispatcher 累积结果（增量合并） |
| `_agent_outputs` | `dict` | `operator.ior` | Dispatcher 结构化输出（增量合并） |
| `iter` | `int` | `operator.add` | 迭代轮次 (自动+1) |
| `feedback_history` | `list[str]` | `operator.add` | Evaluator 增量反馈（自动追加） |
| `eval_action` | `str` | 覆盖 | 最新评估决策 |
| `eval_thought` | `str` | 覆盖 | 最新评估思考 |
| `final_text` | `str` | 覆盖 | 最终回复文本 |
| `thinking_chain` | `list[dict]` | 覆盖 | 完整思维链历史 |

### Planner 节点

`orchestrator/nodes/planner.py`

- 使用 `LLM.with_structured_output(PlanOutput)` 生成结构化规划
- 分析输入：用户查询 + 文件上下文 + 对话历史 + 历史反馈
- 输出：`PlanOutput(rationale, tasks[])`
- 每个 task 包含 `target` (agent_id) 和 `instruction` (执行指令)
- 无任务时 (`tasks=[]`) 表示直接对话，跳过 Dispatcher + Evaluator
- 降级路径：不支持结构化输出的模型 → 原始 LLM + JSON 解析

### Dispatcher 节点

`orchestrator/nodes/dispatcher.py`

- 通过 `aiohttp` 异步发送 A2A JSON-RPC `message/send` 请求
- 任务按序执行（非并发），后序任务自动接收前序累积的 `prior_results` 和 `prior_structured`
- 支持文件上下文传递：`file_ctx` 随请求发送给 Agent
- 异常处理覆盖：HTTP 错误、超时、Agent 返回错误，均有结构化兜底

### Evaluator 节点

`orchestrator/nodes/evaluator.py`

- 使用 `LLM.with_structured_output(EvalResult)` 评估累积结果
- 输出：`EvalResult(thought, action, feedback)`
- 支持模型降级（同 Planner）
- **防死锁三重保障**：

| 机制 | 说明 | 代码位置 |
|------|------|----------|
| 查重熔断 | 检测 feedback 是否与历史重复 → 自动 PARTIAL_ACCEPT | System prompt 规则 #3 |
| 动态宽容度 | iter >= 4 时放宽至 70% 完成即可放行 | System prompt 动态注入 `tolerance_note` |
| 硬性熔断 | iter >= 5 时 Graph 层面强制路由到 Final Reply | `graph.py` route_after_eval |

### Final Reply 节点

`orchestrator/nodes/final_reply.py`

- 综合所有 Sub Agents 的累积结果
- 进行来源归因，标注每个 Agent 的贡献
- 生成面向用户的自然语言回复
- 处理两种情况：有 Sub Agents 结果（综合多源） / 无结果（直接对话）

---

## Web 前端

位于 `web/` 目录，基于原生 JavaScript + Jinja2 模板的单页应用。

**功能**：

| 功能 | 说明 |
|------|------|
| 聊天界面 | 类似 ChatGPT 的消息流布局，支持 Markdown 渲染 |
| 角色切换 | 右上角下拉菜单切换 RBAC 角色 |
| 文件上传 | 支持多文件选择和拖放，自动判断图片/文档 |
| 思维链展开 | 点击「查看完整 Agent 思考过程」查看每轮迭代详情 |
| 历史会话 | 侧边栏列出历史会话，支持切换、删除 |
| 快速操作 | 欢迎页面的快捷按钮（知识库问答、文档总结、图片识别等） |

---

## 日志系统

`engine/logging_config.py` 提供统一的日志配置：

- **双输出**：同时输出到控制台和 `log.txt`
- **级别控制**：通过 `LOG_LEVEL` 环境变量控制日志详细程度
- **关键日志点**：

| 日志点 | 级别 | 内容 |
|--------|------|------|
| Planner Rationale | INFO | 规划思路摘要 |
| Planner Tasks | INFO | 生成的任务列表 (target) |
| Dispatcher A2A 调用 | INFO | 调用的 Agent ID 和任务指令 |
| Dispatcher 结果 | INFO | 结果 keys 和结构化输出 keys |
| Evaluator 决策 | INFO | Action + 思考摘要 + 当前轮次 |
| Evaluator 反馈 | INFO | NEEDS_REVISION 时的具体反馈 |
| Final Reply | INFO | 生成回复长度 |
| 路由决策 | INFO | Planner → Dispatcher/Final_Reply, Evaluator → Planner/Final_Reply |
| 硬性熔断 | INFO | 5 轮上限触发 |
| Agent 发现 | INFO | 发现 N 个远程 Agent |
| 文件上传 | INFO | 文件名、路径、类型 |

---

## 常见问题

**Q：如何切换 LLM 模型或供应商？**

A：修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL`。支持所有 OpenAI-compatible API。例如切换到 DeepSeek：`LLM_BASE_URL=https://api.deepseek.com/v1` + `LLM_MODEL=deepseek-chat`。

**Q：数据库连接失败能运行吗？**

A：可以。系统检测到 `DATABASE_URL` 不可用时会以降级模式运行：LangGraph 使用内存 Checkpointer (`MemorySaver`)，会话无法持久化但核心对话功能不受影响。

**Q：没有 Dify 服务可以运行吗？**

A：可以。`general_chat` Agent 不需要 Dify 即可进行通用对话。图片识别和文档总结功能需要配置 Dify API Key 和对应的 Dify App。`expense_assistant` 则需要 Dify 后端。

**Q：如何调试 Agent 的调用过程？**

A：设置 `LOG_LEVEL=DEBUG` 后重启。关键信息在日志中均有输出：Planner 的 Rationale、Dispatcher 的 A2A 请求/响应、Evaluator 的评估决策和思考过程。前端也可展开「查看完整 Agent 思考过程」查看思维链。

**Q：Agent 调用超时怎么处理？**

A：Dispatcher 节点对每个 Agent 设置了 60 秒超时。超时的 Agent 不会阻塞整体流程，其错误信息会传入 Evaluator 进行综合判断。可在 `dispatcher.py` 的 `_a2a_send_message` 中调整 `timeout` 参数。

**Q：Web UI 无法访问怎么办？**

A：确认 `main.py` 正常运行且无报错，端口未被占用。检查 `web/templates/index.html` 和 `web/static/` 目录是否存在。

**Q：如何扩展系统的 Agent 数量？**

A：参考「添加新 Agent」部分。每个 Agent 需要：`agent_card.yaml`（能力声明）、`subagent.py`（执行逻辑）、注册到 `a2a_agents.yaml`、配置 RBAC 权限。添加后重启主服务即可自动发现。
