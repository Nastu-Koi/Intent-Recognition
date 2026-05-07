# Agents 开发手册

本目录保存各个 Agent 的本地实现。每个 Agent 由 `agent_a2a_service.py` 独立进程加载，对外暴露为 A2A Agent，通过 JSON-RPC 协议与主编排服务通信。

## 架构说明

```
┌─────────────────────────────────────────────────┐
│               agent_a2a_service.py               │
│                                                   │
│  A2A_AGENT_ID=xxx A2A_PORT=yyyy                  │
│                                                   │
│  启动时:                                           │
│    1. 加载 agents/<AGENT_ID>/agent_card.yaml       │
│    2. 动态导入 execution.module.class_name         │
│    3. 创建 SubAgent 实例                           │
│    4. 启动 FastAPI 服务:                            │
│       GET  /health                                 │
│       GET  /.well-known/agent-card.json            │
│       POST /a2a/<agent_id>  (JSON-RPC)            │
│                                                   │
│  JSON-RPC 方法:                                    │
│    - message/send  接收任务并执行                   │
│    - tasks/get     查询任务状态                     │
└─────────────────────────────────────────────────┘
```

## 当前内置 Agent

| Agent ID | 名称 | 类型 | 端口 | 说明 |
|----------|------|------|------|------|
| `general_chat` | 通用对话助手 | 本地 Agent | 8101 | 日常对话 + 图片识别 + 文档总结（内部 ReAct 循环） |
| `dify_expense_assistant` | 报销助手 | Dify Agent | 8102 | 财务报销政策与流程咨询 |

### general_chat — 通用对话助手

基于 LLM Tool Calling 的本地 Agent，支持内部 ReAct 循环：

```
输入 + 文件上下文 → LLM(with_tools) ──→ 直接回复（无需工具）
                                    └──→ 调用工具（图片识别 / 文档总结）
                                           └──→ 上传到 Dify → Dify API → 结果返回 LLM
                                                    └──→ 综合生成回复
```

工具函数封装在 `agents/general_chat/tools.py`：

| 工具 | 功能 | 链路 |
|------|------|------|
| `image_recognition` | 图片 OCR / 场景分析 / 发票识别 | 上传到 Dify → Dify Vision API |
| `document_summary` | 文档总结 / 要点提炼 | 上传到 Dify → Dify Doc Summary API |

tools.py 内部自动完成「上传文件到 Dify 获取 file_id → 调用 Dify App」的完整流程。

### dify_expense_assistant — 报销助手

通过 Dify API 调用 Dify Workflow App 提供报销咨询服务。支持 Chat 和 Workflow 两种模式，通过 `DIFY_DIFY_EXPENSE_ASSISTANT_APP_TYPE` 环境变量控制。

## Agent 目录结构

```
agents/<agent_id>/
├── agent_card.yaml     # 必需：能力声明（metadata / capabilities / execution 等）
├── subagent.py         # 必需：业务执行器（继承 DifySubAgent，实现 execute 方法）
└── __init__.py         # 可选：包标识
```

## 启动方式

```bash
# 启动通用对话助手
A2A_AGENT_ID=general_chat A2A_PORT=8101 python agent_a2a_service.py

# 启动报销助手
A2A_AGENT_ID=dify_expense_assistant A2A_PORT=8102 python agent_a2a_service.py

# 启动主编排服务
python main.py
```

## 添加新 Agent

系统支持两种类型的 Agent，根据需要选择：

### 1. 本地 Agent（纯 Python 实现，不依赖外部服务）

适合不需要 Dify 后端的场景，可以直接在 `execute()` 中编写任意业务逻辑。

**创建目录与能力声明**

```bash
mkdir -p agents/my_agent
touch agents/my_agent/__init__.py
```

**agent_card.yaml**

```yaml
metadata:
  agent_id: my_agent
  name: 我的助手
  description: "处理特定业务..."
  version: "1.0.0"
  category: "general"

capabilities:
  skills:
    - my_skill
  keywords:
    - 关键词
  intent_patterns:
    - 意图
  priority: 50

execution:
  module: "agents.my_agent.subagent"
  class_name: "MyAgent"
  mode: "sync"
```

**subagent.py**

```python
from engine.dify_subagent import DifySubAgent

class MyAgent(DifySubAgent):
    def __init__(self):
        super().__init__(agent_id="my_agent")

    def execute(self, input_data):
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        # 业务逻辑...
        return {"status": "success", "result": "处理结果", "agent": self.agent_id}
```

### 2. Dify Agent（封装 Dify App）

适合已有 Dify App 的场景，继承 `DifySubAgent` 并调用 `query_dify_app()` 即可。

**subagent.py**

```python
from engine.dify_subagent import DifySubAgent
from engine.dify_client import query_dify_app

class MyDifyAgent(DifySubAgent):
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

### 注册到主编排服务

**注册 A2A 发现端点**

编辑 `.config/a2a_agents.yaml`：

```yaml
agents:
  - id: my_agent
    card_url: "http://127.0.0.1:8103/.well-known/agent-card.json"
```

**配置 RBAC 权限**

编辑 `.config/role_permissions.yaml`，将新 Agent 添加到对应角色的 `accessible_agents` 列表。

**启动**

```bash
A2A_AGENT_ID=my_agent A2A_PORT=8103 python agent_a2a_service.py
```

重启主编排服务 `python main.py`，即可在 `/agents` 端点和 Planner 的任务规划中看到新 Agent。

## SubAgent 返回值规范

`execute()` 方法必须返回字典，格式如下：

```python
# 成功
{"status": "success", "result": "回复文本", "agent": "agent_id"}

# 失败
{"status": "error", "error": "错误信息", "agent": "agent_id"}

# 成功 + 结构化数据（结果会透传到下游 Agent 的 prior_structured）
{
    "status": "success",
    "result": "回复文本",
    "agent": "agent_id",
    "file_id": "xxx",           # 自定义结构化字段
    "uploaded_files": [...]     # 自定义结构化字段
}
```

`status` 为 `success` 时，`result` 将作为 Agent 的回复文本返回。额外的结构化字段（不含 `status` / `result` / `error` / `query` / `agent`）会作为 `structured_output` 透传给后续 Agent 的 `prior_structured`。
