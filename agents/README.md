# Agents README

本目录保存各个 Dify SubAgent 的本地实现。项目采用 A2A-native + LangGraph 编排架构：

```text
FastAPI 入口 (main.py)
  -> RBAC 角色验证 & Agent Card 发现
  -> LangGraph Planner-Evaluator 循环
     -> Planner 根据 Agent Card 规划任务
     -> Workers 通过 A2A JSON-RPC 调用远程 SubAgent
     -> Evaluator 评估结果 (最多 5 轮迭代)
     -> Final Reply 综合回复
```

每个 SubAgent 由 `agent_a2a_service.py` 独立进程加载，对外暴露为 A2A Agent。

## 当前内置 Agent

```text
dify_file_uploader  -> Dify 文件上传器       -> 默认端口 8101
dify_doc_summary    -> Dify 文档总结         -> 默认端口 8102
dify_knowledge_qa   -> Dify 知识库问答       -> 默认端口 8103
dify_vision         -> Dify 图片识别         -> 默认端口 8104
```

### 各 Agent 功能说明

| Agent ID | 功能 | Dify App 类型 | 依赖 |
|---|---|---|---|
| `dify_file_uploader` | 上传文件到 Dify，获取 file_id | Files API | 无 |
| `dify_doc_summary` | 文档内容智能总结与摘要提取 | Workflow | 可选 file_id |
| `dify_knowledge_qa` | 基于知识库的企业内部问答 | Chat | 无 |
| `dify_vision` | 图片识别、OCR、发票识别 | Chat (Vision) | 需要 file_id |

### 任务依赖关系

```text
用户上传图片 -> dify_file_uploader (获取 file_id) -> dify_vision (识别)
用户上传文档 -> dify_file_uploader (获取 file_id) -> dify_doc_summary (总结)
用户提问    -> dify_knowledge_qa (直接问答)
```

Planner 会自动识别依赖关系：如果需要 file_id 的 Agent，会先调度 dify_file_uploader。

## 单个 Agent 目录结构

```text
agents/dify_xxx/
  agent_card.yaml     # 必需：能力声明，转换为 A2A Agent Card
  subagent.py         # 必需：业务执行器 (继承 DifySubAgent)
  __init__.py         # 包初始化
```

## 启动方式

```bash
# 启动各 SubAgent (各自独立进程)
A2A_AGENT_ID=dify_file_uploader A2A_PORT=8101 python agent_a2a_service.py
A2A_AGENT_ID=dify_doc_summary A2A_PORT=8102 python agent_a2a_service.py
A2A_AGENT_ID=dify_knowledge_qa A2A_PORT=8103 python agent_a2a_service.py
A2A_AGENT_ID=dify_vision A2A_PORT=8104 python agent_a2a_service.py

# 启动主编排服务 (LangGraph + FastAPI)
python main.py
```

## 如何添加新 Dify Agent

### 1. 创建目录

```bash
mkdir -p agents/dify_new_agent
touch agents/dify_new_agent/__init__.py
```

### 2. 创建 agent_card.yaml

```yaml
metadata:
  agent_id: dify_new_agent
  name: 新 Dify 助手
  description: "新助手描述"
  version: "1.0.0"

capabilities:
  skills:
    - new_skill
  keywords:
    - 关键词1
  intent_patterns:
    - 新意图

execution:
  module: "agents.dify_new_agent.subagent"
  class_name: "DifyNewAgent"
  mode: "sync"

scope:
  - 新业务范围

examples:
  - 新助手能处理什么问题
```

### 3. 创建 subagent.py

```python
from engine.dify_subagent import DifySubAgent
from engine.dify_client import query_dify_app

class DifyNewAgent(DifySubAgent):
    def __init__(self):
        super().__init__(agent_id="dify_new_agent")

    def execute(self, input_data):
        query = input_data.get("query", "")
        result = query_dify_app(
            agent_id=self.agent_id,
            query=query,
        )
        return {"status": "success", "result": result, "agent": self.agent_id}
```

### 4. 注册到 A2A 发现配置

编辑 `.config/a2a_agents.yaml`：

```yaml
agents:
  - id: dify_new_agent
    card_url: "http://127.0.0.1:8105/.well-known/agent-card.json"
```

### 5. 配置 RBAC 权限

编辑 `.config/role_permissions.yaml`，将新 Agent 添加到对应角色的 `accessible_agents` 列表。

### 6. 启动

```bash
A2A_AGENT_ID=dify_new_agent A2A_PORT=8105 python agent_a2a_service.py
```
