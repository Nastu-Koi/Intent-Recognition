# Intent-Recognition
<img width="690" height="631" alt="Agent架构" src="https://github.com/user-attachments/assets/b70165ec-dc1d-4665-bd84-cac0617cc6fa" />

### 快速启动指南

**1. 创建环境变量文件**
首先，从示例文件复制并创建你的 `.env` 配置文件：

``
cp .env.example .env
``

**2. 配置 API Key**

编辑 .env 文件，填写相关的 API 密钥与基础配置, 并确保数据库正在运行。

**3. 启动 Dify SubAgents**
你需要打开 4 个独立的终端窗口，分别运行以下命令来启动对应的子代理服务：


终端 1：文件上传代理

``
A2A_AGENT_ID=dify_file_uploader A2A_PORT=8101 python agent_a2a_service.py
``

终端 2：文档摘要代理

``
A2A_AGENT_ID=dify_doc_summary A2A_PORT=8102 python agent_a2a_service.py
``

终端 3：知识库问答代理

``
A2A_AGENT_ID=dify_knowledge_qa A2A_PORT=8103 python agent_a2a_service.py
``

终端 4：视觉代理

``
A2A_AGENT_ID=dify_vision A2A_PORT=8104 python agent_a2a_service.py
``

**4. 启动主编排服务**
打开第 5 个终端窗口，启动主控程序：

``
python main.py
``

**5. 访问服务**
所有服务成功启动后，即可在浏览器中访问系统：

``
URL: http://localhost:8000
``
