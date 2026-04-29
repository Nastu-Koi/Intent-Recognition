# ============================================================================
# Intent-Recognition FastAPI 应用 Dockerfile
# ============================================================================
# 构建指令:
#   docker build -t intent-recognition:latest .
# 运行指令:
#   docker run --env-file .env -p 8000:8000 intent-recognition:latest
# ============================================================================

FROM python:3.11-slim

# ─── 设置工作目录 ───
WORKDIR /app

# ─── 使用国内镜像源加速 apt 更新 ───
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list || true

# ─── 安装系统依赖（使用重试机制） ───
RUN apt-get update || apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ─── 复制 requirements.txt 并安装 Python 依赖 ───
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── 复制应用代码 ───
COPY . .

# ─── 创建必要的目录 ───
RUN mkdir -p uploads logs

# ─── 暴露端口 ───
EXPOSE 8000

# ─── 健康检查 ───
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# ─── 启动应用 ───
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
