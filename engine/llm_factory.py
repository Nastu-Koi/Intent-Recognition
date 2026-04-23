"""
LLM 配置加载模块 (OpenAI-Compatible 统一格式)

所有 LLM 调用统一使用 ChatOpenAI (langchain_openai)。
通过 base_url 参数兼容所有 OpenAI-compatible 供应商：
  - OpenAI:     https://api.openai.com/v1
  - 阿里云 Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1
  - Anthropic:  通过 OpenAI-compatible 代理
  - DeepSeek:   https://api.deepseek.com/v1
  - 本地 Ollama: http://localhost:11434/v1

配置优先级: 环境变量 > llm_config.yml > 默认值
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

_LOADED_ENV_FILES: set[str] = set()


def load_env_file(env_file: str = ".env"):
    """
    加载 .env 文件中的环境变量

    优先级：已存在的环境变量 > .env 文件 > 默认值
    """
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / env_file
        env_key = str(env_path.resolve())
        if env_key in _LOADED_ENV_FILES:
            return

        if env_path.exists():
            load_dotenv(env_path, override=False)
            _LOADED_ENV_FILES.add(env_key)
            print(f"✅ 已加载环境变量文件: {env_file}")
        else:
            print(f"⚠️  未找到 .env 文件: {env_file}")
            print(f"   提示：复制 .env.example 为 .env 并填写你的 API Key")

    except ImportError:
        print("⚠️  未安装 python-dotenv，跳过 .env 文件加载")


def load_config(config_path: str = ".config/llm_config.yml") -> Dict[str, Any]:
    """从 YAML 配置文件加载配置。"""
    project_root = Path(__file__).parent.parent
    config_file = project_root / config_path

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def substitute_env_vars(value: str) -> str:
    """替换配置中的 ${ENV_VAR} 引用。"""
    if not isinstance(value, str):
        return value

    import re
    pattern = r'\$\{([^}]+)\}'

    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))

    return re.sub(pattern, replace_env, value)


def init_llm_model(config: Dict[str, Any]):
    """
    使用 ChatOpenAI 统一初始化 LLM 模型 (OpenAI-Compatible)。

    通过 base_url 参数适配不同的 LLM 供应商：
      - 阿里云 Qwen: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
      - OpenAI:      base_url="https://api.openai.com/v1" (默认)
      - DeepSeek:    base_url="https://api.deepseek.com/v1"
      - 本地 Ollama:  base_url="http://localhost:11434/v1"
    """
    from langchain_openai import ChatOpenAI

    llm_config = config.get('llm', {})

    # 从环境变量或配置文件读取参数
    api_key = os.getenv(
        'LLM_API_KEY',
        substitute_env_vars(llm_config.get('api_key', ''))
    )

    base_url = os.getenv(
        'LLM_BASE_URL',
        llm_config.get('base_url', 'https://api.openai.com/v1')
    )

    model_name = os.getenv(
        'LLM_MODEL',
        llm_config.get('model', 'gpt-4o')
    )

    temperature = float(os.getenv(
        'LLM_TEMPERATURE',
        llm_config.get('temperature', 0.0)
    ))

    max_tokens = int(llm_config.get('max_tokens', 4096))
    timeout = int(llm_config.get('timeout', 300))

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return model


def get_max_iterations(config_path: str = ".config/llm_config.yml") -> int:
    """从配置文件读取最大迭代轮次。"""
    if 'MAX_ITERATIONS' in os.environ:
        return int(os.getenv('MAX_ITERATIONS'))

    config = load_config(config_path)
    agent_config = config.get('agent', {})
    return int(agent_config.get('max_iterations', 10))


def get_llm_model(config_path: str = ".config/llm_config.yml"):
    """
    一站式入口：加载 .env → 加载 YAML 配置 → 初始化 ChatOpenAI 模型。

    使用示例：
    ```python
    from engine.llm_factory import get_llm_model
    model = get_llm_model()
    ```
    """
    load_env_file(".env")
    config = load_config(config_path)
    return init_llm_model(config)
