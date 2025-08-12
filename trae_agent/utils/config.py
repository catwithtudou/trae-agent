# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Trae Agent 配置管理模块

本模块实现了 Trae Agent 的完整配置系统，支持：
- 多种 LLM 提供商的统一配置管理
- 灵活的模型配置和参数设置
- MCP (Model Context Protocol) 服务器配置
- Agent 配置和工具管理
- 配置值的优先级解析（CLI > 环境变量 > 配置文件 > 默认值）
- YAML 和 JSON 格式的向后兼容性
"""

import os
from dataclasses import dataclass, field

import yaml

from trae_agent.utils.legacy_config import LegacyConfig


class ConfigError(Exception):
    """配置相关的异常类"""
    pass


@dataclass
class ModelProvider:
    """
    LLM 模型提供商配置类
    
    用于配置不同的 AI 模型提供商（如 OpenAI、Anthropic、Google 等）的连接信息。
    对于官方提供商，base_url 是可选的；对于 Azure，api_version 是必需的。
    
    Attributes:
        api_key: API 密钥，用于身份验证
        provider: 提供商名称（如 'openai', 'anthropic', 'google' 等）
        base_url: 自定义 API 基础 URL（可选，用于私有部署或代理）
        api_version: API 版本（Azure 专用）
    """

    api_key: str
    provider: str
    base_url: str | None = None
    api_version: str | None = None


@dataclass
class ModelConfig:
    """
    LLM 模型配置类
    
    定义了与特定 AI 模型交互时的所有参数和设置。
    
    Attributes:
        model: 模型名称（如 'gpt-4', 'claude-3-sonnet' 等）
        model_provider: 关联的模型提供商配置
        max_tokens: 最大生成 token 数量
        temperature: 温度参数，控制输出的随机性（0.0-1.0）
        top_p: 核采样参数，控制输出的多样性
        top_k: Top-K 采样参数
        parallel_tool_calls: 是否支持并行工具调用
        max_retries: 最大重试次数
        supports_tool_calling: 是否支持工具调用功能
        candidate_count: 候选响应数量（Gemini 专用）
        stop_sequences: 停止序列列表
    """

    model: str
    model_provider: ModelProvider
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    parallel_tool_calls: bool
    max_retries: int
    supports_tool_calling: bool = True
    candidate_count: int | None = None  # Gemini specific field
    stop_sequences: list[str] | None = None

    def resolve_config_values(
        self,
        *,
        model_providers: dict[str, ModelProvider] | None = None,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        解析和覆盖配置值
        
        当通过 CLI 参数或环境变量提供配置值时，它们会覆盖配置文件中的值。
        优先级：CLI 参数 > 环境变量 > 配置文件 > 默认值
        
        Args:
            model_providers: 可用的模型提供商字典
            provider: CLI 指定的提供商名称
            model: CLI 指定的模型名称
            model_base_url: CLI 指定的基础 URL
            api_key: CLI 指定的 API 密钥
        """
        # 解析模型名称，CLI 参数优先
        self.model = str(resolve_config_value(cli_value=model, config_value=self.model))

        # 处理模型提供商的变更
        # 用户如果想要更改模型提供商，需要满足以下条件之一：
        # 1. 确保提供商名称在 model_providers 字典中可用
        # 2. 如果不可用，则需要提供 base_url 和 api_key 来注册新的提供商
        if provider:
            if model_providers and provider in model_providers:
                # 使用已配置的提供商
                self.model_provider = model_providers[provider]
            elif api_key is None:
                raise ConfigError("To register a new model provider, an api_key should be provided")
            else:
                # 创建新的提供商配置
                self.model_provider = ModelProvider(
                    api_key=api_key,
                    provider=provider,
                    base_url=model_base_url,
                )

        # 将提供商名称映射到对应的环境变量名称
        # 例如：'openai' -> 'OPENAI_API_KEY', 'OPENAI_BASE_URL'
        env_var_api_key = str(self.model_provider.provider).upper() + "_API_KEY"
        env_var_api_base_url = str(self.model_provider.provider).upper() + "_BASE_URL"

        # 按优先级解析 API 密钥：CLI > 环境变量 > 配置文件
        resolved_api_key = resolve_config_value(
            cli_value=api_key,
            config_value=self.model_provider.api_key,
            env_var=env_var_api_key,
        )

        # 按优先级解析基础 URL：CLI > 环境变量 > 配置文件
        resolved_api_base_url = resolve_config_value(
            cli_value=model_base_url,
            config_value=self.model_provider.base_url,
            env_var=env_var_api_base_url,
        )

        # 应用解析后的值
        if resolved_api_key:
            self.model_provider.api_key = str(resolved_api_key)

        if resolved_api_base_url:
            self.model_provider.base_url = str(resolved_api_base_url)


@dataclass
class MCPServerConfig:
    """
    MCP (Model Context Protocol) 服务器配置类
    
    MCP 是一个标准协议，允许 AI 应用程序与外部数据源和工具安全地连接。
    此配置类支持多种传输方式来连接 MCP 服务器。
    
    传输方式：
    - stdio: 通过标准输入输出与本地进程通信
    - sse: 通过 Server-Sent Events 与远程服务器通信
    - http: 通过 HTTP 流式传输
    - websocket: 通过 WebSocket 连接
    """
    
    # stdio 传输方式配置
    command: str | None = None          # 要执行的命令
    args: list[str] | None = None       # 命令参数列表
    env: dict[str, str] | None = None   # 环境变量
    cwd: str | None = None              # 工作目录

    # SSE 传输方式配置
    url: str | None = None              # SSE 服务器 URL

    # HTTP 流式传输配置
    http_url: str | None = None         # HTTP 服务器 URL
    headers: dict[str, str] | None = None  # HTTP 请求头

    # WebSocket 传输配置
    tcp: str | None = None              # TCP 连接地址

    # 通用配置
    timeout: int | None = None          # 连接超时时间（秒）
    trust: bool | None = None           # 是否信任服务器证书

    # 元数据
    description: str | None = None      # 服务器描述信息


@dataclass
class AgentConfig:
    """
    Agent 配置基类
    
    定义了所有 Agent 类型共享的基础配置项。这是一个抽象基类，
    具体的 Agent 类型（如 TraeAgent）会继承并扩展这个配置。
    
    Attributes:
        allow_mcp_servers: 允许使用的 MCP 服务器名称列表
        mcp_servers_config: MCP 服务器配置字典，键为服务器名称
        max_steps: Agent 执行的最大步数限制
        model: 使用的 LLM 模型配置
        tools: Agent 可用的工具名称列表
    """

    allow_mcp_servers: list[str]
    mcp_servers_config: dict[str, MCPServerConfig]
    max_steps: int
    model: ModelConfig
    tools: list[str]


@dataclass
class TraeAgentConfig(AgentConfig):
    """
    Trae Agent 专用配置类
    
    继承自 AgentConfig，添加了 Trae Agent 特有的配置项。
    Trae Agent 是专门用于软件工程任务的智能代理。
    
    Attributes:
        enable_lakeview: 是否启用 LakeView 功能（Agent 步骤的简洁总结）
        tools: 默认工具集合，包含软件工程任务常用的工具
            - bash: 执行 shell 命令
            - str_replace_based_edit_tool: 基于字符串替换的文件编辑工具
            - sequentialthinking: 结构化思考工具
            - task_done: 任务完成标记工具
    """

    enable_lakeview: bool = True
    tools: list[str] = field(
        default_factory=lambda: [
            "bash",                        # Shell 命令执行工具
            "str_replace_based_edit_tool", # 文件编辑工具
            "sequentialthinking",          # 结构化思考工具
            "task_done",                   # 任务完成工具
        ]
    )

    def resolve_config_values(
        self,
        *,
        max_steps: int | None = None,
    ):
        """
        解析 TraeAgent 特有的配置值
        
        Args:
            max_steps: CLI 指定的最大步数
        """
        resolved_value = resolve_config_value(cli_value=max_steps, config_value=self.max_steps)
        if resolved_value:
            self.max_steps = int(resolved_value)


@dataclass
class LakeviewConfig:
    """
    LakeView 功能配置类
    
    LakeView 是 Trae Agent 的一个重要功能，它使用独立的 LLM 模型
    对 Agent 的执行步骤进行简洁的总结，帮助用户快速理解 Agent 的行为。
    
    Attributes:
        model: 用于生成 LakeView 总结的 LLM 模型配置
               通常使用较小、快速的模型来降低成本
    """

    model: ModelConfig


@dataclass
class Config:
    """
    Trae Agent 总配置类
    
    这是整个 Trae Agent 系统的顶级配置类，负责管理所有子系统的配置，
    包括模型提供商、模型配置、Agent 配置和 LakeView 配置等。
    
    Attributes:
        lakeview: LakeView 功能配置（可选）
        model_providers: 模型提供商配置字典
        models: 模型配置字典
        trae_agent: Trae Agent 配置
    """

    lakeview: LakeviewConfig | None = None
    model_providers: dict[str, ModelProvider] | None = None
    models: dict[str, ModelConfig] | None = None

    trae_agent: TraeAgentConfig | None = None

    @classmethod
    def create(
        cls,
        *,
        config_file: str | None = None,
        config_string: str | None = None,
    ) -> "Config":
        """
        从配置文件或配置字符串创建 Config 实例
        
        支持 YAML 和 JSON 格式的配置文件。对于 JSON 格式，会自动转换为
        新的 YAML 格式配置系统以保持向后兼容性。
        
        Args:
            config_file: 配置文件路径（YAML 或 JSON 格式）
            config_string: 配置字符串（YAML 格式）
            
        Returns:
            Config: 解析后的配置实例
            
        Raises:
            ConfigError: 配置解析错误或验证失败
        """
        if config_file and config_string:
            raise ConfigError("Only one of config_file or config_string should be provided")

        # Parse YAML config from file or string
        try:
            if config_file is not None:
                if config_file.endswith(".json"):
                    return cls.create_from_legacy_config(config_file=config_file)
                with open(config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
            elif config_string is not None:
                yaml_config = yaml.safe_load(config_string)
            else:
                raise ConfigError("No config file or config string provided")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}") from e

        config = cls()

        # Parse model providers
        model_providers = yaml_config.get("model_providers", None)
        if model_providers is not None and len(model_providers.keys()) > 0:
            config_model_providers: dict[str, ModelProvider] = {}
            for model_provider_name, model_provider_config in model_providers.items():
                config_model_providers[model_provider_name] = ModelProvider(**model_provider_config)
            config.model_providers = config_model_providers
        else:
            raise ConfigError("No model providers provided")

        # Parse models and populate model_provider fields
        models = yaml_config.get("models", None)
        if models is not None and len(models.keys()) > 0:
            config_models: dict[str, ModelConfig] = {}
            for model_name, model_config in models.items():
                if model_config["model_provider"] not in config_model_providers:
                    raise ConfigError(f"Model provider {model_config['model_provider']} not found")
                config_models[model_name] = ModelConfig(**model_config)
                config_models[model_name].model_provider = config_model_providers[
                    model_config["model_provider"]
                ]
            config.models = config_models
        else:
            raise ConfigError("No models provided")

        # Parse lakeview config
        lakeview = yaml_config.get("lakeview", None)
        if lakeview is not None:
            lakeview_model_name = lakeview.get("model", None)
            if lakeview_model_name is None:
                raise ConfigError("No model provided for lakeview")
            lakeview_model = config_models[lakeview_model_name]
            config.lakeview = LakeviewConfig(
                model=lakeview_model,
            )
        else:
            config.lakeview = None

        mcp_servers_config = {
            k: MCPServerConfig(**v) for k, v in yaml_config.get("mcp_servers", {}).items()
        }
        allow_mcp_servers = yaml_config.get("allow_mcp_servers", [])

        # Parse agents
        agents = yaml_config.get("agents", None)
        if agents is not None and len(agents.keys()) > 0:
            for agent_name, agent_config in agents.items():
                agent_model_name = agent_config.get("model", None)
                if agent_model_name is None:
                    raise ConfigError(f"No model provided for {agent_name}")
                try:
                    agent_model = config_models[agent_model_name]
                except KeyError as e:
                    raise ConfigError(f"Model {agent_model_name} not found") from e
                match agent_name:
                    case "trae_agent":
                        trae_agent_config = TraeAgentConfig(
                            **agent_config,
                            mcp_servers_config=mcp_servers_config,
                            allow_mcp_servers=allow_mcp_servers,
                        )
                        trae_agent_config.model = agent_model
                        if trae_agent_config.enable_lakeview and config.lakeview is None:
                            raise ConfigError("Lakeview is enabled but no lakeview config provided")
                        config.trae_agent = trae_agent_config
                    case _:
                        raise ConfigError(f"Unknown agent: {agent_name}")
        else:
            raise ConfigError("No agent configs provided")
        return config

    def resolve_config_values(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int | None = None,
    ):
        """
        解析和应用运行时配置值
        
        将 CLI 参数、环境变量等运行时配置值应用到配置对象中，
        遵循优先级规则：CLI > 环境变量 > 配置文件 > 默认值
        
        Args:
            provider: 指定的模型提供商名称
            model: 指定的模型名称
            model_base_url: 指定的模型 API 基础 URL
            api_key: 指定的 API 密钥
            max_steps: 指定的最大执行步数
            
        Returns:
            Config: 更新后的配置实例（支持链式调用）
        """
        if self.trae_agent:
            self.trae_agent.resolve_config_values(
                max_steps=max_steps,
            )
            self.trae_agent.model.resolve_config_values(
                model_providers=self.model_providers,
                provider=provider,
                model=model,
                model_base_url=model_base_url,
                api_key=api_key,
            )
        return self

    @classmethod
    def create_from_legacy_config(
        cls,
        *,
        legacy_config: LegacyConfig | None = None,
        config_file: str | None = None,
    ) -> "Config":
        """
        从旧版配置格式创建 Config 实例
        
        为了保持向后兼容性，支持从旧版 JSON 配置格式转换为新的
        YAML 配置格式。这个方法会自动映射旧版配置结构到新版。
        
        Args:
            legacy_config: 旧版配置对象
            config_file: 旧版配置文件路径
            
        Returns:
            Config: 转换后的新版配置实例
            
        Raises:
            ConfigError: 配置转换错误
        """
        if legacy_config and config_file:
            raise ConfigError("Only one of legacy_config or config_file should be provided")

        if config_file:
            legacy_config = LegacyConfig(config_file)
        elif not legacy_config:
            raise ConfigError("No legacy_config or config_file provided")

        model_provider = ModelProvider(
            api_key=legacy_config.model_providers[legacy_config.default_provider].api_key,
            base_url=legacy_config.model_providers[legacy_config.default_provider].base_url,
            api_version=legacy_config.model_providers[legacy_config.default_provider].api_version,
            provider=legacy_config.default_provider,
        )

        model_config = ModelConfig(
            model=legacy_config.model_providers[legacy_config.default_provider].model,
            model_provider=model_provider,
            max_tokens=legacy_config.model_providers[legacy_config.default_provider].max_tokens,
            temperature=legacy_config.model_providers[legacy_config.default_provider].temperature,
            top_p=legacy_config.model_providers[legacy_config.default_provider].top_p,
            top_k=legacy_config.model_providers[legacy_config.default_provider].top_k,
            parallel_tool_calls=legacy_config.model_providers[
                legacy_config.default_provider
            ].parallel_tool_calls,
            max_retries=legacy_config.model_providers[legacy_config.default_provider].max_retries,
            candidate_count=legacy_config.model_providers[
                legacy_config.default_provider
            ].candidate_count,
            stop_sequences=legacy_config.model_providers[
                legacy_config.default_provider
            ].stop_sequences,
        )
        mcp_servers_config = {
            k: MCPServerConfig(**vars(v)) for k, v in legacy_config.mcp_servers.items()
        }
        trae_agent_config = TraeAgentConfig(
            max_steps=legacy_config.max_steps,
            enable_lakeview=legacy_config.enable_lakeview,
            model=model_config,
            allow_mcp_servers=legacy_config.allow_mcp_servers,
            mcp_servers_config=mcp_servers_config,
        )

        if trae_agent_config.enable_lakeview:
            lakeview_config = LakeviewConfig(
                model=model_config,
            )
        else:
            lakeview_config = None

        return cls(
            trae_agent=trae_agent_config,
            lakeview=lakeview_config,
            model_providers={
                legacy_config.default_provider: model_provider,
            },
            models={
                "default_model": model_config,
            },
        )


def resolve_config_value(
    *,
    cli_value: int | str | float | None,
    config_value: int | str | float | None,
    env_var: str | None = None,
) -> int | str | float | None:
    """
    配置值解析函数
    
    按照优先级顺序解析配置值：CLI 参数 > 环境变量 > 配置文件 > 默认值
    这是 Trae Agent 配置系统的核心函数，确保用户可以通过多种方式
    灵活地覆盖配置值。
    
    Args:
        cli_value: 命令行参数提供的值（最高优先级）
        config_value: 配置文件中的值
        env_var: 环境变量名称（可选）
        
    Returns:
        解析后的配置值，如果所有来源都为空则返回 None
        
    Example:
        >>> resolve_config_value(
        ...     cli_value=None,
        ...     config_value="config_model",
        ...     env_var="OPENAI_MODEL"
        ... )
        # 如果环境变量 OPENAI_MODEL 存在，返回其值
        # 否则返回 "config_model"
    """
    # 优先级 1: CLI 参数（最高优先级）
    if cli_value is not None:
        return cli_value

    # 优先级 2: 环境变量
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    # 优先级 3: 配置文件值
    if config_value is not None:
        return config_value

    # 优先级 4: 默认值（None）
    return None
