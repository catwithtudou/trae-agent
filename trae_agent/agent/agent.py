# Agent 模块 - 智能代理的统一入口和管理器
# 本模块实现了代理系统的门面模式(Facade Pattern)，为不同类型的智能代理提供统一的接口
# 主要功能：
# 1. 代理类型管理和工厂创建
# 2. 轨迹记录和任务执行管理
# 3. CLI 控制台集成和用户交互
# 4. MCP (Model Context Protocol) 工具初始化
# 5. 异步任务协调和资源清理

import asyncio
import contextlib
from enum import Enum

from trae_agent.utils.cli.cli_console import CLIConsole
from trae_agent.utils.config import AgentConfig, Config
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class AgentType(Enum):
    """代理类型枚举
    
    定义系统支持的智能代理类型，采用枚举模式确保类型安全
    当前支持的代理类型：
    - TraeAgent: 专门用于软件工程任务的代理，支持代码编辑、文件操作等
    
    设计优势：
    - 类型安全：防止无效的代理类型
    - 可扩展：便于添加新的代理类型
    - 文档化：明确系统支持的代理种类
    """
    TraeAgent = "trae_agent"


class Agent:
    """智能代理的统一管理器和门面类
    
    本类实现了门面模式(Facade Pattern)，为复杂的代理系统提供简化的统一接口
    主要职责：
    1. 代理实例的创建和配置管理
    2. 轨迹记录系统的初始化和管理
    3. CLI 控制台的集成和交互管理
    4. 异步任务的协调和执行
    5. 资源的生命周期管理和清理
    
    设计模式：
    - 门面模式：简化复杂子系统的接口
    - 工厂模式：根据类型创建不同的代理实例
    - 依赖注入：通过构造函数注入配置和依赖
    """
    
    def __init__(
        self,
        agent_type: AgentType | str,
        config: Config,
        trajectory_file: str | None = None,
        cli_console: CLIConsole | None = None,
    ):
        """初始化智能代理管理器
        
        Args:
            agent_type: 代理类型，支持枚举或字符串形式
            config: 系统配置对象，包含各种代理的配置信息
            trajectory_file: 轨迹记录文件路径，None 时自动生成
            cli_console: CLI 控制台实例，用于用户交互和状态显示
        
        设计说明：
        - 支持字符串到枚举的自动转换，提高易用性
        - 轨迹记录支持自动路径生成，简化使用
        - 采用依赖注入模式，便于测试和扩展
        """
        # 类型转换：支持字符串形式的代理类型输入
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)
        self.agent_type: AgentType = agent_type

        # 轨迹记录系统初始化
        # 轨迹记录用于追踪代理的执行过程，便于调试和分析
        if trajectory_file is not None:
            self.trajectory_file: str = trajectory_file
            self.trajectory_recorder: TrajectoryRecorder = TrajectoryRecorder(trajectory_file)
        else:
            # 自动生成轨迹文件路径，使用时间戳确保唯一性
            self.trajectory_recorder = TrajectoryRecorder()
            self.trajectory_file = self.trajectory_recorder.get_trajectory_path()

        # 代理实例创建 - 工厂模式实现
        # 根据代理类型创建相应的代理实例，支持不同类型的专业化代理
        match self.agent_type:
            case AgentType.TraeAgent:
                # TraeAgent 专门用于软件工程任务
                if config.trae_agent is None:
                    raise ValueError("trae_agent_config is required for TraeAgent")
                # 延迟导入，避免循环依赖
                from .trae_agent import TraeAgent

                self.agent_config: AgentConfig = config.trae_agent

                # 创建 TraeAgent 实例并配置 CLI 控制台
                self.agent: TraeAgent = TraeAgent(self.agent_config)
                self.agent.set_cli_console(cli_console)

        # CLI 控制台配置
        # 根据配置决定是否启用 LakeView 可视化功能
        if cli_console:
            if config.trae_agent.enable_lakeview:
                # 启用 LakeView 可视化，提供更丰富的执行状态展示
                cli_console.set_lakeview(config.lakeview)
            else:
                # 禁用 LakeView，使用简单的文本输出
                cli_console.set_lakeview(None)

        # 设置轨迹记录器，用于追踪代理执行过程
        self.agent.set_trajectory_recorder(self.trajectory_recorder)

    async def run(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """执行代理任务的主要方法
        
        这是代理系统的核心执行方法，协调各个组件完成复杂的任务执行流程
        
        Args:
            task: 要执行的任务描述
            extra_args: 额外的任务参数，如项目路径、问题描述等
            tool_names: 指定使用的工具名称列表，None 时使用默认工具
        
        Returns:
            AgentExecution: 任务执行结果，包含执行步骤、状态、结果等信息
        
        执行流程：
        1. 任务初始化和工具配置
        2. MCP 工具发现和初始化
        3. 任务详情展示
        4. 异步执行任务和 CLI 控制台
        5. 资源清理和结果返回
        """
        # 步骤1: 任务初始化
        # 创建新任务并配置工具，这是执行流程的起点
        self.agent.new_task(task, extra_args, tool_names)

        # 步骤2: MCP 工具初始化
        # MCP (Model Context Protocol) 允许代理动态发现和使用外部工具
        if self.agent.allow_mcp_servers:
            if self.agent.cli_console:
                self.agent.cli_console.print("Initialising MCP tools...")
            # 异步初始化 MCP 工具，可能涉及网络通信
            await self.agent.initialise_mcp()

        # 步骤3: 任务详情展示
        # 向用户展示任务配置信息，提供透明度和可追踪性
        if self.agent.cli_console:
            task_details = {
                "Task": task,  # 任务描述
                "Model Provider": self.agent_config.model.model_provider.provider,  # 模型提供商
                "Model": self.agent_config.model.model,  # 具体模型名称
                "Max Steps": str(self.agent_config.max_steps),  # 最大执行步数
                "Trajectory File": self.trajectory_file,  # 轨迹记录文件
                "Tools": ", ".join([tool.name for tool in self.agent.tools]),  # 可用工具列表
            }
            # 添加额外参数到任务详情中
            if extra_args:
                for key, value in extra_args.items():
                    task_details[key.capitalize()] = value
            # 在控制台显示任务详情
            self.agent.cli_console.print_task_details(task_details)

        # 步骤4: 异步任务协调
        # 同时启动 CLI 控制台和代理任务执行，实现并发处理
        cli_console_task = (
            asyncio.create_task(self.agent.cli_console.start()) if self.agent.cli_console else None
        )

        try:
            # 执行核心任务，这是代理的主要工作流程
            execution = await self.agent.execute_task()
        finally:
            # 步骤5: 资源清理
            # 确保 MCP 客户端被正确清理，即使任务执行失败也要执行清理
            # 使用 contextlib.suppress 避免清理过程中的异常影响主流程
            with contextlib.suppress(Exception):
                await self.agent.cleanup_mcp_clients()

        # 等待 CLI 控制台任务完成
        # 这确保了所有用户界面更新都已完成
        if cli_console_task:
            await cli_console_task

        # 返回执行结果，包含完整的执行轨迹和状态信息
        return execution
