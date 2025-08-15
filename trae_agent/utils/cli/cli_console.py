# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Trae Agent 的基础 CLI 控制台类模块。

本模块定义了 CLI 控制台的抽象基类和相关的数据结构，为不同类型的控制台实现
（如简单文本控制台、Rich 富文本控制台）提供统一的接口规范。

主要组件：
- ConsoleMode: 控制台运行模式枚举
- ConsoleType: 控制台类型枚举  
- ConsoleStep: 控制台步骤数据类
- CLIConsole: 抽象基类，定义控制台接口
- generate_agent_step_table: 生成代理步骤表格的工具函数
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from rich.panel import Panel
from rich.table import Table

from trae_agent.agent.agent_basics import AgentExecution, AgentStep, AgentStepState
from trae_agent.utils.config import LakeviewConfig
from trae_agent.utils.lake_view import LakeView


class ConsoleMode(Enum):
    """控制台操作模式枚举。
    
    定义了控制台的两种主要运行模式，用于区分不同的使用场景。
    """

    RUN = "run"  # 执行单个任务后退出模式
    INTERACTIVE = "interactive"  # 交互模式，可接受用户多次输入任务


class ConsoleType(Enum):
    """可用的控制台类型枚举。
    
    定义了不同的控制台实现类型，支持从简单文本到富文本界面的不同展示方式。
    """

    SIMPLE = "simple"  # 简单的基于文本的控制台
    RICH = "rich"  # 基于 Rich 库的富文本控制台，支持 TUI


# 代理状态信息映射表：将代理执行状态映射到对应的颜色和表情符号
# 这种设计模式将枚举值与视觉表示解耦，便于统一管理和修改
AGENT_STATE_INFO = {
    AgentStepState.THINKING: ("blue", "🤔"),      # 思考状态：蓝色 + 思考表情
    AgentStepState.CALLING_TOOL: ("yellow", "🔧"), # 调用工具状态：黄色 + 工具表情
    AgentStepState.REFLECTING: ("magenta", "💭"),  # 反思状态：洋红色 + 思考表情
    AgentStepState.COMPLETED: ("green", "✅"),     # 完成状态：绿色 + 完成表情
    AgentStepState.ERROR: ("red", "❌"),          # 错误状态：红色 + 错误表情
}


@dataclass
class ConsoleStep:
    """表示控制台步骤的数据类。
    
    封装了单个控制台步骤的所有相关信息，包括代理步骤数据、显示状态和异步任务。
    使用 dataclass 装饰器自动生成构造函数和其他魔术方法。
    
    Attributes:
        agent_step: 关联的代理执行步骤对象
        agent_step_printed: 标记该步骤是否已经打印到控制台
        lake_view_panel_generator: 异步任务，用于生成 LakeView 面板（可选）
    """

    agent_step: AgentStep  # 代理步骤对象，包含执行的详细信息
    agent_step_printed: bool = False  # 打印状态标记，避免重复显示
    lake_view_panel_generator: asyncio.Task[Panel | None] | None = None  # 异步面板生成任务


class CLIConsole(ABC):
    """CLI 控制台的抽象基类。
    
    使用抽象基类模式定义控制台的通用接口，强制子类实现核心方法。
    这种设计确保了不同控制台实现的一致性，同时允许灵活的定制化。
    
    设计模式：
    - 抽象基类模式：定义接口契约
    - 模板方法模式：提供通用逻辑，抽象化变化点
    - 策略模式：支持不同的控制台实现策略
    """

    def __init__(
        self, mode: ConsoleMode = ConsoleMode.RUN, lakeview_config: LakeviewConfig | None = None
    ):
        """初始化 CLI 控制台。

        Args:
            mode: 控制台操作模式（运行模式或交互模式）
            lakeview_config: LakeView 配置对象，用于可视化展示（可选）
        """
        self.mode: ConsoleMode = mode  # 设置控制台运行模式
        self.set_lakeview(lakeview_config)  # 配置 LakeView 可视化组件
        self.console_step_history: dict[int, ConsoleStep] = {}  # 步骤历史记录字典
        self.agent_execution: AgentExecution | None = None  # 当前代理执行对象

    @abstractmethod
    async def start(self):
        """启动控制台显示。
        
        抽象方法，子类必须实现具体的启动逻辑。
        使用 async 支持异步操作，适合需要等待用户输入或网络请求的场景。
        """
        pass

    @abstractmethod
    def update_status(
        self, agent_step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ):
        """更新控制台的代理状态显示。

        Args:
            agent_step: 当前代理步骤信息，包含执行状态和结果
            agent_execution: 完整的代理执行信息，包含所有步骤
        
        注意：两个参数都是可选的，允许灵活的状态更新方式。
        """
        pass

    @abstractmethod
    def print_task_details(self, details: dict[str, str]):
        """打印初始任务配置详情。
        
        用于在任务开始时显示配置信息，帮助用户了解当前任务的参数设置。
        
        Args:
            details: 任务详情字典，键值对形式的配置信息
        """
        pass

    @abstractmethod
    def print(self, message: str, color: str = "blue", bold: bool = False):
        """向控制台打印消息。
        
        提供统一的消息输出接口，支持颜色和样式设置。
        
        Args:
            message: 要打印的消息内容
            color: 文本颜色，默认为蓝色
            bold: 是否加粗显示，默认为 False
        """
        pass

    @abstractmethod
    def get_task_input(self) -> str | None:
        """获取用户输入的任务（用于交互模式）。

        Returns:
            用户输入的任务字符串，如果用户想要退出则返回 None
        
        注意：返回 None 表示用户希望退出交互模式。
        """
        pass

    @abstractmethod
    def get_working_dir_input(self) -> str:
        """获取用户输入的工作目录（用于交互模式）。

        Returns:
            工作目录路径字符串
        
        用于交互模式下让用户指定任务执行的工作目录。
        """
        pass

    @abstractmethod
    def stop(self):
        """停止控制台并清理资源。
        
        抽象方法，子类需要实现具体的清理逻辑，如：
        - 关闭文件句柄
        - 停止异步任务
        - 释放网络连接等
        """
        pass

    def set_lakeview(self, lakeview_config: LakeviewConfig | None = None):
        """设置控制台的 LakeView 配置。
        
        LakeView 是一个可视化组件，用于展示代理执行过程的图形化信息。
        使用空对象模式处理可选配置，避免空指针异常。
        
        Args:
            lakeview_config: LakeView 配置对象，为 None 时禁用 LakeView 功能
        """
        if lakeview_config:
            # 有配置时创建 LakeView 实例
            self.lake_view: LakeView | None = LakeView(lakeview_config)
        else:
            # 无配置时设置为 None，采用空对象模式
            self.lake_view = None


def generate_agent_step_table(agent_step: AgentStep) -> Table:
    """生成代理步骤的 Rich 表格显示。
    
    这是一个工具函数，将代理执行步骤转换为美观的表格格式。
    使用条件渲染模式，只显示存在的信息，避免空行干扰。
    
    Args:
        agent_step: 代理执行步骤对象
        
    Returns:
        Rich Table 对象，可直接用于控制台显示
        
    设计特点：
    - 状态可视化：使用颜色和 emoji 直观展示执行状态
    - 条件渲染：根据数据存在性动态添加表格行
    - 嵌套表格：工具调用信息使用子表格展示，层次清晰
    """
    # 从状态映射表获取对应的颜色和表情符号，提供默认值避免KeyError
    color, emoji = AGENT_STATE_INFO.get(agent_step.state, ("white", "❓"))

    # 创建主表格：无表头设计，固定宽度确保显示一致性
    table = Table(show_header=False, width=120)
    table.add_column("Step Number", style="cyan", width=15)  # 左列：标签列
    table.add_column(f"{agent_step.step_number}", style="green", width=105)  # 右列：内容列

    # 添加状态行：使用 Rich 标记语法设置颜色和样式
    table.add_row(
        "Status",
        f"[{color}]{emoji} Step {agent_step.step_number}: {agent_step.state.value.title()}[/{color}]",
    )

    # 条件渲染：只在有 LLM 响应时添加对应行
    if agent_step.llm_response and agent_step.llm_response.content:
        table.add_row("LLM Response", f"💬 {agent_step.llm_response.content}")

    # 条件渲染：只在有工具调用时添加工具信息
    if agent_step.tool_calls:
        # 生成工具名称列表，使用 Rich 标记语法高亮显示
        tool_names = [f"[cyan]{call.name}[/cyan]" for call in agent_step.tool_calls]
        table.add_row("Tools", f"🔧 {', '.join(tool_names)}")

        # 为每个工具调用创建详细的嵌套表格
        for tool_call in agent_step.tool_calls:
            # 创建工具调用子表格：显示参数和结果
            tool_call_table = Table(show_header=False, width=100)
            tool_call_table.add_column("Arguments", style="green", width=50)
            tool_call_table.add_column("Result", style="green", width=50)
            
            # 查找对应的工具执行结果
            tool_result_str = ""
            for tool_result in agent_step.tool_results or []:
                if tool_result.call_id == tool_call.call_id:
                    tool_result_str = tool_result.result or ""
                    break
            
            # 添加参数和结果到子表格
            tool_call_table.add_row(f"{tool_call.arguments}", f"{tool_result_str}")
            # 将子表格作为一行添加到主表格中
            table.add_row(tool_call.name, tool_call_table)

    # 条件渲染：只在有反思内容时添加反思行
    if agent_step.reflection:
        table.add_row("Reflection", f"💭 {agent_step.reflection}")

    # 条件渲染：只在有错误时添加错误行
    if agent_step.error:
        table.add_row("Error", f"❌ {agent_step.error}")

    return table
