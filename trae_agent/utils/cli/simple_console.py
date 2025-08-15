# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""简单 CLI 控制台实现模块。

本模块提供了基于纯文本的轻量级控制台实现，适用于自动化脚本、批处理任务
和不需要复杂 UI 交互的场景。

主要特点：
- 轻量级：最小化资源消耗，启动快速
- 兼容性强：在各种终端环境下都能正常工作
- 简洁输出：纯文本格式，易于日志记录和解析
- 非阻塞：支持异步操作，不影响主程序流程

适用场景：
- CI/CD 管道中的自动化任务
- 服务器端批处理脚本
- 需要简洁输出的命令行工具
- 资源受限的环境
"""

import asyncio
from typing import override

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from trae_agent.agent.agent_basics import AgentExecution, AgentState, AgentStep, AgentStepState
from trae_agent.utils.cli.cli_console import (
    AGENT_STATE_INFO,
    CLIConsole,
    ConsoleMode,
    ConsoleStep,
    generate_agent_step_table,
)
from trae_agent.utils.config import LakeviewConfig


class SimpleCLIConsole(CLIConsole):
    """简单文本 CLI 控制台实现。

    继承自 CLIConsole 抽象基类，提供轻量级的纯文本控制台功能。
    专注于基本的文本输出，不包含复杂的格式化和交互功能。
    
    设计理念：
    - 最小化依赖：只使用 Python 标准库功能
    - 高性能：避免复杂的渲染逻辑，减少 CPU 和内存消耗
    - 广泛兼容：在任何支持文本输出的环境中都能工作
    - 易于调试：输出格式简单，便于问题排查
    
    适用场景：
    - 自动化脚本和批处理任务
    - 不需要富文本 UI 的环境
    - 资源受限的系统
    - 需要将输出重定向到文件的场景
    """

    def __init__(
        self, mode: ConsoleMode = ConsoleMode.RUN, lakeview_config: LakeviewConfig | None = None
    ):
        """初始化简单控制台。
        
        调用父类构造函数完成基础初始化，并设置简单控制台特有的属性。

        Args:
            mode: 控制台操作模式（RUN 或 INTERACTIVE）
            lakeview_config: LakeView 集成配置（可选，用于可视化功能）
            
        注意：
        - 简单控制台在交互模式下功能有限
        - LakeView 配置主要用于与可视化组件的集成
        """
        super().__init__(mode, lakeview_config)
        # Rich 控制台实例，用于格式化输出和表格显示
        self.console: Console = Console()

    @override
    def update_status(
        self, agent_step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ):
        """更新控制台状态显示。
        
        处理代理步骤和执行状态的更新，管理步骤历史记录，并在适当时机打印步骤信息。
        支持 LakeView 可视化功能的异步生成。
        
        Args:
            agent_step: 代理步骤对象（可选）
            agent_execution: 代理执行对象（可选）
            
        处理逻辑：
        1. 更新步骤历史记录
        2. 检查步骤是否完成或出错
        3. 打印步骤更新信息
        4. 异步生成 LakeView 面板（如果启用）
        """
        if agent_step:
            if agent_step.step_number not in self.console_step_history:
                # 更新步骤历史记录
                self.console_step_history[agent_step.step_number] = ConsoleStep(agent_step)

            if (
                agent_step.state in [AgentStepState.COMPLETED, AgentStepState.ERROR]
                and not self.console_step_history[agent_step.step_number].agent_step_printed
            ):
                # 打印步骤更新信息
                self._print_step_update(agent_step, agent_execution)
                self.console_step_history[agent_step.step_number].agent_step_printed = True

                # 如果启用了 LakeView，在后台生成 LakeView 面板
                if (
                    self.lake_view
                    and not self.console_step_history[
                        agent_step.step_number
                    ].lake_view_panel_generator
                ):
                    self.console_step_history[
                        agent_step.step_number
                    ].lake_view_panel_generator = asyncio.create_task(
                        self._create_lakeview_step_display(agent_step)
                    )

        # 更新代理执行状态
        self.agent_execution = agent_execution

    @override
    async def start(self):
        """启动控制台并等待执行完成。
        
        控制台启动后会持续监控代理执行状态，直到执行完成或出错。
        执行完成后会打印 LakeView 摘要（如果启用）和执行摘要。
        
        执行流程：
        1. 循环等待代理执行完成
        2. 打印 LakeView 摘要（如果启用）
        3. 打印执行摘要
        
        注意：
        - 使用异步睡眠避免阻塞其他任务
        - 支持完成和错误两种终止状态
        """
        # 等待代理执行完成或出错
        while self.agent_execution is None or (
            self.agent_execution.agent_state != AgentState.COMPLETED
            and self.agent_execution.agent_state != AgentState.ERROR
        ):
            await asyncio.sleep(1)  # 异步等待，避免阻塞

        # 如果启用了 LakeView，打印 LakeView 摘要
        if self.lake_view and self.agent_execution:
            await self._print_lakeview_summary()

        # 打印执行摘要
        if self.agent_execution:
            self._print_execution_summary()

    def _print_step_update(
        self, agent_step: AgentStep, agent_execution: AgentExecution | None = None
    ):
        """打印步骤更新信息。
        
        生成并显示包含步骤详细信息的表格，包括基本信息、令牌使用情况等。
        
        Args:
            agent_step: 代理步骤对象
            agent_execution: 代理执行对象（可选）
            
        显示内容：
        - 步骤基本信息（通过 generate_agent_step_table 生成）
        - 当前步骤的令牌使用情况
        - 总令牌使用情况（如果可用）
        """
        # 生成步骤基本信息表格
        table = generate_agent_step_table(agent_step)

        # 添加当前步骤的令牌使用信息
        if agent_step.llm_usage:
            table.add_row(
                "Token Usage",
                f"Input: {agent_step.llm_usage.input_tokens} Output: {agent_step.llm_usage.output_tokens}",
            )

        # 添加总令牌使用信息（如果执行对象存在且包含令牌信息）
        if agent_execution and agent_execution.total_tokens:
            table.add_row(
                "Total Tokens",
                f"Input: {agent_execution.total_tokens.input_tokens} Output: {agent_execution.total_tokens.output_tokens}",
            )

        # 使用 Rich 控制台打印表格
        self.console.print(table)

    async def _print_lakeview_summary(self):
        """打印所有已完成步骤的 LakeView 摘要。
        
        遍历所有步骤历史记录，等待 LakeView 面板生成完成，并显示摘要信息。
        使用格式化的标题和分隔线提高可读性。
        
        显示格式：
        - 标题和分隔线
        - 每个步骤的 LakeView 面板（如果生成成功）
        
        注意：
        - 异步等待面板生成完成
        - 只显示成功生成的面板
        """
        # 打印 LakeView 摘要标题
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]Lakeview Summary[/bold cyan]")
        self.console.print("=" * 60)

        # 遍历所有步骤历史记录
        for step in self.console_step_history.values():
            if step.lake_view_panel_generator:
                # 等待 LakeView 面板生成完成
                lake_view_panel = await step.lake_view_panel_generator
                if lake_view_panel:
                    # 打印生成的 LakeView 面板
                    self.console.print(lake_view_panel)

    def _print_execution_summary(self):
        """打印最终执行摘要。
        
        显示代理执行的完整摘要信息，包括任务描述、执行状态、步骤数量、
        执行时间、令牌使用情况和最终结果。
        
        摘要内容：
        - 任务描述（截断长文本）
        - 执行成功状态
        - 执行步骤数量
        - 总执行时间
        - 令牌使用统计
        - 最终结果（使用 Markdown 格式）
        
        注意：
        - 如果没有执行对象则直接返回
        - 根据执行成功状态调整显示样式
        """
        if not self.agent_execution:
            return

        # 打印执行摘要标题
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold green]Execution Summary[/bold green]")
        self.console.print("=" * 60)

        # 创建摘要表格
        table = Table(show_header=False, width=60)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=40)

        # 添加任务描述（长文本截断处理）
        table.add_row(
            "Task",
            self.agent_execution.task[:50] + "..."
            if len(self.agent_execution.task) > 50
            else self.agent_execution.task,
        )
        # 添加执行成功状态
        table.add_row("Success", "✅ Yes" if self.agent_execution.success else "❌ No")
        # 添加步骤数量
        table.add_row("Steps", str(len(self.agent_execution.steps)))
        # 添加执行时间
        table.add_row("Execution Time", f"{self.agent_execution.execution_time:.2f}s")

        # 添加令牌使用统计（如果可用）
        if self.agent_execution.total_tokens:
            total_tokens = (
                self.agent_execution.total_tokens.input_tokens
                + self.agent_execution.total_tokens.output_tokens
            )
            table.add_row("Total Tokens", str(total_tokens))
            table.add_row("Input Tokens", str(self.agent_execution.total_tokens.input_tokens))
            table.add_row("Output Tokens", str(self.agent_execution.total_tokens.output_tokens))

        # 打印摘要表格
        self.console.print(table)

        # 显示最终结果（如果存在）
        if self.agent_execution.final_result:
            self.console.print(
                Panel(
                    Markdown(self.agent_execution.final_result),
                    title="Final Result",
                    # 根据执行成功状态设置边框颜色
                    border_style="green" if self.agent_execution.success else "red",
                )
            )

    @override
    def print_task_details(self, details: dict[str, str]):
        """打印初始任务配置详情。
        
        将任务配置信息格式化为面板显示，提供清晰的任务概览。
        
        Args:
            details: 包含任务详情的字典
            
        显示格式：
        - 使用 Rich Panel 组件
        - 蓝色边框样式
        - 键值对格式，键名加粗显示
        
        注意：
        - 自动去除末尾换行符
        - 支持 Rich 标记语法
        """
        # 构建可渲染的内容字符串
        renderable = ""
        for key, value in details.items():
            # 键名加粗，值正常显示
            renderable += f"[bold]{key}:[/bold] {value}\n"
        # 去除末尾的换行符
        renderable = renderable.strip()
        
        # 使用 Panel 组件显示任务详情
        self.console.print(
            Panel(
                renderable,
                title="Task Details",
                border_style="blue",  # 蓝色边框
            )
        )

    @override
    def print(self, message: str, color: str = "blue", bold: bool = False):
        """向控制台打印消息。
        
        支持颜色和粗体格式化的消息打印功能。
        
        Args:
            message: 要打印的消息内容
            color: 文本颜色（默认蓝色）
            bold: 是否加粗显示（默认否）
            
        格式化处理：
        1. 根据 bold 参数决定是否加粗
        2. 根据 color 参数设置文本颜色
        3. 使用 Rich 控制台进行格式化输出
        
        注意：
        - 使用 Rich 标记语法进行格式化
        - 支持所有 Rich 支持的颜色名称
        """
        # 根据 bold 参数决定是否加粗
        message = f"[bold]{message}[/bold]" if bold else message
        # 根据 color 参数设置颜色
        message = f"[{color}]{message}[/{color}]"
        # 使用 Rich 控制台打印格式化消息
        self.console.print(message)

    @override
    def get_task_input(self) -> str | None:
        """获取用户任务输入（用于交互模式）。
        
        在交互模式下提示用户输入任务描述，支持退出命令。
        
        Returns:
            用户输入的任务字符串，如果用户选择退出或发生异常则返回 None
            
        交互逻辑：
        - 仅在交互模式下工作
        - 显示格式化的任务输入提示
        - 支持 "exit" 和 "quit" 退出命令
        - 处理用户中断异常
        
        注意：
        - 非交互模式直接返回 None
        - 异常处理确保程序稳定性
        """
        # 非交互模式直接返回 None
        if self.mode != ConsoleMode.INTERACTIVE:
            return None

        # 显示任务输入提示
        self.console.print("\n[bold blue]Task:[/bold blue] ", end="")
        try:
            # 获取用户输入
            task = input()
            # 检查退出命令
            if task.lower() in ["exit", "quit"]:
                return None
            return task
        except (EOFError, KeyboardInterrupt):
            # 处理用户中断异常
            return None

    @override
    def get_working_dir_input(self) -> str:
        """获取用户工作目录输入（用于交互模式）。
        
        在交互模式下提示用户输入工作目录路径。
        
        Returns:
            用户输入的工作目录路径，如果发生异常或非交互模式则返回空字符串
            
        交互逻辑：
        - 仅在交互模式下显示提示
        - 显示格式化的目录输入提示
        - 处理用户中断异常
        
        注意：
        - 非交互模式返回空字符串
        - 异常情况下返回空字符串作为默认值
        """
        # 非交互模式返回空字符串
        if self.mode != ConsoleMode.INTERACTIVE:
            return ""

        # 显示工作目录输入提示
        self.console.print("[bold blue]Working Directory:[/bold blue] ", end="")
        try:
            # 获取用户输入的工作目录
            return input()
        except (EOFError, KeyboardInterrupt):
            # 异常情况下返回空字符串
            return ""

    @override
    def stop(self):
        """停止控制台并清理资源。
        
        简单控制台的停止操作，由于使用标准输出，不需要特殊的清理工作。
        
        注意：
        - 简单控制台不需要显式清理
        - 保留此方法以保持接口一致性
        - 未来可能添加清理逻辑
        """
        # 简单控制台不需要显式清理资源
        # 保留此方法以保持与父类接口的一致性
        pass

    async def _create_lakeview_step_display(self, agent_step: AgentStep) -> Panel | None:
        """为步骤创建 LakeView 显示面板。
        
        异步生成包含步骤可视化信息的 Rich Panel，用于 LakeView 摘要显示。
        
        Args:
            agent_step: 代理步骤对象
            
        Returns:
            格式化的 Panel 对象，如果 LakeView 未启用或生成失败则返回 None
            
        生成过程：
        1. 检查 LakeView 是否可用
        2. 调用 LakeView 创建步骤信息
        3. 根据步骤状态设置面板样式
        4. 构建包含 emoji 和描述的面板
        
        注意：
        - 异步操作，不阻塞主流程
        - 面板宽度固定为 80 字符
        - 根据步骤状态动态设置边框颜色
        """
        # 检查 LakeView 是否可用
        if self.lake_view is None:
            return None

        # 异步创建 LakeView 步骤信息
        lake_view_step = await self.lake_view.create_lakeview_step(agent_step)

        # 检查步骤信息是否生成成功
        if lake_view_step is None:
            return None

        # 根据步骤状态获取显示颜色
        color, _ = AGENT_STATE_INFO.get(agent_step.state, ("white", "❓"))

        # 创建并返回格式化的面板
        return Panel(
            f"""[{lake_view_step.tags_emoji}] The agent [bold]{lake_view_step.desc_task}[/bold]
{lake_view_step.desc_details}""",
            title=f"Step {agent_step.step_number} (Lakeview)",
            border_style=color,  # 根据步骤状态设置边框颜色
            width=80,            # 固定面板宽度
        )
