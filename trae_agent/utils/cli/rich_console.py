# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""富文本 CLI 控制台实现模块，提供增强的 UI 功能。

本模块基于 Rich 库实现了功能丰富的终端用户界面，提供实时更新、
彩色输出、表格显示、进度指示等高级功能。

主要特点：
- 实时 UI 更新：使用 Rich Live 组件实现动态界面刷新
- 丰富的视觉效果：支持颜色、样式、表格、面板等多种显示元素
- 交互式体验：提供更好的用户交互和状态反馈
- 状态可视化：通过颜色和图标直观显示执行状态
- 异步支持：非阻塞的 UI 更新机制

适用场景：
- 交互式开发环境
- 需要实时状态监控的应用
- 用户友好的命令行工具
- 演示和调试场景

技术依赖：
- Rich 库：提供终端 UI 组件和渲染能力
- asyncio：支持异步 UI 更新任务
"""

import asyncio
import os
from typing import override

from rich.panel import Panel
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

from trae_agent.agent.agent_basics import AgentExecution, AgentStep, AgentStepState
from trae_agent.utils.cli.cli_console import (
    AGENT_STATE_INFO,
    CLIConsole,
    ConsoleMode,
    ConsoleStep,
    generate_agent_step_table,
)
from trae_agent.utils.config import LakeviewConfig


class TokenDisplay(Static):
    """实时显示令牌使用情况的组件。
    
    继承自 Textual 的 Static 组件，用于在界面上显示代理执行过程中的
    令牌消耗统计信息，包括总令牌数、输入令牌数和输出令牌数。
    
    功能特点：
    - 实时更新：响应式显示令牌使用变化
    - 格式化显示：使用千位分隔符提高可读性
    - 状态指示：通过颜色区分有无令牌使用
    - 详细统计：分别显示输入、输出和总计令牌数
    """

    total_tokens: reactive[int] = reactive(0)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)

    @override
    def render(self) -> Text:
        """渲染令牌显示内容。
        
        根据当前的令牌使用情况生成格式化的显示文本。
        如果有令牌使用则显示详细统计，否则显示默认状态。
        
        Returns:
            格式化的 Rich Text 对象，包含令牌使用信息
        """
        if self.total_tokens > 0:
            return Text(
                f"Tokens: {self.total_tokens:,} total | "
                + f"Input: {self.input_tokens:,} | "
                + f"Output: {self.output_tokens:,}",
                style="bold blue",
            )
        return Text("Tokens: 0 total", style="dim")

    def update_tokens(self, agent_execution: AgentExecution):
        """从代理执行对象更新令牌计数。
        
        提取代理执行过程中的令牌使用统计信息，并更新显示组件的
        响应式属性，触发界面自动刷新。
        
        Args:
            agent_execution: 包含令牌使用统计的代理执行对象
        """
        if agent_execution and agent_execution.total_tokens:
            self.input_tokens = agent_execution.total_tokens.input_tokens
            self.output_tokens = agent_execution.total_tokens.output_tokens
            self.total_tokens = self.input_tokens + self.output_tokens


class RichConsoleApp(App[None]):
    """富文本控制台的 Textual 应用程序。
    
    基于 Textual 框架构建的终端用户界面应用，提供交互式的代理执行环境。
    支持实时显示执行状态、处理用户输入、展示令牌使用等功能。
    
    界面布局：
    - 顶部：应用标题和时钟
    - 中部：代理执行日志显示区域
    - 底部：任务输入区域和令牌统计
    - 底栏：快捷键提示
    
    交互功能：
    - 任务输入和执行
    - 实时状态监控
    - 帮助和状态查询
    - 优雅的退出处理
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #execution_container {
        height: 1fr;
        border: solid $primary;
    }

    #input_container {
        height: auto;
        max-height: 5;
        border: solid $secondary;
    }

    #footer_container {
        height: 1;
        background: $background 50%;
    }

    RichLog {
        scrollbar-size: 1 1;
        scrollbar-size-horizontal: 1;
    }

    Input {
        height: 3;
    }

    .task_display {
        background: $surface;
        color: $text;
        padding: 1;
        height: auto;
        max-height: 3;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, console_impl: "RichCLIConsole"):
        """初始化富文本控制台应用。
        
        设置应用的基本属性和组件引用，准备用户界面的各个部分。
        
        Args:
            console_impl: 富文本控制台实现的引用，用于访问配置和代理
        """
        super().__init__()
        # 控制台实现的引用，用于访问代理和配置
        self.console_impl: "RichCLIConsole" = console_impl
        # UI 组件引用，在 compose 和 on_mount 中初始化
        self.execution_log: RichLog | None = None      # 执行日志显示组件
        self.task_input: Input | None = None           # 任务输入组件
        self.task_display: Static | None = None        # 任务显示组件
        self.token_display: TokenDisplay | None = None # 令牌统计显示组件
        # 应用状态
        self.current_task: str | None = None           # 当前执行的任务
        self.is_running_task: bool = False             # 是否正在执行任务

    @override
    def compose(self) -> ComposeResult:
        """构建用户界面布局。
        
        定义应用的整体布局结构，包括各个容器和组件的层次关系。
        根据控制台模式（交互式或运行模式）调整界面元素。
        
        Returns:
            生成器，产生界面组件的层次结构
        """
        yield Header(show_clock=True)

        # Top container for agent execution
        with Container(id="execution_container"):
            yield RichLog(id="execution_log", wrap=True, markup=True)

        # Bottom container for input/task display
        with Container(id="input_container"):
            if self.console_impl.mode == ConsoleMode.INTERACTIVE:
                yield Input(placeholder="Enter your task...", id="task_input")
                yield Static("", id="task_display", classes="task_display")
            else:
                yield Static("", id="task_display", classes="task_display")

        # Footer container for token usage
        with Container(id="footer_container"):
            yield TokenDisplay(id="token_display")

        yield Footer()

    def on_mount(self) -> None:
        """应用挂载时的回调函数。
        
        在应用启动后初始化各个组件的引用，设置焦点，
        并根据运行模式显示初始内容。
        """
        self.title = "Trae Agent CLI"

        self.execution_log = self.query_one("#execution_log", RichLog)
        self.token_display = self.query_one("#token_display", TokenDisplay)
        self.task_display = self.query_one("#task_display", Static)

        if self.console_impl.mode == ConsoleMode.INTERACTIVE:
            self.task_input = self.query_one("#task_input", Input)
            _ = self.task_input.focus()

        # Show initial task in RUN mode
        if self.console_impl.mode == ConsoleMode.RUN and self.console_impl.initial_task:
            self.task_display.update(
                Panel(self.console_impl.initial_task, title="Task", border_style="blue")
            )

    @on(Input.Submitted, "#task_input")
    def handle_task_input(self, event: Input.Submitted) -> None:
        """处理交互模式下的任务输入提交。
        
        响应用户在输入框中提交的内容，支持任务执行、内置命令处理等功能。
        包括帮助、清屏、状态查询、退出等命令的处理逻辑。
        
        Args:
            event: 输入提交事件，包含用户输入的内容
        """
        if self.is_running_task:
            return

        task = event.value.strip()
        if not task:
            return

        if task.lower() in ["exit", "quit"]:
            self.exit()
            return

        if task.lower() == "help":
            if self.execution_log:
                _ = self.execution_log.write(
                    Panel(
                        """[bold]Available Commands:[/bold]

• Type any task description to execute it
• 'status' - Show agent status
• 'clear' - Clear the execution log
• 'exit' or 'quit' - End the session""",
                        title="Help",
                        border_style="yellow",
                    )
                )
            event.input.value = ""
            return

        if task.lower() == "clear":
            if self.execution_log:
                _ = self.execution_log.clear()
            event.input.value = ""
            return

        if task.lower() == "status":
            if hasattr(self.console_impl, "agent") and self.console_impl.agent:
                agent_info = getattr(self.console_impl.agent, "agent_config", None)
                if agent_info and self.execution_log:
                    _ = self.execution_log.write(
                        Panel(
                            f"""[bold]Provider:[/bold] {agent_info.model.model_provider.provider}
[bold]Model:[/bold] {agent_info.model.model}
[bold]Working Directory:[/bold] {os.getcwd()}""",
                            title="Agent Status",
                            border_style="blue",
                        )
                    )
            else:
                if self.execution_log:
                    _ = self.execution_log.write("[yellow]Agent not initialized[/yellow]")
            event.input.value = ""
            return

        # Execute the task
        self.current_task = task
        if self.task_display:
            _ = self.task_display.update(Panel(task, title="Current Task", border_style="green"))
        event.input.value = ""
        self.is_running_task = True

        # Start task execution
        _ = asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: str):
        """使用代理执行任务。
        
        异步执行用户提交的任务，处理执行过程中的状态更新和错误处理。
        根据控制台模式决定执行完成后的行为（退出或继续等待输入）。
        
        Args:
            task: 要执行的任务描述字符串
        """
        try:
            if not hasattr(self.console_impl, "agent") or not self.console_impl.agent:
                if self.execution_log:
                    _ = self.execution_log.write("[red]Error: Agent not available[/red]")
                return

            # Get working directory
            working_dir = os.getcwd()
            if self.console_impl.mode == ConsoleMode.INTERACTIVE:
                # For interactive mode, we might want to ask for working directory
                # For now, use current directory
                pass

            task_args = {
                "project_path": working_dir,
                "issue": task,
                "must_patch": "false",
            }

            if self.execution_log:
                _ = self.execution_log.write(f"[blue]Executing task: {task}[/blue]")

            # Execute the task
            _ = await self.console_impl.agent.run(task, task_args)

            if self.execution_log:
                _ = self.execution_log.write("[green]Task completed successfully![/green]")

        except Exception as e:
            if self.execution_log:
                _ = self.execution_log.write(f"[red]Error executing task: {e}[/red]")
        finally:
            self.is_running_task = False
            if self.console_impl.mode == ConsoleMode.RUN:
                # In run mode, exit after task completion
                await asyncio.sleep(1)  # Brief pause to show completion
                _ = self.exit()
            else:
                # In interactive mode, clear task display and re-enable input
                if self.task_display:
                    _ = self.task_display.update("")
                if self.task_input:
                    _ = self.task_input.focus()

    def log_agent_step(self, agent_step: AgentStep):
        """将代理步骤记录到执行日志中。
        
        格式化代理执行步骤的信息，并以面板形式显示在执行日志中。
        包括步骤状态的颜色标识和详细信息表格。
        
        Args:
            agent_step: 要记录的代理执行步骤对象
        """
        color, _ = AGENT_STATE_INFO.get(agent_step.state, ("white", "❓"))

        # Create step display
        step_content = generate_agent_step_table(agent_step)

        if self.execution_log:
            _ = self.execution_log.write(
                Panel(step_content, title=f"Step {agent_step.step_number}", border_style=color)
            )

    async def action_quit(self) -> None:
        """退出应用程序。
        
        处理应用退出逻辑，设置控制台实现的退出标志，
        并优雅地关闭 Textual 应用。
        """
        self.console_impl.should_exit = True
        _ = self.exit()


class RichCLIConsole(CLIConsole):
    """使用 Textual 构建 TUI 界面的富文本 CLI 控制台。
    
    继承自 CLIConsole 抽象基类，使用 Textual 框架实现功能丰富的
    终端用户界面。提供交互式和运行两种模式，支持实时状态显示、
    任务执行、令牌统计等功能。
    
    核心特性：
    - TUI 界面：基于 Textual 的现代终端用户界面
    - 双模式支持：交互式模式和单次运行模式
    - 实时更新：动态显示执行状态和进度
    - 异步执行：非阻塞的任务执行和 UI 更新
    - 状态管理：完整的步骤历史和状态跟踪
    
    适用场景：
    - 开发和调试环境
    - 演示和展示
    - 交互式任务执行
    - 需要可视化反馈的自动化任务
    """

    def __init__(
        self, mode: ConsoleMode = ConsoleMode.RUN, lakeview_config: LakeviewConfig | None = None
    ):
        """初始化富文本 CLI 控制台。
        
        设置控制台的运行模式、配置参数和内部状态。
        准备 Textual 应用和相关的代理执行环境。
        
        Args:
            mode: 控制台运行模式（RUN 或 INTERACTIVE）
            lakeview_config: LakeView 可视化配置（可选）
        """
        super().__init__(mode, lakeview_config)
        self.app: RichConsoleApp | None = None
        self.should_exit: bool = False
        self.initial_task: str | None = None
        self._is_running: bool = False

        # Agent context for interactive mode
        self.agent = None
        self.trae_agent_config = None
        self.config_file = None
        self.trajectory_file = None

    @override
    async def start(self):
        """启动富文本控制台应用程序。
        
        创建并运行 Textual 应用，提供完整的 TUI 界面。
        防止重复启动，确保应用的生命周期管理正确。
        """
        # Prevent multiple starts of the same app
        if self._is_running:
            return

        self._is_running = True

        try:
            if self.app is None:
                self.app = RichConsoleApp(self)

            # Run the textual app
            await self.app.run_async()
        finally:
            self._is_running = False

    @override
    def update_status(
        self, agent_step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ):
        """更新控制台的代理状态显示。
        
        处理代理执行过程中的状态更新，包括步骤进度和令牌使用统计。
        维护步骤历史记录，并在适当时机触发 UI 更新。
        
        Args:
            agent_step: 代理执行步骤对象（可选）
            agent_execution: 代理执行对象（可选）
        """
        if agent_step and self.app:
            if agent_step.step_number not in self.console_step_history:
                # update step history
                self.console_step_history[agent_step.step_number] = ConsoleStep(agent_step)

            if (
                agent_step.state in [AgentStepState.COMPLETED, AgentStepState.ERROR]
                and not self.console_step_history[agent_step.step_number].agent_step_printed
            ):
                self.app.log_agent_step(agent_step)
                self.console_step_history[agent_step.step_number].agent_step_printed = True

        if agent_execution:
            self.agent_execution = agent_execution
            if self.app and self.app.token_display:
                self.app.token_display.update_tokens(agent_execution)

    @override
    def print_task_details(self, details: dict[str, str]):
        """打印初始任务配置详情。
        
        在执行日志中显示任务的配置信息，以面板形式呈现
        键值对格式的详细信息。
        
        Args:
            details: 包含任务配置信息的字典
        """
        if self.app and self.app.execution_log:
            content = "\n".join([f"[bold]{key}:[/bold] {value}" for key, value in details.items()])
            _ = self.app.execution_log.write(
                Panel(content, title="Task Details", border_style="blue")
            )

    @override
    def print(self, message: str, color: str = "blue", bold: bool = False):
        """向控制台打印消息。
        
        在执行日志中显示格式化的消息，支持颜色和样式设置。
        
        Args:
            message: 要打印的消息内容
            color: 消息颜色（默认蓝色）
            bold: 是否使用粗体样式（默认否）
        """
        if self.app and self.app.execution_log:
            formatted_message = f"[bold]{message}[/bold]" if bold else message
            formatted_message = f"[{color}]{formatted_message}[/{color}]"
            _ = self.app.execution_log.write(formatted_message)

    @override
    def get_task_input(self) -> str | None:
        """获取用户任务输入（交互模式）。
        
        在富文本控制台中，用户输入由 TUI 界面处理，
        此方法不直接使用。
        
        Returns:
            None，因为输入由 TUI 事件处理
        """
        # This method is not used in rich console as input is handled by the TUI
        return None

    @override
    def get_working_dir_input(self) -> str:
        """获取用户工作目录输入（交互模式）。
        
        目前返回当前工作目录，未来可以通过对话框增强。
        
        Returns:
            当前工作目录路径
        """
        # For now, return current directory. Could be enhanced with a dialog
        return os.getcwd()

    @override
    def stop(self):
        """停止控制台并清理资源。
        
        设置退出标志并关闭 Textual 应用，确保资源正确释放。
        """
        self.should_exit = True
        if self.app:
            _ = self.app.exit()

    def set_agent_context(self, agent, trae_agent_config, config_file, trajectory_file) -> None:
        """设置交互模式下任务执行的代理上下文。
        
        配置代理实例和相关的配置文件，为交互式任务执行做准备。
        
        Args:
            agent: 代理实例
            trae_agent_config: Trae 代理配置
            config_file: 配置文件路径
            trajectory_file: 轨迹文件路径
        """
        self.agent = agent
        self.trae_agent_config = trae_agent_config
        self.config_file = config_file
        self.trajectory_file = trajectory_file

    def set_initial_task(self, task: str):
        """设置运行模式的初始任务。
        
        在 RUN 模式下，设置要执行的任务内容，
        该任务将在应用启动后自动显示和执行。
        
        Args:
            task: 要执行的任务描述
        """
        self.initial_task = task
