# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Trae Agent 的命令行界面模块。

本模块提供了 Trae Agent 的主要命令行接口，包括：
- run: 执行单个任务
- interactive: 启动交互式会话
- show_config: 显示配置信息
- tools: 显示可用工具列表
"""

# 标准库导入
import asyncio  # 异步编程支持
import os       # 操作系统接口
import sys      # 系统特定的参数和函数
import traceback  # 异常追踪
from pathlib import Path  # 面向对象的文件系统路径

# 第三方库导入
import click  # 命令行界面创建工具
from dotenv import load_dotenv  # 环境变量加载
from rich.console import Console  # 富文本控制台
from rich.panel import Panel      # 富文本面板组件
from rich.table import Table      # 富文本表格组件

# 项目内部导入
from trae_agent.agent import Agent  # 主代理类
from trae_agent.utils.cli import CLIConsole, ConsoleFactory, ConsoleMode, ConsoleType  # CLI 相关工具
from trae_agent.utils.config import Config, TraeAgentConfig  # 配置管理

# 加载环境变量（从 .env 文件）
_ = load_dotenv()

# 创建全局控制台实例，用于输出信息
console = Console()


def resolve_config_file(config_file: str) -> str:
    """
    解析配置文件路径，提供向后兼容性支持。
    
    首先尝试指定的文件，如果 YAML 文件不存在则回退到 JSON 文件。
    这样可以确保用户从旧版本升级时配置文件仍然可用。
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        str: 解析后的有效配置文件路径
        
    Raises:
        SystemExit: 当配置文件不存在时退出程序
    """
    # 检查是否为 YAML 格式的配置文件
    if config_file.endswith(".yaml") or config_file.endswith(".yml"):
        yaml_path = Path(config_file)
        # 生成对应的 JSON 文件路径作为备选
        json_path = Path(config_file.replace(".yaml", ".json").replace(".yml", ".json"))
        
        # 优先使用 YAML 文件
        if yaml_path.exists():
            return str(yaml_path)
        # 如果 YAML 不存在，尝试使用 JSON 文件（向后兼容）
        elif json_path.exists():
            console.print(f"[yellow]YAML config not found, using JSON config: {json_path}[/yellow]")
            return str(json_path)
        # 两个文件都不存在，报错退出
        else:
            console.print(
                "[red]Error: Config file not found. Please specify a valid config file in the command line option --config-file[/red]"
            )
            sys.exit(1)
    else:
        # 非 YAML 文件直接返回原路径
        return config_file


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Trae Agent - 基于大语言模型的软件工程任务智能代理。
    
    这是主命令组，包含了所有子命令的入口点。
    """
    pass


@cli.command()
@click.argument("task", required=False)
@click.option("--file", "-f", "file_path", help="Path to a file containing the task description.")
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option("--max-steps", help="Maximum number of execution steps", type=int)
@click.option("--working-dir", "-w", help="Working directory for the agent")
@click.option("--must-patch", "-mp", is_flag=True, help="Whether to patch the code")
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--trajectory-file", "-t", help="Path to save trajectory file")
@click.option("--patch-path", "-pp", help="Path to patch file")
@click.option(
    "--console-type",
    "-ct",
    default="simple",
    type=click.Choice(["simple", "rich"], case_sensitive=False),
    help="Type of console to use (simple or rich)",
)
@click.option(
    "--agent-type",
    "-at",
    type=click.Choice(["trae_agent"], case_sensitive=False),
    help="Type of agent to use (trae_agent)",
    default="trae_agent",
)
def run(
    task: str | None,
    file_path: str | None,
    patch_path: str,
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
    working_dir: str | None = None,
    must_patch: bool = False,
    config_file: str = "trae_config.yaml",
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str | None = "trae_agent",
):
    """
    执行单个任务的主函数。
    
    这是 Trae Agent 的核心功能，用于执行用户指定的软件工程任务。
    支持通过命令行参数或配置文件来设置各种选项。
    
    Args:
        task: 要执行的任务描述，与 file_path 二选一
        file_path: 包含任务描述的文件路径，与 task 二选一
        patch_path: 补丁文件的保存路径
        provider: LLM 提供商（如 openai, anthropic 等）
        model: 要使用的具体模型名称
        model_base_url: 模型 API 的基础 URL
        api_key: API 密钥
        max_steps: 最大执行步数限制
        working_dir: 代理的工作目录
        must_patch: 是否必须生成补丁文件
        config_file: 配置文件路径
        trajectory_file: 轨迹记录文件路径
        console_type: 控制台类型（simple 或 rich）
        agent_type: 代理类型（目前仅支持 trae_agent）
        
    Returns:
        None: 函数执行完成后程序结束
        
    Raises:
        SystemExit: 当参数错误或执行失败时退出程序
    """

    # 应用配置文件的向后兼容性处理
    config_file = resolve_config_file(config_file)

    # 处理任务输入：支持直接传入任务字符串或从文件读取
    if file_path:
        # 不能同时指定任务字符串和文件路径
        if task:
            console.print(
                "[red]Error: Cannot use both a task string and the --file argument.[/red]"
            )
            sys.exit(1)
        # 从文件读取任务描述
        try:
            task = Path(file_path).read_text()
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            sys.exit(1)
    elif not task:
        # 必须提供任务描述或文件路径之一
        console.print(
            "[red]Error: Must provide either a task string or use the --file argument.[/red]"
        )
        sys.exit(1)

    # 创建并解析配置
    # 首先从配置文件加载基础配置，然后用命令行参数覆盖
    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    # 验证代理类型参数
    if not agent_type:
        console.print("[red]Error: agent_type is required.[/red]")
        sys.exit(1)

    # 创建 CLI 控制台
    console_mode = ConsoleMode.RUN
    # 根据用户指定或推荐选择控制台类型
    if console_type:
        selected_console_type = (
            ConsoleType.SIMPLE if console_type.lower() == "simple" else ConsoleType.RICH
        )
    else:
        # 如果未指定，使用工厂方法获取推荐的控制台类型
        selected_console_type = ConsoleFactory.get_recommended_console_type(console_mode)

    # 创建控制台实例
    cli_console = ConsoleFactory.create_console(
        console_type=selected_console_type, mode=console_mode
    )

    # 对于 Rich 控制台，在运行模式下设置初始任务
    if selected_console_type == ConsoleType.RICH and hasattr(cli_console, "set_initial_task"):
        cli_console.set_initial_task(task)

    # 创建代理实例
    agent = Agent(agent_type, config, trajectory_file, cli_console)

    # 处理工作目录设置
    if working_dir:
        # 如果指定了工作目录，尝试切换到该目录
        try:
            os.chdir(working_dir)
            console.print(f"[blue]Changed working directory to: {working_dir}[/blue]")
        except Exception as e:
            console.print(f"[red]Error changing directory: {e}[/red]")
            sys.exit(1)
    else:
        # 如果未指定，使用当前工作目录
        working_dir = os.getcwd()

    # 确保工作目录是绝对路径
    if not Path(working_dir).is_absolute():
        console.print(
            f"[red]Working directory must be an absolute path: {working_dir}, it should start with `/`[/red]"
        )
        sys.exit(1)

    try:
        # 准备任务执行参数
        task_args = {
            "project_path": working_dir,  # 项目路径
            "issue": task,                # 任务描述
            "must_patch": "true" if must_patch else "false",  # 是否必须生成补丁
            "patch_path": patch_path,     # 补丁文件路径
        }

        # 为 Rich 控制台设置代理上下文（如果适用）
        if selected_console_type == ConsoleType.RICH and hasattr(cli_console, "set_agent_context"):
            cli_console.set_agent_context(agent, config.trae_agent, config_file, trajectory_file)

        # 代理将处理启动适当的控制台并执行任务
        _ = asyncio.run(agent.run(task, task_args))

        # 任务执行完成，显示轨迹文件保存位置
        console.print(f"\n[green]Trajectory saved to: {agent.trajectory_file}[/green]")

    except KeyboardInterrupt:
        # 用户中断执行
        console.print("\n[yellow]Task execution interrupted by user[/yellow]")
        console.print(f"[blue]Partial trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)
    except Exception as e:
        # 处理其他异常
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        console.print(traceback.format_exc())
        console.print(f"[blue]Trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)


@cli.command()
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--max-steps", help="Maximum number of execution steps", type=int, default=20)
@click.option("--trajectory-file", "-t", help="Path to save trajectory file")
@click.option(
    "--console-type",
    "-ct",
    type=click.Choice(["simple", "rich"], case_sensitive=False),
    help="Type of console to use (simple or rich)",
)
@click.option(
    "--agent-type",
    "-at",
    type=click.Choice(["trae_agent"], case_sensitive=False),
    help="Type of agent to use (trae_agent)",
    default="trae_agent",
)
def interactive(
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    config_file: str = "trae_config.yaml",
    max_steps: int | None = None,
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str | None = "trae_agent",
):
    """
    启动与 Trae Agent 的交互式会话。
    
    在交互模式下，用户可以连续输入多个任务，代理会逐一执行。
    支持两种控制台类型：简单控制台和富文本控制台。
    
    Args:
        provider: LLM 提供商
        model: 要使用的具体模型名称
        model_base_url: 模型 API 的基础 URL
        api_key: API 密钥
        config_file: 配置文件路径
        max_steps: 最大执行步数限制
        trajectory_file: 轨迹记录文件路径
        console_type: 控制台类型（simple 或 rich）
        agent_type: 代理类型
    """
    # 应用配置文件的向后兼容性处理
    config_file = resolve_config_file(config_file)

    # 创建并解析配置
    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    # 验证 trae_agent 配置是否存在
    if config.trae_agent:
        trae_agent_config = config.trae_agent
    else:
        console.print("[red]Error: trae_agent configuration is required in the config file.[/red]")
        sys.exit(1)

    # 为交互模式创建 CLI 控制台
    console_mode = ConsoleMode.INTERACTIVE
    # 根据用户指定或推荐选择控制台类型
    if console_type:
        selected_console_type = (
            ConsoleType.SIMPLE if console_type.lower() == "simple" else ConsoleType.RICH
        )
    else:
        # 获取交互模式下推荐的控制台类型
        selected_console_type = ConsoleFactory.get_recommended_console_type(console_mode)

    # 创建控制台实例，传入 lakeview 配置用于可能的可视化功能
    cli_console = ConsoleFactory.create_console(
        console_type=selected_console_type, lakeview_config=config.lakeview, mode=console_mode
    )

    # 验证代理类型参数
    if not agent_type:
        console.print("[red]Error: agent_type is required.[/red]")
        sys.exit(1)

    # 创建代理实例
    agent = Agent(agent_type, config, trajectory_file, cli_console)

    # 获取实际的轨迹文件路径（可能是自动生成的）
    trajectory_file = agent.trajectory_file

    # 根据控制台类型启动不同的交互循环
    if selected_console_type == ConsoleType.SIMPLE:
        # 对于简单控制台，使用传统的交互循环
        asyncio.run(
            _run_simple_interactive_loop(
                agent, cli_console, trae_agent_config, config_file, trajectory_file
            )
        )
    else:
        # 对于富文本控制台，启动处理交互的文本应用
        asyncio.run(
            _run_rich_interactive_loop(
                agent, cli_console, trae_agent_config, config_file, trajectory_file
            )
        )


async def _run_simple_interactive_loop(
    agent: Agent,
    cli_console: CLIConsole,
    trae_agent_config: TraeAgentConfig,
    config_file: str,
    trajectory_file: str | None,
):
    """
    运行简单控制台的交互循环。
    
    这个函数实现了传统的命令行交互模式，用户可以：
    - 输入任务描述来执行任务
    - 使用特殊命令如 'help', 'status', 'clear', 'exit' 等
    - 查看代理状态和配置信息
    
    Args:
        agent: 代理实例
        cli_console: CLI 控制台实例
        trae_agent_config: Trae Agent 配置
        config_file: 配置文件路径
        trajectory_file: 轨迹文件路径
    """
    # 主交互循环
    while True:
        try:
            # 获取用户输入的任务
            task = cli_console.get_task_input()
            if task is None:
                console.print("[green]Goodbye![/green]")
                break

            # 处理帮助命令
            if task.lower() == "help":
                console.print(
                    Panel(
                        """[bold]Available Commands:[/bold]

• Type any task description to execute it
• 'status' - Show agent status
• 'clear' - Clear the screen
• 'exit' or 'quit' - End the session""",
                        title="Help",
                        border_style="yellow",
                    )
                )
                continue

            # 获取工作目录输入
            working_dir = cli_console.get_working_dir_input()

            # 处理状态查询命令
            if task.lower() == "status":
                console.print(
                    Panel(
                        f"""[bold]Provider:[/bold] {agent.agent_config.model.model_provider.provider}
    [bold]Model:[/bold] {agent.agent_config.model.model}
    [bold]Available Tools:[/bold] {len(agent.agent.tools)}
    [bold]Config File:[/bold] {config_file}
    [bold]Working Directory:[/bold] {os.getcwd()}""",
                        title="Agent Status",
                        border_style="blue",
                    )
                )
                continue

            # 处理清屏命令
            if task.lower() == "clear":
                console.clear()
                continue

            # 为此任务设置轨迹记录
            console.print(f"[blue]Trajectory will be saved to: {trajectory_file}[/blue]")

            # 准备任务执行参数
            task_args = {
                "project_path": working_dir,  # 项目路径
                "issue": task,                # 任务描述
                "must_patch": "false",        # 交互模式下默认不强制生成补丁
            }

            # 执行任务
            console.print(f"\n[blue]Executing task: {task}[/blue]")

            # 启动控制台和执行任务（并发进行）
            console_task = asyncio.create_task(cli_console.start())
            execution_task = asyncio.create_task(agent.run(task, task_args))

            # 等待执行完成
            _ = await execution_task
            _ = await console_task

            # 显示轨迹保存信息
            console.print(f"\n[green]Trajectory saved to: {trajectory_file}[/green]")

        except KeyboardInterrupt:
            # 处理用户中断（Ctrl+C）
            console.print("\n[yellow]Use 'exit' or 'quit' to end the session[/yellow]")
        except EOFError:
            # 处理输入结束（Ctrl+D）
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            # 处理其他异常
            console.print(f"[red]Error: {e}[/red]")


async def _run_rich_interactive_loop(
    agent: Agent,
    cli_console: CLIConsole,
    trae_agent_config: TraeAgentConfig,
    config_file: str,
    trajectory_file: str | None,
):
    """
    运行富文本控制台的交互循环。
    
    这个函数为富文本控制台设置代理上下文，然后启动控制台 UI。
    富文本控制台提供更丰富的用户界面，包括实时状态显示、
    语法高亮、进度条等功能。
    
    Args:
        agent: 代理实例
        cli_console: CLI 控制台实例
        trae_agent_config: Trae Agent 配置
        config_file: 配置文件路径
        trajectory_file: 轨迹文件路径
    """
    # 在富文本控制台中设置代理，以便它可以处理任务执行
    if hasattr(cli_console, "set_agent_context"):
        cli_console.set_agent_context(agent, trae_agent_config, config_file, trajectory_file)

    # 启动控制台 UI - 这将处理整个交互过程
    await cli_console.start()


@cli.command()
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option("--max-steps", help="Maximum number of execution steps", type=int)
def show_config(
    config_file: str,
    provider: str | None,
    model: str | None,
    model_base_url: str | None,
    api_key: str | None,
    max_steps: int | None,
):
    """
    显示当前配置设置。
    
    这个命令用于查看和验证 Trae Agent 的配置信息，包括：
    - 通用设置（提供商、最大步数等）
    - 模型配置（模型名称、API 设置等）
    - 安全信息（API 密钥的部分显示）
    
    Args:
        config_file: 配置文件路径
        provider: LLM 提供商（命令行覆盖）
        model: 模型名称（命令行覆盖）
        model_base_url: 模型 API 基础 URL（命令行覆盖）
        api_key: API 密钥（命令行覆盖）
        max_steps: 最大步数（命令行覆盖）
    """
    # 应用配置文件的向后兼容性处理
    config_file = resolve_config_file(config_file)

    # 检查配置文件是否存在
    config_path = Path(config_file)
    if not config_path.exists():
        console.print(
            Panel(
                f"""[yellow]No configuration file found at: {config_file}[/yellow]

Using default settings and environment variables.""",
                title="Configuration Status",
                border_style="yellow",
            )
        )

    # 创建并解析配置，命令行参数会覆盖配置文件中的设置
    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    # 验证 trae_agent 配置是否存在
    if config.trae_agent:
        trae_agent_config = config.trae_agent
    else:
        console.print("[red]Error: trae_agent configuration is required in the config file.[/red]")
        sys.exit(1)

    # 显示通用设置
    general_table = Table(title="General Settings")
    general_table.add_column("Setting", style="cyan")
    general_table.add_column("Value", style="green")

    general_table.add_row(
        "Default Provider", str(trae_agent_config.model.model_provider.provider or "Not set")
    )
    general_table.add_row("Max Steps", str(trae_agent_config.max_steps or "Not set"))

    console.print(general_table)

    # 显示提供商特定设置
    provider_config = trae_agent_config.model.model_provider
    provider_table = Table(title=f"{provider_config.provider.title()} Configuration")
    provider_table.add_column("Setting", style="cyan")
    provider_table.add_column("Value", style="green")

    # 添加模型相关配置
    provider_table.add_row("Model", trae_agent_config.model.model or "Not set")
    provider_table.add_row("Base URL", provider_config.base_url or "Not set")
    provider_table.add_row("API Version", provider_config.api_version or "Not set")
    
    # API 密钥显示（出于安全考虑只显示前后几位）
    provider_table.add_row(
        "API Key",
        f"Set ({provider_config.api_key[:4]}...{provider_config.api_key[-4:]})"
        if provider_config.api_key
        else "Not set",
    )
    
    # 模型参数配置
    provider_table.add_row("Max Tokens", str(trae_agent_config.model.max_tokens))
    provider_table.add_row("Temperature", str(trae_agent_config.model.temperature))
    provider_table.add_row("Top P", str(trae_agent_config.model.top_p))

    # Anthropic 特有的 Top K 参数
    if trae_agent_config.model.model_provider.provider == "anthropic":
        provider_table.add_row("Top K", str(trae_agent_config.model.top_k))

    console.print(provider_table)


@cli.command()
def tools():
    """
    显示可用工具及其描述。
    
    这个命令列出所有注册的工具，包括工具名称和功能描述。
    工具是 Trae Agent 执行任务时可以调用的功能模块，
    如文件操作、代码编辑、命令执行等。
    """
    from .tools import tools_registry

    # 创建工具列表表格
    tools_table = Table(title="Available Tools")
    tools_table.add_column("Tool Name", style="cyan")
    tools_table.add_column("Description", style="green")

    # 遍历工具注册表，显示每个工具的信息
    for tool_name in tools_registry:
        try:
            # 尝试实例化工具并获取其信息
            tool = tools_registry[tool_name]()
            tools_table.add_row(tool.name, tool.description)
        except Exception as e:
            # 如果工具加载失败，显示错误信息
            tools_table.add_row(tool_name, f"[red]Error loading: {e}[/red]")

    console.print(tools_table)


def main():
    """
    CLI 的主入口点。
    
    这是程序的入口函数，当模块作为脚本运行时会调用此函数。
    它启动 Click 命令行界面，处理用户输入的命令和参数。
    """
    cli()


if __name__ == "__main__":
    # 当模块作为脚本直接运行时，调用主函数
    main()
