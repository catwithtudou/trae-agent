# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Base Agent class for LLM-based agents."""

import contextlib
from abc import ABC, abstractmethod

from trae_agent.agent.agent_basics import AgentExecution, AgentState, AgentStep, AgentStepState
from trae_agent.tools import tools_registry
from trae_agent.tools.base import Tool, ToolCall, ToolExecutor, ToolResult
from trae_agent.tools.ckg.ckg_database import clear_older_ckg
from trae_agent.utils.cli import CLIConsole
from trae_agent.utils.config import AgentConfig, ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.llm_clients.llm_client import LLMClient
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class BaseAgent(ABC):
    """
    LLM-based agent的抽象基类。
    这个类为所有具体的agent实现提供了一个通用的结构和核心逻辑。
    它封装了与LLM的交互、工具的执行、任务状态的管理以及执行轨迹的记录。
    """

    def __init__(self, agent_config: AgentConfig):
        """
        初始化agent。

        Args:
            agent_config: 包含模型参数和其他设置的配置对象。
        """
        # 初始化LLM客户端，用于与大语言模型进行通信
        self._llm_client = LLMClient(agent_config.model)
        self._model_config = agent_config.model
        # 设置agent执行任务的最大步数
        self._max_steps = agent_config.max_steps
        # 存储初始的对话消息，通常是系统提示
        self._initial_messages: list[LLMMessage] = []
        # 当前agent需要执行的任务描述
        self._task: str = ""
        # 根据配置加载agent可用的工具
        self._tools: list[Tool] = [
            tools_registry[tool_name](model_provider=self._model_config.model_provider.provider)
            for tool_name in agent_config.tools
        ]
        # 初始化工具执行器
        self._tool_caller: ToolExecutor = ToolExecutor([])
        # 用于在命令行界面显示agent状态的控制台
        self._cli_console: CLIConsole | None = None

        # 初始化轨迹记录器，用于记录agent的每一步操作
        self._trajectory_recorder: TrajectoryRecorder | None = None

        # CKG工具特定：清除旧的CKG数据库
        clear_older_ckg()

    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端实例。"""
        return self._llm_client

    @property
    def trajectory_recorder(self) -> TrajectoryRecorder | None:
        """获取此agent的轨迹记录器。"""
        return self._trajectory_recorder

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """
        设置此agent的轨迹记录器。
        同时也会为LLM客户端设置记录器，以确保LLM的交互也被记录。
        """
        self._trajectory_recorder = recorder
        # Also set it on the LLM client
        self._llm_client.set_trajectory_recorder(recorder)

    @property
    def cli_console(self) -> CLIConsole | None:
        """获取此agent的CLI控制台。"""
        return self._cli_console

    def set_cli_console(self, cli_console: CLIConsole | None) -> None:
        """设置此agent的CLI控制台。"""
        self._cli_console = cli_console

    @property
    def tools(self) -> list[Tool]:
        """获取此agent可用的工具列表。"""
        return self._tools

    @property
    def task(self) -> str:
        """获取agent的当前任务。"""
        return self._task

    @task.setter
    def task(self, value: str):
        """设置agent的当前任务。"""
        self._task = value

    @property
    def initial_messages(self) -> list[LLMMessage]:
        """获取agent的初始消息列表。"""
        return self._initial_messages

    @property
    def model_config(self) -> ModelConfig:
        """获取agent的模型配置。"""
        return self._model_config

    @property
    def max_steps(self) -> int:
        """获取agent执行任务的最大步数。"""
        return self._max_steps

    @abstractmethod
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """
        创建一个新任务。这是一个抽象方法，需要在子类中实现。
        子类需要根据任务描述和额外参数来设置agent的初始状态和消息。
        """
        pass

    async def execute_task(self) -> AgentExecution:
        """
        执行一个任务。这是agent的核心驱动循环（ReAct循环）。
        它会迭代执行“思考-行动”的步骤，直到任务完成、达到最大步数或发生错误。

        Returns:
            一个AgentExecution对象，包含了任务执行的完整结果和步骤。
        """
        import time

        start_time = time.time()
        # 初始化任务执行记录对象
        execution = AgentExecution(task=self._task, steps=[])
        step: AgentStep | None = None

        try:
            # 从初始消息开始
            messages = self._initial_messages
            step_number = 1

            # ReAct循环：在最大步数限制内持续执行
            while step_number <= self._max_steps:
                # 1. 初始化当前步骤
                step = AgentStep(step_number=step_number, state=AgentStepState.THINKING)
                try:
                    # 2. 运行LLM进行思考和规划，决定下一步行动
                    messages = await self._run_llm_step(step, messages, execution)
                    # 3. 完成并记录当前步骤
                    self._finalize_step(
                        step, messages, execution
                    )  # record trajectory for this step and update the CLI console
                    # 4. 检查任务是否已完成
                    if execution.agent_state == AgentState.COMPLETED:
                        break
                    step_number += 1
                except Exception as error:
                    # 如果步骤执行出错，记录错误并终止循环
                    execution.agent_state = AgentState.ERROR
                    step.state = AgentStepState.ERROR
                    step.error = str(error)
                    self._finalize_step(step, messages, execution)
                    break

            # 如果循环结束是因为超过了最大步数
            if step_number > self._max_steps and not execution.success:
                execution.final_result = "任务执行超过最大步数限制仍未完成。"

        except Exception as e:
            # 捕获整个任务执行过程中的意外异常
            execution.final_result = f"Agent执行失败: {str(e)}"

        execution.execution_time = time.time() - start_time

        # 清理任何可能存在的MCP（多模态内容协议）客户端
        with contextlib.suppress(Exception):
            await self.cleanup_mcp_clients()

        # 更新最终的CLI状态
        self._update_cli_console(step, execution)
        return execution

    async def _run_llm_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> list["LLMMessage"]:
        """
        执行单个LLM步骤，包括思考、调用工具或判断任务是否完成。
        这是ReAct模式中“Reason”和“Act”的结合点。

        Args:
            step: 当前的AgentStep对象。
            messages: 发送给LLM的消息历史。
            execution: 当前的AgentExecution对象。

        Returns:
            更新后的消息列表，用于下一次迭代。
        """
        # 状态：思考中
        step.state = AgentStepState.THINKING
        self._update_cli_console(step, execution)
        # 调用LLM获取响应（思考过程和行动计划）
        llm_response = self._llm_client.chat(messages, self._model_config, self._tools)
        step.llm_response = llm_response

        # 在CLI中显示LLM的响应
        self._update_cli_console(step, execution)

        # 更新token使用统计
        self._update_llm_usage(llm_response, execution)

        # 检查LLM是否认为任务已经完成
        if self.llm_indicates_task_completed(llm_response):
            # 如果任务确实完成了，更新状态并返回
            if self._is_task_completed(llm_response):
                execution.agent_state = AgentState.COMPLETED
                execution.final_result = llm_response.content
                execution.success = True
                return messages
            else:
                # 如果LLM误判了任务完成，则发送一条消息提示它继续
                execution.agent_state = AgentState.RUNNING
                return [LLMMessage(role="user", content=self.task_incomplete_message())]
        else:
            # 如果任务未完成，则处理工具调用（Act）
            tool_calls = llm_response.tool_calls
            return await self._tool_call_handler(tool_calls, step)

    def _finalize_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> None:
        """
        完成一个步骤的收尾工作，包括记录、更新状态和UI。

        Args:
            step: 已执行完毕的步骤。
            messages: 当前的消息历史。
            execution: 任务执行对象。
        """
        step.state = AgentStepState.COMPLETED
        # 记录这一步的轨迹
        self._record_handler(step, messages)
        # 更新CLI显示
        self._update_cli_console(step, execution)
        # 将完成的步骤添加到执行历史中
        execution.steps.append(step)

    def reflect_on_result(self, tool_results: list[ToolResult]) -> str | None:
        """
        对工具执行结果进行反思。这是ReAct+Reflect模式中的“Reflect”部分。
        如果工具执行失败，可以生成反思信息，指导LLM在下一步中纠正错误。
        子类可以重写此方法以实现更复杂的反思逻辑。

        Args:
            tool_results: 工具执行的结果列表。

        Returns:
            一个包含反思内容的字符串，或者在没有需要反思的情况下返回None。
        """
        if len(tool_results) == 0:
            return None

        # 默认的反思逻辑：如果工具执行失败，生成一条提示信息
        reflection = "\n".join(
            f"工具执行失败，错误信息: {tool_result.error}。请考虑尝试不同的方法或修正参数。"
            for tool_result in tool_results
            if not tool_result.success
        )

        return reflection

    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """
        检查LLM的响应是否表明任务已完成。
        子类可以重写此方法以适应不同模型的表达方式。

        Args:
            llm_response: LLM的响应。

        Returns:
            如果LLM表示任务完成，则为True。
        """
        completion_indicators = [
            "task completed",
            "task finished",
            "done",
            "completed successfully",
            "finished successfully",
        ]

        response_lower = llm_response.content.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    def _is_task_completed(self, llm_response: LLMResponse) -> bool:  # pyright: ignore[reportUnusedParameter]
        """
        根据LLM的响应，最终确认任务是否真的完成了。
        子类可以重写此方法以加入更严格的完成条件检查（例如，运行测试）。

        Args:
            llm_response: LLM的响应。

        Returns:
            如果任务确实完成，则为True。
        """
        return True

    def task_incomplete_message(self) -> str:
        """
        当LLM错误地认为任务已完成时，返回一条消息提示它继续工作。
        子类可以重写此方法以提供更具体的指令。
        """
        return "任务尚未完成，请继续。"

    @abstractmethod
    async def cleanup_mcp_clients(self) -> None:
        """
        清理MCP客户端。在使用MCP的子类中必须实现此方法。
        """
        pass

    def _update_cli_console(
        self, step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ) -> None:
        """如果存在CLI控制台，则更新其显示状态。"""
        if self.cli_console:
            self.cli_console.update_status(step, agent_execution)

    def _update_llm_usage(self, llm_response: LLMResponse, execution: AgentExecution):
        """更新LLM token的使用统计。"""
        if not llm_response.usage:
            return
        # if execution.total_tokens is None then set it to be llm_response.usage else sum it up
        # execution.total_tokens is not None
        if not execution.total_tokens:
            execution.total_tokens = llm_response.usage
        else:
            execution.total_tokens += llm_response.usage

    def _record_handler(self, step: AgentStep, messages: list[LLMMessage]) -> None:
        """如果存在轨迹记录器，则记录当前agent步骤的详细信息。"""
        if self.trajectory_recorder:
            self.trajectory_recorder.record_agent_step(
                step_number=step.step_number,
                state=step.state.value,
                llm_messages=messages,
                llm_response=step.llm_response,
                tool_calls=step.tool_calls,
                tool_results=step.tool_results,
                reflection=step.reflection,
                error=step.error,
            )

    async def _tool_call_handler(
        self, tool_calls: list[ToolCall] | None, step: AgentStep
    ) -> list[LLMMessage]:
        """
        处理LLM生成的工具调用请求。这是ReAct模式中的“Act”部分。

        Args:
            tool_calls: LLM请求调用的工具列表。
            step: 当前的AgentStep对象。

        Returns:
            一个消息列表，其中包含了工具执行的结果，将用于构建下一次LLM调用的上下文。
        """
        messages: list[LLMMessage] = []
        # 如果没有工具调用，说明LLM可能陷入了困境，返回一条提示信息
        if not tool_calls or len(tool_calls) <= 0:
            messages = [
                LLMMessage(
                    role="user",
                    content="看起来你没有完成任务，也没有调用任何工具。",
                )
            ]
            return messages

        # 状态：正在调用工具
        step.state = AgentStepState.CALLING_TOOL
        step.tool_calls = tool_calls
        self._update_cli_console(step)

        # 根据配置并行或串行执行工具
        if self._model_config.parallel_tool_calls:
            tool_results = await self._tool_caller.parallel_tool_call(tool_calls)
        else:
            tool_results = await self._tool_caller.sequential_tool_call(tool_calls)
        step.tool_results = tool_results
        self._update_cli_console(step)
        # 将工具执行结果封装成消息，添加到对话历史中
        for tool_result in tool_results:
            # Add tool result to conversation
            message = LLMMessage(role="user", tool_result=tool_result)
            messages.append(message)

        # 对工具结果进行反思
        reflection = self.reflect_on_result(tool_results)
        if reflection:
            step.state = AgentStepState.REFLECTING
            step.reflection = reflection

            # 显示反思内容
            self._update_cli_console(step)

            # 将反思内容也作为一条消息添加到历史中，让LLM看到
            messages.append(LLMMessage(role="assistant", content=reflection))

        return messages
