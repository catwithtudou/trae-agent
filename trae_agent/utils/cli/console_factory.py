# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""控制台工厂模块，用于创建不同类型的 CLI 控制台。

本模块实现了工厂模式，提供统一的接口来创建不同类型的控制台实例。
支持简单文本控制台和富文本 TUI 控制台的创建，并根据使用场景提供推荐的控制台类型。

主要组件：
- ConsoleFactory: 控制台工厂类，负责创建和推荐控制台实例

设计模式：
- 工厂模式：封装对象创建逻辑，提供统一的创建接口
- 策略模式：根据不同的模式和类型选择合适的控制台实现
"""

from trae_agent.utils.config import LakeviewConfig

from .cli_console import CLIConsole, ConsoleMode, ConsoleType
from .rich_console import RichCLIConsole
from .simple_console import SimpleCLIConsole


class ConsoleFactory:
    """CLI 控制台工厂类。
    
    使用工厂模式封装不同类型控制台的创建逻辑，提供统一的创建接口。
    支持根据控制台类型和运行模式创建相应的控制台实例，并提供智能推荐功能。
    
    设计优势：
    - 解耦客户端代码与具体控制台实现
    - 集中管理控制台创建逻辑
    - 提供类型安全的对象创建
    - 支持未来扩展新的控制台类型
    """

    @staticmethod
    def create_console(
        console_type: ConsoleType,
        mode: ConsoleMode = ConsoleMode.RUN,
        lakeview_config: LakeviewConfig | None = None,
    ) -> CLIConsole:
        """根据类型和模式创建控制台实例。
        
        工厂方法的核心实现，根据传入的控制台类型创建对应的实例。
        使用静态方法避免不必要的实例化，提高性能。

        Args:
            console_type: 要创建的控制台类型（SIMPLE 或 RICH）
            mode: 控制台操作模式（RUN 或 INTERACTIVE）
            lakeview_config: LakeView 配置对象，用于可视化功能（可选）

        Returns:
            CLIConsole 抽象基类的具体实现实例

        Raises:
            ValueError: 当 console_type 不被支持时抛出异常
            
        注意：
        - 简单控制台适合脚本化和批处理场景
        - 富文本控制台适合交互式和可视化场景
        """

        # 根据控制台类型创建对应的实例
        if console_type == ConsoleType.SIMPLE:
            # 创建简单文本控制台：轻量级，适合自动化场景
            return SimpleCLIConsole(mode=mode, lakeview_config=lakeview_config)
        elif console_type == ConsoleType.RICH:
            # 创建富文本 TUI 控制台：功能丰富，适合交互场景
            return RichCLIConsole(mode=mode, lakeview_config=lakeview_config)
        else:
            # 不支持的控制台类型，抛出异常
            raise ValueError(f"Unsupported console type: {console_type}")

    @staticmethod
    def get_recommended_console_type(mode: ConsoleMode) -> ConsoleType:
        """根据运行模式获取推荐的控制台类型。
        
        智能推荐功能，根据不同的使用场景提供最适合的控制台类型。
        这种设计让用户无需深入了解各种控制台的特性就能获得最佳体验。

        Args:
            mode: 控制台操作模式

        Returns:
            推荐的控制台类型
            
        推荐策略：
        - 交互模式：推荐富文本控制台，提供更好的用户体验
        - 运行模式：推荐简单控制台，减少资源消耗和复杂性
        """
        # 交互模式推荐富文本控制台：支持实时交互和美观界面
        if mode == ConsoleMode.INTERACTIVE:
            return ConsoleType.RICH
        # 运行模式推荐简单控制台：轻量级，适合自动化脚本
        else:
            return ConsoleType.SIMPLE
