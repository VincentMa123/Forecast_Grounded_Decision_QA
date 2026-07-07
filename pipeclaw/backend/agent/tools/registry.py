"""
工具注册器
用于管理所有可用的 Agent 工具，并生成 OpenAI function calling 所需的 schema
"""
from typing import Dict, Any, Callable, List, Optional
import inspect


class ToolRegistry:
    """工具注册器 - 管理所有 Agent 工具"""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: Dict[str, Any],
        returns: Optional[str] = None
    ):
        """
        注册一个工具

        Args:
            name: 工具名称
            fn: 工具函数
            description: 工具描述
            parameters: 参数定义（JSON Schema格式）
            returns: 返回值描述
        """
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "parameters": parameters,
            "returns": returns or "操作结果"
        }

    def get(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        tool = self._tools.get(name)
        return tool["fn"] if tool else None

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具完整信息"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册工具名称"""
        return list(self._tools.keys())

    def openai_tools_schema(self) -> List[Dict[str, Any]]:
        """
        生成 OpenAI function calling 所需的 tools schema

        Returns:
            符合 OpenAI API 格式的工具定义列表
        """
        tools = []
        for name, info in self._tools.items():
            tool_schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"]
                }
            }
            tools.append(tool_schema)
        return tools

    def execute(self, name: str, **kwargs) -> Any:
        """
        执行工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        fn = self.get(name)
        if not fn:
            raise ValueError(f"工具 '{name}' 未注册")

        try:
            return fn(**kwargs)
        except Exception as e:
            return {
                "error": str(e),
                "tool": name,
                "params": kwargs
            }


# 全局工具注册器实例
tool_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    returns: Optional[str] = None
):
    """
    装饰器：注册工具函数

    Usage:
        @register_tool(
            name="get_node_flow",
            description="获取节点流量数据",
            parameters={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "日期 (YYYY-MM-DD)"}
                },
                "required": ["date"]
            }
        )
        def get_node_flow(date: str):
            # 实现
            pass
    """
    def decorator(fn: Callable):
        tool_registry.register(
            name=name,
            fn=fn,
            description=description,
            parameters=parameters,
            returns=returns
        )
        return fn
    return decorator


def create_json_schema_from_params(
    properties: Dict[str, Dict[str, str]],
    required: List[str]
) -> Dict[str, Any]:
    """
    辅助函数：从简化的参数定义创建 JSON Schema

    Args:
        properties: 参数属性定义
        required: 必需参数列表

    Returns:
        JSON Schema 格式的参数定义
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }
