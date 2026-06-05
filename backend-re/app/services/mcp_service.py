"""MCP tool loading for LangChain agents."""

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings


_mcp_client: MultiServerMCPClient | None = None
_amap_tools = None


def get_mcp_client() -> MultiServerMCPClient:
    """Return the singleton MCP client configured for Amap."""

    global _mcp_client

    if _mcp_client is None:
        settings = get_settings()
        _mcp_client = MultiServerMCPClient(
            {
                "amap": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["amap-mcp-server"],
                    "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
                }
            }
        )

    return _mcp_client


async def get_amap_tools():
    """Load Amap MCP tools once and reuse them."""

    global _amap_tools

    if _amap_tools is None:
        client = get_mcp_client()
        _amap_tools = await client.get_tools()
        print("✅ LangChain MCP高德工具初始化成功")
        print(f"   工具数量: {len(_amap_tools)}")
        for tool in _amap_tools[:8]:
            print(f"     - {tool.name}")
        if len(_amap_tools) > 8:
            print(f"     ... 还有 {len(_amap_tools) - 8} 个工具")

    return _amap_tools


def filter_tools(tools, *name_parts: str):
    """Return tools whose names match any exact name or suffix part."""

    selected = []
    for tool in tools:
        for part in name_parts:
            if tool.name == part or tool.name.endswith(part) or part in tool.name:
                selected.append(tool)
                break
    return selected
