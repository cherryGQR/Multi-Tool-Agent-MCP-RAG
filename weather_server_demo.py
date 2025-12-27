from mcp.server.fastmcp import FastMCP
import os  # 新增导入

# 通过环境变量设置服务参数（在创建FastMCP实例前设置）
os.environ["FASTMCP_HOST"] = "0.0.0.0"  # 允许外部访问
os.environ["FASTMCP_PORT"] = "8009"      # 指定端口
os.environ["FASTMCP_STREAMABLE_HTTP_PATH"] = "/mcp"  # 保持与客户端配置一致

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",  # 传输协议
    )