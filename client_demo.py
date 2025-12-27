from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

CLIENT_CONFIG = (
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/home/app.e0022971/Agent/MCP/math_server_demo.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure your start your weather server on port 8000
            "url": "http://localhost:8009/mcp",
            "transport": "streamable_http",
        }
    }
)

llm = ChatOpenAI(
    max_retries=2,
    model="QWQ-32B",
    api_key="QWQ_32B_RTAwMjI5NzEmUVdRXzMyQiYyMDI1LzUvOA==",
    # temperature=0.7,
    base_url="http://t-llmserver.ai.cdtp.com/v1/",
    # streaming=True  # 开启流式输出
)

llm_test = ChatOpenAI(
    max_retries=2,
    model="DeepSeek-R1-fp8",
    api_key="sglang_DeepSeek_R1_fp8_test",
    # temperature=0.7,
    base_url="http://llmserver.ai.cxmt.com/v1/",
    # streaming=True  # 开启流式输出
)
temp = f"""
域名：http://llmserver.ai.cxmt.com 
model-name: DeepSeek-R1-fp8
密钥：sglang_DeepSeek_R1_fp8_test
"""


# llm = ChatOpenAI(
#         max_retries=2,
#         model="DeepSeek-V3-fp16",
#         api_key="DeepSeek_V3_fp16_RTAwMjI5NzEmRGVlcFNlZWtfVjNfZnAxNiYyMDI1LzUvMTk=",
#         #temperature=0.7,
#         base_url="http://t-llmserver.ai.cdtp.com/v1/",
#         #streaming=True  # 开启流式输出
#     )

async def main():
    # 初始化客户端
    client = MultiServerMCPClient(CLIENT_CONFIG)

    # 获取工具时需要异步等待
    tools = await client.get_tools()

    # 创建agent
    agent = create_react_agent(
        model=llm_test,
        tools=tools
    )

    # 执行数学查询
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "请帮忙计算 (3 + 5) x 12"}]}
    )
    print("Math Response:", math_response)

    # 执行天气查询
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "帮我查询nyc天气?"}]}
    )
    print("Weather Response:", weather_response)


if __name__ == "__main__":
    # 启动异步事件循环
    asyncio.run(main())