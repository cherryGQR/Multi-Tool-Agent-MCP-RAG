import asyncio
from typing import List, Dict, Any
from vector_store import ToolVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient


class ToolManager:
    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530"):
        self.vector_store = ToolVectorStore(host=milvus_host, port=milvus_port)
        self.mcp_client = None
        self.available_servers = {}

    def set_mcp_client(self, client: MultiServerMCPClient):
        """设置MCP客户端"""
        self.mcp_client = client

    def set_available_servers(self, server_config: Dict):
        """设置可用的服务器配置"""
        self.available_servers = server_config

    async def get_relevant_tools(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """根据用户查询获取相关工具"""
        print(f"正在检索与查询相关的工具: '{user_query}'")

        # 从向量数据库检索相关工具
        relevant_tools = await self.vector_store.search_similar_tools(user_query, top_k=top_k)

        if not relevant_tools:
            print("未找到相关工具，返回空列表")
            return []

        print(f"找到 {len(relevant_tools)} 个相关工具:")
        for i, tool in enumerate(relevant_tools, 1):
            print(
                f"  {i}. {tool['tool_name']} (服务器: {tool['server_name']}) - 相似度: {tool['similarity_score']:.3f}")

        return relevant_tools

    async def get_filtered_mcp_tools(self, user_query: str) -> List:
        """获取经过筛选的MCP工具"""
        if not self.mcp_client:
            raise ValueError("MCP客户端未设置")

        # 获取相关工具信息
        relevant_tools = await self.get_relevant_tools(user_query)

        if not relevant_tools:
            print("未找到相关工具，返回空工具列表")
            return []

        # 获取所有可用工具
        all_tools = await self.mcp_client.get_tools()
        print(f"所有可用工具数量: {len(all_tools)}")

        # 根据检索结果筛选工具
        filtered_tools = []
        relevant_tool_names = {tool['tool_name'] for tool in relevant_tools}

        for tool in all_tools:
            if tool.name in relevant_tool_names:
                filtered_tools.append(tool)
                print(f"已选择工具: {tool.name}")

        print(f"最终筛选出的工具数量: {len(filtered_tools)}")
        return filtered_tools

    def initialize_tool_database(self):
        """初始化工具数据库"""
        self.vector_store.initialize_default_tools()


# 全局工具管理器实例
tool_manager = ToolManager()