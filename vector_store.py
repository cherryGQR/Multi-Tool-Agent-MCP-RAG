import asyncio
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import json


class ToolVectorStore:
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "mcp_tools"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 的维度

        # 连接Milvus
        self._connect()
        self._create_collection()

    def _connect(self):
        """连接Milvus数据库"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            print(f"成功连接到Milvus: {self.host}:{self.port}")
        except Exception as e:
            print(f"连接Milvus失败: {e}")
            raise

    def _create_collection(self):
        """创建集合（如果不存在）"""
        if not utility.has_collection(self.collection_name):
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="tool_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="server_name", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)  # 存储额外元数据
            ]

            schema = CollectionSchema(fields=fields, description="MCP工具向量数据库")
            self.collection = Collection(name=self.collection_name, schema=schema)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print(f"创建新的集合: {self.collection_name}")
        else:
            self.collection = Collection(self.collection_name)
            print(f"加载现有集合: {self.collection_name}")

        # 加载集合到内存
        self.collection.load()

    def _get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        if not text:
            return [0.0] * self.dimension
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def add_tool(self, tool_data: Dict[str, Any]):
        """添加工具到向量数据库"""
        try:
            # 准备数据
            tool_id = f"{tool_data['server_name']}_{tool_data['tool_name']}"
            description = tool_data.get('description', '')

            # 生成嵌入向量
            embedding = self._get_embedding(description)

            # 准备插入数据
            data = [
                [tool_id],  # id
                [tool_data['tool_name']],  # tool_name
                [description],  # description
                [tool_data['server_name']],  # server_name
                [embedding],  # embedding
                [json.dumps(tool_data.get('metadata', {}))]  # metadata
            ]

            # 插入数据
            result = self.collection.insert(data)
            print(f"成功添加工具: {tool_data['tool_name']} (ID: {tool_id})")
            return result

        except Exception as e:
            print(f"添加工具失败 {tool_data['tool_name']}: {e}")
            return None

    async def search_similar_tools(self, query: str, top_k: int = 3, score_threshold: float = 0.7) -> List[
        Dict[str, Any]]:
        """搜索相似工具"""
        try:
            # 生成查询向量
            query_embedding = self._get_embedding(query)

            # 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["tool_name", "description", "server_name", "metadata"]
            )

            # 处理结果
            similar_tools = []
            for hits in results:
                for hit in hits:
                    if hit.score < score_threshold:  # L2距离越小越好
                        continue

                    tool_info = {
                        "tool_name": hit.entity.get("tool_name"),
                        "description": hit.entity.get("description"),
                        "server_name": hit.entity.get("server_name"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "similarity_score": 1.0 / (1.0 + hit.score)  # 转换为相似度分数
                    }
                    similar_tools.append(tool_info)

            # 按相似度排序
            similar_tools.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_tools[:top_k]

        except Exception as e:
            print(f"搜索工具失败: {e}")
            return []

    def initialize_default_tools(self):
        """初始化默认工具数据"""
        default_tools = [
            {
                "tool_name": "add",
                "description": "Add two numbers together. Use this for mathematical addition operations.",
                "server_name": "math",
                "metadata": {
                    "category": "mathematics",
                    "parameters": ["a: int", "b: int"],
                    "examples": ["calculate 3 + 5", "add two numbers"]
                }
            },
            {
                "tool_name": "multiply",
                "description": "Multiply two numbers. Use this for mathematical multiplication operations.",
                "server_name": "math",
                "metadata": {
                    "category": "mathematics",
                    "parameters": ["a: int", "b: int"],
                    "examples": ["calculate 3 * 5", "multiply two numbers"]
                }
            },
            {
                "tool_name": "get_weather",
                "description": "Get weather information for a specific location. Use this for weather queries and forecasts.",
                "server_name": "weather",
                "metadata": {
                    "category": "weather",
                    "parameters": ["location: str"],
                    "examples": ["check weather in Beijing", "what's the weather like in Shanghai"]
                }
            }
        ]

        print("初始化默认工具数据...")
        for tool in default_tools:
            self.add_tool(tool)

        # 刷新集合
        self.collection.flush()
        print("默认工具数据初始化完成")