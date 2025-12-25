"""
RAG接口文件 - 与百度千帆知识库交互
"""
import requests
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class RAGInterface:
    """RAG知识库搜索接口"""
    
    def __init__(self, 
                 url: str = "https://qianfan.baidubce.com/v2/knowledgebases/search",
                 knowledgebase_ids: List[str] = None,
                 authorization: str = None,
                 top_k: int = 10,
                 enable_graph: bool = False,
                 enable_expansion: bool = False):
        """
        初始化RAG接口
        
        Args:
            url: 知识库搜索API地址
            knowledgebase_ids: 知识库ID列表
            authorization: 授权token
            top_k: 返回top_k个结果
            enable_graph: 是否启用图谱
            enable_expansion: 是否启用扩展
        """
        self.url = url
        self.knowledgebase_ids = knowledgebase_ids or ["8d5b089e-c511-4321-8c90-2507ab38676a"]
        self.authorization = authorization or 'Bearer bce-v3/ALTAK-1l8HQWyl8VcezSx4yLKDH/6317ab81f76d0284c2060e313334925906337472'
        self.top_k = top_k
        self.enable_graph = enable_graph
        self.enable_expansion = enable_expansion
        
    def search(self, query: str) -> Dict[str, Any]:
        """
        搜索知识库
        
        Args:
            query: 查询文本
            
        Returns:
            搜索结果字典
        """
        payload = json.dumps({
            "query": [
                {
                    "type": "text",
                    "text": query
                }
            ],
            "knowledgebase_ids": self.knowledgebase_ids,
            "enable_graph": self.enable_graph,
            "enable_expansion": self.enable_expansion,
            "top_k": self.top_k
        }, ensure_ascii=False)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.authorization
        }
        
        try:
            response = requests.post(
                self.url, 
                headers=headers, 
                data=payload.encode("utf-8"),
                timeout=30
            )
            response.encoding = "utf-8"
            return response.json()
        except Exception as e:
            print(f"RAG搜索错误: {e}")
            return {"error": str(e), "results": []}
    
    def batch_search(self, queries: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        并发搜索多个query
        
        Args:
            queries: 查询文本列表
            max_workers: 最大并发数
            
        Returns:
            搜索结果列表
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(self.search, query): query for query in queries}
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    result['query'] = query  # 保存原始query
                    results.append(result)
                except Exception as e:
                    print(f"查询 '{query}' 搜索失败: {e}")
                    results.append({"query": query, "error": str(e), "results": []})
        return results
    
    def extract_contexts(self, search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从搜索结果中提取上下文信息
        
        Args:
            search_result: 搜索结果
            
        Returns:
            上下文列表，每个元素包含content和score
        """
        contexts = []
        if "error" in search_result:
            return contexts
            
        # 根据实际API返回格式解析
        # 这里需要根据实际返回格式调整
        if "data" in search_result:
            for item in search_result.get("data", []):
                context = {
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "metadata": item.get("metadata", {})
                }
                contexts.append(context)
        elif "results" in search_result:
            for item in search_result.get("results", []):
                context = {
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "metadata": item.get("metadata", {})
                }
                contexts.append(context)
        
        return contexts

