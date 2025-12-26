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
        # RAG返回格式：{"chunks": [{"content": [{"type": "text", "text": "..."}], "rerank": {"score": ...}, ...}]}
        if "chunks" in search_result:
            for chunk in search_result.get("chunks", []):
                # 提取content数组中的文本内容
                content_texts = []
                content_array = chunk.get("content", [])
                if isinstance(content_array, list):
                    for content_item in content_array:
                        if isinstance(content_item, dict):
                            # 提取text字段
                            text = content_item.get("text", "")
                            if text:
                                content_texts.append(text)
                
                # 合并所有文本内容
                combined_content = "\n".join(content_texts) if content_texts else ""
                
                # 优先使用rerank.score，如果没有则使用recall.score
                rerank_score = chunk.get("rerank", {}).get("score", 0.0) if isinstance(chunk.get("rerank"), dict) else 0.0
                recall_score = chunk.get("recall", {}).get("score", 0.0) if isinstance(chunk.get("recall"), dict) else 0.0
                score = rerank_score if rerank_score > 0 else recall_score
                
                if combined_content:  # 只有当有内容时才添加
                    context = {
                        "content": combined_content,
                        "score": score,
                        "chunk_id": chunk.get("chunk_id", ""),
                        "metadata": chunk.get("meta", {})
                    }
                    contexts.append(context)
        elif "data" in search_result:
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

