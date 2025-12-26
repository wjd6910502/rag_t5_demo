"""
通用大模型接口文件 - 用于query生成和质量判断
支持阿里云通义千问等大模型API
使用qwen_agent方式调用（与app.py保持一致）
"""
import json
import os
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwen_agent.agents import Assistant

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入配置文件
try:
    import config
    DEFAULT_LLM_CONFIG = config.LLM_CONFIG
except ImportError:
    try:
        import config_example
        DEFAULT_LLM_CONFIG = config_example.LLM_CONFIG
    except ImportError:
        DEFAULT_LLM_CONFIG = {}


def count_chinese_chars(text: str) -> int:
    """计算中文字符数量（不包括标点符号和空格）"""
    import re
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return len(chinese_only)


class LLMInterface:
    """通用大模型接口（用于query生成和质量判断）"""
    
    def __init__(self,
                 api_url: str = None,
                 api_key: str = None,
                 model_name: str = "qwen-plus",
                 max_retries: int = 3):
        """
        初始化大模型接口
        
        Args:
            api_url: API地址（如果为None，使用默认的阿里云通义千问地址）
            api_key: API密钥
            model_name: 模型名称
            max_retries: 最大重试次数
        """
        # 配置LLM，使用与app.py相同的方式
        # 默认使用阿里云通义千问API
        # 优先级：参数 > 配置文件 > 环境变量 > 默认值
        model_server = api_url or DEFAULT_LLM_CONFIG.get("api_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = api_key or DEFAULT_LLM_CONFIG.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")
        
        # 基础LLM配置
        self.llm_cfg = {
            'model': model_name,
            'model_server': model_server,
            'api_key': api_key,
            # 配置生成参数，降低重复性
            'generate_cfg': {
                'temperature': 0.9,  # 提高温度增加多样性
                'top_p': 0.9,  # 使用nucleus sampling
                'repetition_penalty': 1.2,  # 增加重复惩罚，降低重复性
            }
        }
        
        # 创建Assistant实例（不带工具）
        self.bot = Assistant(llm=self.llm_cfg, files=[])
        self.max_retries = max_retries
        
    def _call_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        调用大模型API（使用qwen_agent方式）
        
        Args:
            prompt: 提示词
            temperature: 温度参数（暂未使用，qwen_agent会自动处理）
            max_tokens: 最大生成token数（暂未使用，qwen_agent会自动处理）
            
        Returns:
            生成的文本
        """
        for attempt in range(self.max_retries):
            try:
                # 使用Assistant.run()方式调用，与app.py保持一致
                response_content = ""
                messages = [{'role': 'user', 'content': prompt}]
                
                for response in self.bot.run(messages=messages):
                    # 处理响应对象
                    if isinstance(response, dict) and 'content' in response:
                        response_content += response['content']
                    elif isinstance(response, list):
                        for item in response:
                            if isinstance(item, dict) and 'content' in item:
                                response_content += item['content']
                
                if response_content:
                    return response_content.strip()
                else:
                    if attempt < self.max_retries - 1:
                        continue
                    return ""
                    
            except Exception as e:
                logger.warning(f"API调用异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    continue
                return ""
        
        return ""
    
    def generate_queries(self, original_query: str, prompt_template: str, num_queries: int = 3) -> List[str]:
        """
        根据原始query生成多个类似的query
        
        Args:
            original_query: 原始查询
            prompt_template: 提示词模板
            num_queries: 生成query的数量
            
        Returns:
            生成的query列表
        """
        prompt = prompt_template.format(query=original_query, num=num_queries)
        response = self._call_api(prompt, temperature=0.8, max_tokens=500)
        
        # 解析返回的多个query（假设以换行或特定分隔符分隔）
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        # 限制数量
        queries = queries[:num_queries]
        
        return queries if queries else [original_query]
    
    def batch_generate_queries(self, 
                               original_query: str, 
                               prompt_template: str, 
                               num_queries: int = 3,
                               max_workers: int = 3) -> List[str]:
        """
        并发生成多个query（通过多次调用）
        
        Args:
            original_query: 原始查询
            prompt_template: 提示词模板
            num_queries: 生成query的总数量
            max_workers: 最大并发数
            
        Returns:
            生成的query列表
        """
        # 每次调用生成一个query，并发多次调用
        def generate_one_query():
            prompt = prompt_template.format(query=original_query, num=1)
            response = self._call_api(prompt, temperature=0.8, max_tokens=200)
            query = response.strip()
            # 不进行长度过滤，直接返回生成的query
            if query:
                return query
            return None
        
        queries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_one_query) for _ in range(num_queries)]
            for future in as_completed(futures):
                try:
                    query = future.result()
                    if query and query not in queries:
                        queries.append(query)
                        if len(queries) >= num_queries:
                            break
                except Exception as e:
                    logger.warning(f"生成query失败: {e}")
        
        # 如果数量不够，添加原始query
        if len(queries) < num_queries:
            if original_query not in queries:
                queries.append(original_query)
        
        # 返回生成的query，不做截断
        return queries[:num_queries] if queries else [original_query]
    
    def evaluate_improvement(self, 
                            original_input: str,
                            original_output: str,
                            improved_output: str,
                            prompt_template: str) -> Dict[str, Any]:
        """
        评估改进效果
        
        Args:
            original_input: 原始输入
            original_output: 原始输出
            improved_output: 改进后的输出
            prompt_template: 评估提示词模板
            
        Returns:
            评估结果字典，包含是否改进、评分等
        """
        # 构建key-value格式的输入输出
        input_output_data = {
            "input": original_input,
            "original_output": original_output,
            "improved_output": improved_output
        }
        
        prompt = prompt_template.format(**input_output_data)
        response = self._call_api(prompt, temperature=0.3, max_tokens=500)
        
        # 解析评估结果（假设返回JSON格式或特定格式）
        try:
            # 尝试解析为JSON
            result = json.loads(response)
        except:
            # 如果不是JSON，尝试解析文本
            result = {
                "improved": "是" in response or "yes" in response.lower() or "true" in response.lower(),
                "score": self._extract_score(response),
                "reason": response
            }
        
        return result
    
    def _extract_score(self, text: str) -> float:
        """从文本中提取评分（0-1之间）"""
        import re
        # 尝试提取0-1之间的分数
        patterns = [
            r'评分[：:]\s*([0-9.]+)',
            r'score[：:]\s*([0-9.]+)',
            r'([0-9.]+)\s*分',
            r'([0-1]\.[0-9]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                if score > 1:
                    score = score / 100.0  # 如果是百分制，转换为0-1
                return min(max(score, 0.0), 1.0)
        return 0.5  # 默认分数

