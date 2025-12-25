"""
T5模型交互接口文件
使用本地HTTP服务接口
"""
import requests
from typing import List, Dict, Any, Optional


class T5ModelInterface:
    """T5模型交互接口 - 使用本地HTTP服务"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 max_len: int = 4096,
                 max_len_generate: int = 512,
                 task: str = "summarize",
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 no_repeat_ngram_size: int = 3,
                 early_stopping: bool = True):
        """
        初始化T5模型接口
        
        Args:
            base_url: T5服务地址
            max_len: 最大输入长度（保留用于兼容性，实际由服务端控制）
            max_len_generate: 最大生成长度
            task: 任务类型 (summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro)
            num_beams: beam search 数量
            length_penalty: 长度惩罚
            no_repeat_ngram_size: 不重复 n-gram 大小
            early_stopping: 是否早停
        """
        self.base_url = base_url.rstrip("/")
        self.max_len = max_len
        self.max_len_generate = max_len_generate
        self.task = task
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        
        # 健康检查
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            print(f"T5服务连接成功: {self.base_url}")
        except Exception as e:
            print(f"警告: T5服务连接失败 ({self.base_url}): {e}")
            print("请确保T5服务已启动")
    
    def generate(self, text: str, system_prompt: str = None) -> str:
        """
        生成文本
        
        Args:
            text: 输入文本
            system_prompt: 系统提示词（可选，会与text拼接）
            
        Returns:
            生成的文本
        """
        # 如果有系统提示词，拼接输入
        if system_prompt:
            input_text = f"{system_prompt}\n{text}"
        else:
            input_text = text
        
        # 构建请求
        url = f"{self.base_url}/generate"
        payload = {
            "text": input_text,
            "task": self.task,
            "max_length": self.max_len_generate,
            "min_length": max(1, self.max_len_generate // 10),  # 最小长度为最大长度的1/10
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "early_stopping": self.early_stopping
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("generated_text", "")
        except requests.exceptions.RequestException as e:
            print(f"T5生成请求失败: {e}")
            return ""
    
    def batch_generate(self, texts: List[str], system_prompt: str = None) -> List[str]:
        """
        批量生成文本
        
        Args:
            texts: 输入文本列表
            system_prompt: 系统提示词（可选）
            
        Returns:
            生成的文本列表
        """
        # 准备批量请求
        requests_list = []
        for text in texts:
            # 如果有系统提示词，拼接输入
            if system_prompt:
                input_text = f"{system_prompt}\n{text}"
            else:
                input_text = text
            
            requests_list.append({
                "text": input_text,
                "task": self.task,
                "max_length": self.max_len_generate,
                "min_length": max(1, self.max_len_generate // 10),
                "num_beams": self.num_beams,
                "length_penalty": self.length_penalty,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "early_stopping": self.early_stopping
            })
        
        # 发送批量请求
        url = f"{self.base_url}/batch_generate"
        try:
            response = requests.post(
                url,
                json=requests_list,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取生成结果
            results = []
            for item in result.get("results", []):
                results.append(item.get("generated_text", ""))
            
            return results
        except requests.exceptions.RequestException as e:
            print(f"T5批量生成请求失败: {e}")
            # 如果批量请求失败，回退到单个请求
            print("回退到单个请求模式...")
            results = []
            for text in texts:
                result = self.generate(text, system_prompt)
                results.append(result)
            return results

