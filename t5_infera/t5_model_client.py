"""
T5 模型服务客户端
用于测试模型服务
"""

import requests
import json
import time
from typing import Optional, Dict, List


class T5ModelClient:
    """T5 模型服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: 服务端地址
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate(
        self,
        text: str,
        task: str = "summarize",
        max_length: int = 200,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True
    ) -> Dict:
        """
        生成文本
        
        Args:
            text: 输入文本
            task: 任务类型 (summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro)
            max_length: 最大生成长度
            min_length: 最小生成长度
            num_beams: beam search 数量
            length_penalty: 长度惩罚
            no_repeat_ngram_size: 不重复 n-gram 大小
            early_stopping: 是否早停
        
        Returns:
            生成结果
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "text": text,
            "task": task,
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": early_stopping
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            result["elapsed_time"] = f"{elapsed_time:.2f}s"
            return result
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def batch_generate(self, requests_list: List[Dict]) -> Dict:
        """
        批量生成文本
        
        Args:
            requests_list: 请求列表，每个请求包含 text 和 task 等参数
        
        Returns:
            批量生成结果
        """
        url = f"{self.base_url}/batch_generate"
        
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json=requests_list,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            elapsed_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            result["elapsed_time"] = f"{elapsed_time:.2f}s"
            return result
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def test_summarization():
    """测试摘要生成"""
    print("\n" + "="*60)
    print("测试摘要生成")
    print("="*60)
    
    client = T5ModelClient()
    
    # 健康检查
    health = client.health_check()
    print(f"健康检查: {json.dumps(health, indent=2, ensure_ascii=False)}")
    
    # 测试文本
    test_text = """
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, 
    and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
    During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
    man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York 
    City was finished in 1930. It was the first structure to reach a height of 300 metres. It is now 
    taller than the Chrysler Building in New York City by 5.2 metres (17 ft). Excluding transmitters, 
    the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
    
    print(f"\n输入文本: {test_text[:100]}...")
    
    result = client.generate(
        text=test_text,
        task="summarize",
        max_length=100,
        min_length=30,
        num_beams=4
    )
    
    print(f"\n生成结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def test_translation():
    """测试翻译"""
    print("\n" + "="*60)
    print("测试翻译 (English to German)")
    print("="*60)
    
    client = T5ModelClient()
    
    test_text = "The weather is nice today."
    
    print(f"输入文本: {test_text}")
    
    result = client.generate(
        text=test_text,
        task="translate_en_to_de",
        max_length=50,
        min_length=5,
        num_beams=4
    )
    
    print(f"\n生成结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def test_batch_generate():
    """测试批量生成"""
    print("\n" + "="*60)
    print("测试批量生成")
    print("="*60)
    
    client = T5ModelClient()
    
    requests_list = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "task": "summarize",
            "max_length": 50,
            "min_length": 10
        },
        {
            "text": "Hello, how are you?",
            "task": "translate_en_to_fr",
            "max_length": 50,
            "min_length": 5
        }
    ]
    
    print(f"批量请求数量: {len(requests_list)}")
    
    result = client.batch_generate(requests_list)
    
    print(f"\n批量生成结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    print("T5 模型服务客户端测试")
    print("="*60)
    
    # 测试摘要
    try:
        test_summarization()
    except Exception as e:
        print(f"摘要测试失败: {e}")
    
    # 测试翻译
    try:
        test_translation()
    except Exception as e:
        print(f"翻译测试失败: {e}")
    
    # 测试批量生成
    try:
        test_batch_generate()
    except Exception as e:
        print(f"批量生成测试失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()

