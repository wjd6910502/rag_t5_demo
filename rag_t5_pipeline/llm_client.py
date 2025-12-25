"""
大模型客户端 - 基于 LLMInterface 的示例客户端
用于演示如何调用大模型进行 query 生成和质量评估
"""
import argparse
import json
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from llm_interface import LLMInterface
from prompts import QUERY_GENERATION_PROMPT, QUALITY_EVALUATION_PROMPT

# 尝试导入配置
try:
    import config
    LLM_CONFIG = config.LLM_CONFIG
except ImportError:
    import config_example
    LLM_CONFIG = config_example.LLM_CONFIG


class LLMClient:
    """大模型客户端封装类"""
    
    def __init__(self, api_url=None, api_key=None, model_name=None, max_retries=3):
        """
        初始化客户端
        
        Args:
            api_url: API地址（可选，从配置读取）
            api_key: API密钥（可选，从配置读取）
            model_name: 模型名称（可选，从配置读取）
            max_retries: 最大重试次数
        """
        # 从配置或参数获取值
        api_url = api_url or LLM_CONFIG.get("api_url")
        api_key = api_key or LLM_CONFIG.get("api_key")
        model_name = model_name or LLM_CONFIG.get("model_name", "qwen-plus")
        
        if not api_key:
            print("警告: API密钥未配置，请在config.py中设置LLM_CONFIG['api_key']")
        
        self.llm = LLMInterface(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            max_retries=max_retries
        )
    
    def generate_queries(self, original_query, num_queries=3, use_batch=False):
        """
        生成多个类似的查询
        
        Args:
            original_query: 原始查询
            num_queries: 生成查询的数量
            use_batch: 是否使用批量生成（并发）
        
        Returns:
            生成的查询列表
        """
        print(f"\n正在生成 {num_queries} 个类似查询...")
        print(f"原始查询: {original_query}")
        
        if use_batch:
            queries = self.llm.batch_generate_queries(
                original_query=original_query,
                prompt_template=QUERY_GENERATION_PROMPT,
                num_queries=num_queries
            )
        else:
            queries = self.llm.generate_queries(
                original_query=original_query,
                prompt_template=QUERY_GENERATION_PROMPT,
                num_queries=num_queries
            )
        
        print(f"\n生成了 {len(queries)} 个查询:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        
        return queries
    
    def evaluate_quality(self, original_input, original_output, improved_output):
        """
        评估改进质量
        
        Args:
            original_input: 原始输入
            original_output: 原始输出
            improved_output: 改进后的输出
        
        Returns:
            评估结果字典
        """
        print("\n正在评估质量改进...")
        print(f"原始输入: {original_input[:100]}...")
        print(f"原始输出: {original_output[:100]}...")
        print(f"改进输出: {improved_output[:100]}...")
        
        result = self.llm.evaluate_improvement(
            original_input=original_input,
            original_output=original_output,
            improved_output=improved_output,
            prompt_template=QUALITY_EVALUATION_PROMPT
        )
        
        print("\n评估结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return result
    
    def chat(self, prompt, temperature=0.7, max_tokens=2000):
        """
        简单的对话接口
        
        Args:
            prompt: 用户输入
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            模型回复
        """
        print(f"\n用户: {prompt}")
        print("\n模型回复:")
        
        response = self.llm._call_api(prompt, temperature=temperature, max_tokens=max_tokens)
        print(response)
        
        return response


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='大模型客户端 - 基于LLMInterface')
    
    # 配置参数
    parser.add_argument('--api_url', type=str, default=None,
                       help='API地址（可选，优先使用config中的配置）')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API密钥（可选，优先使用config中的配置）')
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称（可选，默认从config读取）')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数（默认3）')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 生成查询命令
    query_parser = subparsers.add_parser('generate', help='生成类似查询')
    query_parser.add_argument('query', type=str, help='原始查询')
    query_parser.add_argument('--num', type=int, default=3, help='生成数量（默认3）')
    query_parser.add_argument('--batch', action='store_true', help='使用批量生成（并发）')
    
    # 评估质量命令
    eval_parser = subparsers.add_parser('evaluate', help='评估质量改进')
    eval_parser.add_argument('--input', type=str, required=True, help='原始输入')
    eval_parser.add_argument('--original', type=str, required=True, help='原始输出')
    eval_parser.add_argument('--improved', type=str, required=True, help='改进后的输出')
    
    # 对话命令
    chat_parser = subparsers.add_parser('chat', help='简单对话')
    chat_parser.add_argument('prompt', type=str, help='用户输入')
    chat_parser.add_argument('--temperature', type=float, default=0.7, help='温度参数（默认0.7）')
    chat_parser.add_argument('--max_tokens', type=int, default=2000, help='最大token数（默认2000）')
    
    # 交互式对话
    interactive_parser = subparsers.add_parser('interactive', help='交互式对话模式')
    
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 创建客户端
    client = LLMClient(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        max_retries=args.max_retries
    )
    
    # 执行相应命令
    if args.command == 'generate':
        client.generate_queries(args.query, num_queries=args.num, use_batch=args.batch)
    
    elif args.command == 'evaluate':
        client.evaluate_quality(args.input, args.original, args.improved)
    
    elif args.command == 'chat':
        client.chat(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens)
    
    elif args.command == 'interactive':
        print("=" * 60)
        print("交互式对话模式（输入 'exit' 或 'quit' 退出）")
        print("=" * 60)
        while True:
            try:
                user_input = input("\n你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("再见！")
                    break
                client.chat(user_input)
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def example_usage():
    """示例用法"""
    print("=" * 60)
    print("LLMClient 使用示例")
    print("=" * 60)
    
    # 创建客户端
    client = LLMClient()
    
    # 示例1: 生成查询
    print("\n【示例1】生成类似查询")
    print("-" * 60)
    queries = client.generate_queries(
        original_query="请介绍下千帆大模型知识库",
        num_queries=3
    )
    
    # 示例2: 评估质量
    print("\n\n【示例2】评估质量改进")
    print("-" * 60)
    result = client.evaluate_quality(
        original_input="问题：什么是人工智能？",
        original_output="AI是一种技术。",
        improved_output="人工智能（AI）是一种计算机科学技术，旨在创建能够执行通常需要人类智能的任务的系统，如学习、推理、问题解决和感知。"
    )
    
    # 示例3: 简单对话
    print("\n\n【示例3】简单对话")
    print("-" * 60)
    response = client.chat("请用一句话解释什么是机器学习")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == '__main__':
    # 如果没有命令行参数，运行示例
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()

