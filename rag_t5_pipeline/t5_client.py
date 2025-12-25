"""
T5模型客户端
用于直接访问和使用T5模型服务
"""
import argparse
import sys
import json
from typing import List, Optional
from t5_model_interface import T5ModelInterface

# 尝试导入配置文件，如果不存在则使用默认配置
try:
    import config
    DEFAULT_CONFIG = config.T5_CONFIG
except ImportError:
    DEFAULT_CONFIG = {
        "base_url": "http://localhost:8000",
        "max_len": 4096,
        "max_len_generate": 512,
        "task": "summarize",
        "num_beams": 4,
        "length_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }


class T5Client:
    """T5模型客户端"""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 max_len: Optional[int] = None,
                 max_len_generate: Optional[int] = None,
                 task: Optional[str] = None,
                 num_beams: Optional[int] = None,
                 length_penalty: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None,
                 early_stopping: Optional[bool] = None):
        """
        初始化T5客户端
        
        Args:
            base_url: T5服务地址（默认从配置文件读取）
            max_len: 最大输入长度
            max_len_generate: 最大生成长度
            task: 任务类型
            num_beams: beam search 数量
            length_penalty: 长度惩罚
            no_repeat_ngram_size: 不重复 n-gram 大小
            early_stopping: 是否早停
        """
        # 使用传入参数或配置文件中的默认值
        config_dict = {
            "base_url": base_url or DEFAULT_CONFIG.get("base_url", "http://localhost:8000"),
            "max_len": max_len or DEFAULT_CONFIG.get("max_len", 4096),
            "max_len_generate": max_len_generate or DEFAULT_CONFIG.get("max_len_generate", 512),
            "task": task or DEFAULT_CONFIG.get("task", "summarize"),
            "num_beams": num_beams or DEFAULT_CONFIG.get("num_beams", 4),
            "length_penalty": length_penalty or DEFAULT_CONFIG.get("length_penalty", 2.0),
            "no_repeat_ngram_size": no_repeat_ngram_size or DEFAULT_CONFIG.get("no_repeat_ngram_size", 3),
            "early_stopping": early_stopping if early_stopping is not None else DEFAULT_CONFIG.get("early_stopping", True)
        }
        
        self.interface = T5ModelInterface(**config_dict)
        self.config = config_dict
    
    def generate(self, text: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            text: 输入文本
            system_prompt: 系统提示词（可选）
            
        Returns:
            生成的文本
        """
        return self.interface.generate(text, system_prompt)
    
    def batch_generate(self, texts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        批量生成文本
        
        Args:
            texts: 输入文本列表
            system_prompt: 系统提示词（可选）
            
        Returns:
            生成的文本列表
        """
        return self.interface.batch_generate(texts, system_prompt)
    
    def interactive_mode(self):
        """交互式模式"""
        print("=" * 60)
        print("T5模型客户端 - 交互式模式")
        print("=" * 60)
        print(f"服务地址: {self.config['base_url']}")
        print(f"任务类型: {self.config['task']}")
        print(f"最大生成长度: {self.config['max_len_generate']}")
        print("\n输入 'quit' 或 'exit' 退出")
        print("输入 'config' 查看当前配置")
        print("输入 'task <task_name>' 切换任务类型 (summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro)")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n请输入文本 (或输入命令): ").strip()
                
                if not user_input:
                    continue
                
                # 处理退出命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("退出交互式模式")
                    break
                
                # 处理配置查看命令
                if user_input.lower() == 'config':
                    print("\n当前配置:")
                    print(json.dumps(self.config, indent=2, ensure_ascii=False))
                    continue
                
                # 处理任务切换命令
                if user_input.lower().startswith('task '):
                    new_task = user_input[5:].strip()
                    valid_tasks = ['summarize', 'translate_en_to_de', 'translate_en_to_fr', 'translate_en_to_ro']
                    if new_task in valid_tasks:
                        self.interface.task = new_task
                        self.config['task'] = new_task
                        print(f"任务类型已切换为: {new_task}")
                    else:
                        print(f"无效的任务类型。有效选项: {', '.join(valid_tasks)}")
                    continue
                
                # 生成文本
                print("\n正在生成...")
                result = self.generate(user_input)
                
                print("\n" + "-" * 60)
                print("生成结果:")
                print("-" * 60)
                print(result)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n退出交互式模式")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="T5模型客户端 - 用于访问和使用T5模型服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python t5_client.py
  
  # 单次生成
  python t5_client.py -t "这是一段需要总结的文本"
  
  # 批量生成（从文件读取）
  python t5_client.py -f input.txt
  
  # 指定任务类型
  python t5_client.py -t "Hello world" --task translate_en_to_de
  
  # 自定义服务地址
  python t5_client.py -t "文本" --base-url http://localhost:9000
        """
    )
    
    parser.add_argument(
        '-t', '--text',
        type=str,
        help='要处理的文本（单次生成模式）'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='包含文本的文件路径（每行一个文本，批量生成模式）'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help=f'T5服务地址（默认: {DEFAULT_CONFIG.get("base_url", "http://localhost:8000")}）'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        choices=['summarize', 'translate_en_to_de', 'translate_en_to_fr', 'translate_en_to_ro'],
        help='任务类型（默认: summarize）'
    )
    
    parser.add_argument(
        '--max-len-generate',
        type=int,
        default=None,
        help=f'最大生成长度（默认: {DEFAULT_CONFIG.get("max_len_generate", 512)}）'
    )
    
    parser.add_argument(
        '--num-beams',
        type=int,
        default=None,
        help=f'Beam search 数量（默认: {DEFAULT_CONFIG.get("num_beams", 4)}）'
    )
    
    parser.add_argument(
        '--length-penalty',
        type=float,
        default=None,
        help=f'长度惩罚（默认: {DEFAULT_CONFIG.get("length_penalty", 2.0)}）'
    )
    
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=None,
        help='系统提示词（可选）'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='进入交互式模式'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出文件路径（仅批量模式）'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='以JSON格式输出结果'
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = T5Client(
        base_url=args.base_url,
        task=args.task,
        max_len_generate=args.max_len_generate,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty
    )
    
    # 交互式模式
    if args.interactive or (not args.text and not args.file):
        client.interactive_mode()
        return
    
    # 单次生成模式
    if args.text:
        try:
            result = client.generate(args.text, args.system_prompt)
            
            if args.json:
                output = {
                    "input": args.text,
                    "output": result,
                    "task": client.config['task'],
                    "config": client.config
                }
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                print("\n" + "=" * 60)
                print("输入文本:")
                print("-" * 60)
                print(args.text)
                print("\n" + "=" * 60)
                print("生成结果:")
                print("-" * 60)
                print(result)
                print("=" * 60)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    # 批量生成模式
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("错误: 输入文件为空", file=sys.stderr)
                sys.exit(1)
            
            print(f"正在处理 {len(texts)} 个文本...")
            results = client.batch_generate(texts, args.system_prompt)
            
            if args.json:
                output = {
                    "inputs": texts,
                    "outputs": results,
                    "task": client.config['task'],
                    "config": client.config
                }
                output_str = json.dumps(output, ensure_ascii=False, indent=2)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output_str)
                    print(f"结果已保存到: {args.output}")
                else:
                    print(output_str)
            else:
                output_lines = []
                for i, (text, result) in enumerate(zip(texts, results), 1):
                    output_lines.append(f"\n{'=' * 60}")
                    output_lines.append(f"文本 {i}/{len(texts)}")
                    output_lines.append(f"{'-' * 60}")
                    output_lines.append(f"输入: {text}")
                    output_lines.append(f"输出: {result}")
                    output_lines.append(f"{'=' * 60}")
                
                output_str = "\n".join(output_lines)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output_str)
                    print(f"结果已保存到: {args.output}")
                else:
                    print(output_str)
                    
        except FileNotFoundError:
            print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

