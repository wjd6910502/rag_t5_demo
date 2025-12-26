"""
RAG知识库客户端
用于直接访问和使用百度千帆知识库搜索服务
"""
import argparse
import sys
import json
from typing import List, Optional
from rag_interface import RAGInterface

# 尝试导入配置文件，如果不存在则使用默认配置
try:
    import config
    DEFAULT_CONFIG = config.RAG_CONFIG
except ImportError:
    DEFAULT_CONFIG = {
        "knowledgebase_ids": ["8d5b089e-c511-4321-8c90-2507ab38676a"],
        "authorization": "Bearer bce-v3/ALTAK-1l8HQWyl8VcezSx4yLKDH/6317ab81f76d0284c2060e313334925906337472",
        "top_k": 10,
        "enable_graph": False,
        "enable_expansion": False
    }


class RAGClient:
    """RAG知识库客户端"""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 knowledgebase_ids: Optional[List[str]] = None,
                 authorization: Optional[str] = None,
                 top_k: Optional[int] = None,
                 enable_graph: Optional[bool] = None,
                 enable_expansion: Optional[bool] = None):
        """
        初始化RAG客户端
        
        Args:
            url: 知识库搜索API地址（默认从配置文件读取）
            knowledgebase_ids: 知识库ID列表
            authorization: 授权token
            top_k: 返回top_k个结果
            enable_graph: 是否启用图谱
            enable_expansion: 是否启用扩展
        """
        # 使用传入参数或配置文件中的默认值
        config_dict = {
            "url": url or "https://qianfan.baidubce.com/v2/knowledgebases/search",
            "knowledgebase_ids": knowledgebase_ids or DEFAULT_CONFIG.get("knowledgebase_ids", []),
            "authorization": authorization or DEFAULT_CONFIG.get("authorization", ""),
            "top_k": top_k if top_k is not None else DEFAULT_CONFIG.get("top_k", 10),
            "enable_graph": enable_graph if enable_graph is not None else DEFAULT_CONFIG.get("enable_graph", False),
            "enable_expansion": enable_expansion if enable_expansion is not None else DEFAULT_CONFIG.get("enable_expansion", False)
        }
        
        self.interface = RAGInterface(**config_dict)
        self.config = config_dict
    
    def search(self, query: str) -> dict:
        """
        搜索知识库
        
        Args:
            query: 查询文本
            
        Returns:
            搜索结果字典
        """
        return self.interface.search(query)
    
    def batch_search(self, queries: List[str], max_workers: int = 5) -> List[dict]:
        """
        批量搜索知识库
        
        Args:
            queries: 查询文本列表
            max_workers: 最大并发数
            
        Returns:
            搜索结果列表
        """
        return self.interface.batch_search(queries, max_workers)
    
    def extract_contexts(self, search_result: dict) -> List[dict]:
        """
        从搜索结果中提取上下文信息
        
        Args:
            search_result: 搜索结果
            
        Returns:
            上下文列表，每个元素包含content和score
        """
        return self.interface.extract_contexts(search_result)
    
    def interactive_mode(self):
        """交互式模式"""
        print("=" * 60)
        print("RAG知识库客户端 - 交互式模式")
        print("=" * 60)
        print(f"API地址: {self.config['url']}")
        print(f"知识库ID: {', '.join(self.config['knowledgebase_ids'])}")
        print(f"Top K: {self.config['top_k']}")
        print(f"启用图谱: {self.config['enable_graph']}")
        print(f"启用扩展: {self.config['enable_expansion']}")
        print("\n输入 'quit' 或 'exit' 退出")
        print("输入 'config' 查看当前配置")
        print("输入 'topk <number>' 设置top_k值")
        print("输入 'graph on/off' 启用/禁用图谱")
        print("输入 'expansion on/off' 启用/禁用扩展")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n请输入查询 (或输入命令): ").strip()
                
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
                
                # 处理top_k设置命令
                if user_input.lower().startswith('topk '):
                    try:
                        new_top_k = int(user_input[5:].strip())
                        if new_top_k > 0:
                            self.interface.top_k = new_top_k
                            self.config['top_k'] = new_top_k
                            print(f"Top K 已设置为: {new_top_k}")
                        else:
                            print("Top K 必须大于 0")
                    except ValueError:
                        print("无效的 Top K 值，请输入数字")
                    continue
                
                # 处理图谱开关命令
                if user_input.lower().startswith('graph '):
                    value = user_input[6:].strip().lower()
                    if value == 'on':
                        self.interface.enable_graph = True
                        self.config['enable_graph'] = True
                        print("图谱已启用")
                    elif value == 'off':
                        self.interface.enable_graph = False
                        self.config['enable_graph'] = False
                        print("图谱已禁用")
                    else:
                        print("无效命令，请输入 'graph on' 或 'graph off'")
                    continue
                
                # 处理扩展开关命令
                if user_input.lower().startswith('expansion '):
                    value = user_input[10:].strip().lower()
                    if value == 'on':
                        self.interface.enable_expansion = True
                        self.config['enable_expansion'] = True
                        print("扩展已启用")
                    elif value == 'off':
                        self.interface.enable_expansion = False
                        self.config['enable_expansion'] = False
                        print("扩展已禁用")
                    else:
                        print("无效命令，请输入 'expansion on' 或 'expansion off'")
                    continue
                
                # 执行搜索
                print("\n正在搜索...")
                result = self.search(user_input)
                
                # 显示结果
                print("\n" + "-" * 60)
                print("搜索结果:")
                print("-" * 60)
                
                if "error" in result:
                    print(f"错误: {result['error']}")
                else:
                    contexts = self.extract_contexts(result)
                    if contexts:
                        print(f"找到 {len(contexts)} 个相关上下文:\n")
                        for i, ctx in enumerate(contexts, 1):
                            print(f"[{i}] 相关性分数: {ctx.get('score', 0.0):.4f}")
                            print(f"内容: {ctx.get('content', '')[:200]}...")
                            if ctx.get('metadata'):
                                print(f"元数据: {json.dumps(ctx['metadata'], ensure_ascii=False)}")
                            print()
                    else:
                        print("未找到相关上下文")
                        print(f"原始响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n退出交互式模式")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                import traceback
                traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG知识库客户端 - 用于访问和使用百度千帆知识库搜索服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python rag_client.py
  
  # 单次搜索
  python rag_client.py -q "请介绍下千帆大模型知识库"
  
  # 批量搜索（从文件读取）
  python rag_client.py -f queries.txt
  
  # 指定top_k值
  python rag_client.py -q "查询文本" --top-k 20
  
  # 启用图谱和扩展
  python rag_client.py -q "查询文本" --enable-graph --enable-expansion
  
  # 自定义知识库ID
  python rag_client.py -q "查询文本" --knowledgebase-id "your-kb-id"
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='查询文本（单次搜索模式）'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='包含查询的文件路径（每行一个查询，批量搜索模式）'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default=None,
        help='知识库搜索API地址（默认: https://qianfan.baidubce.com/v2/knowledgebases/search）'
    )
    
    parser.add_argument(
        '--knowledgebase-id',
        type=str,
        action='append',
        help='知识库ID（可多次使用指定多个ID）'
    )
    
    parser.add_argument(
        '--authorization',
        type=str,
        default=None,
        help='授权token（默认从配置文件读取）'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help=f'返回top_k个结果（默认: {DEFAULT_CONFIG.get("top_k", 10)}）'
    )
    
    parser.add_argument(
        '--enable-graph',
        action='store_true',
        help='启用图谱'
    )
    
    parser.add_argument(
        '--enable-expansion',
        action='store_true',
        help='启用扩展'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='批量搜索时的最大并发数（默认: 5）'
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
    
    parser.add_argument(
        '--extract-contexts',
        action='store_true',
        help='仅输出提取的上下文信息（不包含原始响应）'
    )
    
    args = parser.parse_args()
    
    # 处理知识库ID
    knowledgebase_ids = args.knowledgebase_id if args.knowledgebase_id else None
    
    # 创建客户端
    client = RAGClient(
        url=args.url,
        knowledgebase_ids=knowledgebase_ids,
        authorization=args.authorization,
        top_k=args.top_k,
        enable_graph=args.enable_graph,
        enable_expansion=args.enable_expansion
    )
    
    # 交互式模式
    if args.interactive or (not args.query and not args.file):
        client.interactive_mode()
        return
    
    # 单次搜索模式
    if args.query:
        try:
            result = client.search(args.query)
            
            if args.json:
                if args.extract_contexts:
                    contexts = client.extract_contexts(result)
                    output = {
                        "query": args.query,
                        "contexts": contexts,
                        "count": len(contexts)
                    }
                else:
                    output = {
                        "query": args.query,
                        "result": result,
                        "contexts": client.extract_contexts(result)
                    }
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                print("\n" + "=" * 60)
                print("查询:")
                print("-" * 60)
                print(args.query)
                print("\n" + "=" * 60)
                print("搜索结果:")
                print("-" * 60)
                
                if "error" in result:
                    print(f"错误: {result['error']}")
                else:
                    contexts = client.extract_contexts(result)
                    if contexts:
                        print(f"找到 {len(contexts)} 个相关上下文:\n")
                        for i, ctx in enumerate(contexts, 1):
                            print(f"[{i}] 相关性分数: {ctx.get('score', 0.0):.4f}")
                            content = ctx.get('content', '')
                            print(f"内容: {content}")
                            if ctx.get('metadata'):
                                print(f"元数据: {json.dumps(ctx['metadata'], ensure_ascii=False, indent=2)}")
                            print()
                    else:
                        print("未找到相关上下文")
                        if not args.extract_contexts:
                            print(f"\n原始响应:")
                            print(json.dumps(result, ensure_ascii=False, indent=2))
                
                print("=" * 60)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 批量搜索模式
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            if not queries:
                print("错误: 输入文件为空", file=sys.stderr)
                sys.exit(1)
            
            print(f"正在处理 {len(queries)} 个查询...")
            results = client.batch_search(queries, max_workers=args.max_workers)
            
            if args.json:
                if args.extract_contexts:
                    output = {
                        "queries": queries,
                        "results": [
                            {
                                "query": r.get('query', ''),
                                "contexts": client.extract_contexts(r),
                                "count": len(client.extract_contexts(r))
                            }
                            for r in results
                        ]
                    }
                else:
                    output = {
                        "queries": queries,
                        "results": results
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
                for i, result in enumerate(results, 1):
                    query = result.get('query', queries[i-1] if i <= len(queries) else '')
                    output_lines.append(f"\n{'=' * 60}")
                    output_lines.append(f"查询 {i}/{len(results)}: {query}")
                    output_lines.append(f"{'-' * 60}")
                    
                    if "error" in result:
                        output_lines.append(f"错误: {result['error']}")
                    else:
                        contexts = client.extract_contexts(result)
                        if contexts:
                            output_lines.append(f"找到 {len(contexts)} 个相关上下文:\n")
                            for j, ctx in enumerate(contexts, 1):
                                output_lines.append(f"[{j}] 相关性分数: {ctx.get('score', 0.0):.4f}")
                                output_lines.append(f"内容: {ctx.get('content', '')}")
                                if ctx.get('metadata'):
                                    output_lines.append(f"元数据: {json.dumps(ctx['metadata'], ensure_ascii=False, indent=2)}")
                                output_lines.append("")
                        else:
                            output_lines.append("未找到相关上下文")
                            if not args.extract_contexts:
                                output_lines.append(f"\n原始响应:")
                                output_lines.append(json.dumps(result, ensure_ascii=False, indent=2))
                    
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
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()

