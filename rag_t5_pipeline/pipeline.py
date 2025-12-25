"""
主Pipeline文件 - 整合RAG、T5模型和通用大模型
"""
import json
from typing import Dict, Any, List
from collections import defaultdict
import sys
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 导入各个接口
from rag_interface import RAGInterface
from t5_model_interface import T5ModelInterface
from llm_interface import LLMInterface
from prompts import (
    QUERY_GENERATION_PROMPT,
    T5_SYSTEM_PROMPT,
    QUALITY_EVALUATION_PROMPT
)


class RAGT5Pipeline:
    """RAG + T5 Pipeline"""
    
    def __init__(self,
                 # RAG配置
                 rag_knowledgebase_ids: List[str] = None,
                 rag_authorization: str = None,
                 rag_top_k: int = 10,
                 # T5配置
                 t5_base_url: str = None,
                 t5_max_len: int = 4096,
                 t5_max_len_generate: int = 512,
                 t5_task: str = "summarize",
                 t5_num_beams: int = 4,
                 t5_length_penalty: float = 2.0,
                 t5_no_repeat_ngram_size: int = 3,
                 t5_early_stopping: bool = True,
                 # LLM配置
                 llm_api_url: str = None,
                 llm_api_key: str = None,
                 llm_model_name: str = "qwen-plus",
                 # Pipeline配置
                 num_query_generations: int = 3,
                 context_merge_top_k: int = 20):
        """
        初始化Pipeline
        
        Args:
            rag_knowledgebase_ids: RAG知识库ID列表
            rag_authorization: RAG授权token
            rag_top_k: RAG返回top_k结果
            t5_base_url: T5服务地址（如 http://localhost:8000）
            t5_max_len: T5最大输入长度（保留用于兼容性）
            t5_max_len_generate: T5最大生成长度
            t5_task: T5任务类型 (summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro)
            t5_num_beams: T5 beam search 数量
            t5_length_penalty: T5 长度惩罚
            t5_no_repeat_ngram_size: T5 不重复 n-gram 大小
            t5_early_stopping: T5 是否早停
            llm_api_url: 通用大模型API地址
            llm_api_key: 通用大模型API密钥
            llm_model_name: 通用大模型名称
            num_query_generations: 生成的query数量
            context_merge_top_k: 合并上下文时保留的top_k
        """
        # 初始化RAG接口
        self.rag = RAGInterface(
            knowledgebase_ids=rag_knowledgebase_ids,
            authorization=rag_authorization,
            top_k=rag_top_k
        )
        
        # 初始化T5模型接口
        if t5_base_url:
            self.t5 = T5ModelInterface(
                base_url=t5_base_url,
                max_len=t5_max_len,
                max_len_generate=t5_max_len_generate,
                task=t5_task,
                num_beams=t5_num_beams,
                length_penalty=t5_length_penalty,
                no_repeat_ngram_size=t5_no_repeat_ngram_size,
                early_stopping=t5_early_stopping
            )
        else:
            self.t5 = None
            logger.warning("警告: T5服务地址未配置，将跳过T5生成步骤")
        
        # 初始化通用大模型接口
        self.llm = LLMInterface(
            api_url=llm_api_url,
            api_key=llm_api_key,
            model_name=llm_model_name
        )
        
        # Pipeline配置
        self.num_query_generations = num_query_generations
        self.context_merge_top_k = context_merge_top_k
    
    def generate_queries(self, original_query: str) -> List[str]:
        """
        步骤1: 生成多个类似的query
        
        Args:
            original_query: 原始查询
            
        Returns:
            生成的query列表
        """
        logger.info(f"[步骤1] 生成类似query，原始query: {original_query}")
        queries = self.llm.batch_generate_queries(
            original_query=original_query,
            prompt_template=QUERY_GENERATION_PROMPT,
            num_queries=self.num_query_generations
        )
        logger.info(f"生成了 {len(queries)} 个query: {queries}")
        return queries
    
    def retrieve_contexts(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        步骤2: 并发请求RAG获取上下文
        
        Args:
            queries: 查询列表
            
        Returns:
            上下文列表（包含content和score）
        """
        logger.info(f"[步骤2] 并发请求RAG，查询数量: {len(queries)}")
        search_results = self.rag.batch_search(queries, max_workers=len(queries))
        
        all_contexts = []
        for result in search_results:
            contexts = self.rag.extract_contexts(result)
            all_contexts.extend(contexts)
            logger.info(f"Query '{result.get('query', 'unknown')}' 获取到 {len(contexts)} 个上下文")
        
        return all_contexts
    
    def merge_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        步骤3: 按照打分merge上下文
        
        Args:
            contexts: 上下文列表
            
        Returns:
            合并后的上下文文本
        """
        logger.info(f"[步骤3] 合并上下文，原始上下文数量: {len(contexts)}")
        
        if not contexts:
            return ""
        
        # 按score排序
        sorted_contexts = sorted(contexts, key=lambda x: x.get('score', 0.0), reverse=True)
        
        # 取top_k
        top_contexts = sorted_contexts[:self.context_merge_top_k]
        
        # 合并文本
        merged_text = "\n\n".join([
            f"[相关性得分: {ctx.get('score', 0.0):.3f}]\n{ctx.get('content', '')}"
            for ctx in top_contexts
        ])
        
        logger.info(f"合并后保留 {len(top_contexts)} 个上下文，总长度: {len(merged_text)} 字符")
        return merged_text
    
    def generate_with_t5(self, query: str, context: str) -> str:
        """
        步骤4: 使用T5模型生成结果
        
        Args:
            query: 查询
            context: 上下文
            
        Returns:
            生成的文本
        """
        if self.t5 is None:
            logger.warning("警告: T5模型未初始化，跳过生成")
            return ""
        
        logger.info(f"[步骤4] 使用T5模型生成，query长度: {len(query)}, context长度: {len(context)}")
        
        # 计算系统提示词模板长度（不含context部分）
        # 使用空字符串格式化来获取模板长度
        prompt_template = T5_SYSTEM_PROMPT.format(context="")
        prompt_template_len = len(prompt_template)
        query_part = f"\n\n问题：{query}"
        query_part_len = len(query_part)
        
        # 计算可用的context长度（留100字符缓冲）
        # 注意：max_len是token数，这里用字符数近似估算
        # 中文字符通常1个字符约等于1-2个token，这里保守估计
        max_context_chars = (self.t5.max_len * 2) - prompt_template_len - query_part_len - 200
        
        # 如果context过长，截断
        if len(context) > max_context_chars and max_context_chars > 0:
            context = context[:max_context_chars]
            logger.info(f"Context被截断到 {max_context_chars} 字符（约 {max_context_chars//2} tokens）")
        
        # 构建系统提示词
        system_prompt = T5_SYSTEM_PROMPT.format(context=context)
        
        # 生成（传入query作为text，system_prompt作为系统提示词）
        result = self.t5.generate(query, system_prompt=system_prompt)
        logger.info(f"生成结果长度: {len(result)} 字符")
        return result
    
    def evaluate_improvement(self, 
                            original_input: str,
                            original_output: str,
                            improved_output: str) -> Dict[str, Any]:
        """
        步骤5: 评估改进效果
        
        Args:
            original_input: 原始输入（query + context）
            original_output: 原始输出（可能是空或初始输出）
            improved_output: 改进后的输出
            
        Returns:
            评估结果
        """
        logger.info(f"[步骤5] 评估改进效果")
        
        # 构建key-value格式
        input_output_data = {
            "input": original_input[:500],  # 截断以避免过长
            "original_output": original_output[:500] if original_output else "无原始输出",
            "improved_output": improved_output[:500]
        }
        
        evaluation = self.llm.evaluate_improvement(
            original_input=input_output_data["input"],
            original_output=input_output_data["original_output"],
            improved_output=input_output_data["improved_output"],
            prompt_template=QUALITY_EVALUATION_PROMPT
        )
        
        logger.info(f"评估结果: {evaluation}")
        return evaluation
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        运行完整Pipeline
        
        Args:
            query: 输入查询
            
        Returns:
            包含所有步骤结果的字典
        """
        results = {
            "original_query": query,
            "generated_queries": [],
            "contexts": [],
            "merged_context": "",
            "t5_output": "",
            "evaluation": {}
        }
        
        try:
            # 步骤1: 生成多个query
            generated_queries = self.generate_queries(query)
            results["generated_queries"] = generated_queries
            logger.info(f"[步骤1输出] 生成的query列表（共{len(generated_queries)}个）:")
            for i, q in enumerate(generated_queries, 1):
                logger.info(f"  {i}. {q}")
            
            # 步骤2: 并发请求RAG
            contexts = self.retrieve_contexts(generated_queries)
            results["contexts"] = contexts
            logger.info(f"[步骤2输出] 获取到的上下文总数: {len(contexts)}")
            if contexts:
                logger.info(f"[步骤2输出] 上下文得分范围: {min(ctx.get('score', 0) for ctx in contexts):.3f} ~ {max(ctx.get('score', 0) for ctx in contexts):.3f}")
                logger.info(f"[步骤2输出] 前3个上下文预览:")
                for i, ctx in enumerate(contexts[:3], 1):
                    content_preview = ctx.get('content', '')[:100] + '...' if len(ctx.get('content', '')) > 100 else ctx.get('content', '')
                    logger.info(f"  {i}. [得分: {ctx.get('score', 0):.3f}] {content_preview}")
            
            # 步骤3: 合并上下文
            merged_context = self.merge_contexts(contexts)
            results["merged_context"] = merged_context
            logger.info(f"[步骤3输出] 合并后的上下文长度: {len(merged_context)} 字符")
            if merged_context:
                preview_length = min(200, len(merged_context))
                preview_text = merged_context[:preview_length]
                if len(merged_context) > preview_length:
                    logger.info(f"[步骤3输出] 合并上下文预览（前{preview_length}字符）:\n{preview_text}...")
                else:
                    logger.info(f"[步骤3输出] 合并上下文内容:\n{preview_text}")
            else:
                logger.warning("[步骤3输出] 合并后的上下文为空")
            
            # 步骤4: T5生成
            if self.t5:
                t5_output = self.generate_with_t5(query, merged_context)
                results["t5_output"] = t5_output
                logger.info(f"[步骤4输出] T5生成结果长度: {len(t5_output)} 字符")
                if t5_output:
                    logger.info(f"[步骤4输出] T5生成结果:\n{t5_output}")
                else:
                    logger.warning("[步骤4输出] T5生成结果为空")
                
                # 步骤5: 评估改进
                original_input = f"Query: {query}\nContext: {merged_context[:200]}"
                evaluation = self.evaluate_improvement(
                    original_input=original_input,
                    original_output="",
                    improved_output=t5_output
                )
                results["evaluation"] = evaluation
                logger.info(f"[步骤5输出] 评估结果:")
                logger.info(f"  是否改进: {evaluation.get('improved', '未知')}")
                if 'score' in evaluation:
                    logger.info(f"  总体评分: {evaluation['score']:.3f}")
                if 'accuracy_score' in evaluation:
                    logger.info(f"  准确性评分: {evaluation.get('accuracy_score', 'N/A'):.3f if isinstance(evaluation.get('accuracy_score'), (int, float)) else 'N/A'}")
                if 'completeness_score' in evaluation:
                    logger.info(f"  完整性评分: {evaluation.get('completeness_score', 'N/A'):.3f if isinstance(evaluation.get('completeness_score'), (int, float)) else 'N/A'}")
                if 'relevance_score' in evaluation:
                    logger.info(f"  相关性评分: {evaluation.get('relevance_score', 'N/A'):.3f if isinstance(evaluation.get('relevance_score'), (int, float)) else 'N/A'}")
                if 'fluency_score' in evaluation:
                    logger.info(f"  流畅性评分: {evaluation.get('fluency_score', 'N/A'):.3f if isinstance(evaluation.get('fluency_score'), (int, float)) else 'N/A'}")
                if 'reason' in evaluation:
                    reason = evaluation.get('reason', '')
                    if reason:
                        reason_preview = reason[:200] + '...' if len(reason) > 200 else reason
                        logger.info(f"  评估原因: {reason_preview}")
                logger.info(f"[步骤5输出] 完整评估结果（JSON）:\n{json.dumps(evaluation, ensure_ascii=False, indent=2)}")
            else:
                logger.warning("跳过T5生成和评估步骤（模型未初始化）")
            
            logger.info("[完成] Pipeline执行完成")
            return results
            
        except Exception as e:
            logger.error(f"[错误] Pipeline执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results["error"] = str(e)
            return results


def main():
    """主函数 - 示例用法"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='RAG + T5 Pipeline')
    
    # RAG参数
    parser.add_argument('--rag_kb_ids', nargs='+', 
                       default=['8d5b089e-c511-4321-8c90-2507ab38676a'],
                       help='RAG知识库ID列表')
    parser.add_argument('--rag_auth', type=str, 
                       default='Bearer bce-v3/ALTAK-1l8HQWyl8VcezSx4yLKDH/6317ab81f76d0284c2060e313334925906337472',
                       help='RAG授权token')
    
    # T5参数
    parser.add_argument('--t5_base_url', type=str, 
                       default='http://localhost:8000',
                       help='T5服务地址')
    parser.add_argument('--t5_max_len', type=int, default=4096,
                       help='T5最大输入长度（保留用于兼容性）')
    parser.add_argument('--t5_max_len_generate', type=int, default=512,
                       help='T5最大生成长度')
    parser.add_argument('--t5_task', type=str, default='summarize',
                       help='T5任务类型 (summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro)')
    parser.add_argument('--t5_num_beams', type=int, default=4,
                       help='T5 beam search 数量')
    parser.add_argument('--t5_length_penalty', type=float, default=2.0,
                       help='T5 长度惩罚')
    parser.add_argument('--t5_no_repeat_ngram_size', type=int, default=3,
                       help='T5 不重复 n-gram 大小')
    parser.add_argument('--t5_early_stopping', type=lambda x: (str(x).lower() == 'true'),
                       default=True,
                       help='T5 是否早停 (true/false)')
    
    # LLM参数
    parser.add_argument('--llm_api_key', type=str, default='',
                       help='通用大模型API密钥')
    parser.add_argument('--llm_api_url', type=str, default=None,
                       help='通用大模型API地址（可选）')
    parser.add_argument('--llm_model', type=str, default='qwen-plus',
                       help='通用大模型名称（如: qwen-plus, qwen-max, qwen-turbo等）')
    
    # Pipeline参数
    parser.add_argument('--num_queries', type=int, default=3,
                       help='生成的query数量')
    parser.add_argument('--context_top_k', type=int, default=20,
                       help='合并上下文时保留的top_k')
    
    # 查询参数
    parser.add_argument('--query', type=str, 
                       default='请介绍下千帆大模型知识库',
                       help='输入查询')
    
    # 日志参数
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径（默认：logs/pipeline_YYYYMMDD_HHMMSS.log）')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志文件目录（默认：logs）')
    
    args = parser.parse_args()
    
    # 配置日志输出到文件和控制台
    from datetime import datetime
    
    # 创建日志文件路径
    if args.log_file:
        log_file = args.log_file
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除已有的处理器（避免重复）
    root_logger.handlers = []
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录日志文件路径
    logger.info(f"日志文件已创建: {log_file}")
    
    # 创建Pipeline
    pipeline = RAGT5Pipeline(
        rag_knowledgebase_ids=args.rag_kb_ids,
        rag_authorization=args.rag_auth,
        rag_top_k=10,
        t5_base_url=args.t5_base_url,
        t5_max_len=args.t5_max_len,
        t5_max_len_generate=args.t5_max_len_generate,
        t5_task=args.t5_task,
        t5_num_beams=args.t5_num_beams,
        t5_length_penalty=args.t5_length_penalty,
        t5_no_repeat_ngram_size=args.t5_no_repeat_ngram_size,
        t5_early_stopping=args.t5_early_stopping,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
        llm_model_name=args.llm_model,
        num_query_generations=args.num_queries,
        context_merge_top_k=args.context_top_k
    )
    
    # 运行Pipeline
    results = pipeline.run(args.query)
    
    # 输出结果
    logger.info("\n" + "="*50)
    logger.info("最终结果:")
    logger.info("="*50)
    logger.info(f"原始查询: {results['original_query']}")
    logger.info(f"生成的查询数量: {len(results['generated_queries'])}")
    logger.info(f"获取的上下文数量: {len(results['contexts'])}")
    logger.info(f"合并后的上下文长度: {len(results['merged_context'])} 字符")
    if results.get('t5_output'):
        logger.info(f"T5生成结果:\n{results['t5_output']}")
    if results.get('evaluation'):
        logger.info(f"评估结果:\n{json.dumps(results['evaluation'], ensure_ascii=False, indent=2)}")
    logger.info("="*50)


if __name__ == '__main__':
    main()

