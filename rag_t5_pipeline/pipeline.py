"""
主Pipeline文件 - 整合RAG、T5模型和通用大模型
"""
import json
import re
from typing import Dict, Any, List, Optional, Union
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
    
    def _extract_valid_json_from_evaluation(self, evaluation: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        从混乱的评估结果文本中提取有效的JSON数据
        
        Args:
            evaluation: 评估结果（可能是字典或格式混乱的JSON字符串）
            
        Returns:
            提取的JSON数据字典
        """
        result = {}
        
        # 首先尝试直接解析
        try:
            if isinstance(evaluation, dict):
                data = evaluation
            elif isinstance(evaluation, str):
                data = json.loads(evaluation)
            else:
                data = evaluation
            
            if isinstance(data, dict):
                # 如果reason字段包含JSON字符串，尝试提取
                if 'reason' in data and isinstance(data['reason'], str):
                    reason_text = data['reason']
                    # 查找最后的完整JSON结构
                    json_pattern = r'\{\s*"improved"\s*:\s*true[^}]*(?:,\s*"[^"]+":\s*[^}]+)*\}'
                    matches = list(re.finditer(json_pattern, reason_text.replace('\\n', ' ')))
                    if matches:
                        last_match = matches[-1]
                        json_str = last_match.group(0)
                        try:
                            nested_data = json.loads(json_str)
                            for key, value in nested_data.items():
                                if key not in result:
                                    result[key] = value
                        except:
                            pass
                
                # 合并外部数据
                for key, value in data.items():
                    if key not in result or result[key] is None:
                        result[key] = value
        except:
            pass
        
        # 如果直接解析失败或数据不完整，尝试手动提取字段
        # 将数据转换为字符串形式以便进行正则匹配
        if isinstance(evaluation, dict):
            text_str = json.dumps(evaluation, ensure_ascii=False)
        elif isinstance(evaluation, str):
            text_str = evaluation
        else:
            text_str = str(evaluation)
        
        # 提取comprehensive_score（取最后一个出现的值）
        if 'comprehensive_score' not in result:
            comp_matches = list(re.finditer(r'"comprehensive_score"\s*:\s*([\d.]+)', text_str))
            if comp_matches:
                result['comprehensive_score'] = float(comp_matches[-1].group(1))
        
        # 也尝试score字段作为comprehensive_score的别名
        if 'comprehensive_score' not in result:
            score_matches = list(re.finditer(r'"score"\s*:\s*([\d.]+)', text_str))
            if score_matches:
                result['comprehensive_score'] = float(score_matches[-1].group(1))
        
        # 提取improvement_percentage（取最后一个出现的值）
        if 'improvement_percentage' not in result:
            imp_matches = list(re.finditer(r'"improvement_percentage"\s*:\s*([\d.]+)', text_str))
            if imp_matches:
                result['improvement_percentage'] = float(imp_matches[-1].group(1))
        
        # 提取各个分数（取最后一个出现的值）
        for field in ['accuracy_score', 'completeness_score', 'relevance_score', 'fluency_score']:
            if field not in result:
                matches = list(re.finditer(rf'"{field}"\s*:\s*([\d.]+)', text_str))
                if matches:
                    result[field] = float(matches[-1].group(1))
        
        # 提取reason（最后一个完整的reason文本）
        if 'reason' not in result or not result['reason']:
            reason_pattern = r'"reason"\s*:\s*"([^"]*改进后的输出[^"]*(?:均有明显改善|整体质量显著提升)[^"]*)"'
            reason_match = re.search(reason_pattern, text_str)
            if reason_match:
                reason_text = reason_match.group(1)
                # 清理reason文本
                reason_text = re.sub(r'\\n', '\n', reason_text)
                reason_text = re.sub(r'\{[^}]*"improved"[^}]*\}', '', reason_text)
                reason_text = re.sub(r'\{\s*"[^"]+"\s*:\s*[^}]+\}', '', reason_text)
                # 提取最后的完整句子
                sentences = re.findall(r'改进后的输出[^。]+。', reason_text)
                if sentences:
                    result['reason'] = sentences[-1].strip()
                else:
                    result['reason'] = reason_text.strip()
        
        if 'improved' not in result:
            result['improved'] = True
            
        return result
    
    def _calculate_original_scores(self, improved_scores: Dict[str, float], 
                                   improvement_percentage: Optional[float]) -> Optional[Dict[str, float]]:
        """
        根据改进后的分数和改进百分比计算原始分数
        
        Args:
            improved_scores: 改进后的分数字典
            improvement_percentage: 改进百分比
            
        Returns:
            原始分数字典（如果无法计算则返回None）
        """
        if improvement_percentage is None:
            return None
        
        factor = 1 + improvement_percentage / 100.0
        original_scores = {}
        for key, value in improved_scores.items():
            if isinstance(value, (int, float)) and value is not None:
                original_scores[key] = value / factor
        return original_scores if original_scores else None
    
    def parse_and_format_evaluation(self, evaluation: Dict[str, Any]) -> str:
        """
        解析并格式化评估结果，输出每个评价维度的前后打分对比、提升比、理由
        
        Args:
            evaluation: 评估结果字典
            
        Returns:
            格式化后的评估结果字符串
        """
        if not evaluation:
            return "评估结果为空"
        
        # 提取有效的JSON数据（处理格式混乱的情况）
        try:
            # 如果evaluation已经是字典，尝试提取有效数据
            extracted_data = self._extract_valid_json_from_evaluation(evaluation)
        except:
            extracted_data = evaluation
        
        # 提取改进后的分数
        improved_scores = {
            'comprehensive_score': extracted_data.get('comprehensive_score') or extracted_data.get('score'),
            'accuracy_score': extracted_data.get('accuracy_score'),
            'completeness_score': extracted_data.get('completeness_score'),
            'relevance_score': extracted_data.get('relevance_score'),
            'fluency_score': extracted_data.get('fluency_score'),
        }
        
        # 获取改进百分比
        improvement_percentage = extracted_data.get('improvement_percentage')
        
        # 计算原始分数
        original_scores = self._calculate_original_scores(improved_scores, improvement_percentage)
        
        # 提取原因
        reason = extracted_data.get('reason', '')
        if reason:
            # 清理reason文本
            reason = re.sub(r'\\n', '\n', reason)
            reason = re.sub(r'\{\s*"improved"[^}]+\}', '', reason)
            reason = re.sub(r'\{[^}]*"improved"[^}]*\}', '', reason)
            # 提取最后的完整句子
            sentences = re.findall(r'改进后的输出[^。]+。', reason)
            if sentences:
                reason = sentences[-1]
            else:
                # 如果找不到完整句子，取最后300个字符
                reason = reason[-300:] if len(reason) > 300 else reason
            reason = reason.strip()
        
        # 构建输出字符串
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("评估结果详细解析")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        # 定义维度名称
        dimensions = {
            'accuracy_score': '准确性 (Accuracy)',
            'completeness_score': '完整性 (Completeness)',
            'relevance_score': '相关性 (Relevance)',
            'fluency_score': '流畅性 (Fluency)',
            'comprehensive_score': '综合得分 (Comprehensive)'
        }
        
        # 输出每个维度的前后对比
        output_lines.append("各评价维度前后对比：")
        output_lines.append("-" * 80)
        
        for key in ['accuracy_score', 'completeness_score', 'relevance_score', 'fluency_score']:
            name = dimensions.get(key, key)
            orig_val = original_scores.get(key) if original_scores else None
            impr_val = improved_scores.get(key)
            
            if orig_val is not None and impr_val is not None:
                diff = impr_val - orig_val
                diff_pct = (diff / orig_val * 100) if orig_val > 0 else 0
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                output_lines.append(
                    f"{name:25s} | 原始: {orig_val:.3f} → 改进后: {impr_val:.3f} {arrow} {diff:+.3f} ({diff_pct:+.2f}%)"
                )
            elif impr_val is not None:
                output_lines.append(f"{name:25s} | 改进后: {impr_val:.3f} (原始分数未知)")
            else:
                output_lines.append(f"{name:25s} | 数据缺失")
        
        output_lines.append("-" * 80)
        output_lines.append("")
        
        # 输出综合得分对比
        output_lines.append("综合得分对比：")
        output_lines.append("-" * 80)
        orig_comp = original_scores.get('comprehensive_score') if original_scores else None
        impr_comp = improved_scores.get('comprehensive_score')
        
        if orig_comp is not None and impr_comp is not None:
            diff = impr_comp - orig_comp
            diff_pct = (diff / orig_comp * 100) if orig_comp > 0 else 0
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            output_lines.append(
                f"综合得分 | 原始: {orig_comp:.3f} → 改进后: {impr_comp:.3f} {arrow} {diff:+.3f} ({diff_pct:+.2f}%)"
            )
            if improvement_percentage is not None:
                output_lines.append(f"改进百分比: {improvement_percentage:.2f}%")
        elif impr_comp is not None:
            output_lines.append(f"综合得分 | 改进后: {impr_comp:.3f}")
            if improvement_percentage is not None:
                output_lines.append(f"改进百分比: {improvement_percentage:.2f}%")
                # 尝试反推原始分数
                if improvement_percentage > 0:
                    factor = 1 + improvement_percentage / 100.0
                    estimated_original = impr_comp / factor
                    output_lines.append(f"估算原始分数: {estimated_original:.3f} (基于改进百分比反推)")
        
        output_lines.append("-" * 80)
        
        # 输出原因
        if reason:
            output_lines.append("")
            output_lines.append("改进原因：")
            output_lines.append("-" * 80)
            output_lines.append(reason)
            output_lines.append("-" * 80)
        
        output_lines.append("")
        output_lines.append("=" * 80)
        
        return "\n".join(output_lines)
    
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
                # 使用query（不带上下文）调用T5模型获取原始输出
                original_output = self.t5.generate(query) if self.t5 else ""
                logger.info(f"[步骤5] 不带上下文的T5生成结果长度: {len(original_output)} 字符")
                if original_output:
                    logger.info(f"[步骤5] 不带上下文的T5生成结果:\n{original_output}")
                
                original_input = f"Query: {query}\nContext: {merged_context[:200]}"
                evaluation = self.evaluate_improvement(
                    original_input=original_input,
                    original_output=original_output,
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
                
                # 解析并格式化评估结果
                formatted_evaluation = self.parse_and_format_evaluation(evaluation)
                logger.info(f"\n[步骤5输出] 格式化评估结果:\n{formatted_evaluation}")
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
        logger.info(f"评估结果（原始JSON）:\n{json.dumps(results['evaluation'], ensure_ascii=False, indent=2)}")
        # 解析并格式化评估结果
        formatted_evaluation = pipeline.parse_and_format_evaluation(results['evaluation'])
        logger.info(f"\n评估结果（格式化）:\n{formatted_evaluation}")
    logger.info("="*50)


if __name__ == '__main__':
    main()

