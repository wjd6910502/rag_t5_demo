"""
使用示例
"""
from pipeline import RAGT5Pipeline
import config  # 需要创建config.py文件，参考config_example.py


def main():
    # 初始化Pipeline
    pipeline = RAGT5Pipeline(
        # RAG配置
        rag_knowledgebase_ids=config.RAG_CONFIG["knowledgebase_ids"],
        rag_authorization=config.RAG_CONFIG["authorization"],
        rag_top_k=config.RAG_CONFIG["top_k"],
        
        # T5配置
        t5_base_url=config.T5_CONFIG["base_url"],
        t5_max_len=config.T5_CONFIG["max_len"],
        t5_max_len_generate=config.T5_CONFIG["max_len_generate"],
        t5_task=config.T5_CONFIG.get("task", "summarize"),
        t5_num_beams=config.T5_CONFIG.get("num_beams", 4),
        t5_length_penalty=config.T5_CONFIG.get("length_penalty", 2.0),
        t5_no_repeat_ngram_size=config.T5_CONFIG.get("no_repeat_ngram_size", 3),
        t5_early_stopping=config.T5_CONFIG.get("early_stopping", True),
        
        # LLM配置
        llm_api_url=config.LLM_CONFIG["api_url"],
        llm_api_key=config.LLM_CONFIG["api_key"],
        llm_model_name=config.LLM_CONFIG["model_name"],
        
        # Pipeline配置
        num_query_generations=config.PIPELINE_CONFIG["num_query_generations"],
        context_merge_top_k=config.PIPELINE_CONFIG["context_merge_top_k"]
    )
    
    # 运行Pipeline
    query = "请介绍下千帆大模型知识库"
    results = pipeline.run(query)
    
    # 输出结果
    print("\n" + "="*50)
    print("Pipeline执行结果")
    print("="*50)
    print(f"原始查询: {results['original_query']}")
    print(f"\n生成的查询 ({len(results['generated_queries'])} 个):")
    for i, q in enumerate(results['generated_queries'], 1):
        print(f"  {i}. {q}")
    
    print(f"\n获取的上下文数量: {len(results['contexts'])}")
    print(f"合并后的上下文长度: {len(results['merged_context'])} 字符")
    
    if results.get('t5_output'):
        print(f"\nT5生成结果:")
        print(f"{results['t5_output']}")
    
    if results.get('evaluation'):
        print(f"\n质量评估结果:")
        import json
        print(json.dumps(results['evaluation'], ensure_ascii=False, indent=2))
    
    if results.get('error'):
        print(f"\n错误: {results['error']}")
    
    print("="*50)


if __name__ == '__main__':
    main()

