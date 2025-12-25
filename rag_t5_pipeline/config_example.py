"""
配置文件示例
复制此文件为config.py并填入实际的配置信息
"""

# RAG配置
RAG_CONFIG = {
    "knowledgebase_ids": ["8d5b089e-c511-4321-8c90-2507ab38676a"],
    "authorization": "Bearer bce-v3/ALTAK-1l8HQWyl8VcezSx4yLKDH/6317ab81f76d0284c2060e313334925906337472",
    "top_k": 10
}

# T5模型配置（使用本地HTTP服务）
T5_CONFIG = {
    "base_url": "http://localhost:8000",  # T5服务地址
    "max_len": 4096,  # 保留用于兼容性，实际由服务端控制
    "max_len_generate": 512,
    "task": "summarize",  # 任务类型: summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro
    "num_beams": 4,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True
}

# 通用大模型配置（使用qwen_agent方式，与app.py保持一致）
LLM_CONFIG = {
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问地址
    #"api_key": "sk-11bd3e1feed241e6aedab31b7ce07ce6",  # API密钥
    "api_key": "sk-581bbf477d74411ca63667289e255321",
    "model_name": "qwen-plus"  # 可选: qwen-plus, qwen-max, qwen-turbo等
}

# Pipeline配置
PIPELINE_CONFIG = {
    "num_query_generations": 3,  # 生成的query数量
    "context_merge_top_k": 20   # 合并上下文时保留的top_k
}

