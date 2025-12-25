"""
提示词文件 - 包含所有提示词模板
"""

# Query生成提示词
QUERY_GENERATION_PROMPT = """请根据以下原始查询，生成{num}个语义相似但表达不同的查询问题。

要求：
1. 保持核心语义不变
2. 使用不同的表达方式
3. 每个查询应该是有意义的完整问题
4. **重要：每个查询的长度必须控制在5-10个字之间**（不包括标点符号）
5. 避免重复和冗余的词语

原始查询：{query}

请直接输出{num}个查询，每行一个，不要添加编号、序号或其他说明："""

# 系统提示词（用于T5模型）
T5_SYSTEM_PROMPT = """你是一个专业的文本生成助手。请根据给定的查询和上下文信息，生成准确、完整的回答。

上下文信息：
{context}

请基于以上上下文回答以下问题："""

# 质量评估提示词
QUALITY_EVALUATION_PROMPT = """请评估以下文本生成结果的质量改进情况。

原始输入：
{input}

原始输出：
{original_output}

改进后的输出：
{improved_output}

请从以下维度进行评估：
1. 准确性：信息是否准确
2. 完整性：信息是否完整
3. 相关性：是否与输入高度相关
4. 流畅性：表达是否流畅自然

请以JSON格式输出评估结果：
{{
    "improved": true/false,
    "score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "fluency_score": 0.0-1.0,
    "reason": "改进原因说明"
}}

请直接输出JSON，不要添加其他说明："""

# 简化的质量评估提示词（如果JSON解析失败）
SIMPLE_QUALITY_EVALUATION_PROMPT = """请判断改进后的输出是否比原始输出更好。

原始输入：{input}
原始输出：{original_output}
改进后的输出：{improved_output}

请回答：
1. 是否改进（是/否）
2. 评分（0-1之间的小数）
3. 简要说明原因

格式：
是否改进：是/否
评分：0.XX
原因：XXX"""

