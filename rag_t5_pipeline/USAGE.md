# Pipeline 使用指南

## 快速开始

### 1. 基本执行（使用默认参数）

```bash
cd rag_t5_pipeline
python pipeline.py
```

这将使用所有默认参数运行，查询内容为：`"请介绍下千帆大模型知识库"`

### 2. 自定义查询

```bash
python pipeline.py --query "你的问题"
```

### 3. 查看所有可用参数

```bash
python pipeline.py --help
```

## 常用执行命令

### 最小配置（仅必需参数）

```bash
python pipeline.py \
    --query "什么是人工智能？" \
    --llm_api_key "sk-your-api-key-here"
```

### 完整配置示例

```bash
python pipeline.py \
    --query "请介绍下千帆大模型知识库" \
    --rag_kb_ids "8d5b089e-c511-4321-8c90-2507ab38676a" \
    --rag_auth "Bearer bce-v3/ALTAK-xxx" \
    --t5_base_url "http://localhost:8000" \
    --t5_task "summarize" \
    --llm_api_key "sk-your-api-key" \
    --llm_model "qwen-plus" \
    --num_queries 3 \
    --context_top_k 20
```

## 参数说明

### 必需参数
- `--llm_api_key`: LLM API密钥（必需，用于生成query和评估）

### 可选参数

#### RAG相关
- `--rag_kb_ids`: 知识库ID列表（可多个，用空格分隔）
- `--rag_auth`: RAG授权token

#### T5相关
- `--t5_base_url`: T5服务地址（默认: `http://localhost:8000`）
- `--t5_task`: 任务类型（默认: `summarize`）
- `--t5_max_len`: 最大输入长度（默认: `4096`）
- `--t5_max_len_generate`: 最大生成长度（默认: `512`）
- `--t5_num_beams`: Beam search数量（默认: `4`）
- `--t5_length_penalty`: 长度惩罚（默认: `2.0`）
- `--t5_no_repeat_ngram_size`: 不重复n-gram大小（默认: `3`）
- `--t5_early_stopping`: 是否早停（默认: `true`）

#### LLM相关
- `--llm_api_url`: API地址（可选，有默认值）
- `--llm_model`: 模型名称（默认: `qwen-plus`）

#### Pipeline相关
- `--num_queries`: 生成的query数量（默认: `3`）
- `--context_top_k`: 合并上下文时保留的top_k（默认: `20`）

## 执行流程

Pipeline执行以下步骤：

1. **生成多个query**: 基于原始query生成多个类似的查询
2. **检索上下文**: 并发请求RAG获取相关上下文
3. **合并上下文**: 按相关性得分合并上下文
4. **T5生成**: 使用T5模型生成结果（如果T5服务已配置）
5. **评估改进**: 评估生成结果的质量改进（如果T5服务已配置）

## 输出说明

程序会输出以下信息：
- 每个步骤的执行日志（INFO级别）
- 最终结果摘要
- T5生成结果（如果启用）
- 评估结果（如果启用）

## 常见问题

### Q: 如何跳过T5生成？
A: 不提供 `--t5_base_url` 参数或设置为空，Pipeline会跳过T5相关步骤。

### Q: 如何增加生成的query数量？
A: 使用 `--num_queries` 参数，例如：`--num_queries 5`

### Q: 如何查看详细的执行日志？
A: 日志已自动配置，所有步骤信息都会输出到控制台。

### Q: 如何指定多个知识库？
A: 使用空格分隔多个ID：`--rag_kb_ids "id1" "id2" "id3"`

## 环境要求

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

确保T5服务已启动（如果使用T5功能）：
```bash
# T5服务应该在 --t5_base_url 指定的地址运行
# 例如：http://localhost:8000
```

