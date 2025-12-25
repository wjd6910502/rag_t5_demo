# RAG + T5 Pipeline

这是一个整合了RAG知识库检索、T5模型生成和通用大模型评估的完整Pipeline系统。

## 项目结构

```
rag_t5_pipeline/
├── __init__.py              # 包初始化文件
├── rag_interface.py         # RAG知识库交互接口
├── t5_model_interface.py    # T5模型交互接口
├── llm_interface.py         # 通用大模型接口（用于query生成和质量判断）
├── prompts.py               # 提示词模板文件
├── pipeline.py              # 主Pipeline文件
└── README.md                # 说明文档
```

## 功能说明

### 1. RAG接口 (`rag_interface.py`)
- 与百度千帆知识库交互
- 支持单个和批量并发搜索
- 提取和解析搜索结果

### 2. T5模型接口 (`t5_model_interface.py`)
- 封装T5模型加载和生成
- 支持单条和批量生成
- 自动处理输入长度限制（4096）

### 3. 通用大模型接口 (`llm_interface.py`)
- 用于生成多个类似的query
- 用于评估生成结果的质量改进
- 使用qwen_agent方式调用阿里云通义千问（与app.py保持一致）

### 4. 提示词文件 (`prompts.py`)
- 包含所有提示词模板
- Query生成提示词
- T5系统提示词
- 质量评估提示词

### 5. 主Pipeline (`pipeline.py`)
整合所有组件，实现完整流程：
1. 根据输入query，在固定提示词下并发请求通用大模型生成多个类似的query
2. 获取到的query分别请求RAG，拿到上下文
3. 按照打分merge上下文，作为最终上下文
4. 在固定系统提示词下，使用query和聚合后的上下文一起作为输入送给T5大模型（限制长度4096）
5. 将输入和输出构建key-value格式，送入通用大模型按照提示词标准判断是否提高

## 使用方法

### 基本使用

```python
from rag_t5_pipeline import RAGT5Pipeline

# 初始化Pipeline
pipeline = RAGT5Pipeline(
    # RAG配置
    rag_knowledgebase_ids=['your-kb-id'],
    rag_authorization='Bearer your-token',
    
    # T5配置
    t5_model_path='./saved_model/summary_model',
    t5_pretrain_model_path='./t5_pegasus_pretrain',
    t5_max_len=4096,
    t5_max_len_generate=512,
    
    # LLM配置
    llm_api_key='your-api-key',
    llm_model_name='qwen-plus',  # 可选: qwen-plus, qwen-max, qwen-turbo等
    
    # Pipeline配置
    num_query_generations=3,
    context_merge_top_k=20
)

# 运行Pipeline
results = pipeline.run("请介绍下千帆大模型知识库")
print(results['t5_output'])
```

### 命令行使用

#### 基本用法（使用默认参数）

```bash
# 进入项目目录
cd rag_t5_pipeline

# 使用默认参数运行
python pipeline.py
```

#### 完整参数示例

```bash
python pipeline.py \
    --query "请介绍下千帆大模型知识库" \
    --rag_kb_ids "8d5b089e-c511-4321-8c90-2507ab38676a" \
    --rag_auth "Bearer your-token" \
    --t5_base_url "http://localhost:8000" \
    --t5_max_len 4096 \
    --t5_max_len_generate 512 \
    --t5_task "summarize" \
    --t5_num_beams 4 \
    --t5_length_penalty 2.0 \
    --t5_no_repeat_ngram_size 3 \
    --t5_early_stopping true \
    --llm_api_key "your-api-key" \
    --llm_api_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --llm_model "qwen-plus" \
    --num_queries 3 \
    --context_top_k 20
```

#### 查看所有参数

```bash
python pipeline.py --help
```

## 配置说明

### RAG配置
- `rag_knowledgebase_ids`: 知识库ID列表
- `rag_authorization`: 授权token
- `rag_top_k`: 每个query返回的top_k结果

### T5配置
- `t5_base_url`: T5服务地址（如 `http://localhost:8000`，必需）
- `t5_max_len`: 最大输入长度（默认4096，保留用于兼容性）
- `t5_max_len_generate`: 最大生成长度（默认512）
- `t5_task`: T5任务类型（默认 `summarize`，可选: `summarize`, `translate_en_to_de`, `translate_en_to_fr`, `translate_en_to_ro`）
- `t5_num_beams`: Beam search 数量（默认4）
- `t5_length_penalty`: 长度惩罚（默认2.0）
- `t5_no_repeat_ngram_size`: 不重复 n-gram 大小（默认3）
- `t5_early_stopping`: 是否早停（默认 `true`）

### LLM配置
- `llm_api_url`: API地址（可选，默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`）
- `llm_api_key`: API密钥（必需，如: `sk-11bd3e1feed241e6aedab31b7ce07ce6`）
- `llm_model_name`: 模型名称（默认 `qwen-plus`，可选: `qwen-plus`, `qwen-max`, `qwen-turbo`等）
- 使用 `qwen_agent.agents.Assistant` 方式调用，与 `app.py` 保持一致

### Pipeline配置
- `num_query_generations`: 生成的query数量（默认3）
- `context_merge_top_k`: 合并上下文时保留的top_k（默认20）

## 依赖要求

```txt
torch
transformers
jieba
requests
numpy
tqdm
qwen-agent  # 用于调用阿里云通义千问大模型
```

## 参数说明

### 命令行参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` | str | `'请介绍下千帆大模型知识库'` | 输入查询 |
| `--rag_kb_ids` | list | `['8d5b089e-c511-4321-8c90-2507ab38676a']` | RAG知识库ID列表 |
| `--rag_auth` | str | (有默认值) | RAG授权token |
| `--t5_base_url` | str | `'http://localhost:8000'` | T5服务地址 |
| `--t5_max_len` | int | `4096` | T5最大输入长度 |
| `--t5_max_len_generate` | int | `512` | T5最大生成长度 |
| `--t5_task` | str | `'summarize'` | T5任务类型 |
| `--t5_num_beams` | int | `4` | T5 beam search 数量 |
| `--t5_length_penalty` | float | `2.0` | T5 长度惩罚 |
| `--t5_no_repeat_ngram_size` | int | `3` | T5 不重复 n-gram 大小 |
| `--t5_early_stopping` | bool | `true` | T5 是否早停 |
| `--llm_api_key` | str | `''` | 通用大模型API密钥 |
| `--llm_api_url` | str | `None` | 通用大模型API地址（可选） |
| `--llm_model` | str | `'qwen-plus'` | 通用大模型名称 |
| `--num_queries` | int | `3` | 生成的query数量 |
| `--context_top_k` | int | `20` | 合并上下文时保留的top_k |

## 执行示例

### 示例1: 最小配置运行
```bash
python pipeline.py \
    --query "什么是人工智能？" \
    --llm_api_key "your-api-key"
```

### 示例2: 指定多个知识库
```bash
python pipeline.py \
    --query "请介绍下千帆大模型知识库" \
    --rag_kb_ids "kb-id-1" "kb-id-2" "kb-id-3" \
    --rag_auth "Bearer your-token" \
    --llm_api_key "your-api-key"
```

### 示例3: 自定义T5配置
```bash
python pipeline.py \
    --query "请介绍下千帆大模型知识库" \
    --t5_base_url "http://localhost:8000" \
    --t5_task "summarize" \
    --t5_num_beams 6 \
    --t5_length_penalty 1.5 \
    --llm_api_key "your-api-key"
```

### 示例4: 增加生成的query数量
```bash
python pipeline.py \
    --query "请介绍下千帆大模型知识库" \
    --num_queries 5 \
    --context_top_k 30 \
    --llm_api_key "your-api-key"
```

## 日志输出

程序使用 Python 标准 `logging` 模块输出日志，日志级别包括：
- **INFO**: 一般信息（步骤进度、结果等）
- **WARNING**: 警告信息（如T5未配置等）
- **ERROR**: 错误信息（异常、失败等）

日志格式：`时间 - 模块名 - 级别 - 消息`

## 注意事项

1. 需要配置正确的API密钥和授权token
2. T5服务需要先启动（通过 `t5_base_url` 访问）
3. 确保网络连接正常，能够访问RAG和LLM API
4. 输入长度会自动截断到4096字符以内
5. 如果T5服务未配置，Pipeline会跳过T5生成步骤，但会继续执行其他步骤

