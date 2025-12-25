# T5 模型服务

基于 PyTorch 的 T5 模型推理服务，支持在 Mac CPU 上运行。

## 功能特性

- ✅ 支持 T5 模型推理（基于 transformers 库）
- ✅ 在 Mac CPU 上运行（无需 GPU）
- ✅ RESTful API 接口（基于 FastAPI）
- ✅ 支持多种任务：摘要生成、翻译等
- ✅ 支持单次和批量生成
- ✅ 完整的客户端测试工具

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 方式一：直接运行

```bash
python t5_model_service.py
```

服务默认运行在 `http://localhost:8000`

### 方式二：使用 uvicorn

```bash
uvicorn t5_model_service:app --host 0.0.0.0 --port 8000
```

### 环境变量配置

- `MODEL_PATH`: 模型路径（默认: `./t5-small`）
- `HOST`: 服务地址（默认: `0.0.0.0`）
- `PORT`: 服务端口（默认: `8000`）

示例：
```bash
MODEL_PATH=./t5-small HOST=0.0.0.0 PORT=8000 python t5_model_service.py
```

## API 接口

### 1. 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 文本生成

```bash
POST /generate
```

请求体：
```json
{
  "text": "输入文本",
  "task": "summarize",
  "max_length": 200,
  "min_length": 30,
  "num_beams": 4,
  "length_penalty": 2.0,
  "no_repeat_ngram_size": 3,
  "early_stopping": true
}
```

支持的任务类型：
- `summarize`: 摘要生成
- `translate_en_to_de`: 英语翻译为德语
- `translate_en_to_fr`: 英语翻译为法语
- `translate_en_to_ro`: 英语翻译为罗马尼亚语

响应：
```json
{
  "generated_text": "生成的文本",
  "input_text": "输入文本",
  "task": "summarize"
}
```

### 3. 批量生成

```bash
POST /batch_generate
```

请求体：数组格式，每个元素为一个生成请求
```json
[
  {
    "text": "文本1",
    "task": "summarize",
    "max_length": 200
  },
  {
    "text": "文本2",
    "task": "translate_en_to_fr",
    "max_length": 100
  }
]
```

## 客户端测试

运行客户端测试脚本：

```bash
python t5_model_client.py
```

客户端会自动测试：
1. 健康检查
2. 摘要生成
3. 翻译功能
4. 批量生成

### 使用客户端类

```python
from t5_model_client import T5ModelClient

# 创建客户端
client = T5ModelClient(base_url="http://localhost:8000")

# 健康检查
health = client.health_check()
print(health)

# 生成文本
result = client.generate(
    text="Your input text here",
    task="summarize",
    max_length=200
)
print(result)
```

## 性能优化建议

1. **CPU 优化**：
   - 模型已配置为使用 CPU（float32）
   - 在 Mac 上运行时，PyTorch 会自动使用 MPS（如果可用）或 CPU

2. **内存管理**：
   - 首次加载模型会占用较多内存
   - 建议至少 8GB RAM

3. **批处理**：
   - 对于多个请求，使用 `/batch_generate` 接口可以提高效率

## 注意事项

1. 首次运行需要下载模型（如果模型不在本地）
2. 模型加载可能需要一些时间（取决于硬件性能）
3. CPU 推理速度较慢，建议设置合理的超时时间
4. 输入文本长度建议不超过 512 tokens

## 故障排查

### 模型加载失败

- 检查模型路径是否正确
- 确认模型文件完整（pytorch_model.bin 或 model.safetensors）
- 检查是否有足够的磁盘空间和内存

### 服务无法启动

- 检查端口是否被占用
- 确认依赖已正确安装
- 查看日志输出获取详细错误信息

### 生成速度慢

- 这是正常的，CPU 推理速度较 GPU 慢
- 可以调整 `num_beams` 参数（减少 beam search 数量）
- 减少 `max_length` 参数

## API 文档

启动服务后，访问以下地址查看交互式 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 示例

### 使用 curl 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 生成摘要
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "task": "summarize",
    "max_length": 100
  }'
```

## 许可证

请参考项目根目录的许可证文件。

