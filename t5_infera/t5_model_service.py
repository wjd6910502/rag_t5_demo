"""
T5 模型推理服务
支持在 Mac CPU 上运行模型推理
"""

import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型和 tokenizer
model = None
tokenizer = None
device = "cpu"  # Mac CPU 模式


def load_model(model_path: str = "./t5-small"):
    """加载 T5 模型和 tokenizer"""
    global model, tokenizer
    
    try:
        logger.info(f"正在加载模型 from {model_path}...")
        
        # 加载 tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        logger.info("Tokenizer 加载完成")
        
        # 加载模型（使用 CPU）
        # 优先使用 safetensors 格式，避免 torch.load 安全漏洞问题
        try:
            model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.float32,  # CPU 使用 float32
                use_safetensors=True  # 优先使用 safetensors 格式
            )
        except Exception as e:
            # 如果 safetensors 不可用，回退到普通加载方式
            logger.warning(f"使用 safetensors 加载失败: {e}，尝试普通加载方式")
            model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.float32  # CPU 使用 float32
            )
        model.eval()  # 设置为评估模式
        model.to(device)
        
        logger.info("模型加载完成")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    model_path = os.getenv("MODEL_PATH", "./t5-small")
    if not os.path.exists(model_path):
        logger.warning(f"模型路径不存在: {model_path}，尝试使用相对路径")
        model_path = "./t5-small"
    
    load_model(model_path)
    yield
    # 关闭时清理（如果需要）


# 初始化 FastAPI 应用
app = FastAPI(title="T5 Model Service", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    """生成请求模型"""
    text: str
    task: Optional[str] = "summarize"  # 支持: summarize, translate_en_to_de, translate_en_to_fr, translate_en_to_ro
    max_length: Optional[int] = 200
    min_length: Optional[int] = 30
    num_beams: Optional[int] = 4
    length_penalty: Optional[float] = 2.0
    no_repeat_ngram_size: Optional[int] = 3
    early_stopping: Optional[bool] = True


class GenerateResponse(BaseModel):
    """生成响应模型"""
    generated_text: str
    input_text: str
    task: str


def get_task_prefix(task: str) -> str:
    """根据任务类型获取前缀"""
    task_prefixes = {
        "summarize": "summarize: ",
        "translate_en_to_de": "translate English to German: ",
        "translate_en_to_fr": "translate English to French: ",
        "translate_en_to_ro": "translate English to Romanian: ",
    }
    return task_prefixes.get(task, "summarize: ")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "T5 Model Service",
        "version": "1.0.0",
        "device": device,
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """文本生成接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 构建输入文本（添加任务前缀）
        prefix = get_task_prefix(request.task)
        input_text = prefix + request.text
        
        # Tokenize
        input_ids = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # 生成参数
        generation_config = {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "num_beams": request.num_beams,
            "length_penalty": request.length_penalty,
            "no_repeat_ngram_size": request.no_repeat_ngram_size,
            "early_stopping": request.early_stopping,
            "do_sample": False,
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **generation_config
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"生成完成: {len(generated_text)} 字符")
        
        return GenerateResponse(
            generated_text=generated_text,
            input_text=request.text,
            task=request.task
        )
    
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/batch_generate")
async def batch_generate_text(requests: List[GenerateRequest]):
    """批量文本生成接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        results = []
        for request in requests:
            # 构建输入文本
            prefix = get_task_prefix(request.task)
            input_text = prefix + request.text
            
            # Tokenize
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)
            
            # 生成参数
            generation_config = {
                "max_length": request.max_length,
                "min_length": request.min_length,
                "num_beams": request.num_beams,
                "length_penalty": request.length_penalty,
                "no_repeat_ngram_size": request.no_repeat_ngram_size,
                "early_stopping": request.early_stopping,
                "do_sample": False,
            }
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    **generation_config
                )
            
            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "generated_text": generated_text,
                "input_text": request.text,
                "task": request.task
            })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"批量生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"启动服务在 {host}:{port}")
    uvicorn.run(app, host=host, port=port)

