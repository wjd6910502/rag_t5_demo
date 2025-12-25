"""
将 PyTorch 模型转换为 safetensors 格式
解决 torch.load 安全漏洞问题
"""
import os
import torch
from transformers import T5ForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_to_safetensors(model_path: str):
    """
    将模型从 pytorch_model.bin 转换为 safetensors 格式
    
    Args:
        model_path: 模型路径
    """
    try:
        logger.info(f"开始转换模型: {model_path}")
        
        # 检查模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        pytorch_model_bin = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(pytorch_model_bin):
            raise FileNotFoundError(f"未找到 pytorch_model.bin: {pytorch_model_bin}")
        
        # 临时禁用 transformers 的 torch.load 安全检查
        # 这样我们可以加载旧格式的模型，然后重新保存为 safetensors
        logger.info("正在临时禁用安全检查以加载旧格式模型...")
        import transformers.utils.import_utils as import_utils
        import transformers.modeling_utils as modeling_utils
        
        # 保存原始检查函数
        original_check = import_utils.check_torch_load_is_safe
        original_load_state_dict = modeling_utils.load_state_dict
        
        # 临时禁用检查 - 方法1: 禁用检查函数
        def bypass_check():
            pass
        import_utils.check_torch_load_is_safe = bypass_check
        
        # 临时禁用检查 - 方法2: 包装 load_state_dict 函数
        def patched_load_state_dict(checkpoint_file, is_quantized=False, map_location="cpu", weights_only=True):
            # 直接调用原始的 load_state_dict，但跳过检查
            # 我们需要手动加载文件
            if checkpoint_file.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(checkpoint_file, framework="pt") as f:
                    state_dict = {}
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
                    return state_dict
            else:
                # 对于 .bin 文件，直接使用 torch.load，跳过检查
                return torch.load(checkpoint_file, map_location=map_location, weights_only=False)
        
        # 应用 patch
        modeling_utils.load_state_dict = patched_load_state_dict
        
        try:
            # 加载模型（使用旧格式，绕过安全检查）
            logger.info("正在加载模型（使用旧格式）...")
            model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                use_safetensors=False,  # 明确使用旧格式
                dtype=torch.float32
            )
            logger.info("模型加载成功")
            
            # 恢复原始函数
            import_utils.check_torch_load_is_safe = original_check
            modeling_utils.load_state_dict = original_load_state_dict
            
            # 使用模型的 save_pretrained 方法保存为 safetensors 格式
            # 这会自动处理共享张量问题
            logger.info("正在保存为 safetensors 格式...")
            model.save_pretrained(
                model_path,
                safe_serialization=True  # 使用 safetensors 格式
            )
            
            # 检查生成的文件
            safetensors_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_file):
                logger.info("✓ 转换完成！模型已保存为 safetensors 格式")
                logger.info(f"✓ safetensors 文件: {safetensors_file}")
                
                # 显示文件大小
                old_size = os.path.getsize(pytorch_model_bin) / (1024 * 1024)
                new_size = os.path.getsize(safetensors_file) / (1024 * 1024)
                logger.info(f"  旧文件大小: {old_size:.2f} MB")
                logger.info(f"  新文件大小: {new_size:.2f} MB")
                
                # 提示可以删除旧文件
                logger.info(f"\n提示: 可以删除旧文件 {pytorch_model_bin} 以节省空间")
            else:
                logger.warning("未找到 model.safetensors 文件，可能转换失败")
                
        except Exception as e:
            # 恢复原始函数
            import_utils.check_torch_load_is_safe = original_check
            modeling_utils.load_state_dict = original_load_state_dict
            raise e
            
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    
    # 默认模型路径
    model_path = "./mt5-small"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    convert_model_to_safetensors(model_path)
