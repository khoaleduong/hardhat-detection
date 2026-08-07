import torch

def get_gpu_info(device_index: int = 0):
    if not torch.cuda.is_available():
        return {
            "device": "CPU",
            "available": False,
        }

    props = torch.cuda.get_device_properties(device_index)
    free_mem, total_mem = torch.cuda.mem_get_info(device_index)

    return {
        "device": torch.cuda.get_device_name(device_index),
        "available": True,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),  # Tổng số GPU trên máy
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "free_memory_gb": round(free_mem / (1024**3), 2),  # VRAM còn trống
        "allocated_memory_gb": round(
            (total_mem - free_mem) / (1024**3), 2
        ),  # VRAM đã dùng
        "multi_processors": props.multi_processor_count,
    }

def estimate_vram_usage(
    model,
    batch_size: int = 32,
    precision_bytes: int = 4,  # 4 cho float32, 2 cho fp16 / bf16
    optimizer_type: str = "adam",
):
    """Ước tính dung lượng VRAM tối thiểu (GB) cần để train mô hình."""
    # 1. Đếm tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())

    # 2. Bộ nhớ cho Trọng số (Weights) & Gradients
    weights_mem = total_params * precision_bytes
    gradients_mem = total_params * precision_bytes

    # 3. Bộ nhớ cho Optimizer States (Adam lưu 2 trạng thái dạng FP32)
    if optimizer_type.lower() in ["adam", "adamw"]:
        optimizer_mem = total_params * 4 * 2
    elif optimizer_type.lower() == "sgd":
        optimizer_mem = 0
    else:
        optimizer_mem = total_params * 4  # Mặc định ước tính 1 state

    # Tổng bộ nhớ cố định (Static Memory)
    static_mem_bytes = weights_mem + gradients_mem + optimizer_mem

    # Đổi sang GB
    static_mem_gb = static_mem_bytes / (1024**3)

    # Ước tính thô bao gồm cả Activations & Cuda Overhead (thường nhân hệ số 1.2 - 1.5)
    # Lưu ý: Activations phụ thuộc lớn vào Batch Size
    estimated_total_gb = static_mem_gb * 1.3

    return {
        "total_params_M": f"{total_params / 1e6:.1f}M",
        "static_vram_gb": round(static_mem_gb, 2),
        "estimated_min_vram_gb": round(estimated_total_gb, 2),
    }