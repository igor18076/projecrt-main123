"""
Утилиты для работы с GPU (NVIDIA CUDA и AMD ROCm)
"""
import os
import torch
import platform
from typing import Optional, Dict, Any

def detect_gpu_device() -> str:
    """
    Автоматическое определение доступного GPU устройства
    
    Returns:
        str: "cuda", "rocm", "directml", или "cpu"
    """
    # Проверяем CUDA (NVIDIA)
    if torch.cuda.is_available():
        return "cuda"
    
    # Проверяем DirectML (AMD на Windows)
    if _is_directml_available():
        return "directml"
    
    # Проверяем ROCm (AMD на Linux)
    if _is_rocm_available():
        return "rocm"
    
    # Fallback на CPU
    return "cpu"

def _is_rocm_available() -> bool:
    """Проверка доступности AMD ROCm (только для Linux)"""
    try:
        # ROCm доступен только на Linux
        if platform.system() != "Linux":
            return False
            
        # Проверяем переменные окружения ROCm
        if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
            return True
        
        # Проверяем наличие ROCm библиотек
        rocm_paths = [
            "/opt/rocm",
            "/usr/local/rocm",
            "/opt/rocm-*"
        ]
        
        for path in rocm_paths:
            if os.path.exists(path):
                return True
        
        return False
    except:
        return False

def _is_directml_available() -> bool:
    """Проверка доступности DirectML для AMD GPU на Windows"""
    try:
        if platform.system() != "Windows":
            return False
            
        # Проверяем наличие torch-directml
        import torch_directml
        return True
    except ImportError:
        return False

def get_device_info(device: str) -> Dict[str, Any]:
    """
    Получение информации об устройстве
    
    Args:
        device: Тип устройства ("cuda", "rocm", "cpu")
    
    Returns:
        Dict с информацией об устройстве
    """
    info = {
        "device_type": device,
        "available": False,
        "name": "Unknown",
        "memory_total": 0,
        "memory_allocated": 0,
        "memory_free": 0
    }
    
    if device == "cuda" and torch.cuda.is_available():
        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated(0)
        info["memory_free"] = info["memory_total"] - info["memory_allocated"]
    
    elif device == "rocm":
        # Для ROCm пока возвращаем базовую информацию
        info["available"] = _is_rocm_available()
        info["name"] = "AMD GPU (ROCm)"
        # ROCm память пока не поддерживается в PyTorch напрямую
    
    elif device == "directml":
        # Для DirectML
        info["available"] = _is_directml_available()
        info["name"] = "AMD GPU (DirectML)"
        try:
            import torch_directml
            info["device_count"] = torch_directml.device_count()
        except:
            info["device_count"] = 1
    
    elif device == "cpu":
        info["available"] = True
        info["name"] = f"CPU ({platform.processor()})"
        info["memory_total"] = _get_cpu_memory()
    
    return info

def _get_cpu_memory() -> int:
    """Получение информации о RAM"""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return 0

def setup_device(device: str = "auto") -> str:
    """
    Настройка устройства для PyTorch
    
    Args:
        device: "auto", "cuda", "rocm", "directml", или "cpu"
    
    Returns:
        str: Финальное устройство
    """
    if device == "auto":
        device = detect_gpu_device()
    
    # Настройка для DirectML (AMD на Windows)
    if device == "directml":
        try:
            import torch_directml
            # DirectML автоматически настраивается
            print("✅ DirectML доступен для AMD GPU")
            # Возвращаем "cpu" но с настроенным DirectML
            # Фактическое устройство будет получено через get_device_object()
            return "cpu"  # PyTorch будет использовать DirectML через torch_directml.device()
        except ImportError:
            print("⚠️  torch-directml не установлен, переключаемся на CPU")
            device = "cpu"
    
    # Настройка для ROCm (AMD на Linux)
    elif device == "rocm":
        # Устанавливаем переменные окружения для ROCm
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Проверяем доступность ROCm
        if not _is_rocm_available():
            print("⚠️  ROCm не найден, переключаемся на CPU")
            device = "cpu"
    
    # Настройка для CUDA
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA недоступна, переключаемся на CPU")
            device = "cpu"
    
    # Настройка для CPU
    elif device == "cpu":
        # Убеждаемся что используем CPU
        torch.set_num_threads(torch.get_num_threads())
    
    return device

def print_device_info(device: str):
    """Вывод информации об устройстве"""
    info = get_device_info(device)
    
    print(f"Устройство: {info['name']}")
    print(f"Доступно: {'✅' if info['available'] else '❌'}")
    
    if info['memory_total'] > 0:
        memory_gb = info['memory_total'] / (1024**3)
        print(f"Память: {memory_gb:.1f} GB")
    
    if device == "cuda" and info['available']:
        print(f"CUDA версия: {torch.version.cuda}")
    elif device == "rocm" and info['available']:
        print("ROCm: AMD GPU поддержка (Linux)")
    elif device == "directml" and info['available']:
        print("DirectML: AMD GPU поддержка (Windows)")

def get_device_object(device: str):
    """
    Получение объекта устройства для PyTorch
    
    Args:
        device: "cuda", "rocm", "directml", или "cpu"
    
    Returns:
        torch.device: Объект устройства
    """
    if device == "cuda":
        return torch.device("cuda")
    elif device == "rocm":
        return torch.device("cuda")  # ROCm использует CUDA API
    elif device == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def optimize_for_device(device: str):
    """
    Оптимизация PyTorch для конкретного устройства
    
    Args:
        device: Тип устройства
    """
    if device == "cuda":
        # Оптимизации для CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    elif device == "rocm":
        # Оптимизации для ROCm
        # ROCm пока не имеет специальных оптимизаций в PyTorch
        pass
        
    elif device == "directml":
        # Оптимизации для DirectML
        # DirectML автоматически оптимизируется
        pass
        
    elif device == "cpu":
        # Оптимизации для CPU
        torch.set_num_threads(torch.get_num_threads())
        torch.backends.mkl.enabled = True
