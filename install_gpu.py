#!/usr/bin/env python3
"""
Универсальный скрипт для установки PyTorch с поддержкой GPU
Автоматически определяет платформу и устанавливает соответствующую версию
"""
import subprocess
import sys
import platform
import os

def run_command(command, description):
    """Выполнение команды с выводом"""
    print(f"\n🔄 {description}...")
    print(f"Команда: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} выполнено успешно")
        if result.stdout:
            print(f"Вывод: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при {description}")
        print(f"Код ошибки: {e.returncode}")
        if e.stdout:
            print(f"Вывод: {e.stdout}")
        if e.stderr:
            print(f"Ошибка: {e.stderr}")
        return False

def detect_gpu_type():
    """Определение типа GPU"""
    print("🔍 Определяем тип GPU...")
    
    try:
        import wmi
        c = wmi.WMI()
        
        gpus = []
        for gpu in c.Win32_VideoController():
            if gpu.Name:
                gpus.append(gpu.Name.lower())
                print(f"Найден GPU: {gpu.Name}")
        
        # Определяем тип GPU
        if any('nvidia' in gpu or 'geforce' in gpu or 'rtx' in gpu or 'gtx' in gpu for gpu in gpus):
            return "nvidia"
        elif any('amd' in gpu or 'radeon' in gpu or 'rx' in gpu for gpu in gpus):
            return "amd"
        else:
            return "unknown"
            
    except ImportError:
        print("⚠️  Модуль wmi не установлен, пропускаем проверку GPU")
        return "unknown"
    except Exception as e:
        print(f"⚠️  Ошибка проверки GPU: {e}")
        return "unknown"

def install_pytorch_cuda():
    """Установка PyTorch с поддержкой CUDA"""
    print("\n🚀 Установка PyTorch с поддержкой NVIDIA CUDA...")
    
    commands = [
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install -r requirements.txt"
    ]
    
    success = True
    for command in commands:
        if not run_command(command, f"Выполнение: {command.split()[2] if len(command.split()) > 2 else command}"):
            success = False
            break
    
    return success

def install_pytorch_directml():
    """Установка PyTorch с поддержкой DirectML (AMD на Windows)"""
    print("\n🚀 Установка PyTorch с поддержкой AMD DirectML...")
    
    commands = [
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "pip install torch-directml",
        "pip install -r requirements.txt"
    ]
    
    success = True
    for command in commands:
        if not run_command(command, f"Выполнение: {command.split()[2] if len(command.split()) > 2 else command}"):
            success = False
            break
    
    return success

def install_pytorch_rocm():
    """Установка PyTorch с поддержкой ROCm (AMD на Linux)"""
    print("\n🚀 Установка PyTorch с поддержкой AMD ROCm...")
    
    commands = [
        "python -m pip install --upgrade pip",
        "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6",
        "pip install -r requirements.txt"
    ]
    
    success = True
    for command in commands:
        if not run_command(command, f"Выполнение: {command.split()[2] if len(command.split()) > 2 else command}"):
            success = False
            break
    
    return success

def install_pytorch_cpu():
    """Установка PyTorch только для CPU"""
    print("\n🚀 Установка PyTorch для CPU...")
    
    commands = [
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio",
        "pip install -r requirements.txt"
    ]
    
    success = True
    for command in commands:
        if not run_command(command, f"Выполнение: {command.split()[2] if len(command.split()) > 2 else command}"):
            success = False
            break
    
    return success

def test_installation():
    """Тестирование установки"""
    print("\n🧪 Тестируем установку...")
    
    test_code = """
import torch
import sys

print(f"Python версия: {sys.version}")
print(f"PyTorch версия: {torch.__version__}")

# Проверяем CUDA
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество устройств: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Устройство {i}: {torch.cuda.get_device_name(i)}")

# Проверяем DirectML (Windows)
try:
    import torch_directml
    print(f"DirectML доступен: {torch_directml.device_count() > 0}")
    if torch_directml.device_count() > 0:
        print(f"DirectML устройств: {torch_directml.device_count()}")
except ImportError:
    print("DirectML: Не установлен")

# Проверяем ROCm (Linux)
if hasattr(torch.version, 'hip'):
    print(f"ROCm версия: {torch.version.hip}")
else:
    print("ROCm версия: Не доступна")

# Тестируем создание тензора
x = torch.randn(2, 3)
print(f"Тензор создан: {x.shape}")

# Тестируем операцию
y = torch.randn(2, 3)
z = x + y
print(f"Операция выполнена: {z.shape}")

print("✅ PyTorch работает корректно")
"""
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def main():
    """Главная функция"""
    print("=" * 60)
    print("УНИВЕРСАЛЬНАЯ УСТАНОВКА PYTORCH С ПОДДЕРЖКОЙ GPU")
    print("=" * 60)
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    print(f"✅ Python версия: {sys.version}")
    print(f"✅ Операционная система: {platform.system()} {platform.version()}")
    
    # Определяем платформу и GPU
    system = platform.system()
    gpu_type = "unknown"
    
    if system == "Windows":
        gpu_type = detect_gpu_type()
    elif system == "Linux":
        # На Linux проверяем ROCm
        if os.path.exists("/opt/rocm") or os.environ.get("ROCM_PATH"):
            gpu_type = "amd"
        else:
            gpu_type = "unknown"
    
    print(f"✅ Обнаружен GPU: {gpu_type.upper()}")
    
    # Выбираем метод установки
    if system == "Windows":
        if gpu_type == "nvidia":
            print("\n🎯 Устанавливаем PyTorch с поддержкой NVIDIA CUDA")
            success = install_pytorch_cuda()
        elif gpu_type == "amd":
            print("\n🎯 Устанавливаем PyTorch с поддержкой AMD DirectML")
            success = install_pytorch_directml()
        else:
            print("\n🎯 GPU не определен, устанавливаем PyTorch для CPU")
            success = install_pytorch_cpu()
    elif system == "Linux":
        if gpu_type == "amd":
            print("\n🎯 Устанавливаем PyTorch с поддержкой AMD ROCm")
            success = install_pytorch_rocm()
        else:
            print("\n🎯 Устанавливаем PyTorch для CPU")
            success = install_pytorch_cpu()
    else:
        print("\n🎯 Неподдерживаемая ОС, устанавливаем PyTorch для CPU")
        success = install_pytorch_cpu()
    
    if not success:
        print("❌ Ошибка при установке PyTorch")
        return False
    
    # Тестируем установку
    if not test_installation():
        print("❌ Ошибка при тестировании установки")
        return False
    
    print("\n" + "=" * 60)
    print("✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)
    
    # Показываем инструкции по использованию
    print("\n🚀 Теперь вы можете запустить проект:")
    print("python main.py train --epochs 10")
    print("\n🌐 Или запустить веб-интерфейс:")
    print("python web_interface.py")
    print("\n🧪 Или протестировать GPU:")
    print("python test_amd_gpu.py")
    
    print("\n📋 Поддерживаемые платформы:")
    print("• Windows + NVIDIA GPU → CUDA")
    print("• Windows + AMD GPU → DirectML")
    print("• Linux + AMD GPU → ROCm")
    print("• Любая ОС без GPU → CPU")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
