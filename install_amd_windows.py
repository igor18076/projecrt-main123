#!/usr/bin/env python3
"""
Скрипт для установки PyTorch с поддержкой AMD GPU на Windows через DirectML
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

def check_windows_version():
    """Проверка версии Windows"""
    print("🔍 Проверяем версию Windows...")
    
    if platform.system() != "Windows":
        print("❌ Этот скрипт предназначен для Windows")
        return False
    
    version = platform.version()
    print(f"✅ Windows версия: {version}")
    
    # DirectML требует Windows 10 версии 1903 или выше
    major, minor, build = version.split('.')
    if int(build) >= 18362:  # Windows 10 1903
        print("✅ Версия Windows поддерживает DirectML")
        return True
    else:
        print("⚠️  Рекомендуется Windows 10 версии 1903 или выше для лучшей поддержки DirectML")
        return True  # Все равно пробуем

def check_amd_gpu():
    """Проверка наличия AMD GPU"""
    print("🔍 Проверяем наличие AMD GPU...")
    
    try:
        import wmi
        c = wmi.WMI()
        
        gpus = []
        for gpu in c.Win32_VideoController():
            if gpu.Name:
                gpus.append(gpu.Name)
                print(f"Найден GPU: {gpu.Name}")
        
        amd_gpus = [gpu for gpu in gpus if 'amd' in gpu.lower() or 'radeon' in gpu.lower()]
        
        if amd_gpus:
            print(f"✅ Найдены AMD GPU: {amd_gpus}")
            return True
        else:
            print("⚠️  AMD GPU не обнаружены, но DirectML может работать с другими GPU")
            return True
            
    except ImportError:
        print("⚠️  Модуль wmi не установлен, пропускаем проверку GPU")
        return True
    except Exception as e:
        print(f"⚠️  Ошибка проверки GPU: {e}")
        return True

def install_pytorch_directml():
    """Установка PyTorch с поддержкой DirectML"""
    print("\n🚀 Установка PyTorch с поддержкой AMD DirectML...")
    
    commands = [
        # Обновляем pip
        "python -m pip install --upgrade pip",
        
        # Устанавливаем PyTorch (CPU версия, совместимая с DirectML)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        
        # Устанавливаем DirectML
        "pip install torch-directml",
        
        # Устанавливаем остальные зависимости
        "pip install -r requirements.txt"
    ]
    
    success = True
    for command in commands:
        if not run_command(command, f"Выполнение: {command.split()[2] if len(command.split()) > 2 else command}"):
            success = False
            break
    
    return success

def test_directml_installation():
    """Тестирование установки DirectML"""
    print("\n🧪 Тестируем установку DirectML...")
    
    test_code = """
import torch
import sys

print(f"Python версия: {sys.version}")
print(f"PyTorch версия: {torch.__version__}")

# Проверяем DirectML
try:
    import torch_directml
    print("✅ torch-directml установлен")
    
    # Проверяем доступность DirectML
    device = torch_directml.device()
    print(f"DirectML устройство: {device}")
    
    # Тестируем создание тензора
    x = torch.randn(2, 3).to(device)
    print(f"DirectML тензор создан: {x.shape}")
    
    # Тестируем операцию
    y = torch.randn(2, 3).to(device)
    z = x + y
    print(f"DirectML операция выполнена: {z.shape}")
    
    print("✅ DirectML работает корректно")
    
except ImportError as e:
    print(f"❌ torch-directml не установлен: {e}")
    return False
except Exception as e:
    print(f"❌ Ошибка при тестировании DirectML: {e}")
    return False

# Проверяем CUDA (должна быть недоступна с CPU версией PyTorch)
print(f"CUDA доступна: {torch.cuda.is_available()}")

return True
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
    print("УСТАНОВКА PYTORCH С ПОДДЕРЖКОЙ AMD GPU НА WINDOWS")
    print("=" * 60)
    
    # Проверяем Windows
    if not check_windows_version():
        return False
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    print(f"✅ Python версия: {sys.version}")
    
    # Проверяем AMD GPU
    check_amd_gpu()
    
    # Устанавливаем PyTorch с DirectML
    if not install_pytorch_directml():
        print("❌ Ошибка при установке PyTorch с DirectML")
        return False
    
    # Тестируем установку
    if not test_directml_installation():
        print("❌ Ошибка при тестировании установки")
        return False
    
    print("\n" + "=" * 60)
    print("✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)
    print("\nТеперь вы можете запустить проект с AMD GPU на Windows:")
    print("python main.py train --epochs 10")
    print("\nДля проверки устройства используйте:")
    print("python -c \"from gpu_utils import detect_gpu_device, print_device_info; device = detect_gpu_device(); print_device_info(device)\"")
    
    print("\n📋 Важные замечания:")
    print("• DirectML работает только на Windows")
    print("• Производительность может отличаться от CUDA/ROCm")
    print("• Для лучшей производительности используйте Linux с ROCm")
    print("• Убедитесь что у вас установлены последние драйверы AMD")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
