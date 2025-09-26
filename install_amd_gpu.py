#!/usr/bin/env python3
"""
Скрипт для установки PyTorch с поддержкой AMD ROCm
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

def check_rocm_installation():
    """Проверка установки ROCm"""
    print("🔍 Проверяем установку ROCm...")
    
    # Проверяем переменные окружения
    rocm_path = os.environ.get("ROCM_PATH")
    hip_path = os.environ.get("HIP_PATH")
    
    if rocm_path or hip_path:
        print(f"✅ ROCm найден в переменных окружения")
        print(f"ROCM_PATH: {rocm_path}")
        print(f"HIP_PATH: {hip_path}")
        return True
    
    # Проверяем стандартные пути
    rocm_paths = [
        "/opt/rocm",
        "/usr/local/rocm",
        "/opt/rocm-*"
    ]
    
    for path in rocm_paths:
        if os.path.exists(path):
            print(f"✅ ROCm найден по пути: {path}")
            return True
    
    print("❌ ROCm не найден")
    print("Пожалуйста, установите ROCm перед продолжением:")
    print("https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
    return False

def install_pytorch_rocm():
    """Установка PyTorch с поддержкой ROCm"""
    print("\n🚀 Установка PyTorch с поддержкой AMD ROCm...")
    
    # Определяем версию ROCm
    rocm_version = "5.6"  # По умолчанию
    
    # Пытаемся определить версию ROCm из системы
    try:
        result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Извлекаем версию из вывода
            output = result.stdout
            if "rocm" in output.lower():
                # Простое извлечение версии
                import re
                version_match = re.search(r'(\d+\.\d+)', output)
                if version_match:
                    rocm_version = version_match.group(1)
                    print(f"Обнаружена версия ROCm: {rocm_version}")
    except:
        print(f"Используем версию ROCm по умолчанию: {rocm_version}")
    
    # Команды для установки
    commands = [
        # Обновляем pip
        "python -m pip install --upgrade pip",
        
        # Устанавливаем PyTorch с ROCm
        f"pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm{rocm_version}",
        
        # Устанавливаем дополнительные зависимости
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

# Проверяем доступность ROCm
if hasattr(torch.version, 'hip'):
    print(f"ROCm версия: {torch.version.hip}")
    print("✅ ROCm поддержка доступна")
else:
    print("❌ ROCm поддержка недоступна")

# Проверяем устройства
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Тестируем создание тензора
try:
    x = torch.randn(2, 3)
    print(f"CPU тензор создан: {x.shape}")
    
    # Пытаемся использовать ROCm
    if hasattr(torch.version, 'hip'):
        try:
            x_rocm = x.to('cuda')  # ROCm использует 'cuda' как устройство
            print(f"ROCm тензор создан: {x_rocm.shape}")
            print("✅ ROCm работает корректно")
        except Exception as e:
            print(f"❌ Ошибка при работе с ROCm: {e}")
    
except Exception as e:
    print(f"❌ Ошибка при создании тензора: {e}")
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
    print("УСТАНОВКА PYTORCH С ПОДДЕРЖКОЙ AMD ROCm")
    print("=" * 60)
    
    # Проверяем операционную систему
    if platform.system() != "Linux":
        print("❌ Этот скрипт предназначен для Linux систем")
        print("Для Windows/macOS используйте стандартную установку PyTorch")
        return False
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    print(f"✅ Python версия: {sys.version}")
    print(f"✅ ОС: {platform.system()} {platform.release()}")
    
    # Проверяем ROCm
    if not check_rocm_installation():
        print("\n⚠️  ROCm не найден, но продолжаем установку...")
        print("PyTorch будет установлен с поддержкой ROCm, но может не работать без ROCm")
    
    # Устанавливаем PyTorch
    if not install_pytorch_rocm():
        print("❌ Ошибка при установке PyTorch")
        return False
    
    # Тестируем установку
    if not test_installation():
        print("❌ Ошибка при тестировании установки")
        return False
    
    print("\n" + "=" * 60)
    print("✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)
    print("\nТеперь вы можете запустить проект с AMD GPU:")
    print("python main.py train --epochs 10")
    print("\nДля проверки устройства используйте:")
    print("python -c \"from gpu_utils import detect_gpu_device, print_device_info; device = detect_gpu_device(); print_device_info(device)\"")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
