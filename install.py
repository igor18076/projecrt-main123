#!/usr/bin/env python3
"""
Скрипт для установки всех зависимостей проекта
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Выполнение команды с выводом"""
    print(f"\n🔄 {description}...")
    print(f"Команда: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - успешно!")
        if result.stdout:
            print(f"Вывод: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ошибка!")
        print(f"Ошибка: {e.stderr}")
        return False

def main():
    """Основная функция установки"""
    print("🚀 Установка зависимостей для системы обучения модели классификации патологий КТ")
    print("=" * 80)
    
    # Проверка Python версии
    if sys.version_info < (3, 10):
        print(f"❌ Требуется Python 3.10+, текущая версия: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} - OK")
    
    # Установка PyTorch Nightly
    pytorch_command = "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
    if not run_command(pytorch_command, "Установка PyTorch Nightly с CUDA 13.0"):
        print("⚠️  Попробуем установить CPU версию PyTorch...")
        pytorch_cpu_command = "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu"
        if not run_command(pytorch_cpu_command, "Установка PyTorch Nightly (CPU версия)"):
            print("❌ Не удалось установить PyTorch. Продолжаем с остальными зависимостями...")
    
    # Установка остальных зависимостей
    if not run_command("pip install -r requirements.txt", "Установка остальных зависимостей"):
        print("❌ Не удалось установить зависимости из requirements.txt")
        sys.exit(1)
    
    # Проверка установки
    print("\n🔍 Проверка установки...")
    check_command = '''
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA доступен: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA версия: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name()}')
except ImportError:
    print('❌ PyTorch не установлен')

try:
    import datasets
    print(f'Datasets: {datasets.__version__}')
except ImportError:
    print('❌ Datasets не установлен')

try:
    import gradio
    print(f'Gradio: {gradio.__version__}')
except ImportError:
    print('❌ Gradio не установлен')
"
'''
    
    run_command(check_command, "Проверка установленных библиотек")
    
    print("\n🎉 Установка завершена!")
    print("\n📋 Следующие шаги:")
    print("1. Запустите веб-интерфейс: python web_interface.py")
    print("2. Или проверьте систему: python quick_start.py --check")
    print("3. Или запустите демо: python demo.py")

if __name__ == "__main__":
    main()
