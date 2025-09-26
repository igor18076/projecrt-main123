"""
Скрипт установки и настройки системы обучения модели классификации патологий КТ
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 10):
        print("❌ Требуется Python 3.10 или выше")
        print(f"Текущая версия: {sys.version}")
        return False
    if sys.version_info >= (3, 12):
        print("⚠️  Python 3.12+ может иметь проблемы совместимости")
        print(f"Текущая версия: {sys.version}")
        print("Рекомендуется использовать Python 3.10-3.11")
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def check_cuda():
    """Проверка доступности CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA доступен - {torch.cuda.get_device_name()}")
            print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA недоступен - будет использоваться CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch не установлен - CUDA проверка пропущена")
        return False

def install_requirements():
    """Установка зависимостей"""
    print("\n📦 Установка зависимостей...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    print("\n📁 Создание директорий...")
    
    directories = [
        "./data",
        "./outputs",
        "./outputs/plots",
        "./outputs/analysis",
        "./checkpoints",
        "./logs",
        "./exported_models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")

def test_imports():
    """Тестирование импортов"""
    print("\n🧪 Тестирование импортов...")
    
    required_modules = [
        "torch",
        "torchvision", 
        "datasets",
        "transformers",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "scikit-learn",
        "pydicom",
        "nibabel",
        "gradio"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Не удалось импортировать: {', '.join(failed_imports)}")
        return False
    
    print("✅ Все модули успешно импортированы")
    return True

def test_model_creation():
    """Тестирование создания модели"""
    print("\n🤖 Тестирование создания модели...")
    
    try:
        from model import create_model
        from config import model_config
        
        model = create_model(model_config.__dict__)
        print("✅ Модель создана успешно")
        
        # Тестируем прямой проход
        import torch
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        output = model(dummy_input)
        print(f"✅ Прямой проход успешен - выходной размер: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка создания модели: {e}")
        return False

def test_data_loading():
    """Тестирование загрузки данных"""
    print("\n📊 Тестирование загрузки данных...")
    
    try:
        from data_loader import get_data_loaders
        
        # Пробуем загрузить небольшой батч
        train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=0)
        
        # Тестируем один батч
        for batch in train_loader:
            print(f"✅ Данные загружены - размер батча: {batch['volume'].shape}")
            break
            
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        print("   Это может быть связано с доступом к датасету CT-RATE")
        return False

def create_config_file():
    """Создание файла конфигурации"""
    print("\n⚙️  Создание конфигурации...")
    
    config_content = """# Конфигурация для обучения модели классификации патологий КТ
# Этот файл можно редактировать для настройки параметров

# Настройки модели
MODEL_TYPE = "resnet3d"
MODEL_NAME = "resnet3d_50"
NUM_CLASSES = 14
INPUT_SIZE = (64, 64, 64)
DROPOUT_RATE = 0.5

# Настройки обучения
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-5
PATIENCE = 10

# Настройки GPU
USE_GPU = True
NUM_WORKERS = 4

# Настройки логирования
USE_WANDB = False
WANDB_PROJECT = "ct-pathology-detection"

# Пути
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
"""
    
    with open("user_config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Файл конфигурации создан: user_config.py")

def print_system_info():
    """Вывод информации о системе"""
    print("\n💻 Информация о системе:")
    print(f"   ОС: {platform.system()} {platform.release()}")
    print(f"   Архитектура: {platform.machine()}")
    print(f"   Python: {sys.version}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   cuDNN: {torch.backends.cudnn.version()}")
    except ImportError:
        print("   PyTorch: не установлен")

def print_usage_instructions():
    """Вывод инструкций по использованию"""
    print("\n🚀 Инструкции по использованию:")
    print("\n1. Обучение модели:")
    print("   python main.py train --epochs 50 --batch-size 8")
    
    print("\n2. Веб-интерфейс:")
    print("   python web_interface.py")
    
    print("\n3. Оценка модели:")
    print("   python main.py evaluate --checkpoint ./checkpoints/best_model.pth")
    
    print("\n4. Предсказание на DICOM файлах:")
    print("   python main.py predict --checkpoint ./checkpoints/best_model.pth --dicom-path ./data/")
    
    print("\n📚 Дополнительная документация:")
    print("   README.md - подробная документация")
    print("   config.py - настройки по умолчанию")
    print("   user_config.py - пользовательские настройки")

def main():
    """Основная функция установки"""
    print("🏥 Установка системы обучения модели классификации патологий КТ")
    print("=" * 70)
    
    # Проверки
    if not check_python_version():
        sys.exit(1)
    
    print_system_info()
    
    # Установка
    if not install_requirements():
        print("\n❌ Установка прервана из-за ошибок")
        sys.exit(1)
    
    create_directories()
    create_config_file()
    
    # Тестирование
    if not test_imports():
        print("\n❌ Установка завершена с ошибками")
        sys.exit(1)
    
    check_cuda()
    
    if not test_model_creation():
        print("\n⚠️  Предупреждение: Модель не создается корректно")
    
    if not test_data_loading():
        print("\n⚠️  Предупреждение: Проблемы с загрузкой данных")
    
    print("\n✅ Установка завершена успешно!")
    print_usage_instructions()
    
    print("\n🎉 Система готова к использованию!")
    print("   Запустите 'python web_interface.py' для веб-интерфейса")

if __name__ == "__main__":
    main()

