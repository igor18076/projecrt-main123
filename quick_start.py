"""
Быстрый старт для обучения модели классификации патологий КТ
"""
import os
import sys
import argparse
import torch
from pathlib import Path

def check_setup():
    """Проверка готовности системы"""
    print("🔍 Проверка готовности системы...")
    
    # Проверяем версию Python
    if sys.version_info < (3, 10):
        print(f"❌ Требуется Python 3.10+, текущая версия: {sys.version}")
        return False
    if sys.version_info >= (3, 12):
        print(f"⚠️  Python 3.12+ может иметь проблемы совместимости")
        print(f"Текущая версия: {sys.version}")
        print("Рекомендуется использовать Python 3.10-3.11")
    print(f"✅ Python {sys.version.split()[0]} - OK")
    
    # Проверяем основные файлы
    required_files = [
        "config.py",
        "model.py", 
        "trainer.py",
        "data_loader.py",
        "main.py",
        "web_interface.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
        return False
    
    # Проверяем зависимости
    try:
        import torch
        import datasets
        import gradio
        print("✅ Основные зависимости установлены")
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        return False
    
    # Проверяем GPU
    if torch.cuda.is_available():
        print(f"✅ GPU доступен: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  GPU недоступен - будет использоваться CPU")
    
    return True

def quick_train():
    """Быстрое обучение с базовыми параметрами"""
    print("\n🚀 Запуск быстрого обучения...")
    
    cmd = [
        sys.executable, "main.py", "train",
        "--epochs", "10",
        "--batch-size", "4",
        "--learning-rate", "1e-4",
        "--visualize"
    ]
    
    print(f"Команда: {' '.join(cmd)}")
    
    try:
        import subprocess
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Обучение завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка обучения: {e}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return False

def launch_web_interface():
    """Запуск веб-интерфейса"""
    print("\n🌐 Запуск веб-интерфейса...")
    
    try:
        import subprocess
        cmd = [sys.executable, "web_interface.py"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска веб-интерфейса: {e}")

def demo_prediction():
    """Демонстрация предсказания на тестовых данных"""
    print("\n🔮 Демонстрация предсказания...")
    
    # Проверяем наличие чекпоинта
    checkpoint_path = "./checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("❌ Обученная модель не найдена")
        print("   Сначала запустите обучение: python quick_start.py --train")
        return False
    
    try:
        from model import load_pretrained_model
        import numpy as np
        
        # Загружаем модель
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pretrained_model(checkpoint_path, device)
        
        # Создаем тестовые данные
        dummy_volume = np.random.rand(64, 64, 64).astype(np.float32)
        dummy_tensor = torch.FloatTensor(dummy_volume).unsqueeze(0).unsqueeze(0)
        
        # Перемещаем данные на устройство
        if device == "directml":
            try:
                import torch_directml
                device_obj = torch_directml.device()
                dummy_tensor = dummy_tensor.to(device_obj)
            except ImportError:
                dummy_tensor = dummy_tensor.to(device)
        else:
            dummy_tensor = dummy_tensor.to(device)
        
        # Предсказание
        model.eval()
        with torch.no_grad():
            logits = model(dummy_tensor)
            probabilities = torch.sigmoid(logits)
        
        # Выводим результаты
        print("Результаты предсказания на тестовых данных:")
        print("-" * 50)
        
        from config import PATHOLOGY_LABELS
        for i, pathology in enumerate(PATHOLOGY_LABELS):
            prob = probabilities[0, i].item()
            if prob > 0.1:  # Показываем только вероятности > 10%
                print(f"{pathology}: {prob:.3f}")
        
        print("✅ Демонстрация завершена")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")
        return False

def show_status():
    """Показать статус системы"""
    print("\n📊 Статус системы:")
    print("-" * 30)
    
    # Проверяем чекпоинты
    checkpoint_dir = Path("./checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        print(f"Чекпоинты: {len(checkpoints)} файлов")
        if checkpoints:
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"Последний: {latest.name}")
    else:
        print("Чекпоинты: не найдены")
    
    # Проверяем выходные файлы
    output_dir = Path("./outputs")
    if output_dir.exists():
        plots = list(output_dir.glob("plots/*.png"))
        analysis = list(output_dir.glob("analysis/*.txt"))
        print(f"Графики: {len(plots)} файлов")
        print(f"Анализ: {len(analysis)} файлов")
    else:
        print("Выходные файлы: не найдены")
    
    # Проверяем экспортированные модели
    export_dir = Path("./exported_models")
    if export_dir.exists():
        exported = list(export_dir.glob("*"))
        print(f"Экспортированные модели: {len(exported)} файлов")
    else:
        print("Экспортированные модели: не найдены")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Быстрый старт системы обучения модели КТ")
    
    parser.add_argument("--check", action="store_true", help="Проверить готовность системы")
    parser.add_argument("--train", action="store_true", help="Запустить быстрое обучение")
    parser.add_argument("--web", action="store_true", help="Запустить веб-интерфейс")
    parser.add_argument("--demo", action="store_true", help="Демонстрация предсказания")
    parser.add_argument("--status", action="store_true", help="Показать статус системы")
    parser.add_argument("--all", action="store_true", help="Выполнить все проверки")
    
    args = parser.parse_args()
    
    if not any([args.check, args.train, args.web, args.demo, args.status, args.all]):
        parser.print_help()
        return
    
    print("🏥 Система обучения модели классификации патологий КТ")
    print("=" * 60)
    
    if args.check or args.all:
        if not check_setup():
            print("\n❌ Система не готова")
            sys.exit(1)
        print("\n✅ Система готова к работе")
    
    if args.status or args.all:
        show_status()
    
    if args.train or args.all:
        if not check_setup():
            print("\n❌ Система не готова для обучения")
            return
        
        if quick_train():
            print("\n🎉 Обучение завершено!")
            print("   Проверьте папку ./checkpoints/ для сохраненных моделей")
            print("   Проверьте папку ./outputs/plots/ для графиков")
    
    if args.demo or args.all:
        demo_prediction()
    
    if args.web or args.all:
        if not check_setup():
            print("\n❌ Система не готова для веб-интерфейса")
            return
        
        print("\n🌐 Запуск веб-интерфейса...")
        print("   Откройте браузер и перейдите по адресу: http://localhost:7860")
        print("   Нажмите Ctrl+C для остановки")
        
        launch_web_interface()

if __name__ == "__main__":
    main()

