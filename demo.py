"""
Демонстрация всех возможностей системы обучения модели классификации патологий КТ
"""
import sys

# Проверка версии Python
if sys.version_info < (3, 10):
    print(f"❌ Требуется Python 3.10+, текущая версия: {sys.version}")
    sys.exit(1)
if sys.version_info >= (3, 12):
    print(f"⚠️  Python 3.12+ может иметь проблемы совместимости")
    print(f"Текущая версия: {sys.version}")
    print("Рекомендуется использовать Python 3.10-3.11")

import os
import time
import torch
import numpy as np
from pathlib import Path

def print_banner():
    """Печать баннера"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║  🏥 СИСТЕМА ОБУЧЕНИЯ МОДЕЛИ КЛАССИФИКАЦИИ ПАТОЛОГИЙ КТ                     ║
    ║                                                                              ║
    ║  Комплексное решение для выявления патологий на компьютерных               ║
    ║  томографиях органов грудной клетки с использованием датасета CT-RATE      ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Проверка системных требований"""
    print("🔍 ПРОВЕРКА СИСТЕМНЫХ ТРЕБОВАНИЙ")
    print("=" * 50)
    
    # Python версия
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (требуется 3.8+)")
        return False
    
    # PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA доступен - {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  CUDA недоступен - будет использоваться CPU")
    except ImportError:
        print("❌ PyTorch не установлен")
        return False
    
    # Другие зависимости
    required_modules = [
        "datasets", "transformers", "matplotlib", "seaborn", 
        "pandas", "numpy", "scikit-learn", "pydicom", "nibabel", "gradio"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Отсутствуют модули: {', '.join(missing_modules)}")
        print("   Установите их командой: pip install -r requirements.txt")
        return False
    
    print("\n✅ Все системные требования выполнены!")
    return True

def demonstrate_model_creation():
    """Демонстрация создания модели"""
    print("\n🤖 ДЕМОНСТРАЦИЯ СОЗДАНИЯ МОДЕЛИ")
    print("=" * 50)
    
    try:
        from model import create_model
        from config import model_config
        
        print("Создание модели ResNet3D-50...")
        model = create_model(model_config.__dict__)
        
        # Информация о модели
        model_info = model.get_model_info()
        print(f"✅ Модель создана успешно!")
        print(f"   Тип: {model_info['model_type']}")
        print(f"   Количество классов: {model_info['num_classes']}")
        print(f"   Размер входа: {model_info['input_size']}")
        print(f"   Параметры: {model_info['total_parameters']:,}")
        print(f"   Обучаемые параметры: {model_info['trainable_parameters']:,}")
        
        # Тест прямого прохода
        print("\nТестирование прямого прохода...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Проверяем DirectML
        if device == "cpu":
            try:
                import torch_directml
                if torch_directml.device_count() > 0:
                    device = "directml"
                    device_obj = torch_directml.device()
                    model = model.to(device_obj)
                else:
                    model = model.to(device)
            except ImportError:
                model = model.to(device)
        else:
            model = model.to(device)
        
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        if device == "directml":
            dummy_input = dummy_input.to(device_obj)
        else:
            dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            start_time = time.time()
            output = model(dummy_input)
            inference_time = time.time() - start_time
        
        print(f"✅ Прямой проход успешен!")
        print(f"   Выходной размер: {output.shape}")
        print(f"   Время инференса: {inference_time*1000:.2f} мс")
        print(f"   Устройство: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Ошибка создания модели: {e}")
        return None

def demonstrate_data_loading():
    """Демонстрация загрузки данных"""
    print("\n📊 ДЕМОНСТРАЦИЯ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 50)
    
    try:
        from data_loader import get_data_loaders
        from config import PATHOLOGY_LABELS
        
        print("Загрузка датасета CT-RATE...")
        print("⚠️  Это может занять некоторое время при первом запуске")
        
        # Создаем загрузчики с маленьким батчем для демонстрации
        train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=0)
        
        print(f"✅ Данные загружены успешно!")
        print(f"   Размер обучающей выборки: {len(train_loader.dataset)}")
        print(f"   Размер валидационной выборки: {len(val_loader.dataset)}")
        print(f"   Количество классов патологий: {len(PATHOLOGY_LABELS)}")
        
        # Показываем пример батча
        print("\nПример батча данных:")
        for batch in train_loader:
            volume = batch['volume']
            labels = batch['labels']
            patient_id = batch['patient_id'][0]
            
            print(f"   Размер объема: {volume.shape}")
            print(f"   Размер меток: {labels.shape}")
            print(f"   ID пациента: {patient_id}")
            
            # Показываем активные патологии
            active_pathologies = []
            for i, label in enumerate(PATHOLOGY_LABELS):
                if labels[0, i].item() > 0.5:
                    active_pathologies.append(label)
            
            if active_pathologies:
                print(f"   Активные патологии: {', '.join(active_pathologies)}")
            else:
                print("   Активные патологии: нет")
            
            break  # Показываем только первый батч
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        print("   Возможные причины:")
        print("   - Нет доступа к интернету для загрузки датасета")
        print("   - Проблемы с доступом к Hugging Face")
        return None, None

def demonstrate_training():
    """Демонстрация обучения (краткая версия)"""
    print("\n🎯 ДЕМОНСТРАЦИЯ ОБУЧЕНИЯ")
    print("=" * 50)
    
    try:
        from trainer import create_trainer
        from config import model_config, training_config
        
        print("Создание тренера...")
        
        # Настраиваем конфигурацию для быстрой демонстрации
        model_config_dict = model_config.__dict__.copy()
        training_config_dict = training_config.__dict__.copy()
        
        training_config_dict.update({
            'batch_size': 2,
            'num_epochs': 3,  # Очень короткое обучение для демонстрации
            'use_wandb': False,
            'save_every_n_epochs': 1,
            'log_every_n_steps': 1
        })
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_trainer(model_config_dict, training_config_dict, device)
        
        print(f"✅ Тренер создан!")
        print(f"   Устройство: {device}")
        print(f"   Размер батча: {training_config_dict['batch_size']}")
        print(f"   Количество эпох: {training_config_dict['num_epochs']}")
        
        print("\nНачинаем краткое обучение (3 эпохи)...")
        print("⚠️  Это займет несколько минут...")
        
        start_time = time.time()
        trainer.train(training_config_dict['num_epochs'])
        training_time = time.time() - start_time
        
        print(f"✅ Обучение завершено за {training_time:.1f} секунд!")
        
        # Показываем результаты
        if trainer.val_losses:
            final_loss = trainer.val_losses[-1]
            print(f"   Финальная потеря: {final_loss:.4f}")
        
        if trainer.val_metrics:
            final_auc = trainer.val_metrics[-1].get('auc_mean', 0)
            print(f"   Финальный AUC: {final_auc:.4f}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return None

def demonstrate_visualization():
    """Демонстрация визуализации"""
    print("\n📈 ДЕМОНСТРАЦИЯ ВИЗУАЛИЗАЦИИ")
    print("=" * 50)
    
    try:
        from visualization import TrainingVisualizer
        
        print("Создание визуализатора...")
        visualizer = TrainingVisualizer()
        
        # Создаем фиктивные данные для демонстрации
        print("Создание демонстрационных данных...")
        
        num_epochs = 10
        train_losses = [0.8 - i * 0.05 + np.random.normal(0, 0.02) for i in range(num_epochs)]
        val_losses = [0.85 - i * 0.04 + np.random.normal(0, 0.03) for i in range(num_epochs)]
        
        train_metrics = []
        val_metrics = []
        
        for epoch in range(num_epochs):
            train_metric = {
                'auc_mean': 0.6 + epoch * 0.02 + np.random.normal(0, 0.01),
                'ap_mean': 0.5 + epoch * 0.015 + np.random.normal(0, 0.01)
            }
            val_metric = {
                'auc_mean': 0.58 + epoch * 0.018 + np.random.normal(0, 0.015),
                'ap_mean': 0.48 + epoch * 0.012 + np.random.normal(0, 0.012)
            }
            
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
        
        print("Создание графиков обучения...")
        curves_path = visualizer.plot_training_curves(
            train_losses, val_losses, train_metrics, val_metrics
        )
        
        print(f"✅ Графики созданы: {curves_path}")
        
        print("Создание графика метрик по классам...")
        class_metrics_path = visualizer.plot_class_metrics(val_metrics[-1], "AUC")
        
        print(f"✅ Метрики по классам созданы: {class_metrics_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка визуализации: {e}")
        return False

def demonstrate_model_export():
    """Демонстрация экспорта модели"""
    print("\n📦 ДЕМОНСТРАЦИЯ ЭКСПОРТА МОДЕЛИ")
    print("=" * 50)
    
    try:
        from model_exporter import export_model
        
        # Проверяем наличие чекпоинта
        checkpoint_path = "./checkpoints/best_model.pth"
        if not os.path.exists(checkpoint_path):
            print("❌ Обученная модель не найдена")
            print("   Сначала запустите демонстрацию обучения")
            return False
        
        print("Экспорт модели в различные форматы...")
        
        exported_files = export_model(
            checkpoint_path=checkpoint_path,
            export_formats=["pytorch", "onnx"],
            model_name="demo_ct_model",
            output_dir="./exported_models"
        )
        
        print("✅ Модель экспортирована!")
        print("Экспортированные файлы:")
        
        for format_name, file_path in exported_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   {format_name}: {file_path} ({size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка экспорта: {e}")
        return False

def demonstrate_web_interface():
    """Демонстрация веб-интерфейса"""
    print("\n🌐 ДЕМОНСТРАЦИЯ ВЕБ-ИНТЕРФЕЙСА")
    print("=" * 50)
    
    try:
        print("Веб-интерфейс готов к запуску!")
        print("\nДля запуска веб-интерфейса выполните:")
        print("   python web_interface.py")
        print("\nПосле запуска откройте браузер и перейдите по адресу:")
        print("   http://localhost:7860")
        print("\nВозможности веб-интерфейса:")
        print("   🎯 Обучение модели с настройкой параметров")
        print("   📊 Оценка обученных моделей")
        print("   🔍 Предсказание на DICOM файлах")
        print("   ℹ️  Информация о системе и патологиях")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка веб-интерфейса: {e}")
        return False

def demonstrate_pathology_labels():
    """Демонстрация поддерживаемых патологий"""
    print("\n🏥 ПОДДЕРЖИВАЕМЫЕ ПАТОЛОГИИ")
    print("=" * 50)
    
    from config import PATHOLOGY_LABELS
    
    print("Модель обучена для выявления следующих патологий:")
    print()
    
    for i, pathology in enumerate(PATHOLOGY_LABELS, 1):
        print(f"{i:2d}. {pathology}")
    
    print(f"\nВсего поддерживается {len(PATHOLOGY_LABELS)} типов патологий")
    print("Модель может предсказывать несколько патологий одновременно")

def main():
    """Главная функция демонстрации"""
    print_banner()
    
    print("🚀 ЗАПУСК ПОЛНОЙ ДЕМОНСТРАЦИИ СИСТЕМЫ")
    print("=" * 60)
    
    # Проверка системных требований
    if not check_system_requirements():
        print("\n❌ Системные требования не выполнены")
        print("   Установите недостающие зависимости: pip install -r requirements.txt")
        return
    
    # Демонстрация создания модели
    model = demonstrate_model_creation()
    if model is None:
        print("\n❌ Не удалось создать модель")
        return
    
    # Демонстрация загрузки данных
    train_loader, val_loader = demonstrate_data_loading()
    if train_loader is None:
        print("\n⚠️  Не удалось загрузить данные, пропускаем обучение")
        print("   Это может быть связано с доступом к датасету CT-RATE")
    else:
        # Демонстрация обучения
        trainer = demonstrate_training()
        if trainer is None:
            print("\n⚠️  Не удалось обучить модель")
        else:
            # Демонстрация экспорта модели
            demonstrate_model_export()
    
    # Демонстрация визуализации
    demonstrate_visualization()
    
    # Демонстрация веб-интерфейса
    demonstrate_web_interface()
    
    # Демонстрация патологий
    demonstrate_pathology_labels()
    
    print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 50)
    
    print("\n📚 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Запустите веб-интерфейс: python web_interface.py")
    print("2. Изучите примеры: python examples.py --all")
    print("3. Прочитайте документацию: README.md")
    print("4. Начните обучение: python main.py train --epochs 50")
    
    print("\n💡 ПОЛЕЗНЫЕ КОМАНДЫ:")
    print("• Быстрая проверка: python quick_start.py --check")
    print("• Статус системы: python quick_start.py --status")
    print("• Веб-интерфейс: python web_interface.py")
    print("• Обучение: python main.py train --epochs 100 --visualize")
    print("• Оценка: python main.py evaluate --checkpoint ./checkpoints/best_model.pth")
    
    print("\n🌟 Спасибо за использование системы!")

if __name__ == "__main__":
    main()

