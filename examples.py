"""
Примеры использования системы обучения модели классификации патологий КТ
"""
import torch
import numpy as np
import os
from pathlib import Path

# Пример 1: Базовое обучение модели
def example_basic_training():
    """Пример базового обучения модели"""
    print("📚 Пример 1: Базовое обучение модели")
    print("-" * 40)
    
    from trainer import create_trainer
    from config import model_config, training_config
    
    # Создаем конфигурацию
    model_config_dict = model_config.__dict__.copy()
    training_config_dict = training_config.__dict__.copy()
    
    # Настраиваем параметры для быстрого тестирования
    training_config_dict.update({
        'batch_size': 4,
        'num_epochs': 5,
        'use_wandb': False,  # Отключаем wandb для примера
        'save_every_n_epochs': 1
    })
    
    # Создаем тренер
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = create_trainer(model_config_dict, training_config_dict, device)
    
    print(f"Устройство: {device}")
    print(f"Количество классов: {trainer.model.num_classes}")
    print(f"Размер батча: {training_config_dict['batch_size']}")
    print(f"Количество эпох: {training_config_dict['num_epochs']}")
    
    # Начинаем обучение
    trainer.train(training_config_dict['num_epochs'])
    
    print("✅ Обучение завершено!")
    return trainer

# Пример 2: Загрузка и использование обученной модели
def example_model_inference():
    """Пример использования обученной модели"""
    print("\n📚 Пример 2: Использование обученной модели")
    print("-" * 40)
    
    from model import load_pretrained_model
    from config import PATHOLOGY_LABELS
    
    # Путь к чекпоинту
    checkpoint_path = "./checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print("❌ Обученная модель не найдена")
        print("   Сначала запустите обучение: python examples.py --train")
        return
    
    # Загружаем модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_model(checkpoint_path, device)
    model.eval()
    
    print(f"Модель загружена на устройство: {device}")
    
    # Создаем тестовые данные
    batch_size = 2
    dummy_volumes = torch.randn(batch_size, 1, 64, 64, 64)
    
    # Перемещаем данные на устройство
    if device == "directml":
        try:
            import torch_directml
            device_obj = torch_directml.device()
            dummy_volumes = dummy_volumes.to(device_obj)
        except ImportError:
            dummy_volumes = dummy_volumes.to(device)
    else:
        dummy_volumes = dummy_volumes.to(device)
    
    # Предсказание
    with torch.no_grad():
        logits = model(dummy_volumes)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
    
    # Выводим результаты
    for i in range(batch_size):
        print(f"\nОбразец {i+1}:")
        print("Предсказанные патологии:")
        
        for j, pathology in enumerate(PATHOLOGY_LABELS):
            prob = probabilities[i, j].item()
            pred = predictions[i, j].item()
            
            if prob > 0.1:  # Показываем только вероятности > 10%
                status = "Да" if pred else "Нет"
                print(f"  {pathology}: {prob:.3f} ({status})")
    
    print("✅ Предсказание завершено!")
    return model

# Пример 3: Работа с DICOM файлами
def example_dicom_processing():
    """Пример работы с DICOM файлами"""
    print("\n📚 Пример 3: Обработка DICOM файлов")
    print("-" * 40)
    
    from data_loader import DICOMDataset, create_dicom_loader
    import pydicom
    
    # Создаем тестовые DICOM данные
    test_dicom_dir = "./test_dicom"
    os.makedirs(test_dicom_dir, exist_ok=True)
    
    # Создаем фиктивные DICOM файлы для демонстрации
    print("Создание тестовых DICOM файлов...")
    
    for i in range(3):
        # Создаем простой DICOM файл
        ds = pydicom.Dataset()
        ds.PatientName = f"TestPatient_{i}"
        ds.PatientID = f"TEST{i:03d}"
        ds.Modality = "CT"
        ds.SeriesDescription = "Test CT Series"
        
        # Создаем тестовые пиксельные данные
        pixel_array = np.random.randint(0, 4096, (64, 64), dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()
        ds.Rows = 64
        ds.Columns = 64
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Сохраняем файл
        dicom_path = os.path.join(test_dicom_dir, f"test_{i:03d}.dcm")
        ds.save_as(dicom_path)
    
    print(f"Создано {len(os.listdir(test_dicom_dir))} тестовых DICOM файлов")
    
    # Загружаем DICOM файлы
    dicom_files = [os.path.join(test_dicom_dir, f) for f in os.listdir(test_dicom_dir)]
    dicom_dataset = DICOMDataset(dicom_files)
    dicom_loader = create_dicom_loader(dicom_files, batch_size=1)
    
    print(f"Загружено {len(dicom_dataset)} DICOM файлов")
    
    # Обрабатываем файлы
    for batch in dicom_loader:
        volume = batch['volume']
        file_path = batch['file_path'][0]
        
        print(f"Обработан файл: {os.path.basename(file_path)}")
        print(f"Размер объема: {volume.shape}")
        print(f"Диапазон значений: [{volume.min():.3f}, {volume.max():.3f}]")
        break  # Обрабатываем только первый файл для примера
    
    print("✅ Обработка DICOM файлов завершена!")
    
    # Очищаем тестовые файлы
    import shutil
    shutil.rmtree(test_dicom_dir)
    print("Тестовые файлы удалены")

# Пример 4: Визуализация результатов
def example_visualization():
    """Пример создания визуализаций"""
    print("\n📚 Пример 4: Создание визуализаций")
    print("-" * 40)
    
    from visualization import TrainingVisualizer, ModelAnalyzer
    
    # Создаем фиктивные данные для демонстрации
    num_epochs = 20
    train_losses = [0.8 - i * 0.03 + np.random.normal(0, 0.02) for i in range(num_epochs)]
    val_losses = [0.85 - i * 0.025 + np.random.normal(0, 0.03) for i in range(num_epochs)]
    
    train_metrics = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        # Создаем фиктивные метрики
        train_metric = {
            'auc_mean': 0.6 + epoch * 0.015 + np.random.normal(0, 0.01),
            'ap_mean': 0.5 + epoch * 0.012 + np.random.normal(0, 0.01)
        }
        val_metric = {
            'auc_mean': 0.58 + epoch * 0.014 + np.random.normal(0, 0.015),
            'ap_mean': 0.48 + epoch * 0.011 + np.random.normal(0, 0.012)
        }
        
        # Добавляем метрики по классам
        for label in PATHOLOGY_LABELS[:5]:  # Только первые 5 классов для примера
            train_metric[f'f1_{label}'] = 0.4 + epoch * 0.01 + np.random.normal(0, 0.05)
            val_metric[f'f1_{label}'] = 0.38 + epoch * 0.009 + np.random.normal(0, 0.06)
        
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
    
    # Создаем визуализатор
    visualizer = TrainingVisualizer()
    
    # Создаем графики
    print("Создание графиков обучения...")
    curves_path = visualizer.plot_training_curves(
        train_losses, val_losses, train_metrics, val_metrics
    )
    print(f"Графики сохранены: {curves_path}")
    
    # Создаем график метрик по классам
    print("Создание графика метрик по классам...")
    class_metrics_path = visualizer.plot_class_metrics(val_metrics[-1], "AUC")
    print(f"Метрики по классам сохранены: {class_metrics_path}")
    
    print("✅ Визуализация завершена!")

# Пример 5: Экспорт модели
def example_model_export():
    """Пример экспорта модели"""
    print("\n📚 Пример 5: Экспорт модели")
    print("-" * 40)
    
    from model_exporter import export_model
    
    # Путь к чекпоинту
    checkpoint_path = "./checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print("❌ Обученная модель не найдена")
        print("   Сначала запустите обучение: python examples.py --train")
        return
    
    print("Экспорт модели в различные форматы...")
    
    # Экспортируем модель
    exported_files = export_model(
        checkpoint_path=checkpoint_path,
        export_formats=["pytorch", "onnx", "torchscript"],
        model_name="example_ct_model",
        output_dir="./exported_models"
    )
    
    print("Экспортированные файлы:")
    for format_name, file_path in exported_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {format_name}: {file_path} ({size:.1f} MB)")
    
    print("✅ Экспорт модели завершен!")

# Пример 6: Полный пайплайн
def example_full_pipeline():
    """Пример полного пайплайна от обучения до предсказания"""
    print("\n📚 Пример 6: Полный пайплайн")
    print("-" * 40)
    
    print("1. Обучение модели...")
    trainer = example_basic_training()
    
    print("\n2. Использование модели...")
    model = example_model_inference()
    
    print("\n3. Создание визуализаций...")
    example_visualization()
    
    print("\n4. Экспорт модели...")
    example_model_export()
    
    print("\n🎉 Полный пайплайн завершен!")

def main():
    """Главная функция для запуска примеров"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Примеры использования системы")
    
    parser.add_argument("--train", action="store_true", help="Пример обучения")
    parser.add_argument("--inference", action="store_true", help="Пример предсказания")
    parser.add_argument("--dicom", action="store_true", help="Пример работы с DICOM")
    parser.add_argument("--visualize", action="store_true", help="Пример визуализации")
    parser.add_argument("--export", action="store_true", help="Пример экспорта")
    parser.add_argument("--all", action="store_true", help="Все примеры")
    
    args = parser.parse_args()
    
    if not any([args.train, args.inference, args.dicom, args.visualize, args.export, args.all]):
        parser.print_help()
        return
    
    print("🏥 Примеры использования системы обучения модели КТ")
    print("=" * 60)
    
    try:
        if args.train or args.all:
            example_basic_training()
        
        if args.inference or args.all:
            example_model_inference()
        
        if args.dicom or args.all:
            example_dicom_processing()
        
        if args.visualize or args.all:
            example_visualization()
        
        if args.export or args.all:
            example_model_export()
        
        print("\n✅ Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

