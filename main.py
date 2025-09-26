"""
Основной скрипт для обучения модели классификации патологий КТ
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
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Optional, List, Dict, Any

from config import model_config, training_config, data_config, PATHOLOGY_LABELS
from trainer import create_trainer, CTTrainer
from visualization import TrainingVisualizer, ModelAnalyzer
from model_exporter import export_model
from data_loader import get_data_loaders, create_dicom_loader, DICOMDataset
from gpu_utils import setup_device, print_device_info, optimize_for_device, get_device_object

def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        training_config.output_dir,
        training_config.checkpoint_dir,
        training_config.log_dir,
        "./exported_models",
        "./outputs/plots",
        "./outputs/analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Создана директория: {directory}")

def save_configs():
    """Сохранение конфигураций"""
    configs = {
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'data_config': data_config.__dict__,
        'pathology_labels': PATHOLOGY_LABELS
    }
    
    config_path = os.path.join(training_config.output_dir, 'configs.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    print(f"Конфигурации сохранены: {config_path}")

def train_model(args):
    """Обучение модели"""
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ КЛАССИФИКАЦИИ ПАТОЛОГИЙ КТ")
    print("=" * 60)
    
    # Настройка
    setup_directories()
    save_configs()
    
    # Определяем устройство
    if args.cpu:
        device = "cpu"
    else:
        device = setup_device(training_config.device)
    
    print(f"Используемое устройство: {device}")
    print_device_info(device)
    optimize_for_device(device)
    
    # Получаем объект устройства для использования в коде
    device_obj = get_device_object(device)
    
    # Создаем тренер
    trainer_config = training_config.__dict__.copy()
    trainer_config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'device': device
    })
    
    model_config_dict = model_config.__dict__.copy()
    if args.model_type:
        model_config_dict['model_type'] = args.model_type
    if args.model_name:
        model_config_dict['model_name'] = args.model_name
    
    trainer = create_trainer(model_config_dict, trainer_config, device)
    
    # Загружаем чекпоинт если указан
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = args.resume
        print(f"Продолжаем обучение с чекпоинта: {resume_checkpoint}")
    
    # Начинаем обучение
    start_time = time.time()
    trainer.train(args.epochs, resume_checkpoint)
    training_time = time.time() - start_time
    
    print(f"\nОбучение завершено за {training_time:.2f} секунд")
    
    # Создаем визуализации
    if args.visualize:
        print("\nСоздание визуализаций...")
        visualizer = TrainingVisualizer()
        
        # Кривые обучения
        curves_path = visualizer.plot_training_curves(
            trainer.train_losses,
            trainer.val_losses,
            trainer.train_metrics,
            trainer.val_metrics
        )
        
        # Метрики по классам
        if trainer.val_metrics:
            class_metrics_path = visualizer.plot_class_metrics(
                trainer.val_metrics[-1], "AUC"
            )
        
        print("Визуализации сохранены в ./outputs/plots/")
    
    # Экспорт модели
    if args.export:
        print("\nЭкспорт модели...")
        best_checkpoint = os.path.join(training_config.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(best_checkpoint):
            exported_files = export_model(
                best_checkpoint,
                export_formats=args.export_formats,
                model_name=args.model_name or "ct_pathology_model"
            )
            print(f"Модель экспортирована: {exported_files}")
        else:
            print("Лучшая модель не найдена для экспорта")

def evaluate_model(args):
    """Оценка модели"""
    print("=" * 60)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 60)
    
    if not args.checkpoint:
        print("Ошибка: Необходимо указать путь к чекпоинту для оценки")
        return
    
    # Загружаем модель
    if args.cpu:
        device = "cpu"
    else:
        device = setup_device(training_config.device)
    
    # Получаем объект устройства
    device_obj = get_device_object(device)
    
    from model import load_pretrained_model
    model = load_pretrained_model(args.checkpoint, device_obj)
    
    # Создаем загрузчики данных
    _, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=training_config.num_workers
    )
    
    # Оценка
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("Выполнение оценки...")
    with torch.no_grad():
        for batch in val_loader:
            volumes = batch['volume'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(volumes)
            predictions, probabilities = model.predict(volumes, threshold=0.5)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Анализ результатов
    analyzer = ModelAnalyzer()
    analysis = analyzer.analyze_predictions(
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    # Сохраняем отчет
    report_path = analyzer.save_analysis_report(analysis)
    
    # Создаем визуализации
    if args.visualize:
        visualizer = TrainingVisualizer()
        
        # ROC кривые
        roc_path = visualizer.plot_roc_curves(
            np.array(all_targets),
            np.array(all_probabilities)
        )
        
        # Precision-Recall кривые
        pr_path = visualizer.plot_precision_recall_curves(
            np.array(all_targets),
            np.array(all_probabilities)
        )
        
        # Матрицы ошибок
        cm_path = visualizer.plot_confusion_matrix(
            np.array(all_targets),
            np.array(all_predictions)
        )
        
        print("Визуализации сохранены в ./outputs/plots/")
    
    # Выводим основные метрики
    print("\nОСНОВНЫЕ МЕТРИКИ:")
    print("-" * 30)
    overall_metrics = analysis['overall_metrics']
    print(f"Точность: {overall_metrics['accuracy']:.4f}")
    print(f"Hamming Loss: {overall_metrics['hamming_loss']:.4f}")
    print(f"Jaccard Score: {overall_metrics['jaccard_score']:.4f}")
    print(f"Средний AUC: {overall_metrics['mean_auc']:.4f}")
    print(f"Средний AP: {overall_metrics['mean_ap']:.4f}")

def predict_dicom(args):
    """Предсказание на DICOM файлах"""
    print("=" * 60)
    print("ПРЕДСКАЗАНИЕ НА DICOM ФАЙЛАХ")
    print("=" * 60)
    
    if not args.checkpoint:
        print("Ошибка: Необходимо указать путь к чекпоинту")
        return
    
    if not args.dicom_path:
        print("Ошибка: Необходимо указать путь к DICOM файлам")
        return
    
    # Загружаем модель
    if args.cpu:
        device = "cpu"
    else:
        device = setup_device(training_config.device)
    
    # Получаем объект устройства
    device_obj = get_device_object(device)
    
    from model import load_pretrained_model
    model = load_pretrained_model(args.checkpoint, device_obj)
    
    # Загружаем DICOM файлы
    dicom_paths = []
    if os.path.isfile(args.dicom_path):
        dicom_paths = [args.dicom_path]
    else:
        for file in os.listdir(args.dicom_path):
            if file.lower().endswith(('.dcm', '.dicom')):
                dicom_paths.append(os.path.join(args.dicom_path, file))
    
    if not dicom_paths:
        print("DICOM файлы не найдены")
        return
    
    print(f"Найдено {len(dicom_paths)} DICOM файлов")
    
    # Создаем датасет
    dicom_dataset = DICOMDataset(dicom_paths)
    dicom_loader = create_dicom_loader(dicom_paths, batch_size=1)
    
    # Предсказания
    model.eval()
    results = []
    
    print("Выполнение предсказаний...")
    with torch.no_grad():
        for batch in dicom_loader:
            volumes = batch['volume'].to(device)
            file_path = batch['file_path'][0]
            
            outputs = model(volumes)
            predictions, probabilities = model.predict(volumes, threshold=0.5)
            
            # Формируем результат
            result = {
                'file_path': file_path,
                'predictions': {},
                'probabilities': {}
            }
            
            for i, label in enumerate(PATHOLOGY_LABELS):
                result['predictions'][label] = bool(predictions[0, i].item())
                result['probabilities'][label] = float(probabilities[0, i].item())
            
            results.append(result)
            
            # Выводим результат
            print(f"\nФайл: {os.path.basename(file_path)}")
            print("Предсказанные патологии:")
            for label, prob in result['probabilities'].items():
                if prob > 0.1:  # Показываем только вероятности > 10%
                    print(f"  {label}: {prob:.3f}")
    
    # Сохраняем результаты
    if args.output:
        results_path = args.output
    else:
        results_path = os.path.join(training_config.output_dir, "dicom_predictions.json")
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены: {results_path}")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Обучение модели классификации патологий КТ")
    
    # Основные команды
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда обучения
    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Размер батча')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Скорость обучения')
    train_parser.add_argument('--model-type', type=str, choices=['resnet3d', 'efficientnet3d'], 
                           default='resnet3d', help='Тип модели')
    train_parser.add_argument('--model-name', type=str, default='resnet3d_50', help='Название модели')
    train_parser.add_argument('--resume', type=str, help='Путь к чекпоинту для продолжения обучения')
    train_parser.add_argument('--visualize', action='store_true', help='Создавать визуализации')
    train_parser.add_argument('--export', action='store_true', help='Экспортировать модель')
    train_parser.add_argument('--export-formats', nargs='+', 
                           choices=['pytorch', 'onnx', 'torchscript'],
                           default=['pytorch', 'onnx'], help='Форматы экспорта')
    train_parser.add_argument('--cpu', action='store_true', help='Использовать CPU вместо GPU')
    
    # Команда оценки
    eval_parser = subparsers.add_parser('evaluate', help='Оценка модели')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Путь к чекпоинту')
    eval_parser.add_argument('--batch-size', type=int, default=8, help='Размер батча')
    eval_parser.add_argument('--visualize', action='store_true', help='Создавать визуализации')
    eval_parser.add_argument('--cpu', action='store_true', help='Использовать CPU вместо GPU')
    
    # Команда предсказания
    predict_parser = subparsers.add_parser('predict', help='Предсказание на DICOM файлах')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Путь к чекпоинту')
    predict_parser.add_argument('--dicom-path', type=str, required=True, help='Путь к DICOM файлам')
    predict_parser.add_argument('--output', type=str, help='Путь для сохранения результатов')
    predict_parser.add_argument('--cpu', action='store_true', help='Использовать CPU вместо GPU')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'predict':
        predict_dicom(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

