"""
Веб-интерфейс для управления обучением модели
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

import gradio as gr
import torch
import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import PATHOLOGY_LABELS, model_config, training_config
from trainer import create_trainer
from visualization import TrainingVisualizer, ModelAnalyzer
from model_exporter import export_model
from data_loader import create_dicom_loader, DICOMDataset
from gpu_utils import detect_gpu_device, print_device_info, setup_device, optimize_for_device, get_device_object
from archive_processor import analyze_dicom_archive

class TrainingManager:
    """Менеджер обучения с поддержкой прерывания"""
    
    def __init__(self):
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.training_progress = {"epoch": 0, "loss": 0.0, "status": "Готов к обучению"}
        self.training_logs = []
    
    def start_training(self, config: Dict[str, Any]) -> str:
        """Запуск обучения в отдельном потоке"""
        if self.is_training:
            return "Обучение уже запущено"
        
        self.is_training = True
        self.training_progress = {"epoch": 0, "loss": 0.0, "status": "Инициализация..."}
        
        def train_worker():
            try:
                # Определяем устройство автоматически
                device = setup_device(config['training_config'].get('device', 'auto'))
                self.training_progress["status"] = f"Используется устройство: {device}"
                
                # Получаем объект устройства
                device_obj = get_device_object(device)
                
                # Создаем тренер
                self.trainer = create_trainer(config['model_config'], config['training_config'], device_obj)
                
                # Начинаем обучение
                self.trainer.train(config['training_config']['num_epochs'])
                
                self.training_progress["status"] = "Обучение завершено"
                self.is_training = False
                
            except Exception as e:
                self.training_progress["status"] = f"Ошибка: {str(e)}"
                self.is_training = False
        
        self.training_thread = threading.Thread(target=train_worker)
        self.training_thread.start()
        
        return "Обучение запущено"
    
    def stop_training(self) -> str:
        """Остановка обучения"""
        if not self.is_training:
            return "Обучение не запущено"
        
        self.is_training = False
        self.training_progress["status"] = "Остановка обучения..."
        
        return "Обучение остановлено"
    
    def get_progress(self) -> Dict[str, Any]:
        """Получение прогресса обучения"""
        if self.trainer and hasattr(self.trainer, 'current_epoch'):
            self.training_progress["epoch"] = self.trainer.current_epoch
            if self.trainer.val_losses:
                self.training_progress["loss"] = self.trainer.val_losses[-1]
        
        return self.training_progress

# Глобальный менеджер обучения
training_manager = TrainingManager()

def create_training_config(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_type: str,
    model_name: str,
    use_gpu: bool
) -> Dict[str, Any]:
    """Создание конфигурации обучения"""
    
    # Автоматическое определение устройства
    if use_gpu:
        device = "auto"  # Автоматическое определение GPU
    else:
        device = "cpu"
    
    training_config_dict = training_config.__dict__.copy()
    training_config_dict.update({
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': epochs,
        'device': device,
        'use_wandb': False  # Отключаем wandb для веб-интерфейса
    })
    
    model_config_dict = model_config.__dict__.copy()
    model_config_dict.update({
        'model_type': model_type,
        'model_name': model_name
    })
    
    return {
        'model_config': model_config_dict,
        'training_config': training_config_dict
    }

def start_training_interface(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_type: str,
    model_name: str,
    use_gpu: bool
) -> str:
    """Интерфейс запуска обучения"""
    
    config = create_training_config(epochs, batch_size, learning_rate, model_type, model_name, use_gpu)
    
    result = training_manager.start_training(config)
    return result

def stop_training_interface() -> str:
    """Интерфейс остановки обучения"""
    return training_manager.stop_training()

def get_training_status() -> str:
    """Получение статуса обучения"""
    progress = training_manager.get_progress()
    return f"Эпоха: {progress['epoch']}, Потери: {progress['loss']:.4f}, Статус: {progress['status']}"

def evaluate_model_interface(checkpoint_path: str, visualize: bool) -> str:
    """Интерфейс оценки модели"""
    
    if not os.path.exists(checkpoint_path):
        return "Ошибка: Чекпоинт не найден"
    
    try:
        # Определяем устройство автоматически
        device = setup_device("auto")
        from model import load_pretrained_model
        model = load_pretrained_model(checkpoint_path, device)
        
        # Создаем загрузчики данных
        from data_loader import get_data_loaders
        _, val_loader = get_data_loaders(batch_size=8, num_workers=2)
        
        # Оценка
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
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
        
        # Формируем отчет
        report = "РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ\n"
        report += "=" * 40 + "\n\n"
        
        overall_metrics = analysis['overall_metrics']
        report += f"Точность: {overall_metrics['accuracy']:.4f}\n"
        report += f"Hamming Loss: {overall_metrics['hamming_loss']:.4f}\n"
        report += f"Jaccard Score: {overall_metrics['jaccard_score']:.4f}\n"
        report += f"Средний AUC: {overall_metrics['mean_auc']:.4f}\n"
        report += f"Средний AP: {overall_metrics['mean_ap']:.4f}\n\n"
        
        report += "МЕТРИКИ ПО КЛАССАМ:\n"
        report += "-" * 20 + "\n"
        for class_name, metrics in analysis['class_metrics'].items():
            report += f"{class_name}:\n"
            report += f"  Precision: {metrics['precision']:.4f}\n"
            report += f"  Recall: {metrics['recall']:.4f}\n"
            report += f"  F1: {metrics['f1']:.4f}\n"
            report += f"  AUC: {metrics['auc']:.4f}\n"
            report += f"  AP: {metrics['ap']:.4f}\n\n"
        
        return report
        
    except Exception as e:
        return f"Ошибка при оценке модели: {str(e)}"

def predict_dicom_interface(dicom_files: List[str], checkpoint_path: str) -> str:
    """Интерфейс предсказания на DICOM файлах"""
    
    if not dicom_files:
        return "Ошибка: Не выбраны DICOM файлы"
    
    if not os.path.exists(checkpoint_path):
        return "Ошибка: Чекпоинт не найден"
    
    try:
        # Определяем устройство автоматически
        device = setup_device("auto")
        from model import load_pretrained_model
        model = load_pretrained_model(checkpoint_path, device)
        
        # Создаем датасет
        dicom_dataset = DICOMDataset(dicom_files)
        dicom_loader = create_dicom_loader(dicom_files, batch_size=1)
        
        # Предсказания
        model.eval()
        results = []
        
        with torch.no_grad():
            for batch in dicom_loader:
                volumes = batch['volume'].to(device)
                file_path = batch['file_path'][0]
                
                outputs = model(volumes)
                predictions, probabilities = model.predict(volumes, threshold=0.5)
                
                # Формируем результат
                result = {
                    'file_path': os.path.basename(file_path),
                    'predictions': {},
                    'probabilities': {}
                }
                
                for i, label in enumerate(PATHOLOGY_LABELS):
                    result['predictions'][label] = bool(predictions[0, i].item())
                    result['probabilities'][label] = float(probabilities[0, i].item())
                
                results.append(result)
        
        # Формируем отчет
        report = "РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ НА DICOM ФАЙЛАХ\n"
        report += "=" * 50 + "\n\n"
        
        for result in results:
            report += f"Файл: {result['file_path']}\n"
            report += "Предсказанные патологии:\n"
            
            # Сортируем по вероятности
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for label, prob in sorted_probs:
                if prob > 0.1:  # Показываем только вероятности > 10%
                    report += f"  {label}: {prob:.3f}\n"
            
            report += "\n"
        
        return report
        
    except Exception as e:
        return f"Ошибка при предсказании: {str(e)}"

def analyze_archive_interface(archive_file, checkpoint_path: str) -> str:
    """Интерфейс анализа архива с DICOM файлами"""
    
    if not archive_file:
        return "Ошибка: Не выбран архив"
    
    if not os.path.exists(checkpoint_path):
        return "Ошибка: Чекпоинт не найден"
    
    try:
        # Определяем устройство автоматически
        device = setup_device("auto")
        from model import load_pretrained_model
        model = load_pretrained_model(checkpoint_path, device)
        
        # Анализируем архив
        result = analyze_dicom_archive(archive_file.name, model, device)
        
        if not result["success"]:
            return f"Ошибка при анализе архива: {result['error']}"
        
        # Формируем отчет
        report = "РЕЗУЛЬТАТЫ АНАЛИЗА АРХИВА С DICOM ФАЙЛАМИ\n"
        report += "=" * 60 + "\n\n"
        
        # Информация об архиве
        archive_info = result["archive_info"]
        report += "ИНФОРМАЦИЯ ОБ АРХИВЕ:\n"
        report += "-" * 30 + "\n"
        report += f"Всего файлов: {archive_info['total_files']}\n"
        report += f"Серий исследований: {len(archive_info['series_info'])}\n"
        report += f"Пациентов: {len(archive_info['patient_info'])}\n\n"
        
        # Сводка анализа
        summary = result["analysis_summary"]
        report += "СВОДКА АНАЛИЗА:\n"
        report += "-" * 20 + "\n"
        report += f"Проанализировано файлов: {summary['total_files_analyzed']}\n"
        report += f"Файлов с патологиями: {summary['files_with_pathology']}\n"
        report += f"Всего патологий обнаружено: {summary['total_pathologies_detected']}\n"
        report += f"Общий статус: {summary['overall_status']}\n"
        report += f"Уверенность: {summary['confidence']:.3f}\n\n"
        
        # Наиболее частые патологии
        if summary['most_common_pathologies']:
            report += "НАИБОЛЕЕ ЧАСТЫЕ ПАТОЛОГИИ:\n"
            report += "-" * 30 + "\n"
            for pathology, count in summary['most_common_pathologies']:
                report += f"{pathology}: {count} файлов\n"
            report += "\n"
        
        # Детальные результаты
        report += "ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ФАЙЛАМ:\n"
        report += "-" * 40 + "\n"
        
        for i, file_result in enumerate(result["detailed_results"][:10]):  # Показываем первые 10 файлов
            report += f"\n{i+1}. Файл: {file_result['file_path']}\n"
            report += f"   Статус: {'Патология' if file_result['has_pathology'] else 'Норма'}\n"
            report += f"   Количество патологий: {file_result['pathology_count']}\n"
            
            if file_result['has_pathology']:
                report += "   Обнаруженные патологии:\n"
                for pathology, prob in file_result['probabilities'].items():
                    if prob > 0.3:  # Показываем только вероятности > 30%
                        report += f"     - {pathology}: {prob:.3f}\n"
        
        if len(result["detailed_results"]) > 10:
            report += f"\n... и еще {len(result['detailed_results']) - 10} файлов\n"
        
        # Рекомендации
        report += "\nРЕКОМЕНДАЦИИ:\n"
        report += "-" * 15 + "\n"
        
        if summary['overall_status'] == "Норма":
            report += "✅ Исследование в пределах нормы. Дополнительных действий не требуется.\n"
        elif summary['overall_status'] == "Сомнительно":
            report += "⚠️ Обнаружены сомнительные изменения. Рекомендуется консультация специалиста.\n"
        else:
            report += "🚨 Обнаружены патологические изменения. Требуется срочная консультация специалиста.\n"
        
        return report
        
    except Exception as e:
        return f"Ошибка при анализе архива: {str(e)}"

def get_available_checkpoints() -> List[str]:
    """Получение списка доступных чекпоинтов"""
    checkpoint_dir = training_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            checkpoints.append(os.path.join(checkpoint_dir, file))
    
    return sorted(checkpoints)

def create_web_interface():
    """Создание веб-интерфейса"""
    
    with gr.Blocks(title="Обучение модели классификации патологий КТ") as interface:
        
        gr.Markdown("# 🏥 Обучение модели классификации патологий КТ")
        gr.Markdown("Система для обучения модели выявления патологий на компьютерных томографиях органов грудной клетки с использованием датасета CT-RATE")
        
        # Информация о системе
        gr.Markdown("""
        ### 🚀 Система готова к работе:
        - ✅ Поддержка NVIDIA CUDA и AMD ROCm GPU
        - ✅ Автоматическое определение доступного устройства
        - ✅ Полный CT-RATE датасет без ограничений
        - ✅ Если CT-RATE недоступен, создаются синтетические данные для тестирования
        """)
        
        with gr.Tabs():
            
            # Вкладка обучения
            with gr.Tab("🎯 Обучение модели"):
                
                gr.Markdown("## Настройки обучения")
                
                with gr.Row():
                    epochs = gr.Slider(1, 200, value=50, step=1, label="Количество эпох")
                    batch_size = gr.Slider(1, 32, value=8, step=1, label="Размер батча")
                    learning_rate = gr.Slider(1e-6, 1e-2, value=1e-4, step=1e-5, label="Скорость обучения")
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["resnet3d", "efficientnet3d"],
                        value="resnet3d",
                        label="Тип модели"
                    )
                    model_name = gr.Dropdown(
                        choices=["resnet3d_18", "resnet3d_50", "efficientnet_b0"],
                        value="resnet3d_50",
                        label="Название модели"
                    )
                    use_gpu = gr.Checkbox(value=True, label="Использовать GPU")
                
                # Убрали ограничения на датасет - теперь используется полный датасет
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Начать обучение", variant="primary")
                    stop_btn = gr.Button("⏹️ Остановить обучение", variant="secondary")
                
                status_text = gr.Textbox(
                    value="Готов к обучению",
                    label="Статус обучения",
                    interactive=False
                )
                
                # Обработчики событий
                start_btn.click(
                    start_training_interface,
                    inputs=[epochs, batch_size, learning_rate, model_type, model_name, use_gpu],
                    outputs=status_text
                )
                
                stop_btn.click(
                    stop_training_interface,
                    outputs=status_text
                )
                
                # Автообновление статуса
                interface.load(
                    get_training_status,
                    outputs=status_text,
                    every=5
                )
            
            # Вкладка оценки
            with gr.Tab("📊 Оценка модели"):
                
                gr.Markdown("## Оценка обученной модели")
                
                checkpoint_path = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Выберите чекпоинт",
                    allow_custom_value=True
                )
                
                visualize_results = gr.Checkbox(value=True, label="Создавать визуализации")
                
                evaluate_btn = gr.Button("📈 Оценить модель", variant="primary")
                
                evaluation_results = gr.Textbox(
                    label="Результаты оценки",
                    lines=20,
                    interactive=False
                )
                
                evaluate_btn.click(
                    evaluate_model_interface,
                    inputs=[checkpoint_path, visualize_results],
                    outputs=evaluation_results
                )
            
            # Вкладка предсказания
            with gr.Tab("🔍 Предсказание на DICOM"):
                
                gr.Markdown("## Предсказание патологий на DICOM файлах")
                
                dicom_files = gr.File(
                    file_count="multiple",
                    file_types=[".dcm", ".DCM"],
                    label="Загрузите DICOM файлы"
                )
                
                checkpoint_path_predict = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Выберите чекпоинт",
                    allow_custom_value=True
                )
                
                predict_btn = gr.Button("🔮 Предсказать патологии", variant="primary")
                
                prediction_results = gr.Textbox(
                    label="Результаты предсказания",
                    lines=20,
                    interactive=False
                )
                
                predict_btn.click(
                    predict_dicom_interface,
                    inputs=[dicom_files, checkpoint_path_predict],
                    outputs=prediction_results
                )
            
            # Вкладка анализа архива
            with gr.Tab("📦 Анализ архива"):
                
                gr.Markdown("## Анализ архива с DICOM файлами")
                gr.Markdown("""
                ### 📋 Возможности:
                - Загрузка ZIP или TAR архивов с DICOM файлами
                - Автоматическое извлечение и анализ всех файлов
                - Определение нормы vs патологии
                - Детальный отчет с рекомендациями
                - Поддержка множественных исследований
                """)
                
                archive_file = gr.File(
                    file_types=[".zip", ".tar", ".tar.gz", ".tgz"],
                    label="Загрузите архив с DICOM файлами"
                )
                
                checkpoint_path_archive = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Выберите чекпоинт для анализа",
                    allow_custom_value=True
                )
                
                analyze_btn = gr.Button("🔬 Анализировать архив", variant="primary")
                
                archive_results = gr.Textbox(
                    label="Результаты анализа архива",
                    lines=25,
                    interactive=False
                )
                
                analyze_btn.click(
                    analyze_archive_interface,
                    inputs=[archive_file, checkpoint_path_archive],
                    outputs=archive_results
                )
            
            # Вкладка информации
            with gr.Tab("ℹ️ Информация"):
                
                gr.Markdown("## Информация о системе")
                
                gr.Markdown(f"""
                ### Поддерживаемые патологии:
                {chr(10).join([f"{i+1}. {label}" for i, label in enumerate(PATHOLOGY_LABELS)])}
                
                ### Системные требования:
                - Python 3.8+
                - PyTorch 2.0+
                - NVIDIA CUDA или AMD ROCm (опционально, для GPU ускорения)
                
                ### Возможности:
                - ✅ Обучение на GPU с возможностью прерывания
                - ✅ Сохранение прогресса обучения
                - ✅ Визуализация процесса обучения
                - ✅ Экспорт модели в различные форматы
                - ✅ Поддержка DICOM файлов
                - ✅ Анализ архивов с DICOM файлами
                - ✅ Определение нормы vs патологии
                - ✅ Веб-интерфейс для управления
                
                ### Датасет:
                Используется датасет CT-RATE с Hugging Face:
                https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
                
                ### Настройка доступа к датасету:
                1. Зарегистрируйтесь на https://huggingface.co
                2. Запросите доступ к датасету CT-RATE
                3. Установите токен: `pip install huggingface_hub`
                4. Войдите в систему: `huggingface-cli login`
                
                ### Альтернатива:
                Если доступ к CT-RATE недоступен, система автоматически создаст синтетические данные для тестирования.
                """)
                
                # Информация о GPU
                device = detect_gpu_device()
                device_info = f"Обнаружено устройство: {device.upper()}\n"
                
                if device == "cuda" and torch.cuda.is_available():
                    device_info += f"GPU: {torch.cuda.get_device_name()}\n"
                    device_info += f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"
                    device_info += f"CUDA версия: {torch.version.cuda}"
                elif device == "rocm":
                    device_info += "AMD GPU с ROCm поддержкой (Linux)\n"
                    if hasattr(torch.version, 'hip'):
                        device_info += f"ROCm версия: {torch.version.hip}"
                elif device == "directml":
                    device_info += "AMD GPU с DirectML поддержкой (Windows)\n"
                    try:
                        import torch_directml
                        device_info += f"DirectML устройств: {torch_directml.device_count()}"
                    except:
                        device_info += "DirectML недоступен"
                elif device == "cpu":
                    device_info += "CPU режим"
                else:
                    device_info += "GPU недоступен"
                
                gr.Textbox(
                    value=device_info,
                    label="Информация об устройстве",
                    interactive=False
                )
    
    return interface

def launch_web_interface(share: bool = False, port: int = 7860):
    """Запуск веб-интерфейса"""
    
    interface = create_web_interface()
    
    print("Запуск веб-интерфейса...")
    print(f"Локальный URL: http://localhost:{port}")
    if share:
        print("Публичный URL будет доступен после запуска")
    
    try:
        # Пробуем запустить с localhost
        interface.launch(
            share=share,
            server_port=port,
            server_name="127.0.0.1",
            enable_queue=True,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Ошибка запуска на localhost: {e}")
        print("🔄 Пробуем альтернативные варианты...")
        
        try:
            # Пробуем с share=True
            print("🌐 Запуск с публичной ссылкой...")
            interface.launch(
                share=True,
                server_port=port,
                server_name="127.0.0.1",
                enable_queue=True,
                show_error=True,
                quiet=False
            )
        except Exception as e2:
            print(f"❌ Ошибка запуска с share=True: {e2}")
            print("🔄 Пробуем другой порт...")
            
            try:
                # Пробуем другой порт
                new_port = 8080
                print(f"🔧 Пробуем порт {new_port}...")
                interface.launch(
                    share=True,
                    server_port=new_port,
                    server_name="127.0.0.1",
                    enable_queue=True,
                    show_error=True,
                    quiet=False
                )
            except Exception as e3:
                print(f"❌ Все попытки запуска неудачны: {e3}")
                print("\n💡 Рекомендации:")
                print("1. Проверьте настройки брандмауэра")
                print("2. Убедитесь что порт не занят")
                print("3. Попробуйте запустить с --share")
                print("4. Проверьте настройки прокси")
                raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Веб-интерфейс для обучения модели")
    parser.add_argument("--share", action="store_true", help="Создать публичную ссылку")
    parser.add_argument("--port", type=int, default=7860, help="Порт для веб-интерфейса")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    
    args = parser.parse_args()
    
    try:
        launch_web_interface(share=args.share, port=args.port)
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\n💡 Попробуйте:")
        print("1. python web_interface.py --share")
        print("2. python web_interface.py --port 8080")
        print("3. python web_interface.py --share --port 8080")

