"""
Экспорт обученной модели в различные форматы
"""
import os
import torch
import torch.onnx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX не установлен. Экспорт в ONNX будет недоступен.")

from model import CTPathologyModel, create_model
from config import PATHOLOGY_LABELS, model_config, training_config

class ModelExporter:
    """Экспорт модели в различные форматы"""
    
    def __init__(self, output_dir: str = "./exported_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_pytorch_model(self, 
                           model: CTPathologyModel,
                           checkpoint_path: str,
                           model_name: str = "ct_pathology_model",
                           include_optimizer: bool = False) -> str:
        """Экспорт модели в формате PyTorch"""
        
        # Загружаем чекпоинт
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Создаем модель
        model_config_dict = checkpoint.get('model_config', model_config.__dict__)
        model = create_model(model_config_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Перемещаем модель на CPU для экспорта
        model = model.to('cpu')
        
        # Подготавливаем данные для экспорта
        export_data = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config_dict,
            'pathology_labels': PATHOLOGY_LABELS,
            'training_config': checkpoint.get('training_config', training_config.__dict__),
            'model_info': model.get_model_info()
        }
        
        if include_optimizer:
            export_data['optimizer_state_dict'] = checkpoint.get('optimizer_state_dict')
            export_data['scheduler_state_dict'] = checkpoint.get('scheduler_state_dict')
        
        # Сохраняем модель
        export_path = os.path.join(self.output_dir, f"{model_name}.pth")
        torch.save(export_data, export_path)
        
        print(f"Модель PyTorch экспортирована: {export_path}")
        return export_path
    
    def export_onnx_model(self, 
                         model: CTPathologyModel,
                         checkpoint_path: str,
                         model_name: str = "ct_pathology_model",
                         input_size: Tuple[int, int, int] = (1, 64, 64, 64),
                         opset_version: int = 11) -> Optional[str]:
        """Экспорт модели в формат ONNX"""
        
        if not ONNX_AVAILABLE:
            print("ONNX не доступен. Установите onnx и onnxruntime для экспорта в ONNX.")
            return None
        
        # Загружаем чекпоинт
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Создаем модель
        model_config_dict = checkpoint.get('model_config', model_config.__dict__)
        model = create_model(model_config_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Создаем пример входа
        dummy_input = torch.randn(input_size)
        
        # Экспорт в ONNX
        export_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['ct_volume'],
                output_names=['pathology_predictions'],
                dynamic_axes={
                    'ct_volume': {0: 'batch_size'},
                    'pathology_predictions': {0: 'batch_size'}
                }
            )
            
            # Проверяем экспортированную модель
            self._verify_onnx_model(export_path, dummy_input.numpy())
            
            # Сохраняем метаданные
            metadata_path = os.path.join(self.output_dir, f"{model_name}_metadata.json")
            metadata = {
                'model_type': 'onnx',
                'input_shape': list(input_size),
                'output_shape': [input_size[0], len(PATHOLOGY_LABELS)],
                'pathology_labels': PATHOLOGY_LABELS,
                'model_config': model_config_dict,
                'opset_version': opset_version
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Модель ONNX экспортирована: {export_path}")
            print(f"Метаданные сохранены: {metadata_path}")
            return export_path
            
        except Exception as e:
            print(f"Ошибка при экспорте в ONNX: {e}")
            return None
    
    def _verify_onnx_model(self, model_path: str, dummy_input: np.ndarray):
        """Проверка экспортированной ONNX модели"""
        try:
            # Загружаем модель
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Тестируем выполнение
            ort_session = ort.InferenceSession(model_path)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            print(f"ONNX модель успешно проверена. Выходной размер: {ort_outputs[0].shape}")
            
        except Exception as e:
            print(f"Ошибка при проверке ONNX модели: {e}")
    
    def export_torchscript_model(self, 
                                model: CTPathologyModel,
                                checkpoint_path: str,
                                model_name: str = "ct_pathology_model",
                                input_size: Tuple[int, int, int] = (1, 64, 64, 64)) -> str:
        """Экспорт модели в TorchScript"""
        
        # Загружаем чекпоинт
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Создаем модель
        model_config_dict = checkpoint.get('model_config', model_config.__dict__)
        model = create_model(model_config_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Создаем пример входа
        dummy_input = torch.randn(input_size)
        
        # Экспорт в TorchScript
        try:
            # Пробуем trace
            traced_model = torch.jit.trace(model, dummy_input)
            export_path = os.path.join(self.output_dir, f"{model_name}_traced.pt")
            traced_model.save(export_path)
            
            # Проверяем модель
            loaded_model = torch.jit.load(export_path)
            test_output = loaded_model(dummy_input)
            print(f"TorchScript модель (trace) экспортирована: {export_path}")
            print(f"Выходной размер: {test_output.shape}")
            
        except Exception as e:
            print(f"Ошибка при trace экспорте: {e}")
            
            # Пробуем script
            try:
                scripted_model = torch.jit.script(model)
                export_path = os.path.join(self.output_dir, f"{model_name}_scripted.pt")
                scripted_model.save(export_path)
                
                # Проверяем модель
                loaded_model = torch.jit.load(export_path)
                test_output = loaded_model(dummy_input)
                print(f"TorchScript модель (script) экспортирована: {export_path}")
                print(f"Выходной размер: {test_output.shape}")
                
            except Exception as e2:
                print(f"Ошибка при script экспорте: {e2}")
                raise e2
        
        return export_path
    
    def export_model_summary(self, 
                           checkpoint_path: str,
                           model_name: str = "ct_pathology_model") -> str:
        """Экспорт сводки модели"""
        
        # Загружаем чекпоинт
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Создаем модель для анализа
        model_config_dict = checkpoint.get('model_config', model_config.__dict__)
        model = create_model(model_config_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Собираем информацию
        summary = {
            'model_info': model.get_model_info(),
            'training_info': {
                'epoch': checkpoint.get('epoch', 0),
                'best_val_loss': checkpoint.get('best_val_loss', 0),
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', [])
            },
            'model_config': model_config_dict,
            'pathology_labels': PATHOLOGY_LABELS,
            'export_timestamp': str(torch.utils.data.get_worker_info())
        }
        
        # Сохраняем сводку
        summary_path = os.path.join(self.output_dir, f"{model_name}_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Сводка модели сохранена: {summary_path}")
        return summary_path
    
    def create_deployment_package(self, 
                                checkpoint_path: str,
                                model_name: str = "ct_pathology_model",
                                formats: List[str] = ["pytorch", "onnx", "torchscript"]) -> str:
        """Создание пакета для развертывания"""
        
        package_dir = os.path.join(self.output_dir, f"{model_name}_deployment")
        os.makedirs(package_dir, exist_ok=True)
        
        exported_files = []
        
        # Экспортируем в различные форматы
        if "pytorch" in formats:
            pytorch_path = self.export_pytorch_model(
                None, checkpoint_path, model_name, include_optimizer=False
            )
            exported_files.append(pytorch_path)
        
        if "onnx" in formats and ONNX_AVAILABLE:
            onnx_path = self.export_onnx_model(None, checkpoint_path, model_name)
            if onnx_path:
                exported_files.append(onnx_path)
        
        if "torchscript" in formats:
            torchscript_path = self.export_torchscript_model(None, checkpoint_path, model_name)
            exported_files.append(torchscript_path)
        
        # Создаем сводку
        summary_path = self.export_model_summary(checkpoint_path, model_name)
        exported_files.append(summary_path)
        
        # Создаем README для развертывания
        readme_path = os.path.join(package_dir, "README.md")
        self._create_deployment_readme(readme_path, model_name, exported_files)
        
        # Создаем requirements.txt для развертывания
        requirements_path = os.path.join(package_dir, "requirements.txt")
        self._create_deployment_requirements(requirements_path)
        
        # Создаем пример использования
        example_path = os.path.join(package_dir, "inference_example.py")
        self._create_inference_example(example_path, model_name)
        
        print(f"Пакет развертывания создан: {package_dir}")
        print(f"Экспортированные файлы: {exported_files}")
        
        return package_dir
    
    def _create_deployment_readme(self, readme_path: str, model_name: str, exported_files: List[str]):
        """Создание README для развертывания"""
        
        readme_content = f"""# {model_name} - Пакет развертывания

## Описание
Модель для классификации патологий на компьютерных томографиях органов грудной клетки.

## Файлы в пакете
"""
        
        for file_path in exported_files:
            filename = os.path.basename(file_path)
            readme_content += f"- {filename}\n"
        
        readme_content += f"""
## Установка зависимостей
```bash
pip install -r requirements.txt
```

## Использование
См. inference_example.py для примера использования модели.

## Поддерживаемые патологии
"""
        
        for i, label in enumerate(PATHOLOGY_LABELS, 1):
            readme_content += f"{i}. {label}\n"
        
        readme_content += """
## Форматы модели
- PyTorch (.pth): Оригинальный формат PyTorch
- ONNX (.onnx): Кроссплатформенный формат для развертывания
- TorchScript (.pt): Оптимизированный формат PyTorch для продакшена

## Требования к входным данным
- Формат: 3D томографические данные
- Размер: 64x64x64 пикселей
- Тип: Float32, нормализованные значения [0, 1]
- Каналы: 1 (grayscale)

## Выходные данные
- Формат: Массив вероятностей для каждого класса патологии
- Размер: [batch_size, {len(PATHOLOGY_LABELS)}]
- Тип: Float32, значения [0, 1]
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _create_deployment_requirements(self, requirements_path: str):
        """Создание requirements.txt для развертывания"""
        
        requirements = """torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pydicom>=2.4.0
nibabel>=5.1.0
opencv-python>=4.8.0
Pillow>=10.0.0
"""
        
        if ONNX_AVAILABLE:
            requirements += "onnx>=1.14.0\nonnxruntime>=1.15.0\n"
        
        with open(requirements_path, 'w') as f:
            f.write(requirements)
    
    def _create_inference_example(self, example_path: str, model_name: str):
        """Создание примера использования модели"""
        
        example_code = f'''"""
Пример использования модели {model_name} для инференса
"""
import torch
import numpy as np
from pathlib import Path
import json

class CTInference:
    def __init__(self, model_path: str, metadata_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем модель
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        self.pathology_labels = checkpoint['pathology_labels']
        
        # Создаем модель
        from model import create_model
        self.model = create_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Загружаем метаданные если доступны
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
    
    def preprocess_volume(self, volume: np.ndarray) -> torch.Tensor:
        """Предобработка CT объема"""
        # Нормализация
        volume = np.clip(volume, -1000, 1000)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Изменение размера до стандартного
        target_size = self.model_config.get('input_size', (64, 64, 64))
        if volume.shape != target_size:
            # Простая интерполяция
            import cv2
            resized = np.zeros(target_size)
            for i in range(target_size[2]):
                z_idx = int(i * volume.shape[2] / target_size[2])
                slice_2d = volume[:, :, z_idx]
                resized_slice = cv2.resize(slice_2d, (target_size[0], target_size[1]))
                resized[:, :, i] = resized_slice
            volume = resized
        
        # Конвертируем в тензор
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        return volume_tensor.to(self.device)
    
    def predict(self, volume: np.ndarray, threshold: float = 0.5) -> dict:
        """Предсказание патологий"""
        with torch.no_grad():
            # Предобработка
            volume_tensor = self.preprocess_volume(volume)
            
            # Инференс
            logits = self.model(volume_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            
            # Конвертируем в numpy
            probs = probabilities.cpu().numpy()[0]
            preds = predictions.cpu().numpy()[0]
            
            # Формируем результат
            result = {{
                'predictions': {{}},
                'probabilities': {{}},
                'summary': {{
                    'total_pathologies': int(np.sum(preds)),
                    'max_probability': float(np.max(probs)),
                    'max_probability_class': self.pathology_labels[np.argmax(probs)]
                }}
            }}
            
            for i, label in enumerate(self.pathology_labels):
                result['predictions'][label] = bool(preds[i])
                result['probabilities'][label] = float(probs[i])
            
            return result

# Пример использования
if __name__ == "__main__":
    # Инициализация
    model_path = "{model_name}.pth"
    inference = CTInference(model_path)
    
    # Пример данных (замените на реальные CT данные)
    dummy_volume = np.random.rand(64, 64, 64).astype(np.float32)
    
    # Предсказание
    result = inference.predict(dummy_volume)
    
    print("Результаты предсказания:")
    print(f"Общее количество патологий: {{result['summary']['total_pathologies']}}")
    print(f"Максимальная вероятность: {{result['summary']['max_probability']:.3f}}")
    print(f"Класс с максимальной вероятностью: {{result['summary']['max_probability_class']}}")
    
    print("\\nДетальные результаты:")
    for label, prob in result['probabilities'].items():
        if prob > 0.1:  # Показываем только вероятности > 10%
            print(f"{{label}}: {{prob:.3f}}")
'''
        
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)

def export_model(checkpoint_path: str, 
                export_formats: List[str] = ["pytorch", "onnx", "torchscript"],
                model_name: str = "ct_pathology_model",
                output_dir: str = "./exported_models") -> Dict[str, str]:
    """Основная функция экспорта модели"""
    
    exporter = ModelExporter(output_dir)
    exported_files = {}
    
    print(f"Начинаем экспорт модели из чекпоинта: {checkpoint_path}")
    
    # Экспорт в различные форматы
    if "pytorch" in export_formats:
        pytorch_path = exporter.export_pytorch_model(None, checkpoint_path, model_name)
        exported_files["pytorch"] = pytorch_path
    
    if "onnx" in export_formats:
        onnx_path = exporter.export_onnx_model(None, checkpoint_path, model_name)
        if onnx_path:
            exported_files["onnx"] = onnx_path
    
    if "torchscript" in export_formats:
        torchscript_path = exporter.export_torchscript_model(None, checkpoint_path, model_name)
        exported_files["torchscript"] = torchscript_path
    
    # Создаем пакет развертывания
    deployment_package = exporter.create_deployment_package(
        checkpoint_path, model_name, export_formats
    )
    exported_files["deployment_package"] = deployment_package
    
    print(f"Экспорт завершен. Файлы сохранены в: {output_dir}")
    return exported_files

