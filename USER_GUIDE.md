# 🏥 Руководство пользователя: Система обучения модели классификации патологий КТ

## 📋 Содержание

- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Использование веб-интерфейса](#использование-веб-интерфейса)
- [Командная строка](#командная-строка)
- [Обучение модели](#обучение-модели)
- [Оценка модели](#оценка-модели)
- [Предсказание на DICOM](#предсказание-на-dicom)
- [Экспорт модели](#экспорт-модели)
- [Устранение неполадок](#устранение-неполадок)

## 🚀 Быстрый старт

### Требования системы
- **Python 3.10+** (рекомендуется Python 3.10-3.11)
- **CUDA 13.0+** (для GPU ускорения)
- **NVIDIA GPU** с поддержкой CUDA 13.0
- 8+ GB RAM
- 10+ GB свободного места на диске

### Установка за 5 минут

```bash
# 1. Клонирование репозитория
git clone <repository-url>
cd ct-pathology-detection

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Проверка системы
python quick_start.py --check

# 4. Запуск веб-интерфейса
python web_interface.py

# 5. Открытие браузера
# http://localhost:7860
```

## 🛠 Установка

### Подробная установка

```bash
# Автоматическая установка всех зависимостей
python install.py

# Или ручная установка:
# 1. Сначала PyTorch Nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. Затем остальные зависимости
pip install -r requirements.txt

# Проверка установки
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"Недоступен\"}')"
```

### Проверка системы

```bash
# Полная проверка системы
python setup.py

# Быстрая проверка
python quick_start.py --check
```

## 🌐 Использование веб-интерфейса

### Запуск веб-интерфейса

```bash
# Локальный запуск
python web_interface.py

# С публичной ссылкой
python web_interface.py --share

# На определенном порту
python web_interface.py --port 8080
```

### Вкладки веб-интерфейса

#### 1. 🎯 Обучение модели
- **Настройка параметров**: эпохи, размер батча, скорость обучения
- **Выбор модели**: ResNet3D-18, ResNet3D-50, EfficientNet3D
- **Мониторинг**: прогресс в реальном времени, графики потерь
- **Управление**: запуск, остановка, сохранение прогресса

#### 2. 📊 Оценка модели
- **Загрузка чекпоинтов**: выбор обученной модели
- **Анализ метрик**: точность, AUC, Precision-Recall
- **Визуализация**: ROC кривые, матрицы ошибок
- **Отчеты**: детальный анализ по классам

#### 3. 🔍 Предсказание на DICOM
- **Загрузка файлов**: одиночные DICOM или папки
- **Предсказание**: анализ патологий
- **Результаты**: вероятности и метки патологий
- **Сохранение**: экспорт результатов в JSON

#### 4. ℹ️ Информация
- **Система**: версии библиотек, GPU информация
- **Патологии**: список поддерживаемых патологий
- **Требования**: системные требования

## 💻 Командная строка

### Основные команды

```bash
# Обучение модели
python main.py train [ОПЦИИ]

# Оценка модели
python main.py evaluate [ОПЦИИ]

# Предсказание на DICOM
python main.py predict [ОПЦИИ]

# Демонстрация системы
python demo.py

# Примеры использования
python examples.py
```

## 🎓 Обучение модели

### Базовое обучение

```bash
# Быстрое обучение (10 эпох)
python main.py train --epochs 10 --batch-size 4

# Полное обучение
python main.py train --epochs 100 --batch-size 8 --learning-rate 1e-4

# Обучение с визуализацией
python main.py train --epochs 50 --visualize --export
```

### Продвинутые опции

```bash
# Обучение с кастомными параметрами
python main.py train \
    --epochs 150 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --model-type resnet3d \
    --model-name resnet3d_50 \
    --patience 15 \
    --visualize \
    --export

# Продолжение обучения
python main.py train \
    --resume ./checkpoints/best_model.pth \
    --epochs 50

# Обучение только на CPU
python main.py train --epochs 20 --cpu
```

### Параметры обучения

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--epochs` | Количество эпох | 100 |
| `--batch-size` | Размер батча | 8 |
| `--learning-rate` | Скорость обучения | 1e-4 |
| `--model-type` | Тип модели (resnet3d, efficientnet3d) | resnet3d |
| `--model-name` | Название модели | resnet3d_50 |
| `--patience` | Терпение для ранней остановки | 10 |
| `--visualize` | Создавать визуализации | False |
| `--export` | Экспортировать модель | False |
| `--resume` | Путь к чекпоинту | None |
| `--cpu` | Использовать CPU | False |

## 📊 Оценка модели

### Оценка обученной модели

```bash
# Базовая оценка
python main.py evaluate --checkpoint ./checkpoints/best_model.pth

# Оценка с визуализацией
python main.py evaluate --checkpoint ./checkpoints/best_model.pth --visualize

# Оценка с кастомным размером батча
python main.py evaluate --checkpoint ./checkpoints/best_model.pth --batch-size 16
```

### Метрики оценки

- **Accuracy**: Общая точность классификации
- **AUC**: Area Under Curve для каждого класса
- **Precision**: Точность предсказаний
- **Recall**: Полнота предсказаний
- **F1-Score**: Гармоническое среднее precision и recall
- **Confusion Matrix**: Матрица ошибок по классам

## 🔍 Предсказание на DICOM

### Загрузка DICOM файлов

```bash
# Предсказание на папке с DICOM файлами
python main.py predict \
    --checkpoint ./checkpoints/best_model.pth \
    --dicom-path ./data/dicom_files/

# Предсказание с сохранением результатов
python main.py predict \
    --checkpoint ./checkpoints/best_model.pth \
    --dicom-path ./data/dicom_files/ \
    --output ./results/predictions.json
```

### Поддерживаемые форматы

- **Одиночные файлы**: .dcm, .DICOM
- **Папки с сериями**: автоматическое объединение
- **NIfTI файлы**: .nii, .nii.gz

### Результаты предсказания

```json
{
  "patient_id": "001",
  "predictions": {
    "Atelectasis": 0.85,
    "Cardiomegaly": 0.12,
    "Consolidation": 0.73,
    "Edema": 0.05,
    "Effusion": 0.91,
    "Emphysema": 0.02,
    "Fibrosis": 0.15,
    "Hernia": 0.01,
    "Infiltration": 0.67,
    "Mass": 0.23,
    "Nodule": 0.45,
    "Pleural_Thickening": 0.08,
    "Pneumonia": 0.89,
    "Pneumothorax": 0.03
  },
  "binary_predictions": {
    "Atelectasis": true,
    "Cardiomegaly": false,
    "Consolidation": true,
    "Edema": false,
    "Effusion": true,
    "Emphysema": false,
    "Fibrosis": false,
    "Hernia": false,
    "Infiltration": true,
    "Mass": false,
    "Nodule": false,
    "Pleural_Thickening": false,
    "Pneumonia": true,
    "Pneumothorax": false
  }
}
```

## 📦 Экспорт модели

### Экспорт в различные форматы

```bash
# Экспорт всех форматов
python main.py train --epochs 50 --export --export-formats pytorch,onnx,torchscript

# Экспорт только PyTorch
python main.py train --epochs 50 --export --export-formats pytorch

# Экспорт существующей модели
python -c "
from model_exporter import export_model
exported_files = export_model(
    checkpoint_path='./checkpoints/best_model.pth',
    export_formats=['pytorch', 'onnx', 'torchscript'],
    model_name='my_ct_model'
)
print('Экспортированные файлы:', exported_files)
"
```

### Поддерживаемые форматы

1. **PyTorch (.pth)**
   - Оригинальный формат PyTorch
   - Сохраняет архитектуру и веса
   - Совместим с PyTorch экосистемой

2. **ONNX (.onnx)**
   - Кроссплатформенный формат
   - Поддержка различных runtime'ов
   - Оптимизация для продакшена

3. **TorchScript (.pt)**
   - Оптимизированный формат PyTorch
   - Независимый от Python
   - Быстрый инференс

## 🏥 Поддерживаемые патологии

Модель обучена для выявления **14 типов патологий**:

| № | Патология | Описание |
|---|-----------|----------|
| 1 | **Atelectasis** | Ателектаз - спадение легочной ткани |
| 2 | **Cardiomegaly** | Кардиомегалия - увеличение сердца |
| 3 | **Consolidation** | Консолидация - уплотнение легочной ткани |
| 4 | **Edema** | Отек - скопление жидкости в тканях |
| 5 | **Effusion** | Выпот - скопление жидкости в плевральной полости |
| 6 | **Emphysema** | Эмфизема - патологическое расширение альвеол |
| 7 | **Fibrosis** | Фиброз - рубцевание легочной ткани |
| 8 | **Hernia** | Грыжа - выпячивание органов |
| 9 | **Infiltration** | Инфильтрация - проникновение патологических веществ |
| 10 | **Mass** | Масса - объемное образование |
| 11 | **Nodule** | Узелок - небольшое округлое образование |
| 12 | **Pleural_Thickening** | Утолщение плевры |
| 13 | **Pneumonia** | Пневмония - воспаление легких |
| 14 | **Pneumothorax** | Пневмоторакс - скопление воздуха в плевральной полости |

## 📊 Датасет CT-RATE

- **Источник**: [Hugging Face CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- **Размер**: 25,692 CT объемов от 21,304 пациентов
- **Формат**: NIfTI файлы с текстовыми отчетами
- **Разделение**: 20,000 пациентов для обучения, 1,304 для валидации
- **Метки**: Мультилейбельные аннотации патологий

## 🔧 Устранение неполадок

### Частые проблемы

#### 1. Ошибка CUDA
```
RuntimeError: CUDA out of memory
```
**Решение**: Уменьшите размер батча или используйте CPU
```bash
python main.py train --batch-size 4 --cpu
```

#### 2. Ошибка загрузки данных
```
FileNotFoundError: CT-RATE dataset not found
```
**Решение**: Убедитесь, что у вас есть доступ к интернету для загрузки датасета

#### 3. Ошибка экспорта ONNX
```
ONNX export failed
```
**Решение**: Установите ONNX и проверьте совместимость версий
```bash
pip install onnx onnxruntime
```

#### 4. Медленное обучение
**Решение**: 
- Используйте GPU: `--device cuda`
- Увеличьте количество workers: `--num-workers 8`
- Используйте смешанную точность

#### 5. Проблемы с Python версией
```
❌ Требуется Python 3.9+, текущая версия: 3.8.x
```
**Решение**: Обновите Python до версии 3.9 или выше

#### 6. Проблемы с CUDA версией
```
⚠️ CUDA версия не поддерживается
```
**Решение**: Установите CUDA 13.0+ или используйте CPU режим

### Логи и отладка

```bash
# Включение подробных логов
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python main.py train --epochs 10 --verbose

# Мониторинг GPU
nvidia-smi -l 1

# Мониторинг памяти
htop
```

### Проверка системы

```bash
# Полная проверка
python setup.py

# Быстрая проверка
python quick_start.py --check

# Проверка PyTorch и CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA доступен: {torch.cuda.is_available()}')
print(f'CUDA версия: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 📚 Дополнительные ресурсы

- [CT-RATE датасет](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- [PyTorch документация](https://pytorch.org/docs/)
- [Gradio документация](https://gradio.app/docs/)
- [Weights & Biases](https://wandb.ai/)

## 🎯 Примеры использования

### Программный интерфейс

```python
from trainer import create_trainer
from model_exporter import export_model
from visualization import TrainingVisualizer

# Создание тренера
trainer = create_trainer(model_config, training_config, device="cuda")

# Обучение
trainer.train(num_epochs=50)

# Экспорт модели
exported_files = export_model(
    checkpoint_path="./checkpoints/best_model.pth",
    export_formats=["pytorch", "onnx"],
    model_name="my_ct_model"
)

# Визуализация
visualizer = TrainingVisualizer()
visualizer.plot_training_curves(
    trainer.train_losses,
    trainer.val_losses,
    trainer.train_metrics,
    trainer.val_metrics
)
```

### Предсказание на новых данных

```python
from model import load_pretrained_model
import torch
import numpy as np

# Загрузка модели
model = load_pretrained_model("./checkpoints/best_model.pth", "cuda")

# Подготовка данных
ct_volume = np.random.rand(64, 64, 64).astype(np.float32)
ct_tensor = torch.FloatTensor(ct_volume).unsqueeze(0).unsqueeze(0)

# Предсказание
with torch.no_grad():
    logits = model(ct_tensor)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()

# Интерпретация результатов
for i, pathology in enumerate(PATHOLOGY_LABELS):
    prob = probabilities[0, i].item()
    pred = predictions[0, i].item()
    print(f"{pathology}: {prob:.3f} ({'Да' if pred else 'Нет'})")
```

---

**🎉 Система готова к использованию!**

Начните с веб-интерфейса для удобного управления:
```bash
python web_interface.py
```

**Примечание**: Эта система предназначена для исследовательских целей. Для клинического использования необходима дополнительная валидация и сертификация.
