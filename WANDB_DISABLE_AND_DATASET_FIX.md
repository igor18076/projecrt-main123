# 🔧 Отключение wandb и исправление обработки датасета

## ❌ Проблемы
1. **Ошибка wandb**: `403 Forbidden` - проблемы с доступом к Weights & Biases
2. **Неправильная обработка датасета**: Код не понимал структуру CT-RATE датасета

## ✅ Решения

### 1. **Отключен wandb**
- **Изменено в `config.py`**: `use_wandb: bool = False`
- **Результат**: Обучение будет работать без wandb, метрики будут выводиться в консоль

### 2. **Исправлена обработка датасета CT-RATE**

#### **Структура датасета:**
```
Ключи: ['VolumeName', 'Medical material', 'Arterial wall calcification', 'Cardiomegaly', ...]
- VolumeName: изображение CT
- Остальные ключи: метки патологий (0/1, True/False)
```

#### **Изменения в `data_loader.py`:**

**Добавлен ключ для изображения:**
```python
possible_keys = ['image', 'ct', 'volume', 'data', 'scan', 'VolumeName']
```

**Исправлена функция `_get_labels()`:**
```python
def _get_labels(self, sample: Dict) -> List[float]:
    labels = [0.0] * len(PATHOLOGY_LABELS)
    
    # Для CT-RATE датасета метки находятся в отдельных ключах
    for i, pathology in enumerate(PATHOLOGY_LABELS):
        if pathology in sample:
            label_value = sample[pathology]
            
            # Конвертируем в float
            if isinstance(label_value, (int, float)):
                labels[i] = float(label_value)
            elif isinstance(label_value, bool):
                labels[i] = 1.0 if label_value else 0.0
            elif isinstance(label_value, str):
                if label_value.lower() in ['true', '1', 'yes', 'positive']:
                    labels[i] = 1.0
                else:
                    labels[i] = 0.0
    
    return labels
```

**Добавлена отладочная информация:**
```python
# Показываем примеры меток для первых образцов
pathology_examples = {}
for pathology in PATHOLOGY_LABELS[:5]:
    if pathology in sample:
        pathology_examples[pathology] = sample[pathology]
print(f"🔍 Примеры меток: {pathology_examples}")
```

## 🎯 Результат

Теперь система:
- ✅ **Работает без wandb** - метрики выводятся в консоль
- ✅ **Правильно обрабатывает CT-RATE** - находит `VolumeName` для изображений
- ✅ **Корректно извлекает метки** - из отдельных ключей патологий
- ✅ **Показывает отладочную информацию** - для понимания структуры данных

## 🚀 Запуск

```bash
python main.py train --epochs 10
```

### Ожидаемый вывод:
```
🔍 Образец 0: доступные ключи: ['VolumeName', 'Medical material', ...]
🔍 Примеры меток: {'Medical material': 0, 'Arterial wall calcification': 1, ...}
✅ DirectML доступен для AMD GPU
Используемое устройство: cpu
Начинаем обучение на 10 эпох
```

## 📊 Поддерживаемые форматы меток

| Тип значения | Пример | Результат |
|--------------|--------|-----------|
| `int` | `1`, `0` | `1.0`, `0.0` |
| `float` | `1.0`, `0.0` | `1.0`, `0.0` |
| `bool` | `True`, `False` | `1.0`, `0.0` |
| `str` | `"true"`, `"false"` | `1.0`, `0.0` |
| `str` | `"1"`, `"0"` | `1.0`, `0.0` |

## 🎉 Готово!

Теперь обучение будет работать корректно с CT-RATE датасетом без wandb! 🚀
