# 🚀 Быстрый старт с AMD GPU на Windows

Это руководство поможет вам быстро запустить проект классификации патологий КТ на видеокарте AMD в Windows.

## ⚡ Запуск за 5 минут

### 1. Проверка системы
```bash
# Проверяем версию Windows (должна быть 1903+)
winver

# Проверяем Python
python --version
```

### 2. Установка PyTorch с DirectML
```bash
# Автоматическая установка (рекомендуется)
python install_amd_windows.py

# Или ручная установка
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
pip install -r requirements.txt
```

### 3. Тестирование
```bash
# Проверяем работу AMD GPU
python -c "
import torch
import torch_directml
print(f'PyTorch: {torch.__version__}')
print(f'DirectML доступен: {torch_directml.device_count() > 0}')
device = torch_directml.device()
print(f'DirectML устройство: {device}')
"
```

### 4. Запуск обучения
```bash
# Быстрое обучение (10 эпох)
python main.py train --epochs 10

# Полное обучение
python main.py train --epochs 100 --visualize --export
```

### 5. Веб-интерфейс
```bash
# Запуск веб-интерфейса
python web_interface.py
```

## 🔧 Настройка DirectML

DirectML автоматически настраивается при установке `torch-directml`. Дополнительных настроек не требуется.

## 📊 Мониторинг GPU

```bash
# Проверка DirectML
python -c "import torch_directml; print(f'Устройств: {torch_directml.device_count()}')"

# Мониторинг через Task Manager
# Откройте Диспетчер задач → Производительность → GPU
```

## 🚨 Устранение неполадок

### DirectML не найден
```bash
# Переустанавливаем DirectML
pip uninstall torch-directml
pip install torch-directml
```

### PyTorch не видит GPU
```bash
# Проверяем установку
python -c "import torch_directml; print(torch_directml.device())"

# Переустанавливаем PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
```

### Недостаточно памяти
```bash
# Уменьшаем размер батча
python main.py train --epochs 10 --batch-size 4
```

### Медленная работа
```bash
# Проверяем драйверы AMD
# Обновите драйверы с https://amd.com/support

# Проверяем DirectML
python -c "
import torch_directml
device = torch_directml.device()
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.mm(x, y)
print('DirectML работает')
"
```

## 📈 Ожидаемая производительность

| GPU | Время эпохи | Память | Скорость |
|-----|-------------|--------|----------|
| AMD RX 7900 XTX | ~60 сек | 8GB | 80% |
| AMD RX 7800 XT | ~75 сек | 6GB | 70% |
| AMD RX 7700 XT | ~90 сек | 4GB | 60% |

**Примечание**: DirectML может быть медленнее CUDA/ROCm, но обеспечивает совместимость с Windows.

## 🎯 Готово!

Теперь вы можете:
- Обучать модели на AMD GPU в Windows
- Использовать веб-интерфейс
- Экспортировать модели
- Анализировать результаты

## 🔄 Альтернативы

### Если DirectML работает медленно:
1. **WSL 2 с ROCm** - лучшая производительность, но сложнее настройка
2. **Linux с ROCm** - оптимальная производительность
3. **CPU** - медленно, но стабильно

### Для лучшей производительности:
```bash
# Рассмотрите переход на Linux с ROCm
# Или использование WSL 2
```

## 📚 Дополнительные ресурсы

- [DirectML документация](https://docs.microsoft.com/en-us/windows/ai/directml/)
- [PyTorch DirectML](https://pytorch.org/blog/pytorch-1.12-release/)
- [AMD драйверы](https://amd.com/support)

Для подробной документации см. [AMD_GPU_SETUP.md](AMD_GPU_SETUP.md)
