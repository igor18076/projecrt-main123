# 🚀 Установка PyTorch Nightly с CUDA 13.0

## 📋 Требования

- **Python 3.10+** (рекомендуется Python 3.10-3.11)
- **CUDA 13.0+** (для GPU ускорения)
- **NVIDIA GPU** с поддержкой CUDA 13.0

## 🛠 Установка PyTorch Nightly

### Быстрая установка

```bash
# Установка PyTorch Nightly с CUDA 13.0
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Установка остальных зависимостей
pip install -r requirements.txt
```

### Альтернативные способы установки

#### 1. Установка через requirements.txt

```bash
# Установка всех зависимостей (включая PyTorch Nightly)
pip install -r requirements.txt
```

#### 2. Установка только PyTorch Nightly

```bash
# Только PyTorch Nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Без torchaudio (если не нужен)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

#### 3. Установка для CPU (если нет GPU)

```bash
# CPU версия PyTorch Nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## 🔧 Проверка установки

### Проверка PyTorch Nightly

```bash
# Проверка версии PyTorch
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

### Проверка системы

```bash
# Полная проверка системы
python setup.py

# Быстрая проверка
python quick_start.py --check
```

## 🐛 Устранение неполадок

### Проблемы с установкой PyTorch Nightly

```bash
# Ошибка: Не удается найти подходящую версию
ERROR: Could not find a version that satisfies the requirement torch

# Решение: Используйте правильный индекс
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Или обновите pip
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Проблемы с CUDA

```bash
# Ошибка: CUDA недоступен
CUDA доступен: False

# Решение: Установите CUDA 13.0
# Скачайте с nvidia.com/cuda-downloads
# Или используйте CPU версию
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Проблемы с зависимостями

```bash
# Ошибка: Конфликт версий
ERROR: pip's dependency resolver does not currently take into account all the packages

# Решение: Создайте чистое окружение
python -m venv pytorch-nightly-env
source pytorch-nightly-env/bin/activate  # Linux/macOS
# или
pytorch-nightly-env\Scripts\activate  # Windows

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -r requirements.txt
```

## 🎯 Преимущества PyTorch Nightly

### 🚀 Новые возможности

- **Последние улучшения производительности**
- **Новые функции и API**
- **Исправления багов**
- **Оптимизации для CUDA 13.0**

### 🔧 Стабильность

- **Ежедневные сборки**
- **Автоматическое тестирование**
- **Обратная совместимость**
- **Поддержка сообщества**

## 📚 Дополнительные ресурсы

- [PyTorch Nightly документация](https://pytorch.org/get-started/locally/)
- [CUDA 13.0 документация](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [PyTorch форум](https://discuss.pytorch.org/)

## ✅ Готово к использованию!

После успешной установки PyTorch Nightly вы можете:

1. **Запустить веб-интерфейс**: `python web_interface.py`
2. **Начать обучение**: `python main.py train --epochs 10`
3. **Запустить демо**: `python demo.py`
4. **Проверить систему**: `python quick_start.py --check`

---

**🎉 PyTorch Nightly установлен!**

Система готова к использованию с самыми последними возможностями PyTorch и CUDA 13.0.
