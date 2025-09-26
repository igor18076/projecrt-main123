# 🛠 Руководство по установке для Python 3.10

## 📋 Требования системы

- **Python 3.10+** (рекомендуется Python 3.10-3.11)
- **CUDA 13.0+** (для GPU ускорения)
- **NVIDIA GPU** с поддержкой CUDA 13.0
- 8+ GB RAM
- 10+ GB свободного места на диске

## 🚀 Установка за 5 минут

### 1. Проверка Python версии

```bash
# Проверка версии Python
python --version

# Должно быть Python 3.10.x или выше
```

### 2. Установка зависимостей

```bash
# Автоматическая установка всех зависимостей
python install.py

# Или ручная установка:
# 1. Сначала PyTorch Nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. Затем остальные зависимости
pip install -r requirements.txt
```

### 3. Проверка установки

```bash
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

# Проверка системы
python quick_start.py --check
```

### 4. Запуск системы

```bash
# Запуск веб-интерфейса
python web_interface.py

# Открытие браузера
# http://localhost:7860
```

## 🔧 Подробная установка

### Установка Python 3.10

#### Windows
```bash
# Скачайте Python 3.10 с официального сайта
# https://www.python.org/downloads/release/python-3100/

# Или используйте Chocolatey
choco install python --version=3.10.0

# Или используйте Anaconda
conda create -n ct-pathology python=3.10
conda activate ct-pathology
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# CentOS/RHEL
sudo yum install python310 python310-pip

# Создание виртуального окружения
python3.10 -m venv ct-pathology-env
source ct-pathology-env/bin/activate
```

#### macOS
```bash
# Используйте Homebrew
brew install python@3.10

# Или используйте Anaconda
conda create -n ct-pathology python=3.10
conda activate ct-pathology
```

### Установка CUDA 13.0

#### Windows
```bash
# Скачайте CUDA Toolkit 13.0 с официального сайта
# https://developer.nvidia.com/cuda-13-0-0-download-archive

# Установите драйверы NVIDIA
# https://www.nvidia.com/drivers/
```

#### Linux
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_linux.run
sudo sh cuda_13.0.0_linux.run

# CentOS/RHEL
sudo yum install cuda-toolkit-13-0
```

#### macOS
```bash
# macOS не поддерживает CUDA
# Используйте CPU версию PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Установка PyTorch

```bash
# GPU версия с CUDA 13.0
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# CPU версия (если нет GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Проверка установки
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🐛 Устранение неполадок

### Проблемы с Python версией

```bash
# Ошибка: Требуется Python 3.10+
❌ Требуется Python 3.10+, текущая версия: 3.9.x

# Решение: Обновите Python
# Windows: Скачайте с python.org
# Linux: sudo apt install python3.10
# macOS: brew install python@3.10
```

### Проблемы с CUDA

```bash
# Ошибка: CUDA недоступен
CUDA доступен: False

# Решение: Установите CUDA 13.0
# Скачайте с nvidia.com/cuda-downloads
# Или используйте CPU версию PyTorch
```

### Проблемы с PyTorch

```bash
# Ошибка: Не удается установить PyTorch
ERROR: Could not find a version that satisfies the requirement torch

# Решение: Используйте правильный индекс
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Или установите CPU версию
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Проблемы с зависимостями

```bash
# Ошибка: Конфликт версий
ERROR: pip's dependency resolver does not currently take into account all the packages

# Решение: Создайте чистое окружение
python -m venv ct-pathology-env
source ct-pathology-env/bin/activate  # Linux/macOS
# или
ct-pathology-env\Scripts\activate  # Windows

pip install -r requirements.txt
```

## ✅ Проверка установки

### Полная проверка системы

```bash
# Запуск полной проверки
python setup.py

# Быстрая проверка
python quick_start.py --check

# Проверка основных компонентов
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')
except ImportError:
    print('❌ PyTorch не установлен')

try:
    import gradio
    print(f'Gradio: {gradio.__version__}')
except ImportError:
    print('❌ Gradio не установлен')

try:
    import datasets
    print(f'Datasets: {datasets.__version__}')
except ImportError:
    print('❌ Datasets не установлен')
"
```

### Проверка веб-интерфейса

```bash
# Запуск веб-интерфейса
python web_interface.py

# Проверка доступности
# Откройте браузер: http://localhost:7860
```

## 🎯 Готово к использованию!

После успешной установки вы можете:

1. **Запустить веб-интерфейс**: `python web_interface.py`
2. **Начать обучение**: `python main.py train --epochs 10`
3. **Запустить демо**: `python demo.py`
4. **Проверить систему**: `python quick_start.py --check`

## 📚 Дополнительные ресурсы

- [Python 3.10 документация](https://docs.python.org/3.10/)
- [PyTorch документация](https://pytorch.org/docs/)
- [CUDA документация](https://docs.nvidia.com/cuda/)
- [Gradio документация](https://gradio.app/docs/)

---

**🎉 Установка завершена!**

Система готова к использованию для обучения моделей выявления патологий на компьютерных томографиях органов грудной клетки.
