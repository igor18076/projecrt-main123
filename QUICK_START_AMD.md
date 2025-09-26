# 🚀 Быстрый старт с AMD GPU

Это руководство поможет вам быстро запустить проект классификации патологий КТ на видеокарте AMD.

## ⚡ Запуск за 5 минут

### 1. Проверка системы
```bash
# Проверяем наличие ROCm
rocm-smi

# Если ROCm не установлен, устанавливаем
sudo apt update
wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/jammy/amdgpu-install_5.6.50600-1_all.deb
sudo dpkg -i amdgpu-install_5.6.50600-1_all.deb
sudo amdgpu-install --usecase=rocm
```

### 2. Установка PyTorch с ROCm
```bash
# Автоматическая установка
python install_amd_gpu.py

# Или ручная установка
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6
pip install -r requirements.txt
```

### 3. Тестирование
```bash
# Проверяем работу AMD GPU
python test_amd_gpu.py
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

## 🔧 Настройка переменных окружения

Добавьте в `~/.bashrc`:
```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
```

## 📊 Мониторинг GPU

```bash
# Мониторинг использования AMD GPU
watch -n 1 rocm-smi

# Детальная информация
rocm-smi --showmeminfo vram
```

## 🚨 Устранение неполадок

### ROCm не найден
```bash
# Проверяем установку
ls -la /opt/rocm

# Перезагружаем переменные
source ~/.bashrc
```

### PyTorch не видит GPU
```bash
# Переустанавливаем PyTorch
pip uninstall torch torchvision torchaudio
python install_amd_gpu.py
```

### Недостаточно памяти
```bash
# Уменьшаем размер батча
python main.py train --epochs 10 --batch-size 4
```

## 📈 Ожидаемая производительность

| GPU | Время эпохи | Память | Скорость |
|-----|-------------|--------|----------|
| AMD RX 7900 XTX | ~45 сек | 8GB | 100% |
| AMD RX 7800 XT | ~55 сек | 6GB | 85% |
| AMD RX 7700 XT | ~65 сек | 4GB | 75% |

## 🎯 Готово!

Теперь вы можете:
- Обучать модели на AMD GPU
- Использовать веб-интерфейс
- Экспортировать модели
- Анализировать результаты

Для подробной документации см. [AMD_GPU_SETUP.md](AMD_GPU_SETUP.md)
