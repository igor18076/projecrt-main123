# 🔧 Устранение проблем с веб-интерфейсом

## ❌ Ошибка: "When localhost is not accessible"

### Причина
Gradio не может получить доступ к localhost из-за настроек сети или брандмауэра.

### Решения

#### 1. Запуск с публичной ссылкой
```bash
python web_interface.py --share
```

#### 2. Использование другого порта
```bash
python web_interface.py --port 8080
```

#### 3. Комбинированный вариант
```bash
python web_interface.py --share --port 8080
```

#### 4. Автоматический запуск
```bash
python start_web.py
```

## 🔧 Другие частые проблемы

### Порт занят
```bash
# Проверка занятых портов (Windows)
netstat -an | findstr :7860
netstat -an | findstr :8080

# Освобождение порта
taskkill /f /im python.exe
```

### Устаревшая версия Gradio
```bash
# Обновление Gradio
pip install --upgrade gradio

# Проверка версии
python -c "import gradio; print(gradio.__version__)"
```

### Проблемы с виртуальным окружением
```bash
# Активация виртуального окружения (Windows)
venv\Scripts\activate

# Активация виртуального окружения (Linux/Mac)
source venv/bin/activate

# Переустановка зависимостей
pip install -r requirements.txt
```

### Проблемы с брандмауэром
1. **Windows**: Добавьте Python в исключения брандмауэра
2. **Linux**: Проверьте iptables или ufw
3. **Mac**: Проверьте настройки безопасности

### Проблемы с прокси
```bash
# Отключение прокси для локальных адресов
set NO_PROXY=localhost,127.0.0.1
python web_interface.py
```

## 🚀 Рекомендуемые команды запуска

### Для разработки
```bash
python web_interface.py --debug
```

### Для демонстрации
```bash
python web_interface.py --share
```

### Для продакшена
```bash
python web_interface.py --port 8080
```

### Автоматический выбор настроек
```bash
python start_web.py
```

## 📊 Диагностика проблем

### Проверка системы
```bash
# Проверка Python
python --version

# Проверка зависимостей
pip list | grep gradio

# Проверка портов
python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 7860)); print('Порт 7860 свободен')"
```

### Логи ошибок
```bash
# Запуск с подробными логами
python web_interface.py --debug 2>&1 | tee web_interface.log
```

## 🎯 Быстрые решения

### Проблема: Веб-интерфейс не запускается
```bash
# Решение 1
python web_interface.py --share

# Решение 2
python start_web.py

# Решение 3
pip install --upgrade gradio
python web_interface.py
```

### Проблема: Ошибка подключения
```bash
# Решение 1
python web_interface.py --port 8080

# Решение 2
python web_interface.py --share --port 8080
```

### Проблема: Медленная работа
```bash
# Решение 1
python web_interface.py --port 8080

# Решение 2
# Отключите антивирус для папки проекта
```

## 🔍 Проверка работоспособности

### Тест 1: Базовый запуск
```bash
python -c "import gradio as gr; print('Gradio работает')"
```

### Тест 2: Создание простого интерфейса
```bash
python -c "
import gradio as gr
def test(x): return x
gr.Interface(test, 'text', 'text').launch(server_name='127.0.0.1', server_port=7861)
"
```

### Тест 3: Проверка портов
```bash
python -c "
import socket
def check_port(port):
    try:
        s = socket.socket()
        s.bind(('127.0.0.1', port))
        s.close()
        return True
    except:
        return False

for port in [7860, 8080, 8081]:
    print(f'Порт {port}: {'свободен' if check_port(port) else 'занят'}')
"
```

## 🎉 Успешный запуск

При успешном запуске вы увидите:
```
🚀 Запуск веб-интерфейса...
Локальный URL: http://localhost:7860
Running on local URL:  http://127.0.0.1:7860
```

Откройте браузер и перейдите по указанному URL для доступа к веб-интерфейсу.
