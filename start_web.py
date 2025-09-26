#!/usr/bin/env python3
"""
Скрипт для запуска веб-интерфейса с автоматическим выбором лучших настроек
"""
import os
import sys
import subprocess
import socket
from pathlib import Path

def check_port_available(port):
    """Проверка доступности порта"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=7860):
    """Поиск доступного порта"""
    for port in range(start_port, start_port + 100):
        if check_port_available(port):
            return port
    return None

def check_gradio_version():
    """Проверка версии Gradio"""
    try:
        import gradio as gr
        version = gr.__version__
        print(f"📦 Gradio версия: {version}")
        
        # Проверяем совместимость
        major_version = int(version.split('.')[0])
        if major_version < 3:
            print("⚠️  Рекомендуется обновить Gradio до версии 3.0+")
            return False
        elif major_version >= 4:
            print("✅ Современная версия Gradio")
        else:
            print("✅ Поддерживаемая версия Gradio")
        
        return True
    except ImportError:
        print("❌ Gradio не установлен")
        return False

def upgrade_gradio():
    """Обновление Gradio"""
    print("🔄 Обновление Gradio...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gradio"])
        print("✅ Gradio обновлен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка обновления Gradio: {e}")
        return False

def launch_web_interface_safe():
    """Безопасный запуск веб-интерфейса"""
    print("🚀 Запуск веб-интерфейса...")
    
    # Проверяем версию Gradio
    if not check_gradio_version():
        print("🔄 Пытаемся обновить Gradio...")
        upgrade_gradio()
    
    # Ищем доступный порт
    port = find_available_port()
    if not port:
        print("❌ Не удалось найти доступный порт")
        return False
    
    print(f"🔧 Используем порт: {port}")
    
    # Пробуем разные варианты запуска
    launch_options = [
        # Вариант 1: Обычный запуск
        {
            "share": False,
            "port": port,
            "description": "Обычный запуск"
        },
        # Вариант 2: С публичной ссылкой
        {
            "share": True,
            "port": port,
            "description": "С публичной ссылкой"
        },
        # Вариант 3: Другой порт
        {
            "share": True,
            "port": 8080,
            "description": "Альтернативный порт"
        }
    ]
    
    for i, options in enumerate(launch_options, 1):
        print(f"\n🔄 Попытка {i}: {options['description']}")
        
        try:
            # Импортируем и запускаем
            from web_interface import launch_web_interface
            launch_web_interface(share=options["share"], port=options["port"])
            return True
            
        except Exception as e:
            print(f"❌ Попытка {i} неудачна: {e}")
            if i < len(launch_options):
                print("🔄 Пробуем следующий вариант...")
            continue
    
    print("\n❌ Все попытки запуска неудачны")
    return False

def main():
    """Главная функция"""
    print("🌐 Запуск веб-интерфейса для проекта CT")
    print("=" * 50)
    
    # Проверяем что мы в правильной директории
    if not Path("web_interface.py").exists():
        print("❌ Файл web_interface.py не найден")
        print("Убедитесь что вы находитесь в корневой папке проекта")
        return False
    
    # Проверяем виртуальное окружение
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Виртуальное окружение активно")
    else:
        print("⚠️  Виртуальное окружение не активно")
        print("Рекомендуется активировать виртуальное окружение")
    
    # Запускаем веб-интерфейс
    success = launch_web_interface_safe()
    
    if not success:
        print("\n💡 Рекомендации по устранению проблем:")
        print("1. Проверьте настройки брандмауэра")
        print("2. Убедитесь что порты 7860 и 8080 свободны")
        print("3. Обновите Gradio: pip install --upgrade gradio")
        print("4. Проверьте настройки прокси")
        print("5. Попробуйте запустить от имени администратора")
        
        print("\n🔧 Альтернативные команды:")
        print("python web_interface.py --share")
        print("python web_interface.py --port 8080")
        print("python web_interface.py --share --port 8080")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Запуск прерван пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)
