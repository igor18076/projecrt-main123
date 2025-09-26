#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы с AMD GPU
"""
import sys
import torch
from gpu_utils import detect_gpu_device, print_device_info, setup_device, optimize_for_device

def test_gpu_detection():
    """Тест определения GPU"""
    print("🔍 Тестирование определения GPU...")
    
    device = detect_gpu_device()
    print(f"Обнаружено устройство: {device}")
    
    print_device_info(device)
    
    return device

def test_pytorch_gpu():
    """Тест PyTorch с GPU поддержкой"""
    print("\n🧪 Тестирование PyTorch с GPU...")
    
    try:
        # Проверяем версии
        print(f"PyTorch версия: {torch.__version__}")
        
        # Проверяем CUDA
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"Количество устройств: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Устройство {i}: {torch.cuda.get_device_name(i)}")
        
        # Проверяем DirectML (Windows)
        try:
            import torch_directml
            print(f"DirectML доступен: {torch_directml.device_count() > 0}")
            if torch_directml.device_count() > 0:
                print(f"DirectML устройств: {torch_directml.device_count()}")
        except ImportError:
            print("DirectML: Не установлен")
        
        # Проверяем ROCm (Linux)
        if hasattr(torch.version, 'hip'):
            print(f"ROCm версия: {torch.version.hip}")
        else:
            print("ROCm версия: Не доступна")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании PyTorch: {e}")
        return False

def test_tensor_operations(device):
    """Тест операций с тензорами"""
    print(f"\n🔢 Тестирование операций с тензорами на {device}...")
    
    try:
        # Создаем тестовый тензор
        x = torch.randn(2, 3, 4, 4, 4)  # 3D тензор как в CT данных
        print(f"CPU тензор создан: {x.shape}")
        
        # Перемещаем на устройство
        if device != "cpu":
            # Для DirectML нужно использовать специальный способ
            if device == "directml":
                try:
                    import torch_directml
                    device_obj = torch_directml.device()
                    x = x.to(device_obj)
                    print(f"Тензор перемещен на DirectML: {x.shape}")
                except ImportError:
                    print("DirectML недоступен, используем CPU")
                    device = "cpu"
            else:
                x = x.to(device)
                print(f"Тензор перемещен на {device}: {x.shape}")
            
            if device != "cpu":
                # Тестируем операции
                y = torch.randn_like(x)
                if device == "directml":
                    y = y.to(device_obj)
                else:
                    y = y.to(device)
                z = x + y
                print(f"Операция сложения выполнена: {z.shape}")
                
                # Тестируем свертку (как в модели)
                conv = torch.nn.Conv3d(3, 16, kernel_size=3, padding=1)
                if device == "directml":
                    conv = conv.to(device_obj)
                else:
                    conv = conv.to(device)
                output = conv(x)
                print(f"3D свертка выполнена: {output.shape}")
            
        else:
            print("CPU режим - базовые операции")
            y = torch.randn_like(x)
            z = x + y
            print(f"Операция сложения выполнена: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании операций: {e}")
        return False

def test_model_creation(device):
    """Тест создания модели"""
    print(f"\n🏗 Тестирование создания модели на {device}...")
    
    try:
        from model import create_model
        
        # Создаем простую модель
        model_config = {
            'model_type': 'resnet3d',
            'model_name': 'resnet3d_18',  # Легкая модель для теста
            'num_classes': 14,
            'input_size': (32, 32, 32),  # Меньший размер для теста
            'dropout_rate': 0.1
        }
        
        model = create_model(model_config)
        
        # Перемещаем модель на устройство
        if device == "directml":
            try:
                import torch_directml
                device_obj = torch_directml.device()
                model = model.to(device_obj)
            except ImportError:
                print("DirectML недоступен, используем CPU")
                device = "cpu"
                model = model.to(device)
        else:
            model = model.to(device)
        
        print(f"Модель создана: {model.get_model_info()}")
        
        # Тестируем прямой проход
        test_input = torch.randn(1, 1, 32, 32, 32)
        if device == "directml":
            test_input = test_input.to(device_obj)
        else:
            test_input = test_input.to(device)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"Прямой проход выполнен: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании модели: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ AMD GPU ПОДДЕРЖКИ")
    print("=" * 60)
    
    # Тест 1: Определение GPU
    device = test_gpu_detection()
    
    # Тест 2: PyTorch с GPU
    pytorch_ok = test_pytorch_gpu()
    
    # Тест 3: Операции с тензорами
    tensor_ok = test_tensor_operations(device)
    
    # Тест 4: Создание модели
    model_ok = test_model_creation(device)
    
    # Результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    print(f"🔍 Определение GPU: {'✅' if device else '❌'}")
    print(f"🧪 PyTorch с GPU: {'✅' if pytorch_ok else '❌'}")
    print(f"🔢 Операции с тензорами: {'✅' if tensor_ok else '❌'}")
    print(f"🏗 Создание модели: {'✅' if model_ok else '❌'}")
    
    all_ok = pytorch_ok and tensor_ok and model_ok
    
    if all_ok:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print(f"Система готова к работе с {device.upper()}")
        
        if device in ["rocm", "directml", "cuda"]:
            print("\n🚀 Для запуска обучения используйте:")
            print("python main.py train --epochs 10")
        else:
            print("\n⚠️  Рекомендуется использовать GPU для лучшей производительности")
            print("python main.py train --epochs 10 --cpu")
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("Проверьте установку ROCm и PyTorch")
        
        if device == "rocm":
            print("\n🔧 Для исправления проблем:")
            print("1. Убедитесь что ROCm установлен: rocm-smi")
            print("2. Переустановите PyTorch: python install_amd_gpu.py")
            print("3. Проверьте переменные окружения ROCm")
        elif device == "directml":
            print("\n🔧 Для исправления проблем:")
            print("1. Установите DirectML: pip install torch-directml")
            print("2. Переустановите PyTorch: python install_amd_windows.py")
            print("3. Проверьте драйверы AMD")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
