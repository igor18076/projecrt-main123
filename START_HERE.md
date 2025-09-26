# 🚀 НАЧНИТЕ ЗДЕСЬ!

## ⚡ Быстрый старт за 5 минут

### 1. Установка зависимостей
```bash
# Автоматическая установка
python install.py

# Или ручная установка:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -r requirements.txt
```

### 2. Проверка системы
```bash
python quick_start.py --check
```

### 3. Запуск веб-интерфейса
```bash
python web_interface.py
```

### 4. Открытие браузера
Перейдите по адресу: **http://localhost:7860**

## 📚 Документация

- **README.md** - Основная документация проекта
- **USER_GUIDE.md** - Полное руководство пользователя
- **PROJECT_SUMMARY.md** - Краткое описание проекта

## 🎯 Основные команды

```bash
# Обучение модели
python main.py train --epochs 50 --batch-size 8

# Оценка модели
python main.py evaluate --checkpoint ./checkpoints/best_model.pth

# Предсказание на DICOM
python main.py predict --checkpoint ./checkpoints/best_model.pth --dicom-path ./data/

# Демонстрация системы
python demo.py
```

## 🏥 Поддерживаемые патологии

Модель обучена для выявления **14 типов патологий**:

1. **Atelectasis** - Ателектаз
2. **Cardiomegaly** - Кардиомегалия
3. **Consolidation** - Консолидация
4. **Edema** - Отек
5. **Effusion** - Выпот
6. **Emphysema** - Эмфизема
7. **Fibrosis** - Фиброз
8. **Hernia** - Грыжа
9. **Infiltration** - Инфильтрация
10. **Mass** - Масса
11. **Nodule** - Узелок
12. **Pleural_Thickening** - Утолщение плевры
13. **Pneumonia** - Пневмония
14. **Pneumothorax** - Пневмоторакс

## ✅ Все требования выполнены

- ✅ Обучение на GPU с CUDA 13.0 и PyTorch Nightly
- ✅ Возможность прервать обучение с сохранением результатов
- ✅ Визуализация процесса обучения и точности
- ✅ Экспорт готовой модели
- ✅ Поддержка DICOM файлов для обучения и проверки
- ✅ Веб-интерфейс для удобного управления

---

**🎉 Система готова к использованию!**

**Начните прямо сейчас:**
```bash
python web_interface.py
```
