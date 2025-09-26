"""
Конфигурация для обучения модели выявления патологий на КТ грудной клетки
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    model_name: str = "resnet3d_50"
    num_classes: int = 14  # Количество классов патологий в CT-RATE
    input_size: tuple = field(default_factory=lambda: (64, 64, 64))  # Размер входного объема
    dropout_rate: float = 0.5
    pretrained: bool = True

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    patience: int = 10  # Early stopping patience
    
    # Пути
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # GPU настройки
    device: str = "auto"  # Автоматическое определение устройства
    num_workers: int = 4
    pin_memory: bool = True
    
    # Сохранение и логирование
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "ct-pathology-detection"
    
    # Экспорт модели
    export_formats: List[str] = field(default_factory=lambda: ["pytorch", "onnx"])

@dataclass
class DataConfig:
    """Конфигурация данных"""
    # CT-RATE датасет
    ct_rate_dataset_name: str = "ibrahimhamamci/CT-RATE"
    train_split: str = "train"
    val_split: str = "validation"
    
    # Предобработка
    normalize_intensity: bool = True
    augment_data: bool = True
    augmentation_prob: float = 0.5
    
    # DICOM настройки
    dicom_supported_formats: List[str] = field(default_factory=lambda: [".dcm", ".DCM"])
    dicom_resize_to: tuple = field(default_factory=lambda: (64, 64, 64))

# Патологии в CT-RATE датасете
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly", 
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# Создание глобальных конфигураций
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()

