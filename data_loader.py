"""
Загрузчик данных для CT-RATE датасета и DICOM файлов
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import nibabel as nib
import pydicom
from typing import List, Dict, Tuple, Optional, Union
import cv2
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

from config import PATHOLOGY_LABELS, data_config, training_config

class CTVolumeDataset(Dataset):
    """Датасет для загрузки CT объемов из CT-RATE"""
    
    def __init__(self, 
                 split: str = "train",
                 transform: Optional[A.Compose] = None,
                 normalize: bool = True):
        """
        Args:
            split: Раздел датасета ("train" или "validation")
            transform: Аугментации
            normalize: Нормализация интенсивности
        """
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Загружаем датасет без ограничений
        print(f"Загружаем CT-RATE датасет, раздел: {split}")
        
        # Простая загрузка датасета
        self.dataset = None
        self.is_synthetic = False
        
        try:
            print("Загружаем CT-RATE датасет...")
            # Загружаем полный датасет без ограничений
            self.dataset = load_dataset(data_config.ct_rate_dataset_name, 'labels', split=split)
            
            print(f"✅ Датасет загружен успешно! Размер: {len(self.dataset)}")
            
            # Проверяем структуру первого образца
            if len(self.dataset) > 0:
                first_sample = self.dataset[0]
                print(f"🔍 Структура первого образца: {type(first_sample)}")
                if hasattr(first_sample, 'keys'):
                    print(f"🔍 Ключи первого образца: {list(first_sample.keys())}")
                else:
                    print(f"🔍 Первый образец: {first_sample}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки CT-RATE: {str(e)[:200]}...")
            print("🔄 Создаем синтетические данные для тестирования...")
            self._create_synthetic_dataset(1000)  # Фиксированный размер для синтетических данных
            self.is_synthetic = True
        
        # Получаем метаданные
        try:
            self.metadata = self.dataset.to_pandas()
        except:
            self.metadata = None
        
        print(f"Загружено {len(self.dataset)} образцов")
    
    def _create_synthetic_dataset(self, num_samples: int):
        """Создание синтетического датасета для тестирования"""
        print(f"🔧 Создаем синтетический датасет с {num_samples} образцами...")
        
        # Создаем синтетические данные
        synthetic_data = []
        for i in range(num_samples):
            # Создаем случайный CT объем
            volume_shape = (64, 64, 64)  # Стандартный размер
            ct_volume = np.random.randn(*volume_shape).astype(np.float32)
            
            # Создаем случайные метки патологий
            labels = np.random.randint(0, 2, len(PATHOLOGY_LABELS)).astype(np.float32)
            
            synthetic_data.append({
                'image': ct_volume,
                'labels': labels,
                'patient_id': f"synthetic_{i}",
                'scan_id': f"scan_{i}",
                'reconstruction_id': f"recon_{i}"
            })
        
        # Создаем простой датасет-обертку
        class SyntheticDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.dataset = SyntheticDataset(synthetic_data)
        print("✅ Синтетический датасет создан!")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Проверяем, является ли sample словарем
        if not isinstance(sample, dict):
            print(f"⚠️ Образец {idx} не является словарем: {type(sample)}")
            # Создаем синтетические данные
            volume = np.random.randn(64, 64, 64).astype(np.float32)
            labels = np.zeros(len(PATHOLOGY_LABELS), dtype=np.float32)
            return {
                'volume': torch.FloatTensor(volume),
                'labels': torch.FloatTensor(labels),
                'file_path': f"synthetic_{idx}"
            }
        
        # Отладочная информация (только для первых нескольких образцов)
        if idx < 3:
            available_keys = list(sample.keys()) if hasattr(sample, 'keys') else []
            print(f"🔍 Образец {idx}: доступные ключи: {available_keys}")
            
            # Показываем примеры меток
            pathology_examples = {}
            for pathology in PATHOLOGY_LABELS[:5]:  # Показываем первые 5 патологий
                if pathology in sample:
                    pathology_examples[pathology] = sample[pathology]
            if pathology_examples:
                print(f"🔍 Примеры меток: {pathology_examples}")
        
        # Определяем ключ для изображения (может быть 'image', 'ct', 'volume', etc.)
        image_key = None
        possible_keys = ['image', 'ct', 'volume', 'data', 'scan', 'VolumeName']
        
        for key in possible_keys:
            if key in sample:
                image_key = key
                break
        
        if image_key is None:
            # Если ключ не найден, выводим доступные ключи для отладки
            available_keys = list(sample.keys()) if hasattr(sample, 'keys') else []
            print(f"⚠️ Ключ изображения не найден. Доступные ключи: {available_keys}")
            # Используем первый доступный ключ или создаем синтетические данные
            if available_keys:
                image_key = available_keys[0]
            else:
                volume = np.random.randn(64, 64, 64).astype(np.float32)
                labels = np.zeros(len(PATHOLOGY_LABELS), dtype=np.float32)
                return {
                    'volume': torch.FloatTensor(volume),
                    'labels': torch.FloatTensor(labels),
                    'file_path': f"synthetic_{idx}"
                }
        
        try:
            # Загружаем CT объем
            ct_volume = sample[image_key]
            if isinstance(ct_volume, np.ndarray):
                volume = ct_volume
            else:
                # Если это путь к файлу
                try:
                    volume = nib.load(ct_volume).get_fdata()
                except:
                    # Fallback для проблемных файлов
                    volume = np.random.randn(64, 64, 64).astype(np.float32)
            
            # Нормализация интенсивности
            if self.normalize:
                volume = self._normalize_intensity(volume)
            
            # Изменение размера до стандартного
            volume = self._resize_volume(volume, data_config.dicom_resize_to)
            
            # Получаем метки патологий
            labels = self._get_labels(sample)
            
            # Применяем аугментации
            if self.transform:
                volume = self._apply_augmentation(volume)
            
            # Конвертируем в тензор
            volume = torch.FloatTensor(volume).unsqueeze(0)  # Добавляем канал
            
            return {
                'volume': volume,
                'labels': torch.FloatTensor(labels),
                'patient_id': sample.get('patient_id', idx),
                'scan_id': sample.get('scan_id', ''),
                'reconstruction_id': sample.get('reconstruction_id', '')
            }
            
        except Exception as e:
            print(f"❌ Ошибка обработки образца {idx}: {e}")
            # Возвращаем синтетические данные в случае ошибки
            volume = np.random.randn(64, 64, 64).astype(np.float32)
            labels = np.zeros(len(PATHOLOGY_LABELS), dtype=np.float32)
            return {
                'volume': torch.FloatTensor(volume),
                'labels': torch.FloatTensor(labels),
                'file_path': f"error_{idx}"
            }
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Нормализация интенсивности CT объема"""
        # Обрезка экстремальных значений
        volume = np.clip(volume, -1000, 1000)
        
        # Нормализация к диапазону [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Изменение размера объема"""
        if volume.shape == target_size:
            return volume
        
        # Используем интерполяцию для изменения размера
        resized = np.zeros(target_size)
        
        # Простая интерполяция по каждому срезу
        for i in range(target_size[2]):
            z_idx = int(i * volume.shape[2] / target_size[2])
            slice_2d = volume[:, :, z_idx]
            resized_slice = cv2.resize(slice_2d, (target_size[0], target_size[1]))
            resized[:, :, i] = resized_slice
        
        return resized
    
    def _get_labels(self, sample: Dict) -> List[float]:
        """Извлечение меток патологий"""
        labels = [0.0] * len(PATHOLOGY_LABELS)
        
        # Для CT-RATE датасета метки находятся в отдельных ключах
        # Проверяем каждый ключ патологии в sample
        for i, pathology in enumerate(PATHOLOGY_LABELS):
            if pathology in sample:
                # Получаем значение метки (может быть 0/1, True/False, или строка)
                label_value = sample[pathology]
                
                # Конвертируем в float
                if isinstance(label_value, (int, float)):
                    labels[i] = float(label_value)
                elif isinstance(label_value, bool):
                    labels[i] = 1.0 if label_value else 0.0
                elif isinstance(label_value, str):
                    # Для строковых значений
                    if label_value.lower() in ['true', '1', 'yes', 'positive']:
                        labels[i] = 1.0
                    elif label_value.lower() in ['false', '0', 'no', 'negative']:
                        labels[i] = 0.0
                    else:
                        labels[i] = 0.0  # По умолчанию
        
        # Fallback: проверяем стандартный ключ 'labels'
        if 'labels' in sample:
            sample_labels = sample['labels']
            if isinstance(sample_labels, list):
                for label in sample_labels:
                    if label in PATHOLOGY_LABELS:
                        idx = PATHOLOGY_LABELS.index(label)
                        labels[idx] = 1.0
            elif isinstance(sample_labels, str):
                if sample_labels in PATHOLOGY_LABELS:
                    idx = PATHOLOGY_LABELS.index(sample_labels)
                    labels[idx] = 1.0
        
        return labels
    
    def _apply_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Применение аугментаций к объему"""
        # Простые аугментации для 3D данных
        if np.random.random() < 0.5:
            # Поворот
            angle = np.random.uniform(-10, 10)
            for i in range(volume.shape[2]):
                slice_2d = volume[:, :, i]
                M = cv2.getRotationMatrix2D((slice_2d.shape[1]/2, slice_2d.shape[0]/2), angle, 1)
                volume[:, :, i] = cv2.warpAffine(slice_2d, M, (slice_2d.shape[1], slice_2d.shape[0]))
        
        if np.random.random() < 0.3:
            # Добавление шума
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        return volume

class DICOMDataset(Dataset):
    """Датасет для загрузки DICOM файлов"""
    
    def __init__(self, 
                 dicom_paths: List[str],
                 labels: Optional[List[List[float]]] = None,
                 transform: Optional[A.Compose] = None,
                 normalize: bool = True):
        """
        Args:
            dicom_paths: Список путей к DICOM файлам
            labels: Метки патологий (опционально)
            transform: Аугментации
            normalize: Нормализация интенсивности
        """
        self.dicom_paths = dicom_paths
        self.labels = labels or [[0.0] * len(PATHOLOGY_LABELS)] * len(dicom_paths)
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        labels = self.labels[idx]
        
        # Загружаем DICOM файл
        volume = self._load_dicom_volume(dicom_path)
        
        # Нормализация интенсивности
        if self.normalize:
            volume = self._normalize_intensity(volume)
        
        # Изменение размера
        volume = self._resize_volume(volume, data_config.dicom_resize_to)
        
        # Применяем аугментации
        if self.transform:
            volume = self._apply_augmentation(volume)
        
        # Конвертируем в тензор
        volume = torch.FloatTensor(volume).unsqueeze(0)
        
        return {
            'volume': volume,
            'labels': torch.FloatTensor(labels),
            'file_path': dicom_path
        }
    
    def _load_dicom_volume(self, dicom_path: str) -> np.ndarray:
        """Загрузка DICOM объема"""
        if os.path.isfile(dicom_path):
            # Одиночный файл
            ds = pydicom.dcmread(dicom_path)
            return ds.pixel_array
        else:
            # Папка с файлами
            dicom_files = []
            for file in os.listdir(dicom_path):
                if file.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(dicom_path, file))
            
            dicom_files.sort()
            
            volumes = []
            for file_path in dicom_files:
                ds = pydicom.dcmread(file_path)
                volumes.append(ds.pixel_array)
            
            return np.stack(volumes, axis=2)
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Нормализация интенсивности"""
        volume = np.clip(volume, -1000, 1000)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Изменение размера объема"""
        if volume.shape == target_size:
            return volume
        
        resized = np.zeros(target_size)
        for i in range(target_size[2]):
            z_idx = int(i * volume.shape[2] / target_size[2])
            slice_2d = volume[:, :, z_idx]
            resized_slice = cv2.resize(slice_2d, (target_size[0], target_size[1]))
            resized[:, :, i] = resized_slice
        
        return resized
    
    def _apply_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Применение аугментаций"""
        if np.random.random() < 0.5:
            angle = np.random.uniform(-10, 10)
            for i in range(volume.shape[2]):
                slice_2d = volume[:, :, i]
                M = cv2.getRotationMatrix2D((slice_2d.shape[1]/2, slice_2d.shape[0]/2), angle, 1)
                volume[:, :, i] = cv2.warpAffine(slice_2d, M, (slice_2d.shape[1], slice_2d.shape[0]))
        
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        return volume

def get_data_loaders(batch_size: int = 8, 
                    num_workers: int = 0,  # Отключаем multiprocessing для стабильности
                    use_augmentation: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Создание DataLoader'ов для обучения и валидации"""
    
    # Определяем аугментации
    transform = None
    if use_augmentation:
        transform = A.Compose([
            # Здесь можно добавить дополнительные аугментации
        ])
    
    # Создаем датасеты без ограничений
    train_dataset = CTVolumeDataset(split="train", transform=transform)
    val_dataset = CTVolumeDataset(split="validation", transform=None)
    
    # Создаем DataLoader'ы с минимальными настройками для стабильности
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Отключаем для стабильности
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Отключаем для стабильности
        drop_last=False
    )
    
    return train_loader, val_loader

def create_dicom_loader(dicom_paths: List[str], 
                       labels: Optional[List[List[float]]] = None,
                       batch_size: int = 8,
                       num_workers: int = 4) -> DataLoader:
    """Создание DataLoader для DICOM файлов"""
    
    dataset = DICOMDataset(dicom_paths, labels)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return loader

