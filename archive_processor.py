"""
Обработчик архивов с DICOM файлами для анализа патологий
"""
import os
import zipfile
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pydicom
import numpy as np
import torch
from data_loader import DICOMDataset, create_dicom_loader

class ArchiveProcessor:
    """Обработчик архивов с DICOM файлами"""
    
    def __init__(self, temp_dir: str = "./temp_archives"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def extract_archive(self, archive_path: str) -> str:
        """
        Извлечение архива во временную папку
        
        Args:
            archive_path: Путь к архиву
            
        Returns:
            Путь к извлеченной папке
        """
        # Создаем уникальную временную папку
        temp_extract_dir = tempfile.mkdtemp(dir=self.temp_dir)
        
        try:
            if archive_path.lower().endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
            elif archive_path.lower().endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_extract_dir)
            else:
                raise ValueError(f"Неподдерживаемый формат архива: {archive_path}")
            
            return temp_extract_dir
            
        except Exception as e:
            # Очищаем временную папку при ошибке
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            raise e
    
    def find_dicom_files(self, directory: str) -> List[str]:
        """
        Поиск DICOM файлов в директории
        
        Args:
            directory: Путь к директории
            
        Returns:
            Список путей к DICOM файлам
        """
        dicom_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(root, file))
        
        return sorted(dicom_files)
    
    def validate_dicom_files(self, dicom_files: List[str]) -> Tuple[List[str], List[str]]:
        """
        Валидация DICOM файлов
        
        Args:
            dicom_files: Список путей к DICOM файлам
            
        Returns:
            Кортеж (валидные_файлы, невалидные_файлы)
        """
        valid_files = []
        invalid_files = []
        
        for file_path in dicom_files:
            try:
                # Пытаемся загрузить DICOM файл
                ds = pydicom.dcmread(file_path)
                
                # Проверяем наличие необходимых тегов
                if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                    valid_files.append(file_path)
                else:
                    invalid_files.append(file_path)
                    
            except Exception as e:
                invalid_files.append(file_path)
        
        return valid_files, invalid_files
    
    def get_dicom_info(self, dicom_files: List[str]) -> Dict[str, Any]:
        """
        Получение информации о DICOM файлах
        
        Args:
            dicom_files: Список путей к DICOM файлам
            
        Returns:
            Словарь с информацией о файлах
        """
        if not dicom_files:
            return {"error": "DICOM файлы не найдены"}
        
        info = {
            "total_files": len(dicom_files),
            "files": [],
            "series_info": {},
            "patient_info": {}
        }
        
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                
                file_info = {
                    "path": os.path.basename(file_path),
                    "size": os.path.getsize(file_path),
                    "modality": getattr(ds, 'Modality', 'Unknown'),
                    "series_uid": getattr(ds, 'SeriesInstanceUID', 'Unknown'),
                    "study_uid": getattr(ds, 'StudyInstanceUID', 'Unknown'),
                    "patient_id": getattr(ds, 'PatientID', 'Unknown'),
                    "study_date": getattr(ds, 'StudyDate', 'Unknown'),
                    "study_time": getattr(ds, 'StudyTime', 'Unknown')
                }
                
                info["files"].append(file_info)
                
                # Группируем по сериям
                series_uid = file_info["series_uid"]
                if series_uid not in info["series_info"]:
                    info["series_info"][series_uid] = {
                        "count": 0,
                        "modality": file_info["modality"],
                        "patient_id": file_info["patient_id"],
                        "study_date": file_info["study_date"]
                    }
                info["series_info"][series_uid]["count"] += 1
                
                # Группируем по пациентам
                patient_id = file_info["patient_id"]
                if patient_id not in info["patient_info"]:
                    info["patient_info"][patient_id] = {
                        "studies": set(),
                        "series_count": 0
                    }
                info["patient_info"][patient_id]["studies"].add(file_info["study_uid"])
                info["patient_info"][patient_id]["series_count"] += 1
                
            except Exception as e:
                info["files"].append({
                    "path": os.path.basename(file_path),
                    "error": str(e)
                })
        
        # Конвертируем set в list для JSON сериализации
        for patient_id in info["patient_info"]:
            info["patient_info"][patient_id]["studies"] = list(info["patient_info"][patient_id]["studies"])
        
        return info
    
    def process_archive(self, archive_path: str) -> Dict[str, Any]:
        """
        Полная обработка архива с DICOM файлами
        
        Args:
            archive_path: Путь к архиву
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            # Извлекаем архив
            extract_dir = self.extract_archive(archive_path)
            
            # Ищем DICOM файлы
            dicom_files = self.find_dicom_files(extract_dir)
            
            if not dicom_files:
                return {
                    "success": False,
                    "error": "DICOM файлы не найдены в архиве",
                    "extract_dir": extract_dir
                }
            
            # Валидируем файлы
            valid_files, invalid_files = self.validate_dicom_files(dicom_files)
            
            # Получаем информацию о файлах
            dicom_info = self.get_dicom_info(valid_files)
            
            result = {
                "success": True,
                "extract_dir": extract_dir,
                "total_files": len(dicom_files),
                "valid_files": len(valid_files),
                "invalid_files": len(invalid_files),
                "dicom_info": dicom_info,
                "valid_file_paths": valid_files,
                "invalid_file_paths": invalid_files
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extract_dir": None
            }
    
    def cleanup(self, extract_dir: str):
        """Очистка временных файлов"""
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
    
    def __del__(self):
        """Очистка при удалении объекта"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def analyze_dicom_archive(archive_path: str, model, device: str = "auto") -> Dict[str, Any]:
    """
    Анализ архива с DICOM файлами на предмет патологий
    
    Args:
        archive_path: Путь к архиву
        model: Обученная модель
        device: Устройство для инференса
        
    Returns:
        Словарь с результатами анализа
    """
    processor = ArchiveProcessor()
    
    try:
        # Обрабатываем архив
        archive_result = processor.process_archive(archive_path)
        
        if not archive_result["success"]:
            return archive_result
        
        if not archive_result["valid_file_paths"]:
            return {
                "success": False,
                "error": "Нет валидных DICOM файлов для анализа"
            }
        
        # Создаем датасет из валидных файлов
        dicom_dataset = DICOMDataset(archive_result["valid_file_paths"])
        dicom_loader = create_dicom_loader(archive_result["valid_file_paths"], batch_size=1)
        
        # Анализируем каждый файл
        model.eval()
        results = []
        
        # Настраиваем устройство для DirectML
        if device == "directml":
            try:
                import torch_directml
                device_obj = torch_directml.device()
            except ImportError:
                device_obj = "cpu"
        else:
            device_obj = device
        
        with torch.no_grad():
            for batch in dicom_loader:
                volumes = batch['volume'].to(device_obj)
                file_path = batch['file_path'][0]
                
                # Получаем предсказания
                outputs = model(volumes)
                predictions, probabilities = model.predict(volumes, threshold=0.5)
                
                # Формируем результат
                result = {
                    'file_path': os.path.basename(file_path),
                    'predictions': {},
                    'probabilities': {},
                    'has_pathology': False,
                    'pathology_count': 0
                }
                
                # Анализируем каждую патологию
                for i, label in enumerate(PATHOLOGY_LABELS):
                    prob = float(probabilities[0, i].item())
                    pred = bool(predictions[0, i].item())
                    
                    result['predictions'][label] = pred
                    result['probabilities'][label] = prob
                    
                    if pred:
                        result['has_pathology'] = True
                        result['pathology_count'] += 1
                
                results.append(result)
        
        # Общий анализ исследования
        total_files = len(results)
        files_with_pathology = sum(1 for r in results if r['has_pathology'])
        total_pathologies = sum(r['pathology_count'] for r in results)
        
        # Определяем общий статус
        if files_with_pathology == 0:
            overall_status = "Норма"
            confidence = 1.0
        elif files_with_pathology < total_files * 0.3:
            overall_status = "Сомнительно"
            confidence = 0.5
        else:
            overall_status = "Патология"
            confidence = 0.8
        
        # Находим наиболее частые патологии
        pathology_counts = {}
        for result in results:
            for pathology, pred in result['predictions'].items():
                if pred:
                    pathology_counts[pathology] = pathology_counts.get(pathology, 0) + 1
        
        most_common_pathologies = sorted(
            pathology_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        analysis_result = {
            "success": True,
            "archive_info": archive_result["dicom_info"],
            "analysis_summary": {
                "total_files_analyzed": total_files,
                "files_with_pathology": files_with_pathology,
                "total_pathologies_detected": total_pathologies,
                "overall_status": overall_status,
                "confidence": confidence,
                "most_common_pathologies": most_common_pathologies
            },
            "detailed_results": results,
            "extract_dir": archive_result["extract_dir"]
        }
        
        return analysis_result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "extract_dir": None
        }
    finally:
        # Очищаем временные файлы
        if 'extract_dir' in locals() and extract_dir:
            processor.cleanup(extract_dir)
