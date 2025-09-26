"""
Тренер для обучения модели классификации патологий КТ
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from typing import Dict, List, Tuple, Optional, Any
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from model import CTPathologyModel, create_model
from data_loader import get_data_loaders, create_dicom_loader
from config import training_config, model_config, data_config, PATHOLOGY_LABELS
from gpu_utils import setup_device, print_device_info, optimize_for_device, get_device_object

class EarlyStopping:
    """Ранняя остановка обучения"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Сохранение лучших весов"""
        self.best_weights = model.state_dict().copy()

class MetricsCalculator:
    """Вычисление метрик для мультилейбельной классификации"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Вычисление всех метрик"""
        metrics = {}
        
        # AUC для каждого класса
        auc_scores = []
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                auc_scores.append(auc)
                metrics[f'auc_{self.class_names[i]}'] = auc
            except ValueError:
                metrics[f'auc_{self.class_names[i]}'] = 0.0
                auc_scores.append(0.0)
        
        # Средний AUC
        metrics['auc_mean'] = np.mean(auc_scores)
        
        # Average Precision для каждого класса
        ap_scores = []
        for i in range(self.num_classes):
            try:
                ap = average_precision_score(y_true[:, i], y_prob[:, i])
                ap_scores.append(ap)
                metrics[f'ap_{self.class_names[i]}'] = ap
            except ValueError:
                metrics[f'ap_{self.class_names[i]}'] = 0.0
                ap_scores.append(0.0)
        
        # Средний Average Precision
        metrics['ap_mean'] = np.mean(ap_scores)
        
        # Точность, полнота, F1 для каждого класса
        for i in range(self.num_classes):
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
            tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'precision_{self.class_names[i]}'] = precision
            metrics[f'recall_{self.class_names[i]}'] = recall
            metrics[f'f1_{self.class_names[i]}'] = f1
        
        return metrics

class CTTrainer:
    """Тренер для модели классификации патологий КТ"""
    
    def __init__(self, 
                 model: CTPathologyModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = "auto"):
        
        # Настраиваем устройство
        if device == "auto":
            device = setup_device(config.get("device", "auto"))
        
        # Получаем объект устройства
        self.device_obj = get_device_object(device)
        self.model = model.to(self.device_obj)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Оптимизируем для устройства
        optimize_for_device(device)
        
        # Оптимизатор и планировщик
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("warmup_epochs", 5),
            eta_min=1e-6
        )
        
        # Ранняя остановка
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=0.001
        )
        
        # Вычисление метрик
        self.metrics_calculator = MetricsCalculator(
            num_classes=model.num_classes,
            class_names=PATHOLOGY_LABELS
        )
        
        # Логирование
        self.setup_logging()
        
        # Состояние обучения
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def setup_logging(self):
        """Настройка логирования"""
        # Создаем директории
        os.makedirs(self.config.get("output_dir", "./outputs"), exist_ok=True)
        os.makedirs(self.config.get("checkpoint_dir", "./checkpoints"), exist_ok=True)
        os.makedirs(self.config.get("log_dir", "./logs"), exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.config.get("log_dir", "./logs"), "tensorboard")
        )
        
        # Weights & Biases
        if self.config.get("use_wandb", True):
            wandb.init(
                project=self.config.get("wandb_project", "ct-pathology-detection"),
                config=self.config,
                name=f"ct_model_{int(time.time())}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            volumes = batch['volume'].to(self.device_obj)
            targets = batch['labels'].to(self.device_obj)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(volumes)
            loss = self.model.compute_loss(outputs, targets)
            
            # Обратный проход
            loss.backward()
            self.optimizer.step()
            
            # Собираем метрики
            total_loss += loss.item()
            
            with torch.no_grad():
                predictions, probabilities = self.model.predict(volumes)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(torch.sigmoid(outputs).cpu().numpy())
            
            # Обновляем прогресс бар
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Логирование
            if batch_idx % self.config.get("log_every_n_steps", 10) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     self.current_epoch * len(self.train_loader) + batch_idx)
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                volumes = batch['volume'].to(self.device_obj)
                targets = batch['labels'].to(self.device_obj)
                
                # Прямой проход
                outputs = self.model(volumes)
                loss = self.model.compute_loss(outputs, targets)
                
                # Собираем метрики
                total_loss += loss.item()
                
                predictions, probabilities = self.model.predict(volumes)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'model_config': self.model.get_model_info(),
            'training_config': self.config
        }
        
        # Сохраняем чекпоинт
        checkpoint_path = os.path.join(
            self.config.get("checkpoint_dir", "./checkpoints"),
            f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучшую модель
        if is_best:
            best_path = os.path.join(
                self.config.get("checkpoint_dir", "./checkpoints"),
                "best_model.pth"
            )
            torch.save(checkpoint, best_path)
            print(f"Сохранена лучшая модель: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка чекпоинта"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        print(f"Загружен чекпоинт эпохи {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None):
        """Основной цикл обучения"""
        
        # Загружаем чекпоинт если указан
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Начинаем обучение на {num_epochs} эпох")
        print(f"Устройство: {self.device}")
        if self.device == "directml":
            try:
                import torch_directml
                print(f"DirectML устройств: {torch_directml.device_count()}")
            except ImportError:
                pass
        print(f"Размер батча: {self.config.get('batch_size', 8)}")
        print(f"Количество классов: {self.model.num_classes}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            print(f"\n--- Эпоха {epoch + 1}/{num_epochs} ---")
            
            # Обучение
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            # Валидация
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Обновляем планировщик
            self.scheduler.step()
            
            # Логирование метрик
            self.log_metrics(train_metrics, val_metrics, epoch)
            
            # Проверяем на лучшую модель
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Сохраняем чекпоинт
            if epoch % self.config.get("save_every_n_epochs", 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Ранняя остановка
            if self.early_stopping(val_metrics['loss'], self.model):
                print(f"Ранняя остановка на эпохе {epoch + 1}")
                break
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train AUC: {train_metrics['auc_mean']:.4f}, Val AUC: {val_metrics['auc_mean']:.4f}")
        
        # Сохраняем финальную модель
        self.save_checkpoint(epoch, is_best)
        
        # Закрываем логирование
        self.writer.close()
        if self.config.get("use_wandb", True):
            wandb.finish()
        
        total_time = time.time() - start_time
        print(f"\nОбучение завершено за {total_time:.2f} секунд")
        print(f"Лучшая валидационная потеря: {self.best_val_loss:.4f}")
    
    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """Логирование метрик"""
        # TensorBoard
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        self.writer.add_scalar('AUC/Train', train_metrics['auc_mean'], epoch)
        self.writer.add_scalar('AUC/Validation', val_metrics['auc_mean'], epoch)
        self.writer.add_scalar('AP/Train', train_metrics['ap_mean'], epoch)
        self.writer.add_scalar('AP/Validation', val_metrics['ap_mean'], epoch)
        
        # Weights & Biases
        if self.config.get("use_wandb", True):
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_auc': train_metrics['auc_mean'],
                'val_auc': val_metrics['auc_mean'],
                'train_ap': train_metrics['ap_mean'],
                'val_ap': val_metrics['ap_mean'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)

def create_trainer(model_config: Dict[str, Any], 
                  training_config: Dict[str, Any],
                  device) -> CTTrainer:
    """Создание тренера"""
    
    # Создаем модель
    model = create_model(model_config)
    
    # Создаем загрузчики данных
    train_loader, val_loader = get_data_loaders(
        batch_size=training_config.get("batch_size", 8),
        num_workers=training_config.get("num_workers", 4)
    )
    
    # Создаем тренер
    trainer = CTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    return trainer

