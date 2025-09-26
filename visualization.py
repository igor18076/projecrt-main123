"""
Визуализация процесса обучения и результатов модели
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

from config import PATHOLOGY_LABELS

class TrainingVisualizer:
    """Визуализация процесса обучения"""
    
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           train_metrics: List[Dict],
                           val_metrics: List[Dict],
                           save_path: Optional[str] = None):
        """Построение кривых обучения"""
        
        epochs = range(1, len(train_losses) + 1)
        
        # Создаем субплоты
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Кривые обучения модели', fontsize=16, fontweight='bold')
        
        # Потери
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Обучение', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Валидация', linewidth=2)
        axes[0, 0].set_title('Функция потерь', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Потери')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        train_aucs = [m['auc_mean'] for m in train_metrics]
        val_aucs = [m['auc_mean'] for m in val_metrics]
        axes[0, 1].plot(epochs, train_aucs, 'b-', label='Обучение', linewidth=2)
        axes[0, 1].plot(epochs, val_aucs, 'r-', label='Валидация', linewidth=2)
        axes[0, 1].set_title('AUC Score', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Precision
        train_aps = [m['ap_mean'] for m in train_metrics]
        val_aps = [m['ap_mean'] for m in val_metrics]
        axes[1, 0].plot(epochs, train_aps, 'b-', label='Обучение', linewidth=2)
        axes[1, 0].plot(epochs, val_aps, 'r-', label='Валидация', linewidth=2)
        axes[1, 0].set_title('Average Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score (средний)
        train_f1s = []
        val_f1s = []
        for train_m, val_m in zip(train_metrics, val_metrics):
            train_f1 = np.mean([train_m.get(f'f1_{label}', 0) for label in PATHOLOGY_LABELS])
            val_f1 = np.mean([val_m.get(f'f1_{label}', 0) for label in PATHOLOGY_LABELS])
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
        
        axes[1, 1].plot(epochs, train_f1s, 'b-', label='Обучение', linewidth=2)
        axes[1, 1].plot(epochs, val_f1s, 'r-', label='Валидация', linewidth=2)
        axes[1, 1].set_title('F1 Score (средний)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Эпоха')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return save_path
    
    def plot_class_metrics(self, 
                         metrics: Dict[str, float],
                         metric_type: str = "AUC",
                         save_path: Optional[str] = None):
        """Построение метрик по классам"""
        
        # Извлекаем метрики для каждого класса
        class_metrics = []
        for label in PATHOLOGY_LABELS:
            metric_key = f'{metric_type.lower()}_{label}'
            if metric_key in metrics:
                class_metrics.append(metrics[metric_key])
            else:
                class_metrics.append(0.0)
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(PATHOLOGY_LABELS)), class_metrics, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(PATHOLOGY_LABELS))))
        
        ax.set_title(f'{metric_type} по классам патологий', fontsize=16, fontweight='bold')
        ax.set_xlabel('Патологии')
        ax.set_ylabel(metric_type)
        ax.set_xticks(range(len(PATHOLOGY_LABELS)))
        ax.set_xticklabels(PATHOLOGY_LABELS, rotation=45, ha='right')
        
        # Добавляем значения на столбцы
        for i, (bar, value) in enumerate(zip(bars, class_metrics)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, f'{metric_type.lower()}_by_class.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return save_path
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str] = None,
                            save_path: Optional[str] = None):
        """Построение матрицы ошибок для мультилейбельной классификации"""
        
        if class_names is None:
            class_names = PATHOLOGY_LABELS
        
        # Вычисляем матрицу ошибок для каждого класса
        fig, axes = plt.subplots(3, 5, figsize=(20, 15))
        fig.suptitle('Матрицы ошибок по классам', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names):
            if i >= len(axes):
                break
                
            # Матрица ошибок для класса i
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            # Нормализация
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Построение
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=['Нет', 'Есть'], yticklabels=['Нет', 'Есть'],
                       ax=axes[i])
            axes[i].set_title(f'{class_name}', fontweight='bold')
            axes[i].set_xlabel('Предсказание')
            axes[i].set_ylabel('Истина')
        
        # Скрываем лишние субплоты
        for i in range(len(class_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, 'confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return save_path
    
    def plot_roc_curves(self, 
                       y_true: np.ndarray,
                       y_prob: np.ndarray,
                       class_names: List[str] = None,
                       save_path: Optional[str] = None):
        """Построение ROC кривых"""
        
        if class_names is None:
            class_names = PATHOLOGY_LABELS
        
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if i >= y_true.shape[1]:
                break
                
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc = np.trapz(tpr, fpr)
            
            plt.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Доля ложных срабатываний (FPR)')
        plt.ylabel('Доля истинных срабатываний (TPR)')
        plt.title('ROC кривые по классам патологий', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return save_path
    
    def plot_precision_recall_curves(self, 
                                   y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   class_names: List[str] = None,
                                   save_path: Optional[str] = None):
        """Построение Precision-Recall кривых"""
        
        if class_names is None:
            class_names = PATHOLOGY_LABELS
        
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if i >= y_true.shape[1]:
                break
                
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap = np.trapz(precision, recall)
            
            plt.plot(recall, precision, color=color, linewidth=2,
                    label=f'{class_name} (AP = {ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Полнота (Recall)')
        plt.ylabel('Точность (Precision)')
        plt.title('Precision-Recall кривые по классам патологий', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, 'precision_recall_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return save_path
    
    def create_interactive_dashboard(self, 
                                   train_losses: List[float],
                                   val_losses: List[float],
                                   train_metrics: List[Dict],
                                   val_metrics: List[Dict],
                                   save_path: Optional[str] = None):
        """Создание интерактивной панели с Plotly"""
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Создаем субплоты
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Функция потерь', 'AUC Score', 'Average Precision', 'F1 Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Потери
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name='Обучение', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name='Валидация', line=dict(color='red')),
            row=1, col=1
        )
        
        # AUC
        train_aucs = [m['auc_mean'] for m in train_metrics]
        val_aucs = [m['auc_mean'] for m in val_metrics]
        fig.add_trace(
            go.Scatter(x=epochs, y=train_aucs, name='Обучение AUC', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_aucs, name='Валидация AUC', line=dict(color='red')),
            row=1, col=2
        )
        
        # Average Precision
        train_aps = [m['ap_mean'] for m in train_metrics]
        val_aps = [m['ap_mean'] for m in val_metrics]
        fig.add_trace(
            go.Scatter(x=epochs, y=train_aps, name='Обучение AP', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_aps, name='Валидация AP', line=dict(color='red')),
            row=2, col=1
        )
        
        # F1 Score
        train_f1s = []
        val_f1s = []
        for train_m, val_m in zip(train_metrics, val_metrics):
            train_f1 = np.mean([train_m.get(f'f1_{label}', 0) for label in PATHOLOGY_LABELS])
            val_f1 = np.mean([val_m.get(f'f1_{label}', 0) for label in PATHOLOGY_LABELS])
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
        
        fig.add_trace(
            go.Scatter(x=epochs, y=train_f1s, name='Обучение F1', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_f1s, name='Валидация F1', line=dict(color='red')),
            row=2, col=2
        )
        
        # Обновляем макет
        fig.update_layout(
            title_text="Интерактивная панель обучения модели",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Обновляем оси
        fig.update_xaxes(title_text="Эпоха")
        fig.update_yaxes(title_text="Значение")
        
        if save_path:
            fig.write_html(save_path)
        else:
            save_path = os.path.join(self.output_dir, 'training_dashboard.html')
            fig.write_html(save_path)
        
        return save_path

class ModelAnalyzer:
    """Анализ результатов модели"""
    
    def __init__(self, output_dir: str = "./outputs/analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_predictions(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray,
                           class_names: List[str] = None) -> Dict[str, any]:
        """Полный анализ предсказаний модели"""
        
        if class_names is None:
            class_names = PATHOLOGY_LABELS
        
        analysis = {}
        
        # Общие метрики
        analysis['overall_metrics'] = self._calculate_overall_metrics(y_true, y_pred, y_prob)
        
        # Метрики по классам
        analysis['class_metrics'] = self._calculate_class_metrics(y_true, y_pred, y_prob, class_names)
        
        # Анализ ошибок
        analysis['error_analysis'] = self._analyze_errors(y_true, y_pred, class_names)
        
        # Статистика предсказаний
        analysis['prediction_stats'] = self._calculate_prediction_stats(y_pred, y_prob, class_names)
        
        return analysis
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Вычисление общих метрик"""
        from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score
        
        metrics = {}
        
        # Точность (для мультилейбельной классификации)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Hamming Loss
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Jaccard Score
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro')
        
        # Средний AUC
        auc_scores = []
        for i in range(y_true.shape[1]):
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.0)
        metrics['mean_auc'] = np.mean(auc_scores)
        
        # Средний Average Precision
        ap_scores = []
        for i in range(y_true.shape[1]):
            try:
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(y_true[:, i], y_prob[:, i])
                ap_scores.append(ap)
            except ValueError:
                ap_scores.append(0.0)
        metrics['mean_ap'] = np.mean(ap_scores)
        
        return metrics
    
    def _calculate_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Вычисление метрик по классам"""
        class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            if i >= y_true.shape[1]:
                break
                
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
            tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            except ValueError:
                auc = 0.0
            
            # Average Precision
            try:
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(y_true[:, i], y_prob[:, i])
            except ValueError:
                ap = 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'ap': ap,
                'support': np.sum(y_true[:, i])
            }
        
        return class_metrics
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, any]:
        """Анализ ошибок модели"""
        error_analysis = {}
        
        # Общее количество ошибок
        total_errors = np.sum(y_true != y_pred)
        total_predictions = y_true.size
        error_rate = total_errors / total_predictions
        
        error_analysis['total_errors'] = total_errors
        error_analysis['total_predictions'] = total_predictions
        error_analysis['error_rate'] = error_rate
        
        # Ошибки по классам
        class_errors = {}
        for i, class_name in enumerate(class_names):
            if i >= y_true.shape[1]:
                break
                
            class_error = np.sum(y_true[:, i] != y_pred[:, i])
            class_total = y_true.shape[0]
            class_error_rate = class_error / class_total
            
            class_errors[class_name] = {
                'errors': class_error,
                'total': class_total,
                'error_rate': class_error_rate
            }
        
        error_analysis['class_errors'] = class_errors
        
        return error_analysis
    
    def _calculate_prediction_stats(self, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, any]:
        """Статистика предсказаний"""
        stats = {}
        
        # Средняя уверенность предсказаний
        mean_confidence = np.mean(y_prob)
        stats['mean_confidence'] = mean_confidence
        
        # Распределение предсказаний по классам
        class_predictions = {}
        for i, class_name in enumerate(class_names):
            if i >= y_pred.shape[1]:
                break
                
            positive_predictions = np.sum(y_pred[:, i])
            total_predictions = y_pred.shape[0]
            positive_rate = positive_predictions / total_predictions
            
            class_predictions[class_name] = {
                'positive_predictions': positive_predictions,
                'total_predictions': total_predictions,
                'positive_rate': positive_rate
            }
        
        stats['class_predictions'] = class_predictions
        
        return stats
    
    def save_analysis_report(self, analysis: Dict[str, any], save_path: Optional[str] = None):
        """Сохранение отчета анализа"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'model_analysis_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ АНАЛИЗА МОДЕЛИ\n")
            f.write("=" * 50 + "\n\n")
            
            # Общие метрики
            f.write("ОБЩИЕ МЕТРИКИ:\n")
            f.write("-" * 20 + "\n")
            for metric, value in analysis['overall_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Метрики по классам
            f.write("МЕТРИКИ ПО КЛАССАМ:\n")
            f.write("-" * 20 + "\n")
            for class_name, metrics in analysis['class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # Анализ ошибок
            f.write("\nАНАЛИЗ ОШИБОК:\n")
            f.write("-" * 20 + "\n")
            error_analysis = analysis['error_analysis']
            f.write(f"Общее количество ошибок: {error_analysis['total_errors']}\n")
            f.write(f"Общий процент ошибок: {error_analysis['error_rate']:.4f}\n")
            
            f.write("\nОшибки по классам:\n")
            for class_name, errors in error_analysis['class_errors'].items():
                f.write(f"  {class_name}: {errors['errors']}/{errors['total']} ({errors['error_rate']:.4f})\n")
        
        print(f"Отчет анализа сохранен: {save_path}")
        return save_path

