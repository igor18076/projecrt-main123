"""
Архитектура модели для классификации патологий на КТ грудной клетки
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any
import timm

from config import model_config, PATHOLOGY_LABELS

class ResNet3D(nn.Module):
    """3D ResNet для анализа медицинских изображений"""
    
    def __init__(self, 
                 model_name: str = "resnet3d_50",
                 num_classes: int = 14,
                 input_size: tuple = (64, 64, 64),
                 dropout_rate: float = 0.5,
                 pretrained: bool = True):
        super(ResNet3D, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Базовая архитектура ResNet3D
        if model_name == "resnet3d_50":
            self.backbone = self._create_resnet3d_50()
        elif model_name == "resnet3d_18":
            self.backbone = self._create_resnet3d_18()
        else:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Инициализация весов
        self._initialize_weights()
    
    def _create_resnet3d_50(self) -> nn.Module:
        """Создание ResNet3D-50"""
        return nn.Sequential(
            # Первый блок
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            # ResNet блоки
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
        )
    
    def _create_resnet3d_18(self) -> nn.Module:
        """Создание ResNet3D-18"""
        return nn.Sequential(
            # Первый блок
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            # ResNet блоки
            self._make_layer(64, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Module:
        """Создание слоя ResNet"""
        layers = []
        
        # Первый блок с изменением размера
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        
        # Остальные блоки
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        # Проверяем размер входа
        if x.dim() == 4:  # Добавляем канал если его нет
            x = x.unsqueeze(1)
        
        # Изменяем размер до стандартного
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=False)
        
        # Проходим через backbone
        x = self.backbone(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Классификация
        x = self.classifier(x)
        
        return x

class BasicBlock3D(nn.Module):
    """Базовый блок ResNet3D"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class EfficientNet3D(nn.Module):
    """3D версия EfficientNet для медицинских изображений"""
    
    def __init__(self, 
                 model_name: str = "efficientnet_b0",
                 num_classes: int = 14,
                 input_size: tuple = (64, 64, 64),
                 dropout_rate: float = 0.5):
        super(EfficientNet3D, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Используем 2D EfficientNet и адаптируем для 3D
        self.backbone_2d = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Адаптация для 3D
        self.conv3d_adapt = nn.Conv3d(1, 3, kernel_size=1)
        
        # Классификатор
        feature_dim = self.backbone_2d.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Адаптация для 2D модели
        x = self.conv3d_adapt(x)  # [B, 3, D, H, W]
        
        # Обрабатываем каждый срез отдельно
        features = []
        for i in range(x.size(2)):
            slice_2d = x[:, :, i, :, :]  # [B, 3, H, W]
            feat = self.backbone_2d(slice_2d)  # [B, feature_dim]
            features.append(feat)
        
        # Объединяем признаки всех срезов
        features = torch.stack(features, dim=1)  # [B, D, feature_dim]
        features = torch.mean(features, dim=1)  # [B, feature_dim]
        
        # Классификация
        output = self.classifier(features)
        
        return output

class CTPathologyModel(nn.Module):
    """Основная модель для классификации патологий КТ"""
    
    def __init__(self, 
                 model_type: str = "resnet3d",
                 model_name: str = "resnet3d_50",
                 num_classes: int = 14,
                 input_size: tuple = (64, 64, 64),
                 dropout_rate: float = 0.5,
                 pretrained: bool = True):
        super(CTPathologyModel, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Выбираем архитектуру модели
        if model_type == "resnet3d":
            self.backbone = ResNet3D(
                model_name=model_name,
                num_classes=num_classes,
                input_size=input_size,
                dropout_rate=dropout_rate,
                pretrained=pretrained
            )
        elif model_type == "efficientnet3d":
            self.backbone = EfficientNet3D(
                model_name=model_name,
                num_classes=num_classes,
                input_size=input_size,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Функция потерь
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        return self.backbone(x)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисление потерь"""
        return self.criterion(outputs, targets)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Предсказание с порогом"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        return predictions, probabilities
    
    def predict_normal_vs_pathology(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Предсказание нормы vs патологии"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            
            # Определяем есть ли патологии
            has_pathology = torch.any(predictions, dim=1)
            pathology_count = torch.sum(predictions, dim=1)
            
            # Вычисляем общую уверенность
            max_prob = torch.max(probabilities, dim=1)[0]
            avg_prob = torch.mean(probabilities, dim=1)
            
            # Определяем статус
            status = []
            confidence = []
            
            for i in range(x.size(0)):
                if pathology_count[i] == 0:
                    status.append("Норма")
                    confidence.append(float(1.0 - avg_prob[i]))  # Высокая уверенность в норме
                elif pathology_count[i] <= 2:
                    status.append("Сомнительно")
                    confidence.append(float(max_prob[i]))
                else:
                    status.append("Патология")
                    confidence.append(float(max_prob[i]))
            
            return {
                "has_pathology": has_pathology.cpu().numpy(),
                "pathology_count": pathology_count.cpu().numpy(),
                "status": status,
                "confidence": confidence,
                "predictions": predictions.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pathology_labels": PATHOLOGY_LABELS
        }

def create_model(model_config: Dict[str, Any]) -> CTPathologyModel:
    """Создание модели по конфигурации"""
    model = CTPathologyModel(
        model_type=model_config.get("model_type", "resnet3d"),
        model_name=model_config.get("model_name", "resnet3d_50"),
        num_classes=model_config.get("num_classes", 14),
        input_size=model_config.get("input_size", (64, 64, 64)),
        dropout_rate=model_config.get("dropout_rate", 0.5),
        pretrained=model_config.get("pretrained", True)
    )
    
    return model

def load_pretrained_model(checkpoint_path: str, device) -> CTPathologyModel:
    """Загрузка предобученной модели"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Создаем модель
    model_config = checkpoint.get("model_config", {})
    model = create_model(model_config)
    
    # Загружаем веса
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    return model

