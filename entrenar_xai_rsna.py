# ============================================================
# entrenar_xai_rsna.py
# ============================================================
# Proyecto:
# Evaluación de explicabilidad en modelos de deep learning
# para apoyo al diagnóstico de neumonía en radiografías de tórax
#
# Funcionalidades:
# 1. Carga del conjunto RSNA Pneumonia Detection Challenge.
# 2. Lectura y preprocesamiento de imágenes DICOM.
# 3. Construcción de variable binaria.
# 4. Partición estratificada train/val/test.
# 5. Entrenamiento de modelo CNN con transfer learning.
# 6. Evaluación predictiva: accuracy, sensibilidad, especificidad,
#    precisión, F1, AUROC, AUPRC.
# 7. Selección de umbral óptimo con validación.
# 8. Curvas ROC, Precision-Recall, pérdida y AUROC.
# 9. Grad-CAM, Grad-CAM++ e Integrated Gradients.
# 10. Evaluación de explicabilidad:
#     - Energía dentro de bounding box.
#     - IoU.
#     - Pointing game.
#     - Estabilidad ante perturbaciones.
#     - Fidelidad por oclusión.
# 11. Generación de ejemplos visuales.
# ============================================================

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
)
from tqdm import tqdm


warnings.filterwarnings("ignore")


# ============================================================
# Configuración
# ============================================================

@dataclass
class Config:
    root: str = "."
    data_dir: str = "rsna-pneumonia-detection-challenge"
    train_images_dir: str = "stage_2_train_images"
    labels_csv: str = "stage_2_train_labels.csv"
    class_info_csv: str = "stage_2_detailed_class_info.csv"
    eda_csv: str = "salida_eda/dataset_eda_rsna.csv"

    output_dir: str = "salida_modelado"
    image_size: int = 224
    model_name: str = "densenet121"
    pretrained: bool = True

    epochs: int = 15
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    threshold_strategy: str = "f1"

    use_class_weights: bool = True
    use_amp: bool = True

    xai_max_samples: int = 150
    xai_top_percent: float = 0.15
    ig_steps: int = 32

    seed: int = 42
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dirs(cfg: Config) -> Dict[str, Path]:
    out = Path(cfg.output_dir)
    dirs = {
        "root": out,
        "models": out / "modelos",
        "figures": out / "figuras",
        "predictions": out / "predicciones",
        "xai": out / "explicabilidad",
        "tables": out / "tablas",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


# ============================================================
# Lectura DICOM
# ============================================================

def read_dicom_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Lee una imagen DICOM y regresa una imagen normalizada en rango [0, 1].

    Returns
    -------
    image_norm:
        Arreglo 2D float32 normalizado.
    original_shape:
        Tupla (alto, ancho) original.
    """
    ds = pydicom.dcmread(str(path))

    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        arr = ds.pixel_array

    arr = arr.astype(np.float32)

    photometric = getattr(ds, "PhotometricInterpretation", "")
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    original_shape = arr.shape

    # Normalización robusta por percentiles para reducir efecto de valores extremos.
    p_low, p_high = np.percentile(arr, (0.5, 99.5))
    if p_high > p_low:
        arr = np.clip(arr, p_low, p_high)
        arr = (arr - p_low) / (p_high - p_low)
    else:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

    return arr.astype(np.float32), original_shape


# ============================================================
# Construcción de metadata
# ============================================================

def _bbox_list_to_json(bboxes: List[Dict]) -> str:
    return json.dumps(bboxes, ensure_ascii=False)


def _bbox_json_to_list(value) -> List[Dict]:
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []

    return []


def build_metadata(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    """
    Construye una tabla a nivel de imagen/patientId.

    Si existe salida_eda/dataset_eda_rsna.csv y contiene split,
    se utiliza esa partición. En caso contrario, se genera una
    partición estratificada 70/15/15.
    """
    root = Path(cfg.root)
    data_path = root / cfg.data_dir
    labels_path = data_path / cfg.labels_csv
    class_info_path = data_path / cfg.class_info_csv
    images_dir = data_path / cfg.train_images_dir
    eda_path = root / cfg.eda_csv

    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró: {labels_path}")

    if not class_info_path.exists():
        raise FileNotFoundError(f"No se encontró: {class_info_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"No se encontró: {images_dir}")

    labels = pd.read_csv(labels_path)
    class_info = pd.read_csv(class_info_path)

    labels["Target"] = labels["Target"].astype(int)

    # Base a nivel patientId.
    patient_base = (
        labels.groupby("patientId", as_index=False)
        .agg(target_binaria=("Target", "max"))
    )

    # Clase clínica detallada.
    class_info_unique = class_info.drop_duplicates(subset=["patientId"])
    patient_base = patient_base.merge(class_info_unique, on="patientId", how="left")

    # Bounding boxes sólo para positivos.
    pos_boxes = labels[labels["Target"] == 1].copy()

    bbox_dict = {}
    for pid, group in pos_boxes.groupby("patientId"):
        boxes = []
        for _, row in group.iterrows():
            if not pd.isna(row["x"]):
                boxes.append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                    }
                )
        bbox_dict[pid] = boxes

    patient_base["bboxes"] = patient_base["patientId"].map(lambda p: bbox_dict.get(p, []))
    patient_base["bbox_count"] = patient_base["bboxes"].apply(len)
    patient_base["image_path"] = patient_base["patientId"].apply(
        lambda p: str(images_dir / f"{p}.dcm")
    )

    # Filtrar imágenes existentes.
    patient_base["image_exists"] = patient_base["image_path"].apply(lambda p: Path(p).exists())
    missing = int((~patient_base["image_exists"]).sum())
    if missing > 0:
        print(f"[ADVERTENCIA] Imágenes faltantes: {missing}. Se eliminarán.")
    patient_base = patient_base[patient_base["image_exists"]].copy()

    # Usar split existente del EDA si está disponible.
    if eda_path.exists():
        eda_df = pd.read_csv(eda_path)
        if {"patientId", "split"}.issubset(eda_df.columns):
            split_df = eda_df[["patientId", "split"]].drop_duplicates(subset=["patientId"])
            patient_base = patient_base.drop(columns=["split"], errors="ignore")
            patient_base = patient_base.merge(split_df, on="patientId", how="left")

    if "split" not in patient_base.columns or patient_base["split"].isna().any():
        patient_base = create_stratified_split(patient_base, seed=cfg.seed)

    # Orden de columnas.
    patient_base["bboxes_json"] = patient_base["bboxes"].apply(_bbox_list_to_json)

    columns = [
        "patientId",
        "image_path",
        "target_binaria",
        "class",
        "split",
        "bbox_count",
        "bboxes_json",
    ]

    metadata = patient_base[columns].copy()
    metadata_path = dirs["tables"] / "metadata_modelado.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"[OK] Metadata guardada en: {metadata_path}")
    print(metadata["split"].value_counts())
    print(metadata.groupby("split")["target_binaria"].value_counts())

    return metadata


def create_stratified_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Crea partición 70/15/15 estratificada por target_binaria.
    """
    df = df.copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["target_binaria"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["target_binaria"],
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


# ============================================================
# Dataset y DataLoader
# ============================================================

def get_transforms(image_size: int, train: bool) -> T.Compose:
    """
    Transformaciones preservando plausibilidad clínica.
    No se usa flip horizontal por defecto para evitar invertir marcadores/lateralidad.
    """
    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomAffine(
                    degrees=7,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03),
                    shear=None,
                ),
                T.ColorJitter(brightness=0.05, contrast=0.05),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class RSNADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = False,
        return_display_image: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.train = train
        self.transform = get_transforms(image_size=image_size, train=train)
        self.return_display_image = return_display_image

    def __len__(self):
        return len(self.df)

    def _scale_bboxes(
        self,
        bboxes: List[Dict],
        original_shape: Tuple[int, int],
    ) -> List[Dict]:
        orig_h, orig_w = original_shape
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h

        scaled = []
        for b in bboxes:
            x = float(b["x"]) * sx
            y = float(b["y"]) * sy
            w = float(b["width"]) * sx
            h = float(b["height"]) * sy

            x = max(0.0, min(x, self.image_size - 1))
            y = max(0.0, min(y, self.image_size - 1))
            w = max(0.0, min(w, self.image_size - x))
            h = max(0.0, min(h, self.image_size - y))

            if w > 0 and h > 0:
                scaled.append({"x": x, "y": y, "width": w, "height": h})

        return scaled

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])

        image_arr, original_shape = read_dicom_image(image_path)

        image_uint8 = (image_arr * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image_uint8).convert("L")

        image_tensor = self.transform(pil_img)

        display_img = pil_img.resize((self.image_size, self.image_size))
        display_arr = np.asarray(display_img).astype(np.float32) / 255.0
        display_tensor = torch.from_numpy(display_arr).unsqueeze(0)

        bboxes = _bbox_json_to_list(row.get("bboxes_json", "[]"))
        bboxes_scaled = self._scale_bboxes(bboxes, original_shape)

        return {
            "image": image_tensor,
            "display_image": display_tensor,
            "label": torch.tensor(float(row["target_binaria"]), dtype=torch.float32),
            "patientId": row["patientId"],
            "bboxes": bboxes_scaled,
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    display_images = torch.stack([item["display_image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    patient_ids = [item["patientId"] for item in batch]
    bboxes = [item["bboxes"] for item in batch]

    return {
        "image": images,
        "display_image": display_images,
        "label": labels,
        "patientId": patient_ids,
        "bboxes": bboxes,
    }


def create_loaders(cfg: Config, metadata: pd.DataFrame):
    train_df = metadata[metadata["split"] == "train"].copy()
    val_df = metadata[metadata["split"] == "val"].copy()
    test_df = metadata[metadata["split"] == "test"].copy()

    train_dataset = RSNADataset(train_df, image_size=cfg.image_size, train=True)
    val_dataset = RSNADataset(val_df, image_size=cfg.image_size, train=False)
    test_dataset = RSNADataset(test_df, image_size=cfg.image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


# ============================================================
# Modelos
# ============================================================

class SimpleCNN(nn.Module):
    """
    Modelo base sencillo para comparación.
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
        return model

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model

    if model_name == "simplecnn":
        return SimpleCNN()

    raise ValueError(
        "Modelo no soportado. Usa: densenet121, resnet50, efficientnet_b0 o simplecnn"
    )


def get_target_layer(model: nn.Module, model_name: str):
    model_name = model_name.lower()

    if model_name == "densenet121":
        return model.features.denseblock4

    if model_name == "resnet50":
        return model.layer4[-1]

    if model_name == "efficientnet_b0":
        return model.features[-1]

    if model_name == "simplecnn":
        return model.features[-2]

    raise ValueError(f"No hay capa objetivo definida para {model_name}")


# ============================================================
# Métricas predictivas
# ============================================================

def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_auprc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(safe_auc(y_true, y_prob)),
        "auprc": float(safe_auprc(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, strategy: str = "f1") -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -np.inf

    for t in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=t)

        if strategy == "f1":
            score = metrics["f1"]
        elif strategy == "youden":
            score = metrics["sensitivity_recall"] + metrics["specificity"] - 1
        elif strategy == "sensitivity":
            score = metrics["sensitivity_recall"]
        else:
            score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t)


# ============================================================
# Entrenamiento y predicción
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp: bool = True,
):
    model.train()

    running_loss = 0.0
    y_true_all = []
    y_prob_all = []

    for batch in tqdm(loader, desc="Entrenamiento", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        y_prob_all.extend(probs.tolist())
        y_true_all.extend(labels.detach().cpu().numpy().ravel().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true_all, y_prob_all, threshold=0.5)
    metrics["loss"] = float(epoch_loss)

    return metrics


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()

    rows = []

    for batch in tqdm(loader, desc="Predicción", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(int)
        patient_ids = batch["patientId"]

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()

        for pid, y, p in zip(patient_ids, labels, probs):
            rows.append(
                {
                    "patientId": pid,
                    "y_true": int(y),
                    "y_prob": float(p),
                }
            )

    return pd.DataFrame(rows)


def evaluate_loader(model, loader, device, threshold: float = 0.5) -> Dict:
    pred_df = predict(model, loader, device)
    metrics = compute_metrics(
        pred_df["y_true"].values,
        pred_df["y_prob"].values,
        threshold=threshold,
    )
    return metrics, pred_df


def train_model(cfg: Config, dirs, train_loader, val_loader, train_df):
    device = get_device(cfg.device)
    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)

    pos = train_df["target_binaria"].sum()
    neg = len(train_df) - pos

    if cfg.use_class_weights:
        pos_weight_value = neg / max(pos, 1)
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    history = []
    best_val_auroc = -np.inf
    best_path = dirs["models"] / f"mejor_modelo_{cfg.model_name}.pt"
    patience_counter = 0

    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Modelo: {cfg.model_name}")
    print(f"[INFO] Pretrained: {cfg.pretrained}")
    print(f"[INFO] pos_weight: {pos_weight_value:.4f}")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nÉpoca {epoch}/{cfg.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=cfg.use_amp,
        )

        val_metrics, _ = evaluate_loader(model, val_loader, device, threshold=0.5)

        scheduler.step(val_metrics["auroc"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_auroc={train_metrics['auroc']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val_auroc": best_val_auroc,
                    "epoch": epoch,
                },
                best_path,
            )

            print(f"[OK] Mejor modelo guardado: {best_path}")
        else:
            patience_counter += 1
            print(f"[INFO] Sin mejora. Paciencia: {patience_counter}/{cfg.patience}")

        if patience_counter >= cfg.patience:
            print("[INFO] Early stopping activado.")
            break

    history_df = pd.DataFrame(history)
    history_path = dirs["tables"] / "historial_entrenamiento.csv"
    history_df.to_csv(history_path, index=False)

    print(f"[OK] Historial guardado en: {history_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history_df, best_path


# ============================================================
# Gráficas predictivas
# ============================================================

def plot_training_history(history_df: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    fig_path_loss = dirs["figures"] / "curva_loss.png"
    fig_path_auroc = dirs["figures"] / "curva_auroc.png"

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Entrenamiento")
    if "val_loss" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_loss, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_auroc"], label="Entrenamiento")
    plt.plot(history_df["epoch"], history_df["val_auroc"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("AUROC")
    plt.title("Curva de AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_auroc, dpi=300)
    plt.close()

    print(f"[OK] Figuras guardadas: {fig_path_loss}, {fig_path_auroc}")


def plot_confusion_matrix(y_true, y_prob, threshold, path: Path) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Matriz de confusión")
    plt.xticks([0, 1], ["Negativo", "Positivo"])
    plt.yticks([0, 1], ["Negativo", "Positivo"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_pr(y_true, y_prob, roc_path: Path, pr_path: Path) -> None:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Sensibilidad")
    plt.title("Curva ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Sensibilidad")
    plt.ylabel("Precisión")
    plt.title("Curva Precisión-Sensibilidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()


# ============================================================
# Explicabilidad: Grad-CAM y Grad-CAM++
# ============================================================

class CAMExplainer:
    def __init__(self, model: nn.Module, target_layer, method: str = "gradcam"):
        self.model = model
        self.target_layer = target_layer
        self.method = method
        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Regresa mapa normalizado en tamaño de entrada.
        Shape: [B, 1, H, W]
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        target = logits[:, 0].sum()
        target.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("No se capturaron activaciones o gradientes.")

        if self.method == "gradcam":
            cam = self._gradcam(activations, gradients)
        elif self.method == "gradcampp":
            cam = self._gradcampp(activations, gradients)
        else:
            raise ValueError("Método CAM no soportado.")

        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = normalize_maps(cam)
        return cam.detach()

    def _gradcam(self, activations, gradients):
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        return cam

    def _gradcampp(self, activations, gradients):
        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8

        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha = alpha_num / (alpha_denom + eps)

        positive_gradients = torch.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        return cam


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza mapas por muestra a rango [0, 1].
    Shape esperado: [B, 1, H, W]
    """
    b = maps.shape[0]
    flat = maps.view(b, -1)
    min_vals = flat.min(dim=1)[0].view(b, 1, 1, 1)
    max_vals = flat.max(dim=1)[0].view(b, 1, 1, 1)

    return (maps - min_vals) / (max_vals - min_vals + eps)


# ============================================================
# Integrated Gradients
# ============================================================

def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    steps: int = 32,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Integrated Gradients para clase positiva.
    Regresa mapa [B, 1, H, W].
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    total_gradients = torch.zeros_like(x)

    for k in range(1, steps + 1):
        alpha = float(k) / steps
        interpolated = baseline + alpha * (x - baseline)
        interpolated.requires_grad_(True)

        logits = model(interpolated)
        score = logits[:, 0].sum()

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        total_gradients += gradients.detach()

    avg_gradients = total_gradients / steps
    ig = (x - baseline) * avg_gradients

    saliency = ig.abs().sum(dim=1, keepdim=True)
    saliency = normalize_maps(saliency)

    return saliency.detach()


# ============================================================
# Métricas de explicabilidad
# ============================================================

def bboxes_to_mask(bboxes: List[Dict], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    for b in bboxes:
        x1 = int(round(b["x"]))
        y1 = int(round(b["y"]))
        x2 = int(round(b["x"] + b["width"]))
        y2 = int(round(b["y"] + b["height"]))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    return mask


def heatmap_to_binary(cam: np.ndarray, top_percent: float = 0.15) -> np.ndarray:
    cam = np.nan_to_num(cam)
    if cam.max() <= cam.min():
        return np.zeros_like(cam, dtype=bool)

    threshold = np.quantile(cam, 1.0 - top_percent)
    return cam >= threshold


def energy_inside_bbox(cam: np.ndarray, mask: np.ndarray) -> float:
    total = cam.sum()
    if total <= 0:
        return np.nan
    return float(cam[mask].sum() / total)


def iou_score(cam_binary: np.ndarray, mask: np.ndarray) -> float:
    intersection = np.logical_and(cam_binary, mask).sum()
    union = np.logical_or(cam_binary, mask).sum()

    if union == 0:
        return np.nan

    return float(intersection / union)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    if cam.max() <= cam.min():
        return np.nan

    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return float(mask[y, x])


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def mean_absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@torch.no_grad()
def predict_probability(model: nn.Module, x: torch.Tensor) -> float:
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].item()
    return float(prob)


def fidelity_occlusion(
    model: nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    top_percent: float = 0.15,
) -> float:
    """
    Ocluye las regiones más relevantes del mapa y mide caída de probabilidad.
    """
    model.eval()

    p_original = predict_probability(model, x)

    mask_np = heatmap_to_binary(cam, top_percent=top_percent)
    mask = torch.from_numpy(mask_np).bool().to(x.device)

    x_occ = x.clone()
    # Valor 0 en espacio normalizado equivale aproximadamente a media.
    x_occ[:, :, mask] = 0.0

    p_occluded = predict_probability(model, x_occ)
    delta = p_original - p_occluded

    return float(delta)


def perturb_tensor(x: torch.Tensor, mode: str = "noise") -> torch.Tensor:
    if mode == "noise":
        return x + torch.randn_like(x) * 0.03

    if mode == "shift":
        return x + 0.05

    return x


# ============================================================
# Visualización XAI
# ============================================================

def tensor_to_display_image(display_tensor: torch.Tensor) -> np.ndarray:
    """
    display_tensor: [1, H, W]
    """
    arr = display_tensor.squeeze(0).detach().cpu().numpy()
    return np.clip(arr, 0, 1)


def draw_bboxes(ax, bboxes: List[Dict]) -> None:
    import matplotlib.patches as patches

    for b in bboxes:
        rect = patches.Rectangle(
            (b["x"], b["y"]),
            b["width"],
            b["height"],
            linewidth=1.8,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def save_xai_figure(
    display_img: np.ndarray,
    bboxes: List[Dict],
    maps: Dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    n = 1 + len(maps)

    plt.figure(figsize=(4.5 * n, 4.5))

    ax = plt.subplot(1, n, 1)
    ax.imshow(display_img, cmap="gray")
    draw_bboxes(ax, bboxes)
    ax.set_title("Imagen + bbox")
    ax.axis("off")

    for idx, (name, cam) in enumerate(maps.items(), start=2):
        ax = plt.subplot(1, n, idx)
        ax.imshow(display_img, cmap="gray")
        ax.imshow(cam, alpha=0.45)
        draw_bboxes(ax, bboxes)
        ax.set_title(name)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Evaluación XAI
# ============================================================

def compute_xai_for_sample(
    model,
    image,
    gradcam_exp,
    gradcampp_exp,
    cfg: Config,
    device,
) -> Dict[str, np.ndarray]:
    image = image.to(device)

    maps = {}

    gradcam = gradcam_exp(image)
    maps["Grad-CAM"] = gradcam[0, 0].detach().cpu().numpy()

    gradcampp = gradcampp_exp(image)
    maps["Grad-CAM++"] = gradcampp[0, 0].detach().cpu().numpy()

    ig = integrated_gradients(model, image, steps=cfg.ig_steps)
    maps["Integrated Gradients"] = ig[0, 0].detach().cpu().numpy()

    return maps


def evaluate_xai(
    cfg: Config,
    dirs: Dict[str, Path],
    model: nn.Module,
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
):
    device = get_device(cfg.device)
    model.eval()

    target_layer = get_target_layer(model, cfg.model_name)

    gradcam_exp = CAMExplainer(model, target_layer, method="gradcam")
    gradcampp_exp = CAMExplainer(model, target_layer, method="gradcampp")

    # Sólo positivos con bbox para métricas de localización.
    pred_map = test_predictions.set_index("patientId").to_dict(orient="index")

    positives = test_df[test_df["target_binaria"] == 1].copy()
    positives["bbox_count"] = positives["bboxes_json"].apply(lambda x: len(_bbox_json_to_list(x)))
    positives = positives[positives["bbox_count"] > 0].copy()

    if len(positives) > cfg.xai_max_samples:
        positives = positives.sample(cfg.xai_max_samples, random_state=cfg.seed)

    xai_dataset = RSNADataset(positives, image_size=cfg.image_size, train=False)
    xai_loader = DataLoader(
        xai_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rows = []

    for batch in tqdm(xai_loader, desc="Evaluación XAI"):
        image = batch["image"].to(device)
        display_image = batch["display_image"][0]
        pid = batch["patientId"][0]
        bboxes = batch["bboxes"][0]

        if len(bboxes) == 0:
            continue

        maps = compute_xai_for_sample(
            model=model,
            image=image,
            gradcam_exp=gradcam_exp,
            gradcampp_exp=gradcampp_exp,
            cfg=cfg,
            device=device,
        )

        h, w = cfg.image_size, cfg.image_size
        bbox_mask = bboxes_to_mask(bboxes, h, w)

        for method_name, cam in maps.items():
            binary = heatmap_to_binary(cam, top_percent=cfg.xai_top_percent)

            row = {
                "patientId": pid,
                "method": method_name,
                "energy_inside_bbox": energy_inside_bbox(cam, bbox_mask),
                "iou": iou_score(binary, bbox_mask),
                "pointing_game": pointing_game(cam, bbox_mask),
                "fidelity_delta_prob": fidelity_occlusion(
                    model=model,
                    x=image,
                    cam=cam,
                    top_percent=cfg.xai_top_percent,
                ),
            }

            # Estabilidad con dos perturbaciones simples.
            perturbed_metrics = []
            for perturb_mode in ["noise", "shift"]:
                x_pert = perturb_tensor(image, mode=perturb_mode)

                if method_name == "Grad-CAM":
                    cam_pert = gradcam_exp(x_pert)[0, 0].detach().cpu().numpy()
                elif method_name == "Grad-CAM++":
                    cam_pert = gradcampp_exp(x_pert)[0, 0].detach().cpu().numpy()
                else:
                    cam_pert = integrated_gradients(
                        model, x_pert, steps=max(8, cfg.ig_steps // 2)
                    )[0, 0].detach().cpu().numpy()

                perturbed_metrics.append(
                    {
                        "corr": pearson_corr(cam, cam_pert),
                        "mad": mean_absolute_difference(cam, cam_pert),
                    }
                )

            row["stability_corr_mean"] = float(
                np.nanmean([m["corr"] for m in perturbed_metrics])
            )
            row["stability_mad_mean"] = float(
                np.nanmean([m["mad"] for m in perturbed_metrics])
            )

            if pid in pred_map:
                row["y_prob"] = pred_map[pid]["y_prob"]
                row["y_true"] = pred_map[pid]["y_true"]

            rows.append(row)

        # Guardar algunos ejemplos visuales.
        if len(rows) <= 30:
            display_np = tensor_to_display_image(display_image)
            prob = pred_map.get(pid, {}).get("y_prob", np.nan)
            title = f"{pid} | y=1 | p={prob:.3f}"
            fig_path = dirs["xai"] / f"ejemplo_xai_{pid}.png"
            save_xai_figure(display_np, bboxes, maps, title, fig_path)

    gradcam_exp.remove_hooks()
    gradcampp_exp.remove_hooks()

    xai_df = pd.DataFrame(rows)
    xai_path = dirs["tables"] / "metricas_explicabilidad.csv"
    xai_df.to_csv(xai_path, index=False)

    summary = (
        xai_df.groupby("method")
        .agg(
            energy_inside_bbox_mean=("energy_inside_bbox", "mean"),
            energy_inside_bbox_std=("energy_inside_bbox", "std"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            pointing_game_mean=("pointing_game", "mean"),
            fidelity_delta_prob_mean=("fidelity_delta_prob", "mean"),
            stability_corr_mean=("stability_corr_mean", "mean"),
            stability_mad_mean=("stability_mad_mean", "mean"),
        )
        .reset_index()
    )

    summary_path = dirs["tables"] / "resumen_metricas_explicabilidad.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Métricas XAI guardadas en: {xai_path}")
    print(f"[OK] Resumen XAI guardado en: {summary_path}")
    print(summary)

    return xai_df, summary


# ============================================================
# Casos cualitativos
# ============================================================

def generate_qualitative_cases(
    cfg: Config,
    dirs: Dict[str, Path],
    model: nn.Module,
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    threshold: float,
    max_cases_per_group: int = 3,
):
    device = get_device(cfg.device)
    model.eval()

    pred = test_predictions.copy()
    pred["y_pred"] = (pred["y_prob"] >= threshold).astype(int)

    conditions = {
        "VP": (pred["y_true"] == 1) & (pred["y_pred"] == 1),
        "VN": (pred["y_true"] == 0) & (pred["y_pred"] == 0),
        "FP": (pred["y_true"] == 0) & (pred["y_pred"] == 1),
        "FN": (pred["y_true"] == 1) & (pred["y_pred"] == 0),
    }

    selected_ids = []

    for group_name, condition in conditions.items():
        subset = pred[condition].copy()
        if len(subset) == 0:
            continue

        if group_name in ["VP", "FP"]:
            subset = subset.sort_values("y_prob", ascending=False)
        else:
            subset = subset.sort_values("y_prob", ascending=True)

        ids = subset.head(max_cases_per_group)["patientId"].tolist()
        selected_ids.extend([(group_name, pid) for pid in ids])

    if not selected_ids:
        print("[INFO] No se encontraron casos cualitativos.")
        return

    target_layer = get_target_layer(model, cfg.model_name)
    gradcam_exp = CAMExplainer(model, target_layer, method="gradcam")
    gradcampp_exp = CAMExplainer(model, target_layer, method="gradcampp")

    pred_map = pred.set_index("patientId").to_dict(orient="index")
    selected_pid_set = [pid for _, pid in selected_ids]
    selected_df = test_df[test_df["patientId"].isin(selected_pid_set)].copy()

    dataset = RSNADataset(selected_df, image_size=cfg.image_size, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    group_by_pid = {pid: group for group, pid in selected_ids}

    for batch in tqdm(loader, desc="Casos cualitativos"):
        image = batch["image"].to(device)
        display_image = batch["display_image"][0]
        pid = batch["patientId"][0]
        bboxes = batch["bboxes"][0]

        maps = compute_xai_for_sample(
            model=model,
            image=image,
            gradcam_exp=gradcam_exp,
            gradcampp_exp=gradcampp_exp,
            cfg=cfg,
            device=device,
        )

        display_np = tensor_to_display_image(display_image)

        group = group_by_pid.get(pid, "caso")
        p = pred_map[pid]["y_prob"]
        yt = pred_map[pid]["y_true"]
        yp = pred_map[pid]["y_pred"]

        title = f"{group} | {pid} | y={yt} | pred={yp} | p={p:.3f}"
        fig_path = dirs["xai"] / f"caso_{group}_{pid}.png"

        save_xai_figure(display_np, bboxes, maps, title, fig_path)

    gradcam_exp.remove_hooks()
    gradcampp_exp.remove_hooks()

    print(f"[OK] Casos cualitativos guardados en: {dirs['xai']}")


# ============================================================
# Main
# ============================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data-dir", type=str, default="rsna-pneumonia-detection-challenge")
    parser.add_argument("--output-dir", type=str, default="salida_modelado")

    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=["densenet121", "resnet50", "efficientnet_b0", "simplecnn"],
    )

    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--xai-max-samples", type=int, default=150)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Entrena y evalúa el modelo, pero no calcula explicabilidad.",
    )

    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_class_weights=not args.no_class_weights,
        use_amp=not args.no_amp,
        xai_max_samples=args.xai_max_samples,
        ig_steps=args.ig_steps,
        device=args.device,
        seed=args.seed,
    )

    cfg.skip_xai = args.skip_xai

    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    dirs = ensure_dirs(cfg)

    cfg_path = dirs["root"] / "config_entrenamiento.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4, ensure_ascii=False)

    metadata = build_metadata(cfg, dirs)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_loaders(
        cfg, metadata
    )

    model, history_df, best_model_path = train_model(
        cfg=cfg,
        dirs=dirs,
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
    )

    plot_training_history(history_df, dirs)

    device = get_device(cfg.device)

    # Evaluación en validación para seleccionar threshold.
    val_pred = predict(model, val_loader, device)
    best_threshold = find_best_threshold(
        val_pred["y_true"].values,
        val_pred["y_prob"].values,
        strategy=cfg.threshold_strategy,
    )

    val_metrics = compute_metrics(
        val_pred["y_true"].values,
        val_pred["y_prob"].values,
        threshold=best_threshold,
    )

    print(f"\n[OK] Umbral óptimo en validación: {best_threshold:.4f}")
    print("[VALIDACIÓN]")
    print(json.dumps(val_metrics, indent=4, ensure_ascii=False))

    # Evaluación en prueba.
    test_pred = predict(model, test_loader, device)
    test_pred["y_pred"] = (test_pred["y_prob"] >= best_threshold).astype(int)

    test_metrics = compute_metrics(
        test_pred["y_true"].values,
        test_pred["y_prob"].values,
        threshold=best_threshold,
    )

    print("\n[PRUEBA]")
    print(json.dumps(test_metrics, indent=4, ensure_ascii=False))

    # Guardar predicciones y métricas.
    val_pred.to_csv(dirs["predictions"] / "predicciones_validacion.csv", index=False)
    test_pred.to_csv(dirs["predictions"] / "predicciones_prueba.csv", index=False)

    metrics_all = {
        "best_model_path": str(best_model_path),
        "best_threshold": best_threshold,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(dirs["tables"] / "metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=4, ensure_ascii=False)

    # Figuras de evaluación predictiva.
    plot_confusion_matrix(
        test_pred["y_true"].values,
        test_pred["y_prob"].values,
        best_threshold,
        dirs["figures"] / "matriz_confusion.png",
    )

    plot_roc_pr(
        test_pred["y_true"].values,
        test_pred["y_prob"].values,
        dirs["figures"] / "curva_roc.png",
        dirs["figures"] / "curva_precision_recall.png",
    )

    print(f"[OK] Métricas y figuras predictivas guardadas en: {dirs['root']}")

    if not getattr(cfg, "skip_xai", False):
        xai_df, xai_summary = evaluate_xai(
            cfg=cfg,
            dirs=dirs,
            model=model,
            test_df=test_df,
            test_predictions=test_pred,
        )

        generate_qualitative_cases(
            cfg=cfg,
            dirs=dirs,
            model=model,
            test_df=test_df,
            test_predictions=test_pred,
            threshold=best_threshold,
            max_cases_per_group=3,
        )

    print("\nProceso completado.")


if __name__ == "__main__":
    main()