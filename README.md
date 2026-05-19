# Tesis de Maestría — Explicabilidad en CNN para apoyo al diagnóstico de neumonía

**Autores:** David Segundo García · Isaac Hernández Ramírez  
**Asesor:** Dr. Daniel Alejandro Cervantes Cabrera  
**Institución:** INFOTEC — Maestría en Ciencia de Datos  

Este repositorio acompaña el trabajo de tesis *"Evaluación de explicabilidad
en modelos de Deep Learning para apoyo al diagnóstico de neumonía en
radiografías de tórax"*, basado en el dataset
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge).

Se entrena una `DenseNet121` preentrenada para clasificación binaria
(neumonía / no neumonía) y se evalúan tres métodos de explicabilidad
post-hoc: **Grad-CAM**, **Grad-CAM++** e **Integrated Gradients**.

---

## Estructura del repositorio

```
tesis_maestria_rsna/
├── data/
│   ├── raw/             # Dataset original (CSVs + DICOMs, no versionados)
│   │   ├── stage_2_train_images/        (26 684 .dcm — gitignored)
│   │   ├── stage_2_test_images/         (.dcm — gitignored)
│   │   ├── stage_2_train_labels.csv
│   │   ├── stage_2_detailed_class_info.csv
│   │   ├── stage_2_sample_submission.csv
│   │   └── kaggle_extra/                # artefactos auxiliares de Kaggle
│   ├── interim/         # Salidas de preprocesamiento (CSVs intermedios)
│   └── processed/       # Dataset limpio y tablas resumen del EDA
├── notebooks/
│   └── 01_explicabilidad_cnn_neumonia.ipynb   # Pipeline completo (Kaggle GPU)
├── src/
│   ├── data/
│   │   └── preprocesar_rsna.py          # Integración + limpieza + split
│   └── eda/
│       └── analisis_exploratorio_rsna.py # Análisis exploratorio
├── models/
│   └── mejor_modelo_densenet121.pt      # Pesos del mejor checkpoint
└── reports/
    ├── eda/             # Reportes de EDA en Markdown / LaTeX
    ├── figures/
    │   ├── eda/         # Figuras del análisis exploratorio
    │   ├── modeling/    # Curvas de entrenamiento, evaluación de prueba
    │   └── explainability/   # Mapas Grad-CAM, Grad-CAM++, IG
    ├── tables/          # Métricas e historial de entrenamiento
    └── predictions/     # Predicciones de validación y prueba
```

---

## Cómo reproducir

### 1. Requisitos

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Descargar el dataset

Descarga el dataset desde Kaggle y colócalo en `data/raw/`:

```
data/raw/
├── stage_2_train_images/
├── stage_2_test_images/
├── stage_2_train_labels.csv
├── stage_2_detailed_class_info.csv
└── stage_2_sample_submission.csv
```

### 3. Pipeline local

```powershell
# Preprocesamiento + integración + split 70/15/15
python src/data/preprocesar_rsna.py

# Análisis exploratorio (genera figuras y tablas)
python src/eda/analisis_exploratorio_rsna.py
```

### 4. Entrenamiento y explicabilidad

El entrenamiento y los métodos de XAI se ejecutan en el notebook
[notebooks/01_explicabilidad_cnn_neumonia.ipynb](notebooks/01_explicabilidad_cnn_neumonia.ipynb)
sobre Kaggle (GPU T4 / P100). Las rutas del notebook ya están configuradas
para `/kaggle/input/...` y `/kaggle/working/...`.

---

## Resultados principales

| Métrica   | Validación | Prueba |
|-----------|-----------:|-------:|
| AUROC     | —          | 0.8899 |
| AUPRC     | —          | 0.7064 |
| F1        | —          | 0.6553 |
| Umbral τ  | —          | 0.73   |

Hiperparámetros: `Adam` lr=1e-4, `batch_size`=32, BCE con `pos_weight`=3.4387,
imagen 224×224, normalización ImageNet, *early stopping* sobre AUROC de validación.

---

## Licencia y uso

Los datos provienen del *RSNA Pneumonia Detection Challenge* y se usan
exclusivamente con fines académicos, sujetos a los
[términos de la competencia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules).
El código de este repositorio se distribuye bajo licencia MIT.
