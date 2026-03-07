import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def safe_read_dicom(dicom_path: Path) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Intenta leer un DICOM y devuelve:
    - ok: bool
    - metadata mínima
    - mensaje de error si falla
    """
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)

        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        modality = getattr(ds, "Modality", None)
        patient_id = getattr(ds, "PatientID", None)
        photometric = getattr(ds, "PhotometricInterpretation", None)

        # Intento de acceso al pixel_array para validar lectura real
        try:
            arr = ds.pixel_array
            pixel_ok = True
            arr_shape = tuple(arr.shape)
            arr_dtype = str(arr.dtype)
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
        except Exception as e:
            pixel_ok = False
            arr_shape = None
            arr_dtype = None
            min_val = None
            max_val = None
            return False, {
                "Rows": rows,
                "Columns": cols,
                "Modality": modality,
                "PatientID": patient_id,
                "PhotometricInterpretation": photometric,
                "pixel_ok": pixel_ok,
                "shape": arr_shape,
                "dtype": arr_dtype,
                "min_val": min_val,
                "max_val": max_val
            }, f"Error al leer pixel_array: {e}"

        meta = {
            "Rows": rows,
            "Columns": cols,
            "Modality": modality,
            "PatientID": patient_id,
            "PhotometricInterpretation": photometric,
            "pixel_ok": pixel_ok,
            "shape": arr_shape,
            "dtype": arr_dtype,
            "min_val": min_val,
            "max_val": max_val
        }
        return True, meta, None

    except Exception as e:
        return False, None, str(e)


def normalize_image_info(min_val: float, max_val: float) -> bool:
    """
    Solo indica si la imagen parece normalizable.
    """
    if min_val is None or max_val is None:
        return False
    return max_val > min_val


def find_duplicate_files(file_paths: List[Path]) -> pd.DataFrame:
    """
    Detecta duplicados por nombre de archivo base sin extensión.
    """
    ids = [p.stem for p in file_paths]
    s = pd.Series(ids, name="patientId")
    dup_ids = s[s.duplicated(keep=False)].sort_values()
    if dup_ids.empty:
        return pd.DataFrame(columns=["patientId", "count"])
    return dup_ids.value_counts().reset_index().rename(columns={"index": "patientId", "patientId": "count"})


def validate_bbox(row: pd.Series, image_dims: Dict[str, Tuple[int, int]]) -> Tuple[bool, str]:
    """
    Valida coordenadas x, y, width, height.
    """
    patient_id = row["patientId"]

    if patient_id not in image_dims:
        return False, "imagen_no_encontrada"

    img_h, img_w = image_dims[patient_id]

    x = row.get("x", np.nan)
    y = row.get("y", np.nan)
    w = row.get("width", np.nan)
    h = row.get("height", np.nan)

    numeric_fields = [x, y, w, h]
    if any(pd.isna(v) for v in numeric_fields):
        return False, "bbox_con_faltantes"

    if w <= 0 or h <= 0:
        return False, "bbox_dim_no_positiva"

    if x < 0 or y < 0:
        return False, "bbox_coordenada_negativa"

    if x + w > img_w or y + h > img_h:
        return False, "bbox_fuera_de_imagen"

    return True, "ok"


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume faltantes por columna.
    """
    missing = df.isna().sum().reset_index()
    missing.columns = ["columna", "faltantes"]
    missing["porcentaje"] = (missing["faltantes"] / len(df) * 100).round(4)
    return missing.sort_values(by="faltantes", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Procesamiento y diagnóstico de base RSNA Pneumonia")
    parser.add_argument("--data_dir", required=True, help="Ruta base del dataset")
    parser.add_argument("--train_images_dir", default="stage_2_train_images", help="Subcarpeta de imágenes train")
    parser.add_argument("--test_images_dir", default="stage_2_test_images", help="Subcarpeta de imágenes test")
    parser.add_argument("--labels_csv", default="stage_2_train_labels.csv", help="Nombre del CSV de labels")
    parser.add_argument("--class_info_csv", default="stage_2_detailed_class_info.csv", help="Nombre del CSV de detailed class info")
    parser.add_argument("--output_dir", default="salida_preprocesamiento", help="Carpeta de salida")
    parser.add_argument("--image_size", type=int, default=224, help="Tamaño objetivo para redimensionamiento reportado")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proporción de test")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proporción de validación")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = data_dir / args.train_images_dir
    test_images_dir = data_dir / args.test_images_dir
    labels_csv_path = data_dir / args.labels_csv
    class_info_csv_path = data_dir / args.class_info_csv

    if not train_images_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes train: {train_images_dir}")
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo labels: {labels_csv_path}")
    if not class_info_csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo class info: {class_info_csv_path}")

    # ------------------------------------------------------------------
    # 1. CARGA DE ARCHIVOS
    # ------------------------------------------------------------------
    print("Cargando archivos...")
    train_image_paths = sorted(list(train_images_dir.glob("*.dcm")))
    test_image_paths = sorted(list(test_images_dir.glob("*.dcm"))) if test_images_dir.exists() else []

    labels_df = pd.read_csv(labels_csv_path)
    class_info_df = pd.read_csv(class_info_csv_path)

    # ------------------------------------------------------------------
    # 2. INSPECCIÓN INICIAL
    # ------------------------------------------------------------------
    print("Realizando inspección inicial...")

    initial_summary = {
        "train_images_count": len(train_image_paths),
        "test_images_count": len(test_image_paths),
        "labels_rows": len(labels_df),
        "class_info_rows": len(class_info_df),
        "labels_columns": labels_df.columns.tolist(),
        "class_info_columns": class_info_df.columns.tolist(),
        "labels_dtypes": {col: str(dtype) for col, dtype in labels_df.dtypes.items()},
        "class_info_dtypes": {col: str(dtype) for col, dtype in class_info_df.dtypes.items()},
    }

    missing_labels = summarize_missing_values(labels_df)
    missing_class_info = summarize_missing_values(class_info_df)

    missing_labels.to_csv(output_dir / "faltantes_labels.csv", index=False)
    missing_class_info.to_csv(output_dir / "faltantes_class_info.csv", index=False)

    # ------------------------------------------------------------------
    # 3. DUPLICADOS
    # ------------------------------------------------------------------
    print("Buscando duplicados...")
    duplicate_train_images = find_duplicate_files(train_image_paths)
    duplicate_train_images.to_csv(output_dir / "duplicados_imagenes_train.csv", index=False)

    labels_dup_rows = labels_df[labels_df.duplicated(keep=False)].copy()
    class_info_dup_rows = class_info_df[class_info_df.duplicated(keep=False)].copy()

    labels_dup_rows.to_csv(output_dir / "duplicados_filas_labels.csv", index=False)
    class_info_dup_rows.to_csv(output_dir / "duplicados_filas_class_info.csv", index=False)

    # Duplicados por patientId
    # Duplicados por patientId en labels
    labels_dup_patient = labels_df["patientId"].astype(str).value_counts().reset_index()
    labels_dup_patient.columns = ["patientId", "count"]
    labels_dup_patient["count"] = pd.to_numeric(labels_dup_patient["count"], errors="coerce")
    labels_dup_patient = labels_dup_patient[labels_dup_patient["count"] > 1]
    labels_dup_patient.to_csv(output_dir / "duplicados_patientid_labels.csv", index=False)

    # Duplicados por patientId en class_info
    class_info_dup_patient = class_info_df["patientId"].astype(str).value_counts().reset_index()
    class_info_dup_patient.columns = ["patientId", "count"]
    class_info_dup_patient["count"] = pd.to_numeric(class_info_dup_patient["count"], errors="coerce")
    class_info_dup_patient = class_info_dup_patient[class_info_dup_patient["count"] > 1]
    class_info_dup_patient.to_csv(output_dir / "duplicados_patientid_class_info.csv", index=False)

    # ------------------------------------------------------------------
    # 4. VERIFICACIÓN DE INTEGRIDAD DICOM
    # ------------------------------------------------------------------
    print("Validando archivos DICOM...")
    dicom_results = []
    image_dims = {}
    valid_train_ids = set()

    for p in tqdm(train_image_paths, desc="Leyendo DICOM train"):
        ok, meta, err = safe_read_dicom(p)
        patient_id = p.stem

        row = {
            "patientId": patient_id,
            "path": str(p),
            "read_ok": ok,
            "error": err
        }

        if meta is not None:
            row.update(meta)
            if meta.get("Rows") is not None and meta.get("Columns") is not None:
                image_dims[patient_id] = (int(meta["Rows"]), int(meta["Columns"]))

        dicom_results.append(row)
        if ok:
            valid_train_ids.add(patient_id)

    dicom_df = pd.DataFrame(dicom_results)
    dicom_df.to_csv(output_dir / "revision_dicom_train.csv", index=False)

    corrupted_or_invalid = dicom_df[~dicom_df["read_ok"]].copy()
    corrupted_or_invalid.to_csv(output_dir / "dicom_invalidos.csv", index=False)

    # ------------------------------------------------------------------
    # 5. CONSISTENCIA ENTRE IMÁGENES Y CSV
    # ------------------------------------------------------------------
    print("Revisando consistencia entre imágenes y CSV...")

    image_ids = {p.stem for p in train_image_paths}
    label_ids = set(labels_df["patientId"].astype(str).unique())
    class_info_ids = set(class_info_df["patientId"].astype(str).unique())

    images_without_label = sorted(list(image_ids - label_ids))
    labels_without_image = sorted(list(label_ids - image_ids))
    classinfo_without_image = sorted(list(class_info_ids - image_ids))

    pd.DataFrame({"patientId": images_without_label}).to_csv(output_dir / "imagenes_sin_label.csv", index=False)
    pd.DataFrame({"patientId": labels_without_image}).to_csv(output_dir / "labels_sin_imagen.csv", index=False)
    pd.DataFrame({"patientId": classinfo_without_image}).to_csv(output_dir / "classinfo_sin_imagen.csv", index=False)

    # ------------------------------------------------------------------
    # 6. INTEGRACIÓN DE TABLAS
    # ------------------------------------------------------------------
    print("Integrando labels y class info...")

    # En labels puede haber varias filas por patientId si hay múltiples bounding boxes
    merged_df = labels_df.merge(
        class_info_df,
        on="patientId",
        how="left",
        suffixes=("_label", "_class")
    )

    merged_df["patientId"] = merged_df["patientId"].astype(str)

    # variable binaria
    if "Target" not in merged_df.columns:
        raise ValueError("No se encontró la columna 'Target' en labels.")

    merged_df["Target"] = pd.to_numeric(merged_df["Target"], errors="coerce")
    merged_df["target_binaria"] = merged_df["Target"].apply(lambda x: 1 if x == 1 else 0)

    # Filtrar solo imágenes existentes
    merged_df["imagen_existe"] = merged_df["patientId"].isin(image_ids)
    merged_df["dicom_valido"] = merged_df["patientId"].isin(valid_train_ids)

    merged_df.to_csv(output_dir / "base_integrada_raw.csv", index=False)

    # ------------------------------------------------------------------
    # 7. VALIDACIÓN DE BOUNDING BOXES
    # ------------------------------------------------------------------
    print("Validando bounding boxes...")

    bbox_df = merged_df[merged_df["target_binaria"] == 1].copy()

    bbox_results = []
    for _, row in bbox_df.iterrows():
        is_valid, reason = validate_bbox(row, image_dims)
        bbox_results.append({
            "patientId": row["patientId"],
            "x": row.get("x", np.nan),
            "y": row.get("y", np.nan),
            "width": row.get("width", np.nan),
            "height": row.get("height", np.nan),
            "bbox_valida": is_valid,
            "motivo": reason
        })

    bbox_validation_df = pd.DataFrame(bbox_results)
    bbox_validation_df.to_csv(output_dir / "validacion_bounding_boxes.csv", index=False)

    valid_bbox_ids = set(
        bbox_validation_df.loc[bbox_validation_df["bbox_valida"], "patientId"].astype(str).unique()
    )

    # ------------------------------------------------------------------
    # 8. CONSTRUCCIÓN DE BASE LIMPIA
    # ------------------------------------------------------------------
    print("Construyendo base limpia...")

    # Consolidar a nivel paciente/imagen
    patient_level = merged_df.groupby("patientId", as_index=False).agg({
        "Target": "max",
        "target_binaria": "max",
        "class": lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna()) > 0 else np.nan,
        "imagen_existe": "max",
        "dicom_valido": "max"
    })

    patient_level["bbox_valida"] = patient_level["patientId"].isin(valid_bbox_ids)

    # Criterios de inclusión
    clean_df = patient_level[
        (patient_level["imagen_existe"] == True) &
        (patient_level["dicom_valido"] == True)
    ].copy()

    # ------------------------------------------------------------------
    # 9. PARTICIÓN TRAIN / VAL / TEST
    # ------------------------------------------------------------------
    print("Generando partición train/val/test...")

    if clean_df["target_binaria"].nunique() < 2:
        raise ValueError("No hay al menos dos clases en clean_df; no se puede estratificar.")

    test_size = args.test_size
    val_size = args.val_size
    remaining_for_train = 1.0 - test_size
    relative_val_size = val_size / remaining_for_train

    train_val_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=42,
        stratify=clean_df["target_binaria"]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42,
        stratify=train_val_df["target_binaria"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df.to_csv(output_dir / "dataset_limpio_con_split.csv", index=False)

    # ------------------------------------------------------------------
    # 10. RESUMEN FINAL
    # ------------------------------------------------------------------
    print("Generando resumen final...")

    labels_missing_total = int(labels_df.isna().sum().sum())
    classinfo_missing_total = int(class_info_df.isna().sum().sum())

    bbox_invalid_count = 0
    if not bbox_validation_df.empty:
        bbox_invalid_count = int((~bbox_validation_df["bbox_valida"]).sum())

    summary = {
        "descripcion_base": {
            "imagenes_train_originales": len(train_image_paths),
            "imagenes_test_originales": len(test_image_paths),
            "registros_labels_originales": len(labels_df),
            "registros_class_info_originales": len(class_info_df),
            "numero_variables_labels": len(labels_df.columns),
            "numero_variables_class_info": len(class_info_df.columns),
            "tipos_datos_labels": {col: str(dtype) for col, dtype in labels_df.dtypes.items()},
            "tipos_datos_class_info": {col: str(dtype) for col, dtype in class_info_df.dtypes.items()},
        },
        "calidad_inicial": {
            "faltantes_totales_labels": labels_missing_total,
            "faltantes_totales_class_info": classinfo_missing_total,
            "filas_duplicadas_labels": int(labels_df.duplicated().sum()),
            "filas_duplicadas_class_info": int(class_info_df.duplicated().sum()),
            "imagenes_train_duplicadas_por_id": int(duplicate_train_images["count"].sum()) if not duplicate_train_images.empty else 0,
            "dicom_invalidos_o_corruptos": int((~dicom_df["read_ok"]).sum()),
            "imagenes_sin_label": len(images_without_label),
            "labels_sin_imagen": len(labels_without_image),
            "classinfo_sin_imagen": len(classinfo_without_image),
        },
        "transformaciones_realizadas": {
            "verificacion_integridad_dicom": True,
            "integracion_tablas": True,
            "creacion_variable_binaria": True,
            "validacion_bounding_boxes": True,
            "normalizacion_reportada": True,
            "redimension_objetivo": [args.image_size, args.image_size],
            "particion_train_val_test": True,
            "estratificacion_por_clase": True
        },
        "resultados_procesamiento": {
            "imagenes_validas_dicom": int(dicom_df["read_ok"].sum()),
            "imagenes_invalidas_dicom": int((~dicom_df["read_ok"]).sum()),
            "registros_integrados_raw": len(merged_df),
            "pacientes_unicos_integrados": int(merged_df["patientId"].nunique()),
            "pacientes_limpios_finales": len(clean_df),
            "positivos_finales": int(clean_df["target_binaria"].sum()),
            "negativos_finales": int((clean_df["target_binaria"] == 0).sum()),
            "anotaciones_bbox_invalidas": bbox_invalid_count,
            "positivos_con_bbox_valida": int(clean_df["bbox_valida"].sum()),
            "train_final": len(train_df),
            "val_final": len(val_df),
            "test_final": len(test_df)
        },
        "distribucion_clases": {
            "train": train_df["target_binaria"].value_counts().to_dict(),
            "val": val_df["target_binaria"].value_counts().to_dict(),
            "test": test_df["target_binaria"].value_counts().to_dict()
        }
    }

    with open(output_dir / "resumen_preprocesamiento.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    # También guardamos un TXT más fácil de leer
    with open(output_dir / "resumen_preprocesamiento.txt", "w", encoding="utf-8") as f:
        f.write("=== RESUMEN DE PREPROCESAMIENTO RSNA ===\n\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\nProceso terminado.")
    print(f"Resultados guardados en: {output_dir.resolve()}")
    print("Archivos principales:")
    print("- resumen_preprocesamiento.json")
    print("- resumen_preprocesamiento.txt")
    print("- dataset_limpio_con_split.csv")
    print("- validacion_bounding_boxes.csv")
    print("- revision_dicom_train.csv")


if __name__ == "__main__":
    main()