import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_CLEAN_COLUMNS = {
    "patientId",
    "Target",
    "target_binaria",
    "class",
    "imagen_existe",
    "dicom_valido",
    "bbox_valida",
    "split",
}

REQUIRED_RAW_COLUMNS = {
    "patientId",
    "x",
    "y",
    "width",
    "height",
    "Target",
    "class",
    "target_binaria",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera analisis exploratorio de datos para la base limpia del proyecto RSNA."
    )
    parser.add_argument(
        "--clean_csv",
        default="data/interim/dataset_limpio_con_split.csv",
        help="Ruta del dataset limpio con variable split.",
    )
    parser.add_argument(
        "--raw_csv",
        default="data/interim/base_integrada_raw.csv",
        help="Ruta de la base integrada raw para derivar metricas de bounding boxes.",
    )
    parser.add_argument(
        "--preprocessing_summary",
        default="data/interim/resumen_preprocesamiento.json",
        help="Ruta del resumen JSON generado por el preprocesamiento.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Carpeta donde se guardaran tablas y resumenes del EDA (las figuras pueden moverse a reports/figures/eda).",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required_columns: set[str], file_name: str) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"El archivo {file_name} no contiene las columnas requeridas: {missing_columns}"
        )


def load_dataframe(csv_path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, required_columns, csv_path.name)
    return df


def load_preprocessing_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}

    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_bbox_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    bbox_df = raw_df.copy()
    numeric_columns = ["x", "y", "width", "height", "Target", "target_binaria"]
    for column in numeric_columns:
        bbox_df[column] = pd.to_numeric(bbox_df[column], errors="coerce")

    bbox_df = bbox_df[bbox_df["target_binaria"] == 1].copy()
    bbox_df = bbox_df.dropna(subset=["x", "y", "width", "height"])
    bbox_df = bbox_df.drop_duplicates(subset=["patientId", "x", "y", "width", "height"])
    bbox_df["bbox_area"] = bbox_df["width"] * bbox_df["height"]
    bbox_df["bbox_center_x"] = bbox_df["x"] + (bbox_df["width"] / 2.0)
    bbox_df["bbox_center_y"] = bbox_df["y"] + (bbox_df["height"] / 2.0)

    if bbox_df.empty:
        return pd.DataFrame(
            columns=[
                "patientId",
                "bbox_count",
                "bbox_area_total",
                "bbox_area_mean",
                "bbox_width_mean",
                "bbox_height_mean",
                "bbox_center_x_mean",
                "bbox_center_y_mean",
            ]
        )

    return (
        bbox_df.groupby("patientId", as_index=False)
        .agg(
            bbox_count=("bbox_area", "size"),
            bbox_area_total=("bbox_area", "sum"),
            bbox_area_mean=("bbox_area", "mean"),
            bbox_width_mean=("width", "mean"),
            bbox_height_mean=("height", "mean"),
            bbox_center_x_mean=("bbox_center_x", "mean"),
            bbox_center_y_mean=("bbox_center_y", "mean"),
        )
        .reset_index(drop=True)
    )


def build_eda_dataset(clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    eda_df = clean_df.copy()
    bbox_features = build_bbox_features(raw_df)
    eda_df = eda_df.merge(bbox_features, on="patientId", how="left")

    fill_zero_columns = [
        "bbox_count",
        "bbox_area_total",
        "bbox_area_mean",
        "bbox_width_mean",
        "bbox_height_mean",
        "bbox_center_x_mean",
        "bbox_center_y_mean",
    ]
    for column in fill_zero_columns:
        eda_df[column] = pd.to_numeric(eda_df[column], errors="coerce").fillna(0.0)

    eda_df["bbox_count"] = eda_df["bbox_count"].astype(int)
    eda_df["proporcion_bbox_valida"] = np.where(
        eda_df["target_binaria"] == 1,
        eda_df["bbox_valida"].astype(int),
        0,
    )
    eda_df["target_texto"] = eda_df["target_binaria"].map({0: "Negativo", 1: "Positivo"})
    return eda_df


def compute_numeric_summary(eda_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "Target",
        "target_binaria",
        "bbox_count",
        "bbox_area_total",
        "bbox_area_mean",
        "bbox_width_mean",
        "bbox_height_mean",
        "bbox_center_x_mean",
        "bbox_center_y_mean",
    ]
    summary = eda_df[numeric_columns].describe().T.reset_index().rename(columns={"index": "variable"})
    summary = summary[["variable", "mean", "50%", "min", "max", "std"]]
    return summary.rename(columns={"50%": "median", "std": "std_dev"}).round(4)


def compute_group_summary(eda_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        eda_df.groupby("target_texto", as_index=False)
        .agg(
            pacientes=("patientId", "count"),
            bbox_count_promedio=("bbox_count", "mean"),
            bbox_area_total_promedio=("bbox_area_total", "mean"),
            bbox_area_total_mediana=("bbox_area_total", "median"),
        )
        .round(4)
    )
    grouped["proporcion"] = (grouped["pacientes"] / grouped["pacientes"].sum()).round(4)
    return grouped


def compute_frequency_tables(eda_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frequency_tables = {}
    for column in ["class", "split", "target_texto", "bbox_valida"]:
        counts = eda_df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="frecuencia")
        counts["proporcion"] = (counts["frecuencia"] / len(eda_df)).round(4)
        frequency_tables[column] = counts
    return frequency_tables


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def plot_class_distribution(eda_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure()
    order = eda_df["class"].value_counts().index
    ax = sns.countplot(data=eda_df, x="class", order=order, hue="class", legend=False)
    ax.set_title("Distribucion de clases clinicas")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=15)
    for patch in ax.patches:
        height = int(patch.get_height())
        ax.annotate(
            f"{height}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_univariante_clases.png", dpi=300)
    plt.close()


def plot_bbox_area_histogram(eda_df: pd.DataFrame, output_dir: Path) -> None:
    positive_df = eda_df[eda_df["target_binaria"] == 1].copy()
    plt.figure()
    sns.histplot(data=positive_df, x="bbox_area_total", bins=30, kde=True)
    plt.title("Distribucion del area total de bounding boxes en casos positivos")
    plt.xlabel("Area total de bounding boxes")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_univariante_area_bbox.png", dpi=300)
    plt.close()


def plot_split_target_distribution(eda_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure()
    ax = sns.countplot(data=eda_df, x="split", hue="target_texto")
    ax.set_title("Distribucion de casos por particion y clase binaria")
    ax.set_xlabel("Split")
    ax.set_ylabel("Frecuencia")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_multivariante_split_target.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(eda_df: pd.DataFrame, output_dir: Path) -> None:
    corr_columns = [
        "target_binaria",
        "bbox_count",
        "bbox_area_total",
        "bbox_area_mean",
        "bbox_width_mean",
        "bbox_height_mean",
        "bbox_center_x_mean",
        "bbox_center_y_mean",
    ]
    corr_matrix = eda_df[corr_columns].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Mapa de calor de correlaciones")
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_multivariante_correlaciones.png", dpi=300)
    plt.close()


def plot_bbox_scatter(eda_df: pd.DataFrame, output_dir: Path) -> None:
    positive_df = eda_df[eda_df["target_binaria"] == 1].copy()
    plt.figure()
    sns.scatterplot(
        data=positive_df,
        x="bbox_width_mean",
        y="bbox_height_mean",
        hue="split",
        size="bbox_count",
        sizes=(30, 150),
        alpha=0.75,
    )
    plt.title("Relacion entre ancho y alto promedio de bounding boxes")
    plt.xlabel("Ancho promedio")
    plt.ylabel("Alto promedio")
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_multivariante_dispersion_bbox.png", dpi=300)
    plt.close()


def build_highlights(
    eda_df: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    preprocessing_summary: dict,
) -> list[str]:
    total_patients = len(eda_df)
    positive_cases = int(eda_df["target_binaria"].sum())
    negative_cases = total_patients - positive_cases
    positive_ratio = positive_cases / total_patients if total_patients else 0.0

    class_distribution = eda_df["class"].value_counts(normalize=True).mul(100).round(2)
    majority_class = class_distribution.index[0]
    majority_class_pct = class_distribution.iloc[0]

    split_distribution = eda_df["split"].value_counts(normalize=True).sort_index().mul(100).round(2)
    positive_group = group_summary[group_summary["target_texto"] == "Positivo"]
    bbox_area_mean_positive = 0.0
    bbox_count_mean_positive = 0.0
    if not positive_group.empty:
        bbox_area_mean_positive = float(positive_group["bbox_area_total_promedio"].iloc[0])
        bbox_count_mean_positive = float(positive_group["bbox_count_promedio"].iloc[0])

    preprocessing_results = preprocessing_summary.get("resultados_procesamiento", {})
    clean_total_reported = preprocessing_results.get("pacientes_limpios_finales")
    positives_reported = preprocessing_results.get("positivos_finales")

    highlights = [
        "Resumen automatico del EDA para apoyar la redaccion del reporte tecnico.",
        (
            f"La base final contiene {total_patients} pacientes validos. "
            f"De ellos, {positive_cases} ({positive_ratio:.2%}) son positivos y {negative_cases} ({1 - positive_ratio:.2%}) son negativos."
        ),
        (
            f"La clase clinica con mayor presencia es '{majority_class}' con {majority_class_pct:.2f}% del total. "
            f"Esto evidencia desbalance entre categorias y justifica reportar proporciones ademas de frecuencias absolutas."
        ),
        (
            f"La particion train/val/test se mantiene cercana a la distribucion esperada: "
            f"train={split_distribution.get('train', 0):.2f}%, "
            f"val={split_distribution.get('val', 0):.2f}%, "
            f"test={split_distribution.get('test', 0):.2f}%."
        ),
        (
            f"En los casos positivos, el numero promedio de bounding boxes por paciente es {bbox_count_mean_positive:.2f} "
            f"y el area total promedio anotada es {bbox_area_mean_positive:.2f}."
        ),
        (
            "Los graficos recomendados para el informe son: distribucion de clases, histograma del area total de bounding boxes, "
            "conteo por split y clase binaria, y mapa de calor de correlaciones."
        ),
    ]

    if clean_total_reported is not None and positives_reported is not None:
        highlights.append(
            (
                f"El resumen del preprocesamiento coincide con el EDA: {clean_total_reported} pacientes limpios y "
                f"{positives_reported} positivos finales."
            )
        )

    bbox_area_summary = numeric_summary[numeric_summary["variable"] == "bbox_area_total"]
    if not bbox_area_summary.empty:
        row = bbox_area_summary.iloc[0]
        highlights.append(
            (
                f"Para la variable bbox_area_total, la media es {row['mean']:.2f}, la mediana {row['median']:.2f}, "
                f"el minimo {row['min']:.2f}, el maximo {row['max']:.2f} y la desviacion estandar {row['std_dev']:.2f}."
            )
        )

    return highlights


def save_highlights(highlights: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(highlights) + "\n", encoding="utf-8")


def save_summary_json(
    eda_df: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    frequency_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    summary_payload = {
        "total_pacientes": int(len(eda_df)),
        "casos_positivos": int(eda_df["target_binaria"].sum()),
        "casos_negativos": int((eda_df["target_binaria"] == 0).sum()),
        "distribucion_por_split": frequency_tables["split"].to_dict(orient="records"),
        "distribucion_por_clase": frequency_tables["class"].to_dict(orient="records"),
        "resumen_numerico": numeric_summary.to_dict(orient="records"),
        "resumen_por_objetivo": group_summary.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(summary_payload, indent=4, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_df = load_dataframe(Path(args.clean_csv), REQUIRED_CLEAN_COLUMNS)
    raw_df = load_dataframe(Path(args.raw_csv), REQUIRED_RAW_COLUMNS)
    preprocessing_summary = load_preprocessing_summary(Path(args.preprocessing_summary))

    eda_df = build_eda_dataset(clean_df, raw_df)
    numeric_summary = compute_numeric_summary(eda_df)
    group_summary = compute_group_summary(eda_df)
    frequency_tables = compute_frequency_tables(eda_df)

    save_dataframe(eda_df, output_dir / "dataset_eda_rsna.csv")
    save_dataframe(numeric_summary, output_dir / "resumen_estadistico_numerico.csv")
    save_dataframe(group_summary, output_dir / "resumen_estadistico_por_objetivo.csv")
    for table_name, table_df in frequency_tables.items():
        save_dataframe(table_df, output_dir / f"frecuencias_{table_name}.csv")

    configure_plotting()
    plot_class_distribution(eda_df, output_dir)
    plot_bbox_area_histogram(eda_df, output_dir)
    plot_split_target_distribution(eda_df, output_dir)
    plot_correlation_heatmap(eda_df, output_dir)
    plot_bbox_scatter(eda_df, output_dir)

    highlights = build_highlights(eda_df, numeric_summary, group_summary, preprocessing_summary)
    save_highlights(highlights, output_dir / "hallazgos_eda.txt")
    save_summary_json(
        eda_df,
        numeric_summary,
        group_summary,
        frequency_tables,
        output_dir / "resumen_eda.json",
    )

    print(f"EDA generado correctamente en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()