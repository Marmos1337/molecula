"""Render the experiment artefacts into reports/RESULTS.md.

Every number in the report is read from the files the pipeline wrote, so the
report cannot drift away from the run that produced it.

    PYTHONPATH=src .venv/bin/python scripts/render_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")
SCHEME_TITLES = {
    "molecule": "molecule — случайное разбиение по молекулам (так же разделено само соревнование)",
    "scaffold": "scaffold — целые скелеты в одной части",
    "cluster": "cluster — целые химические семейства (Butina, Tanimoto 0.4)",
    "publication": "publication — целые источники (штамм/условия эксперимента)",
}


def table(frame: pd.DataFrame, columns: dict[str, str], floats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, rule]
    for row in frame.to_dict("records"):
        cells = []
        for key in columns:
            value = row.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("—")
            elif key in floats:
                cells.append(format(value, floats[key]))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    config = json.loads((REPORTS / "experiments_config.json").read_text())
    summary = pd.read_csv(REPORTS / "experiments_summary.csv")
    paired = pd.read_csv(REPORTS / "experiments_paired.csv")

    out: list[str] = []
    out.append("# Сравнительное исследование и независимая проверка\n")
    out.append(
        "Все числа ниже получены одним прогоном `make experiments` и лежат в "
        "`reports/experiments.csv` (строка на фолд), `reports/experiments_summary.csv`, "
        "`reports/experiments_paired.csv` и `reports/oof_predictions.csv`. "
        "Таблицы в этом файле не набираются руками: их печатает `scripts/render_results.py`.\n"
    )

    out.append("## Как считалось\n")
    out.append(
        f"- Метрика: {config['metric']}.\n"
        f"- Схем разбиения: {len(config['schemes'])} ({', '.join(config['schemes'])}); "
        f"зерно случайности: {', '.join(str(s) for s in config['seeds'])}; "
        f"фолдов в каждой: {config['folds']}.\n"
        f"- Это {len(config['schemes']) * len(config['seeds']) * config['folds']} обучений на каждый подход, "
        f"и все подходы видели ровно одни и те же фолды.\n"
        f"- Обучающих строк соревнования: {config['n_official_rows']}; "
        f"собранных строк ChEMBL для пополнения: {config['n_external_rows']}.\n"
        "- Внешние данные добавляются только в обучающую часть фолда, и из них выбрасывается "
        "всё, чей скелет лежит в проверочной части этого же фолда.\n"
        f"- Время прогона: {config['elapsed_seconds']:.0f} с.\n"
    )

    out.append("\n## Почему средний MSE здесь почти ничего не решает\n")
    analysis_path = REPORTS / "error_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        concentration = analysis["metric_concentration"]["share_from_top"]
        largest = analysis["metric_concentration"]["largest_si"]
        out.append(
            f"MSE считается на сыром SI, поэтому вклад молекулы растёт как квадрат её значения. "
            f"В обучающем наборе самая большая молекула (SI = {largest[0]:,.0f}) даёт "
            f"{concentration['1']:.1%} всей суммы квадратов, пять самых больших — {concentration['5']:.1%}. "
            "Значит метрика фолда определяется тем, в какой фолд попала эта горстка молекул, "
            "а не тем, насколько хорош подход. Поэтому ниже рядом со средним MSE приводится "
            "парная разность: та же пара подходов на тех же фолдах, где общий разброс сокращается.\n"
        )

    out.append("\n## Сравнение подходов\n")
    for scheme in config["schemes"]:
        part = summary[summary["scheme"] == scheme].copy()
        if part.empty:
            continue
        out.append(f"\n### {SCHEME_TITLES.get(scheme, scheme)}\n")
        out.append(table(
            part,
            {
                "candidate": "подход",
                "mse_mean": "MSE (среднее по фолдам)",
                "mse_sd": "σ по фолдам",
                "rmse_log10_mean": "RMSE log10",
                "mae_mean": "MAE",
                "spearman_mean": "Спирмен",
            },
            {"mse_mean": ",.0f", "mse_sd": ",.0f", "rmse_log10_mean": ".3f",
             "mae_mean": ".1f", "spearman_mean": ".3f"},
        ))
        out.append("")

    out.append("\n## Парные разности с предсказанием нулём\n")
    out.append(
        "Отрицательная разность — подход лучше опорного на том же фолде. "
        "Разности берутся по каждому обучению отдельно, поэтому общий разброс, "
        "из-за которого средние MSE неразличимы, здесь сокращается.\n"
    )
    for scheme in config["schemes"]:
        part = paired[paired["scheme"] == scheme].copy()
        if part.empty:
            continue
        out.append(f"\n### {SCHEME_TITLES.get(scheme, scheme)}\n")
        out.append(table(
            part,
            {
                "candidate": "подход",
                "mean_delta_mse": "ΔMSE (среднее)",
                "sd_delta_mse": "σ разности",
                "fits_better": "обучений лучше",
                "fits": "обучений всего",
            },
            {"mean_delta_mse": ",.0f", "sd_delta_mse": ",.0f"},
        ))
        out.append("")

    by_molecule_path = REPORTS / "experiments_paired_by_molecule.csv"
    if by_molecule_path.exists():
        by_molecule = pd.read_csv(by_molecule_path)
        out.append("\n## То же сравнение, но по молекулам\n")
        out.append(
            "Фолды — не выборка из 15: три зерна перераскладывают одни и те же молекулы, "
            "и каждая считалась бы трижды. Здесь единица — молекула (квадрат ошибки усреднён "
            "по зёрнам), поэтому выборка действительно парная.\n\n"
            "`p` — критерий знаковых рангов Уилкоксона, описательный и без поправки на "
            "множественность сравнений. В схемах на перенос молекулы одной серии зависимы, "
            "так что он оптимистичен. И он отвечает на вопрос «становится ли лучше типичной "
            "молекуле», тогда как MSE — среднее, которое решают единицы: подход может "
            "выигрывать по этому критерию и всё равно иметь худший MSE.\n"
        )
        for scheme in config["schemes"]:
            part = by_molecule[by_molecule["scheme"] == scheme].copy()
            if part.empty:
                continue
            out.append(f"\n### {SCHEME_TITLES.get(scheme, scheme)}\n")
            columns = {
                "candidate": "подход",
                "mean_delta_squared_error": "Δ квадрата ошибки (среднее)",
                "median_delta_squared_error": "медиана Δ",
                "molecules_better": "молекул лучше",
                "molecules": "молекул всего",
            }
            if "wilcoxon_p" in part.columns:
                columns["wilcoxon_p"] = "p"
            out.append(table(part, columns, {
                "mean_delta_squared_error": ",.0f",
                "median_delta_squared_error": ",.2f",
                "wilcoxon_p": ".3g",
            }))
            out.append("")

    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        out.append("\n## Разбор ошибок выбранного подхода\n")
        out.append(
            f"Подход: `{analysis['candidate']}`, схема `{analysis['scheme']}`, "
            f"{analysis['n_molecules']} молекул, прогноз каждой получен вне обучения "
            f"и усреднён по зёрнам {', '.join(str(s) for s in analysis['seeds_averaged'])}.\n"
        )
        out.append("\n### По диапазонам SI\n")
        out.append(table(
            pd.DataFrame(analysis["by_si_range"]),
            {"range": "диапазон", "n": "молекул", "mse": "MSE", "mae": "MAE",
             "median_abs_log10_error": "медиана |ошибки| log10",
             "mean_si": "среднее SI", "mean_prediction": "средний прогноз"},
            {"mse": ",.0f", "mae": ".1f", "median_abs_log10_error": ".2f",
             "mean_si": ".1f", "mean_prediction": ".1f"},
        ))
        out.append("\n### По сходству с обучающим набором\n")
        out.append(table(
            pd.DataFrame(analysis["by_similarity_to_training"]),
            {"similarity": "сходство", "n": "молекул", "mse": "MSE",
             "median_abs_log10_error": "медиана |ошибки| log10"},
            {"mse": ",.0f", "median_abs_log10_error": ".2f"},
        ))
        out.append("\n### По культуре клеток\n")
        out.append(table(
            pd.DataFrame(analysis["by_culture"]),
            {"culture": "культура", "n": "молекул", "mse": "MSE",
             "median_abs_log10_error": "медиана |ошибки| log10"},
            {"mse": ",.0f", "median_abs_log10_error": ".2f"},
        ))

        domain = analysis["applicability_domain"]
        out.append("\n### Область применимости — проверка показателя\n")
        verdict = {
            True: "показатель подтверждён: сходство действительно отделяет точные прогнозы от неточных",
            False: "показатель НЕ подтверждён",
            None: "проверить не удалось",
        }[domain.get("separates_error")]
        out.append(
            f"- Показатель: {domain['indicator']}, порог {domain['threshold']}.\n"
            f"- Молекул: {domain['n_points']}, ниже порога {domain['share_below_threshold']:.1%}.\n"
            f"- Спирмен «сходство ↔ |ошибка| в log10»: {domain['spearman_similarity_vs_abs_log_error']:.3f} "
            f"(p = {domain['spearman_p_value']:.2g}).\n"
            f"- **Вывод: {verdict}.**\n"
        )

        out.append("\n### Молекулы с наибольшим вкладом в ошибку\n")
        out.append(table(
            pd.DataFrame(analysis["worst_molecules"]),
            {"inchikey": "InChIKey", "si": "измеренный SI", "prediction": "прогноз",
             "similarity": "сходство с обучением"},
            {"si": ",.1f", "prediction": ",.1f", "similarity": ".2f"},
        ))

    path = REPORTS / "RESULTS.md"
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
