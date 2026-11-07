"""HTTP service for Molecula.

Inference happens here and nowhere else: the browser sends SMILES and gets back
what the saved model said. A row the model cannot read comes back with
``status: "invalid"`` and an error message - never with a zero.

    uvicorn molecula.api:app --port 8000
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .chem import InvalidSmilesError, standardize, svg_structure
from .model import DEFAULT_MODEL_PATH, SavedModel, Prediction

log = logging.getLogger("molecula.api")

MAX_BATCH_ROWS = 2000
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
WEB_ROOT = PACKAGE_ROOT / "web"


# --------------------------------------------------------------------------
# request / response shapes
# --------------------------------------------------------------------------


class PredictRecord(BaseModel):
    # Deliberately permissive: a record with a number where a SMILES should be,
    # or a missing id, has to come back as one invalid row - not as a 422 that
    # throws away the whole batch.
    id: Any = None
    smiles: Any = None


class PredictRequest(BaseModel):
    records: list[PredictRecord] = Field(default_factory=list)
    include_structure: bool = False
    include_neighbours: bool = True


class NeighbourOut(BaseModel):
    inchikey: str
    smiles: str
    si: float
    similarity: float
    dois: str | None = None
    documents: str | None = None


class PredictResultOut(BaseModel):
    id: Any
    status: Literal["ok", "invalid"]
    prediction_si: float | None = None
    error: str | None = None
    # everything below is extra context, the contract above is the required part
    prediction_log10_si: float | None = None
    input_smiles: str | None = None
    canonical_smiles: str | None = None
    inchikey: str | None = None
    nearest_similarity: float | None = None
    applicability: Literal["in_domain", "out_of_domain"] | None = None
    reliability_assessed: bool | None = None
    neighbours: list[NeighbourOut] | None = None
    structure_svg: str | None = None
    computed_at: str | None = None


class PredictResponse(BaseModel):
    model_version: str
    dataset_version: str
    results: list[PredictResultOut]


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

app = FastAPI(
    title="Molecula",
    version=__version__,
    description="Predicted selectivity index (SI = CC50/IC50) against vaccinia virus, from SMILES.",
)

_model: SavedModel | None = None


def get_model() -> SavedModel:
    global _model
    if _model is None:
        path = Path(DEFAULT_MODEL_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _model = SavedModel.load(path)
        log.info("loaded model %s", _model.model_version)
    return _model


def _reliability_assessed(model: SavedModel) -> bool:
    domain = model.metrics.get("applicability_domain", {})
    return bool(domain.get("separates_error"))


def _to_out(
    identifier: Any,
    molecule: Any,
    prediction: Prediction,
    model: SavedModel,
    include_structure: bool,
    include_neighbours: bool,
) -> PredictResultOut:
    return PredictResultOut(
        id=identifier,
        status="ok",
        prediction_si=prediction.si,
        prediction_log10_si=prediction.log10_si,
        input_smiles=molecule.input_smiles,
        canonical_smiles=prediction.canonical_smiles,
        inchikey=prediction.inchikey,
        nearest_similarity=prediction.nearest_similarity,
        applicability="in_domain" if prediction.in_domain else "out_of_domain",
        reliability_assessed=_reliability_assessed(model),
        neighbours=(
            [
                NeighbourOut(
                    inchikey=n.inchikey,
                    smiles=n.smiles,
                    si=n.si,
                    similarity=n.similarity,
                    dois=n.dois if isinstance(n.dois, str) else None,
                    documents=n.documents if isinstance(n.documents, str) else None,
                )
                for n in prediction.neighbours
            ]
            if include_neighbours
            else None
        ),
        structure_svg=svg_structure(molecule.mol) if include_structure else None,
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        error=None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = get_model()
    if not request.records:
        raise HTTPException(status_code=400, detail="no records supplied")
    if len(request.records) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"{len(request.records)} rows exceeds the limit of {MAX_BATCH_ROWS} per request",
        )

    results: list[PredictResultOut] = []
    for position, record in enumerate(request.records, start=1):
        identifier = record.id if record.id is not None else f"row-{position}"
        text = record.smiles if isinstance(record.smiles, str) else None
        try:
            molecule = standardize(text or "")
        except InvalidSmilesError as exc:
            results.append(
                PredictResultOut(
                    id=identifier,
                    status="invalid",
                    prediction_si=None,
                    error=str(exc),
                    input_smiles=text,
                )
            )
            continue
        try:
            prediction = model.predict_molecule(
                molecule, n_neighbours=3 if request.include_neighbours else 1
            )
            results.append(
                _to_out(
                    identifier,
                    molecule,
                    prediction,
                    model,
                    request.include_structure,
                    request.include_neighbours,
                )
            )
        except Exception as exc:  # a failure must never look like a prediction
            log.exception("inference failed for id=%s", identifier)
            results.append(
                PredictResultOut(
                    id=identifier,
                    status="invalid",
                    prediction_si=None,
                    error=f"inference failed: {exc}",
                    input_smiles=text,
                )
            )

    return PredictResponse(
        model_version=model.model_version,
        dataset_version=model.dataset_version,
        results=results,
    )


@app.post("/api/parse-csv")
async def parse_csv(file: UploadFile = File(...)) -> JSONResponse:
    """Read an uploaded CSV into {id, smiles} rows, keeping the file's order."""
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file is larger than 8 MB")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = payload.decode("cp1251")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="cannot decode the file as UTF-8 or CP1251")

    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            delimiter=delimiter,
            dtype=str,
            keep_default_na=False,
            index_col=False,  # a ragged row must not silently shift every column
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot read this file as a table: {exc}")
    if frame.empty:
        raise HTTPException(status_code=400, detail="the file has no data rows")

    lowered = {c.lower().strip(): c for c in frame.columns}
    smiles_column = next(
        (lowered[k] for k in ("smiles", "canonical_smiles", "structure", "molecule") if k in lowered),
        None,
    )
    if smiles_column is None:
        raise HTTPException(
            status_code=400,
            detail=f"no SMILES column found. Columns in the file: {list(frame.columns)}",
        )
    id_column = next(
        (lowered[k] for k in ("id", "mol_id", "molecule_id", "name", "row_id", "index") if k in lowered),
        None,
    )

    records = []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        records.append(
            {
                "id": (str(row[id_column]).strip() or f"row-{position}") if id_column else f"row-{position}",
                "smiles": str(row[smiles_column]).strip(),
            }
        )
    if len(records) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413, detail=f"the file has {len(records)} rows, the limit is {MAX_BATCH_ROWS}"
        )

    return JSONResponse(
        {
            "filename": file.filename,
            "id_column": id_column,
            "smiles_column": smiles_column,
            "records": records,
        }
    )


@app.post("/api/structure")
def structure(payload: dict[str, Any]) -> dict:
    """Structure drawing for one molecule, in the theme the UI asks for."""
    try:
        molecule = standardize(payload.get("smiles") or "")
    except InvalidSmilesError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "canonical_smiles": molecule.canonical_smiles,
        "svg": svg_structure(molecule.mol, dark=bool(payload.get("dark"))),
    }


@app.get("/api/model")
def model_info() -> dict:
    model = get_model()
    info = model.info()
    info["reliability_assessed"] = _reliability_assessed(model)
    info["limitations"] = LIMITATIONS
    return info


@app.get("/api/dataset")
def dataset_info() -> dict:
    """Dataset provenance for the "Data and model" screen."""
    model = get_model()
    processed = PROJECT_ROOT / "data" / "processed"

    summary: dict[str, Any] = {}
    summary_path = processed / "collection_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    sources: list[dict] = []
    sources_path = processed / "sources.csv"
    if sources_path.exists():
        frame = pd.read_csv(sources_path).fillna("")
        used_documents: set[str] = set()
        for row in model.training_rows:
            for document in str(row.get("documents") or "").split(";"):
                document = document.strip()
                if document:
                    used_documents.add(document)
        frame = frame[frame["document_chembl_id"].isin(used_documents)] if used_documents else frame
        sources = frame.sort_values("year", ascending=False).to_dict("records")

    exclusions_path = processed / "exclusions.csv"
    exclusion_summary: dict[str, int] = {}
    if exclusions_path.exists():
        exclusions = pd.read_csv(exclusions_path)
        if not exclusions.empty and "reason" in exclusions:
            cleaned = exclusions["reason"].astype(str).str.replace(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", "N", regex=True)
            exclusion_summary = cleaned.value_counts().head(25).to_dict()

    audit: dict = {}
    audit_path = PROJECT_ROOT / "reports" / "official_data_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())

    analysis: dict = {}
    analysis_path = PROJECT_ROOT / "reports" / "error_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

    by_source: dict[str, int] = {}
    for row in model.training_rows:
        if row.get("used_in_training"):
            by_source[str(row.get("source", "unknown"))] = by_source.get(str(row.get("source", "unknown")), 0) + 1

    return {
        "model_version": model.model_version,
        "dataset_version": model.dataset_version,
        "n_training_molecules": len(model.training_rows),
        "training_rows_by_source": by_source,
        "official_data_audit": audit,
        "error_analysis": analysis,
        "collection": summary,
        "metrics": model.metrics,
        "sources": sources,
        "exclusion_summary": exclusion_summary,
        "limitations": LIMITATIONS,
    }


@app.get("/api/health")
def health() -> dict:
    try:
        model = get_model()
    except Exception as exc:
        return {"status": "no-model", "detail": str(exc)}
    return {"status": "ok", "model_version": model.model_version}


LIMITATIONS = [
    "Это прогноз модели для индекса селективности, а не доказательство того, что молекула "
    "работает как лекарство. Всё, что выглядит интересно, дальше проверяется в лаборатории.",
    "Модель обучена на наборе организаторов соревнования (320 молекул) и использует собранный "
    "из ChEMBL набор как дополнение и как справочник происхождения. Прогноз считается только "
    "по структуре: колонки CC50 и IC50, из которых SI вычисляется напрямую, не читаются нигде — "
    "условие задачи это запрещает.",
    "В наборе организаторов 26 % строк имеют SI = 0: так записаны соединения, для которых IC50 "
    "не установлена. Это соглашение о записи, а не измеренное нулевое отношение, и модель учится "
    "воспроизводить именно его.",
    "Официальная метрика — MSE на сыром SI, а значения растянуты до 36 000. Поэтому почти вся "
    "метрика определяется несколькими самыми большими молекулами: в обучающем наборе одна "
    "молекула даёт 94 % суммы квадратов. Средняя ошибка по выборке мало что говорит о том, "
    "насколько хорош прогноз для обычной молекулы.",
    "Дополнительный набор из ChEMBL — это опубликованные измерения противовирусной активности против "
    "вируса осповакцины и цитотоксичности на клетках-хозяевах. IC50 и CC50 соединяются в SI "
    "только внутри одной публикации и только при совпадающей (или не указанной с одной стороны) "
    "клеточной линии, поэтому отношение никогда не считается по двум несвязанным экспериментам.",
    "Измерения, записанные как «>» или «<» концентрации, исключены: истинное отношение по ним "
    "неизвестно. Они перечислены в отчёте исключений.",
    "Значения в µg/mL пересчитаны в микромоли по молекулярной массе структуры из той же записи; "
    "неверная структура в источнике даёт неверный пересчёт.",
    "Там, где ChEMBL не сохранил знак отношения, число принято за точное; такие записи помечены "
    "в данных полем relation_assumed_exact.",
    "Наборы небольшие и смещены в сторону нуклеозидных аналогов, поэтому для химически далёких "
    "молекул это экстраполяция. Интерфейс помечает такие структуры как вне привычной области модели.",
    "Опубликованные SI одной молекулы расходятся между лабораториями; в наборе хранится медиана "
    "log10(SI) по структуре, а разброс сохраняется отдельным полем.",
]


if WEB_ROOT.exists():
    @app.get("/")
    def index() -> HTMLResponse:
        # The stylesheet and the script are stamped with their own file time,
        # so an updated service never leaves a browser running the old page.
        html = (WEB_ROOT / "index.html").read_text()
        for asset in ("styles.css", "app.js"):
            stamp = int((WEB_ROOT / asset).stat().st_mtime)
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
        return HTMLResponse(html)

    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
