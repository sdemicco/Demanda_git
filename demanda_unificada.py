# -*- coding: utf-8 -*-
# demanda_ecm_app.py
# App 1 página: EMAE (OLS para 12m) + Demanda I+C (ECM) + escenarios 2026–2031 con ECM
# Incluye selector de método para la tasa base (12M rodantes vs. año calendario) y
# muestra de ancla (último año completo) + tasa base en UI y en escenarios.

import io
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Demanda I+C con ECM (corto y largo plazo)", layout="wide", page_icon="📈")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_NAME  = "variables.csv"  # demanda
EMAE_NAME = "emae.xlsx"      # emae

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def repo_file_bytes(name: str) -> Optional[bytes]:
    p = DATA_DIR / name
    return p.read_bytes() if p.exists() else None

def to_monthly_index(df: pd.DataFrame, date_col: str = "indice_tiempo") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        cands = [c for c in df.columns if any(k in c.lower() for k in ["fecha", "indice", "date", "time", "period"])]
        date_col = cands[0] if cands else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col).asfreq("MS")
    return df

def monthly_fill(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    tmp = pd.DataFrame({"v": s})
    tmp["m"] = tmp.index.month
    med = tmp.groupby("m")["v"].transform("median")
    return s.fillna(med).ffill().bfill()

def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))
def mape(a, b):
    a = np.asarray(a); b = np.asarray(b); eps = 1e-8
    return float(np.mean(np.abs((a - b) / np.maximum(np.abs(a), eps)))) * 100.0
def r2_score(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ss_res = np.sum((a - b)**2); ss_tot = np.sum((a - np.mean(a))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def next_12_from_last(last_ts: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_ts) + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=12, freq="MS")

def seasonal_factors_from_history(x: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"x": x})
    df["year"] = df.index.year
    df["month"] = df.index.month
    year_avg = df.groupby("year")["x"].transform("mean")
    ratio = df["x"] / year_avg
    S = ratio.groupby(df["month"]).mean()
    S = S / S.mean()
    return S.reindex(range(1, 13)).values

def last_full_year(x: pd.Series) -> int:
    last_year = x.index.max().year
    return last_year if len(x.loc[str(last_year)]) == 12 else last_year - 1

# -----------------------------------------------------------------------------
# Carga de series
# -----------------------------------------------------------------------------
def load_emae_from_excel(xlsx_bytes: bytes) -> pd.Series:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    # detectar columna de fechas
    date_col = None
    for c in df.columns:
        cl = str(c).lower()
        if np.issubdtype(df[c].dtype, np.datetime64) or any(k in cl for k in ["fecha","indice","date","time","period"]):
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df = to_monthly_index(df, date_col)
    # detectar columna de valores
    val_col = None
    for c in df.columns:
        if "emae" in c.lower(): val_col = c; break
    if val_col is None:
        nums = df.select_dtypes(include=[np.number]).columns
        val_col = nums[0] if len(nums) else df.columns[0]
    s = monthly_fill(pd.to_numeric(df[val_col], errors="coerce").asfreq("MS"))
    s.name = "EMAE"
    return s

def load_dem_ic_from_csv(csv_bytes: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = to_monthly_index(df, "indice_tiempo")
    cand = [c for c in df.columns if ("demanda" in c.lower()) and ("comercial" in c.lower() or "industrial" in c.lower())]
    col = cand[0] if cand else ("Demanda_Comercial_Industrial" if "Demanda_Comercial_Industrial" in df.columns else df.select_dtypes(include=[np.number]).columns[0])
    s = monthly_fill(pd.to_numeric(df[col], errors="coerce").asfreq("MS"))
    s.name = "DemIC"
    return s

# -----------------------------------------------------------------------------
# EMAE — OLS(log) t + C(Month) para pronóstico 12 meses
# -----------------------------------------------------------------------------
def fit_emae_ols_log(y: pd.Series, test_frac: float = 0.10):
    y = pd.Series(y).dropna().astype(float)
    if y.index.freq is None: y = y.asfreq("MS")
    df = pd.DataFrame(index=y.index)
    df["y"] = y.values
    df["t"] = np.arange(1, len(df)+1)
    df["Month"] = df.index.month
    df["log_y"] = np.log(df["y"])

    steps_test = min(12, max(1, int(np.floor(len(df) * test_frac))))
    train = df.iloc[:-steps_test].copy()
    test  = df.iloc[-steps_test:].copy()

    model = smf.ols("log_y ~ t + C(Month)", data=train).fit()
    pred = np.exp(model.predict(test))
    hold = {
        "MAE": mae(test["y"].values, pred.values),
        "RMSE": rmse(test["y"].values, pred.values),
        "MAPE_%": mape(test["y"].values, pred.values),
        "R2": r2_score(test["y"].values, pred.values)
    }

    # reentrenar con todo y proyectar 12m
    model_all = smf.ols("log_y ~ t + C(Month)", data=df).fit()
    fut_idx = next_12_from_last(df.index.max())
    fut = pd.DataFrame(index=fut_idx)
    fut["t"] = np.arange(df["t"].iloc[-1] + 1, df["t"].iloc[-1] + 1 + len(fut))
    fut["Month"] = fut.index.month
    pr = model_all.get_prediction(fut).summary_frame(alpha=0.20)
    emae_mean = np.exp(pr["mean"].values)
    emae_lo   = np.exp(pr["obs_ci_lower"].values)
    emae_hi   = np.exp(pr["obs_ci_upper"].values)
    emae_fc   = pd.Series(emae_mean, index=fut_idx, name="EMAE_fc")
    return model_all, hold, emae_fc, emae_lo, emae_hi

# -----------------------------------------------------------------------------
# ECM: long-run (cointegración) + short-run (Δlog + ECT_L1 + C(Month))
# -----------------------------------------------------------------------------
def fit_ecm(y_dem: pd.Series, x_emae: pd.Series, test_frac: float = 0.10):
    # Alinear y preparar
    y = pd.Series(y_dem).astype(float).asfreq("MS")
    x = pd.Series(x_emae).astype(float).asfreq("MS")
    idx = y.index.intersection(x.index)
    y = y.loc[idx]; x = x.loc[idx]
    # Logs y diferencias
    log_y = np.log(y)
    log_x = np.log(x)
    dlog_y = log_y.diff()
    dlog_x = log_x.diff()

    # Cointegración (long-run) sin estacionalidad (const + log_x)
    df_lr = pd.DataFrame({"log_y": log_y, "log_x": log_x}).dropna()
    lr_model = smf.ols("log_y ~ log_x", data=df_lr).fit()
    ect = (df_lr["log_y"] - lr_model.predict(df_lr))  # residuo long-run
    ect = ect.reindex(log_y.index)
    ect_l1 = ect.shift(1)

    # ECM corto plazo
    df_ecm = pd.DataFrame({
        "dlog_y": dlog_y,
        "dlog_x": dlog_x,
        "ECT_L1": ect_l1,
        "Month": y.index.month
    }).dropna()

    steps_test = min(12, max(1, int(np.floor(len(df_ecm) * test_frac))))
    train = df_ecm.iloc[:-steps_test].copy()
    test_idx = df_ecm.index[-steps_test:]  # sólo el índice temporal del test

    ecm_model = smf.ols("dlog_y ~ dlog_x + ECT_L1 + C(Month)", data=train).fit()

    # Backtest dinámico en niveles para el tramo de test
    # partimos del nivel observado en t0-1 (último del train)
    start_prev = test_idx[0] - pd.offsets.MonthBegin(1)
    log_y_prev = log_y.loc[start_prev]  # nivel verdadero hasta el arranque
    pred_levels = []
    for t in test_idx:
        # drivers en t
        dx_t = (log_x.loc[t] - log_x.loc[t - pd.offsets.MonthBegin(1)])
        ect_prev = log_y_prev - lr_model.predict(pd.DataFrame({"log_x":[log_x.loc[t - pd.offsets.MonthBegin(1)]]})).iloc[0]
        row = pd.DataFrame({"dlog_x":[dx_t], "ECT_L1":[ect_prev], "Month":[t.month]})
        dly_hat = float(ecm_model.predict(row))
        log_y_prev = log_y_prev + dly_hat
        pred_levels.append(np.exp(log_y_prev))
    pred_levels = pd.Series(pred_levels, index=test_idx, name="Dem_pred")

    # Métricas en niveles contra y reales de test
    y_test = y.loc[test_idx]
    hold = {
        "MAE": mae(y_test.values, pred_levels.values),
        "RMSE": rmse(y_test.values, pred_levels.values),
        "MAPE_%": mape(y_test.values, pred_levels.values),
        "R2": r2_score(y_test.values, pred_levels.values)
    }
    return lr_model, ecm_model, hold

def forecast_ecm(lr_model, ecm_model, y_hist: pd.Series, x_hist: pd.Series, x_future: pd.Series):
    """
    Simulación recursiva mensual con ECM usando un índice mensual continuo
    para evitar KeyError en el primer mes futuro (t-1).
    """
    # Asegurar frecuencia mensual y tipos
    y_hist = pd.Series(y_hist, dtype="float64").asfreq("MS")
    x_hist = pd.Series(x_hist, dtype="float64").asfreq("MS")
    x_future = pd.Series(x_future, dtype="float64").asfreq("MS")

    # Último mes observado en demanda (origen de la simulación)
    last_hist = y_hist.index.max()

    # Índice continuo desde el último mes histórico hasta el último futuro
    full_idx = pd.date_range(last_hist, x_future.index[-1], freq="MS")

    # EMAE combinado sobre índice continuo (relleno hacia adelante)
    x_all = pd.concat([x_hist, x_future])
    x_all = x_all.reindex(full_idx).ffill()

    # Logs y diferencia mensual de log(EMAE)
    log_x_all = np.log(x_all.clip(lower=1e-12))
    dlog_x_all = log_x_all.diff()  # Δlog(EMAE)

    # Estado inicial: nivel de log(y) en el último histórico
    log_y_prev = np.log(float(y_hist.loc[last_hist]))

    # Parámetros long-run (log_y* = a + b log_x)
    a = float(lr_model.params["Intercept"])
    b = float(lr_model.params["log_x"])

    # Simulación mes a mes desde el primer mes futuro
    out = []
    for t in x_future.index:
        # Δlog(EMAE) del mes t; si es NaN (por primer paso), usar 0.0
        dx_t = float(dlog_x_all.get(t, 0.0))
        if not np.isfinite(dx_t):
            dx_t = 0.0

        # ECT_{t-1} = log(y)_{t-1} - (a + b log(x)_{t-1})
        log_x_prev = float(log_x_all.loc[t - pd.offsets.MonthBegin(1)])
        ect_prev = log_y_prev - (a + b * log_x_prev)

        row = pd.DataFrame({"dlog_x": [dx_t], "ECT_L1": [ect_prev], "Month": [t.month]})
        dly_hat = float(ecm_model.predict(row))

        # Actualizar el nivel de log(y) y guardar pronóstico en niveles
        log_y_prev = log_y_prev + dly_hat
        out.append((t, np.exp(log_y_prev)))

    return pd.Series([v for _, v in out], index=[i for i, _ in out], name="Dem_fc")

# -----------------------------------------------------------------------------
# Escenarios de EMAE (anual YoY base ± pp) → camino mensual (con estacionalidad)
# -----------------------------------------------------------------------------
def annual_targets_base(x_hist: pd.Series, years, variation_pp: int = 2, base_method: str = "rolling12"):
    """
    Calcula la tasa base de crecimiento anual del EMAE y arma los targets (base/min/max)
    para los años de 'years'. Devuelve (targets, meta).

    base_method:
      - "rolling12": promedio de los últimos 12 YoY mensuales (momentum reciente)
      - "calendar":  YoY anual calendario (promedio del último año completo vs el anterior)
    """
    x = pd.Series(x_hist).astype(float).asfreq("MS")

    # Ancla: promedio del último año calendario completo
    ay = last_full_year(x)
    anchor_level = float(x.loc[str(ay)].mean())

    # Tasa base según método
    if base_method == "calendar":
        df = x.to_frame("x"); df["year"] = df.index.year
        a = df.groupby("year")["x"].mean().dropna()
        if len(a) >= 2:
            g_base = (a.iloc[-1] / a.iloc[-2]) - 1.0
        else:
            g_base = 0.015  # fallback
    else:  # "rolling12"
        yoy_m = x.pct_change(12).dropna()
        g_base = float(yoy_m.tail(12).mean()) if len(yoy_m) > 0 else 0.015

    delta = variation_pp / 100.0
    targets = {
        "base": {y: g_base for y in years},
        "min":  {y: g_base - delta for y in years},
        "max":  {y: g_base + delta for y in years},
    }
    meta = {
        "anchor_year": int(ay),
        "anchor_level": float(anchor_level),
        "base_method": base_method,
        "g_base": float(g_base),
        "variation_pp": int(variation_pp),
    }
    return targets, meta

def monthly_path_from_annual(x_hist: pd.Series, targets: Dict[int, float]) -> pd.Series:
    S = seasonal_factors_from_history(x_hist)
    ay = last_full_year(x_hist)
    anchor = float(x_hist.loc[str(ay)].mean())  # promedio del último año completo

    rows = []
    lvl_year = anchor
    for yy in sorted(targets.keys()):
        lvl_year = lvl_year * (1.0 + targets[yy])
        for m in range(1, 13):
            rows.append({"indice_tiempo": pd.Timestamp(f"{yy}-{m:02d}-01"), "EMAE": lvl_year * S[m-1]})
    df = pd.DataFrame(rows).set_index("indice_tiempo").asfreq("MS")
    return df["EMAE"]

def demand_from_paths_ecm(lr_model, ecm_model, y_hist: pd.Series, x_hist: pd.Series, paths: Dict[str, pd.Series]):
    monthly_out = []
    annual_out = {}
    for scen, x_path in paths.items():
        dem_fc = forecast_ecm(lr_model, ecm_model, y_hist, x_hist, x_path)
        dfm = pd.DataFrame({"Dem_pred": dem_fc}, index=dem_fc.index)
        dfm["escenario"] = scen
        dfm["year"] = dfm.index.year
        monthly_out.append(dfm)
        annual_out[scen] = dfm.groupby("year")["Dem_pred"].sum()
    monthly = pd.concat(monthly_out).reset_index().rename(columns={"index":"indice_tiempo"})
    annual = pd.DataFrame(annual_out); annual.index.name = "Year"
    return monthly, annual

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Demanda I+C con ECM — corto (12m) y largo plazo (escenarios)")

use_repo = st.checkbox("Usar archivos del repo (/data)", value=True)
csv_bytes  = repo_file_bytes(CSV_NAME)  if use_repo else None
emae_bytes = repo_file_bytes(EMAE_NAME) if use_repo else None

if not use_repo:
    c1, c2 = st.columns(2)
    with c1:
        up1 = st.file_uploader("CSV con Demanda I+C (variables.csv)", type=["csv"])
        if up1 is not None: csv_bytes = up1.getvalue()
    with c2:
        up2 = st.file_uploader("EMAE (Excel .xlsx)", type=["xlsx"])
        if up2 is not None: emae_bytes = up2.getvalue()

if (csv_bytes is None) or (emae_bytes is None):
    st.info("Cargá **variables.csv** y **emae.xlsx** (o marcá la opción de usar /data).")
    st.stop()

# Series
y_dem = load_dem_ic_from_csv(csv_bytes)
x_emae = load_emae_from_excel(emae_bytes)

# 1) EMAE OLS — 12 meses
st.header("EMAE — OLS(log) t + C(Month) (pronóstico 12 meses)")
emae_model, emae_hold, emae_fc, emae_lo, emae_hi = fit_emae_ols_log(x_emae, test_frac=0.10)
st.write("**Métricas holdout 10% (nivel)**:", ", ".join([f"{k}: {v:.3f}" for k,v in emae_hold.items()]))

figE, axE = plt.subplots(figsize=(7.0, 3.0))
axE.plot(x_emae.index, x_emae.values, alpha=0.45, label="Histórico")
axE.plot(emae_fc.index, emae_fc.values, linewidth=2, label="Pronóstico 12m")
axE.fill_between(emae_fc.index, emae_lo, emae_hi, alpha=0.15, label="PI80")
axE.set_xlabel("Mes"); axE.set_ylabel("EMAE")
axE.legend()
st.pyplot(figE)

st.divider()

# 2) ECM Demanda I+C
st.header("Demanda I+C — ECM (Δlog + ECTₜ₋₁ + estacionalidad)")

lr_model, ecm_model, dem_hold = fit_ecm(y_dem, x_emae, test_frac=0.10)
st.write("**Métricas holdout 10% (nivel)**:",
         ", ".join([f"{k}: {v:.3f}" for k,v in dem_hold.items()]))

# Mostrar elasticidad de largo plazo y coeficientes ECM
st.write(f"**Elasticidad de largo plazo β (log(EMAE))**: {lr_model.params['log_x']:.3f} "
         f"(p={lr_model.pvalues['log_x']:.4f})")
st.write("**ECM (corto plazo)** — principales coeficientes:")
show = ecm_model.params[["dlog_x","ECT_L1"]].to_frame("coef")
show["p_value"] = ecm_model.pvalues[show.index]
st.dataframe(show.style.format({"coef":"{:.4f}", "p_value":"{:.4f}"}), use_container_width=True)

# --- Toggle y preview de ancla/tasa base para escenarios ---
st.subheader("Cómo se construye el escenario base del EMAE")
opt = st.radio("Método para la tasa base YoY", ["12M rodantes", "Año calendario"],
               index=0, horizontal=True,
               help="Define cómo se estima la tasa base de crecimiento anual del EMAE.")
base_method = "rolling12" if opt == "12M rodantes" else "calendar"
st.session_state["base_method"] = base_method  # lo usamos en la sección de escenarios

def _preview_anchor_and_base(x_hist: pd.Series, base_method: str):
    x = pd.Series(x_hist).astype(float).asfreq("MS")
    ay = last_full_year(x)
    anchor_level = float(x.loc[str(ay)].mean())
    if base_method == "calendar":
        df = x.to_frame("x"); df["year"] = df.index.year
        a = df.groupby("year")["x"].mean().dropna()
        g_base = (a.iloc[-1] / a.iloc[-2]) - 1.0 if len(a) >= 2 else 0.015
        label = "Año calendario"
    else:
        yoy_m = x.pct_change(12).dropna()
        g_base = float(yoy_m.tail(12).mean()) if len(yoy_m) > 0 else 0.015
        label = "12M rodantes"
    return ay, anchor_level, g_base, label

ay_prev, anchor_prev, g_prev, label_prev = _preview_anchor_and_base(x_emae, base_method)
st.caption(
    f"**Ancla EMAE**: promedio del último año completo **{ay_prev}** · "
    f"**Tasa base** ({label_prev}): **{g_prev*100:.2f}%**. "
    f"Los ±pp de los escenarios se aplican sobre esta tasa y luego se distribuyen por mes con la estacionalidad histórica."
)

# Pronóstico Dem 12m con ECM usando EMAE_fc
dem_fc_12 = forecast_ecm(lr_model, ecm_model, y_dem, x_emae, emae_fc)

figS, axS = plt.subplots(figsize=(7.6, 3.2))
hist6_idx = pd.date_range(dem_fc_12.index[0] - pd.DateOffset(months=6), periods=6, freq="MS")
hist6 = y_dem.reindex(hist6_idx)
axS.plot(hist6.index, hist6.values, linestyle=":", label="Histórico (6m)")
axS.plot(dem_fc_12.index, dem_fc_12.values, label="Pronóstico Dem I+C (ECM, 12m)")
axS.axvline(dem_fc_12.index.min(), color="gray", linestyle=":", linewidth=1)
axS.set_xlabel("Mes"); axS.set_ylabel("Demanda I+C")
plt.setp(axS.get_xticklabels(), rotation=45, ha="right")
axS.legend()
st.pyplot(figS)

st.divider()

# 3) Largo Plazo — Escenarios 2026–2031 (ECM)
st.header("Largo Plazo — EMAE escenarios (Base ± pp) → Demanda I+C 2026–2031 (ECM)")
variation_pp = st.radio("Banda EMAE para escenarios (± pp)", [1,2,3,4,5,6], index=1, horizontal=True)
years = list(range(2026, 2032))

base_method = st.session_state.get("base_method", "rolling12")
targets, meta = annual_targets_base(x_emae, years, variation_pp=int(variation_pp), base_method=base_method)
paths = {k: monthly_path_from_annual(x_emae, v) for k, v in targets.items()}
monthly_dem, annual_dem = demand_from_paths_ecm(lr_model, ecm_model, y_dem, x_emae, paths)

# Gráfico EMAE YoY anual (histórico + escenarios)
st.subheader("%Δ EMAE YoY anual — histórico y escenarios")
df_e = x_emae.to_frame("EMAE"); df_e["year"] = df_e.index.year
hist_annual = df_e.groupby("year")["EMAE"].mean()
yoy_hist = (hist_annual.pct_change()*100).dropna()
annual_proj = pd.DataFrame({
    "base":[targets["base"][y]*100 for y in years],
    "min":[targets["min"][y]*100 for y in years],
    "max":[targets["max"][y]*100 for y in years]}, index=years)

figA, axA = plt.subplots(figsize=(7.6,3.0))
axA.plot(yoy_hist.index, yoy_hist.values, marker="o", label="Histórico")
axA.plot(annual_proj.index, annual_proj["base"], marker="o", label="Base")
axA.plot(annual_proj.index, annual_proj["min"], marker="o", label=f"Min (−{variation_pp} pp)")
axA.plot(annual_proj.index, annual_proj["max"], marker="o", label=f"Max (+{variation_pp} pp)")
axA.axhline(0, color="gray", linewidth=0.8)
axA.set_ylabel("%Δ anual EMAE"); axA.set_xlabel("Año")
axA.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False)
plt.tight_layout()
st.pyplot(figA)

# Meta info (ancla y base)
st.caption(
    f"Escenarios construidos con **ancla {meta['anchor_year']}** (promedio anual), "
    f"**tasa base {meta['g_base']*100:.2f}%** ({'12M rodantes' if meta['base_method']=='rolling12' else 'Año calendario'}), "
    f"±{meta['variation_pp']} pp."
)

# Demanda anual — histórico (sin año incompleto) + escenarios
st.subheader("Demanda I+C anual — histórico (sin año incompleto) + escenarios (ECM)")

df_d = y_dem.to_frame("Dem"); df_d["year"] = df_d.index.year
last_full = last_full_year(y_dem)
dem_hist_annual = df_d[df_d["year"] <= last_full].groupby("year")["Dem"].sum()

x_hist = dem_hist_annual.index.astype(int).astype(str).tolist()
y_hist = dem_hist_annual.values.tolist()
x_scen = [str(y) for y in annual_dem.index.tolist()]

figB, axB = plt.subplots(figsize=(9.6, 4.2))
axB.bar(x_hist, y_hist, label="Histórico", color="#4C78A8")
# barras escenarios (mismo x, distinta serie)
axB.bar(x_scen, annual_dem["base"].values, alpha=0.75, label="Base", color="#E45756")
axB.bar(x_scen, annual_dem["min"].values,  alpha=0.60, label="Min",  color="#F58518")
axB.bar(x_scen, annual_dem["max"].values,  alpha=0.60, label="Max",  color="#54A24B")

axB.set_xlabel("Año"); axB.set_ylabel("Demanda I+C (suma anual)")
axB.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False)

# Etiquetas YoY — histórico
hist_yoy = (dem_hist_annual.pct_change() * 100).round(1)
for i, (yr, val) in enumerate(dem_hist_annual.items()):
    if yr in hist_yoy.index and not np.isnan(hist_yoy.loc[yr]):
        axB.text(i, val, f"{hist_yoy.loc[yr]:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")

# Etiquetas YoY — escenarios (desfase horizontal para que no se solapen)
offset_base = 0.00
offset_min  = -0.15
offset_max  = +0.15
ser_base = pd.concat([dem_hist_annual, annual_dem["base"]])
ser_min  = pd.concat([dem_hist_annual, annual_dem["min"]])
ser_max  = pd.concat([dem_hist_annual, annual_dem["max"]])

for j, yr in enumerate(annual_dem.index):
    # posición x base de la barra de escenarios = len(hist) + j
    x0 = len(x_hist) + j
    for ser, col, dx in [(ser_base, "#E45756", offset_base),
                         (ser_min,  "#F58518", offset_min),
                         (ser_max,  "#54A24B", offset_max)]:
        yoy = (ser.pct_change() * 100).round(1)
        val = ser.loc[yr]
        if yr in yoy.index and not np.isnan(yoy.loc[yr]):
            axB.text(x0 + dx, val, f"{yoy.loc[yr]:.1f}%", ha="center", va="bottom", fontsize=8, color=col)

plt.tight_layout()
st.pyplot(figB)

# Descargas
c1, c2 = st.columns(2)
with c1:
    st.download_button("CSV — Demanda anual histórica",
        data=dem_hist_annual.to_frame("Dem_hist_anual").to_csv().encode("utf-8"),
        file_name="demanda_hist_anual.csv", mime="text/csv")
with c2:
    out_annual = annual_dem.copy(); out_annual.index.name="Year"
    st.download_button("CSV — Demanda anual escenarios (ECM) 2026–2031",
        data=out_annual.to_csv().encode("utf-8"),
        file_name="demanda_ecm_escenarios_2026_2031.csv", mime="text/csv")
