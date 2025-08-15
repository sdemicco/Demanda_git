# -*- coding: utf-8 -*-
# demanda_unificada.py
# -------------------------------------------------------
# App unificada:
#   - Pestaña 1: Demanda Industrial + Comercial (I+C)
#   - Pestaña 2: Demanda Residencial (OLS log + Temp^9 + SARIMA Temp)
#   - Pestaña 3: Crecimiento 2026→2031 (I+C / Res / Total)
# -------------------------------------------------------

import io
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="Demanda — I+C & Residencial (Unificada)", layout="wide")

# =============================================================================
# Rutas de datos del repo
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_NAME  = "variables.csv"  # mismo CSV para I+C y Residencial
EMAE_NAME = "emae.xlsx"      # EMAE para I+C

def repo_file_bytes(name: str) -> Optional[bytes]:
    p = DATA_DIR / name
    if p.exists():
        return p.read_bytes()
    return None

# =============================================================================
# Utilidades comunes
# =============================================================================
def to_monthly_index(df: pd.DataFrame, date_col: str = "indice_tiempo") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        cands = [c for c in df.columns if ("fecha" in c.lower()) or ("time" in c.lower()) or ("indice" in c.lower())]
        date_col = cands[0] if cands else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col).asfreq("MS")
    return df

def monthly_fill(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    df = pd.DataFrame({"v": s})
    df["m"] = df.index.month
    med = df.groupby("m")["v"].transform("median")
    s = s.fillna(med).ffill().bfill()
    return s

def choose_transform(y: pd.Series):
    y = pd.to_numeric(y, errors="coerce").astype(float)
    if y.min() >= 0 and y.max() > 0:
        return "log1p", np.log1p(y)
    return "none", y

def invert_transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log1p":
        return np.expm1(values)
    return values

def fit_fast(y: pd.Series, exog: Optional[pd.DataFrame] = None, seasonal_periods: int = 12, maxiter: int = 180):
    tried = [((1,1,1),(0,1,1,seasonal_periods)),
             ((0,1,1),(1,1,1,seasonal_periods)),
             ((0,1,1),(0,1,1,seasonal_periods))]
    for order, sorder in tried:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y, exog=exog, order=order, seasonal_order=sorder,
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = mod.fit(disp=False, maxiter=maxiter)
            return res, order, sorder
        except Exception:
            continue
    mod = sm.tsa.statespace.SARIMAX(
        y, exog=exog, order=(0,1,1), seasonal_order=(0,1,1,seasonal_periods),
        enforce_stationarity=False, enforce_invertibility=False
    )
    res = mod.fit(disp=False, maxiter=maxiter)
    return res, (0,1,1), (0,1,1,seasonal_periods)

def next_12_from_last(last_ts: pd.Timestamp) -> pd.DatetimeIndex:
    start = (pd.Timestamp(last_ts) + pd.offsets.MonthBegin(1))
    return pd.date_range(start, periods=12, freq="MS")

def mae(a, b):  return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def mape(a, b):
    a = np.asarray(a); b = np.asarray(b); eps=1e-8
    return float(np.mean(np.abs((a-b)/np.maximum(np.abs(a), eps))))*100
def r2_score(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ss_res = np.sum((a-b)**2); ss_tot = np.sum((a-np.mean(a))**2)
    return float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")

def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, seasonal_periods: int = 12) -> Dict[str, float]:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    mae_v  = mae(y_t, y_p)
    rmse_v = rmse(y_t, y_p)
    mape_v = mape(y_t, y_p)
    r2_v   = r2_score(y_t, y_p)
    if len(y_t) > seasonal_periods:
        naive = y_t[:-seasonal_periods]
        naive_next = y_t[seasonal_periods:]
        mae_naive = mae(naive_next, naive)
        mase_v = mae_v / (mae_naive if mae_naive != 0 else 1e-8)
    else:
        mase_v = np.nan
    return {"MAE": mae_v, "RMSE": rmse_v, "MAPE_%": mape_v, "R2": r2_v, "MASE": mase_v}

def growth_2026_2031(s):
    s = pd.Series(s).dropna()
    return float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0)

# =============================================================================
# I+C (EMAE como exógena)
# =============================================================================
def load_dem_ic(csv_bytes: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = to_monthly_index(df, "indice_tiempo")
    cand = [c for c in df.columns if ("comercial" in c.lower() or "industrial" in c.lower()) and "demanda" in c.lower()]
    if cand:
        col = cand[0]
    else:
        col = "Demanda_Comercial_Industrial" if "Demanda_Comercial_Industrial" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    y = monthly_fill(pd.to_numeric(df[col], errors="coerce").asfreq("MS"))
    return y

def load_emae(xlsx_bytes: bytes) -> pd.Series:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    date_col = None
    for c in df.columns:
        if (np.issubdtype(df[c].dtype, np.datetime64)) or ("fecha" in c.lower()) or ("indice" in c.lower()) or ("time" in c.lower()):
            date_col = c; break
    if date_col is None:
        date_col = df.columns[0]
    df = to_monthly_index(df, date_col)
    val_col = None
    for c in df.columns:
        if "emae" in c.lower():
            val_col = c; break
    if val_col is None:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        val_col = nums[0] if nums else df.columns[0]
    x = monthly_fill(pd.to_numeric(df[val_col], errors="coerce").asfreq("MS"))
    return x

def rolling_backtest_last_n_ic(y: pd.Series, x: pd.Series, last_n: int = 12, seasonal_periods: int = 12) -> Dict[str, float]:
    idx_all = y.dropna().index.intersection(x.dropna().index)
    y_all = y.loc[idx_all]; x_all = x.loc[idx_all]
    preds, trues = [], []
    min_hist = seasonal_periods * 2
    if y_all.shape[0] < (last_n + min_hist):
        last_n = max(1, min(last_n, y_all.shape[0] - seasonal_periods - 1))
    for t in range(last_n, 0, -1):
        split_end = y_all.index[-t-1]
        target_ts = y_all.index[-t]
        y_tr = y_all.loc[:split_end]
        x_tr = x_all.loc[:split_end].to_frame(name="EMAE")
        kind_y, y_log = choose_transform(y_tr)
        res, _, _ = fit_fast(y_log, exog=x_tr, seasonal_periods=seasonal_periods)
        Xf = pd.DataFrame({"EMAE": [x_all.loc[target_ts]]}, index=pd.DatetimeIndex([target_ts]))
        pred = res.get_forecast(steps=1, exog=Xf)
        y_hat = invert_transform(pred.predicted_mean.values, kind_y)[0]
        preds.append(y_hat); trues.append(y_all.loc[target_ts])
    return metrics_from_predictions(np.array(trues), np.array(preds), seasonal_periods=seasonal_periods)

def forecast_short_term_ic(y_hist: pd.Series, x_hist: pd.Series, alpha: float = 0.20, history_months: int = 6):
    idx = y_hist.dropna().index.intersection(x_hist.dropna().index)
    yh = y_hist.loc[idx]; xh = x_hist.loc[idx].to_frame(name="EMAE")
    kind_y, y_log = choose_transform(yh)
    fit_all, _, _ = fit_fast(y_log, exog=xh, seasonal_periods=12)
    kind_e, x_log = choose_transform(xh["EMAE"])
    fit_e, _, _ = fit_fast(pd.Series(x_log, index=xh.index), exog=None, seasonal_periods=12)
    horizon = next_12_from_last(yh.index.max())
    pred_e = fit_e.get_forecast(steps=len(horizon))
    emae_base = invert_transform(pred_e.predicted_mean.values, kind_e)
    def fc_with(exog_vec):
        Xf = pd.DataFrame({"EMAE": exog_vec}, index=horizon)
        pred = fit_all.get_forecast(steps=len(horizon), exog=Xf)
        mean = invert_transform(pred.predicted_mean.values, kind_y)
        ci = pred.conf_int(alpha=alpha)
        lo = invert_transform(ci.iloc[:,0].values, kind_y)
        hi = invert_transform(ci.iloc[:,1].values, kind_y)
        return mean, lo, hi
    y_base_mean, y_base_lo, y_base_hi = fc_with(emae_base)
    hist_idx = pd.date_range(horizon[0] - pd.DateOffset(months=history_months),
                             periods=history_months, freq="MS")
    hist_vals = yh.reindex(hist_idx)
    hist_df = pd.DataFrame({"indice_tiempo": hist_idx, "Dem_IC_observado": hist_vals.values}).dropna()
    fc_df = pd.DataFrame({
        "indice_tiempo": horizon,
        "Dem_IC_base": y_base_mean,
        "PI80_lo_base": y_base_lo,
        "PI80_hi_base": y_base_hi,
        "EMAE_base": emae_base
    })
    oos_metrics = rolling_backtest_last_n_ic(y_hist, x_hist, last_n=12, seasonal_periods=12)
    return hist_df, fc_df, oos_metrics

def long_horizon_annual_sum_ic(y_hist: pd.Series, x_hist: pd.Series,
                               years=(2026, 2027, 2028, 2029, 2030, 2031),
                               alpha: float = 0.20, variation_pp: int = 2):
    idx = y_hist.dropna().index.intersection(x_hist.dropna().index)
    yh = y_hist.loc[idx]; xh = x_hist.loc[idx].to_frame(name="EMAE")
    kind_y, y_log = choose_transform(yh)
    fit_all, _, _ = fit_fast(y_log, exog=xh, seasonal_periods=12)
    emae_df = xh["EMAE"].to_frame()
    emae_df["year"] = emae_df.index.year
    emae_df["month"]= emae_df.index.month
    annual_avg = emae_df.groupby("year")["EMAE"].transform("mean")
    seasonal_ratio = emae_df["EMAE"] / annual_avg
    S = (seasonal_ratio.groupby(emae_df["month"]).mean() / seasonal_ratio.groupby(emae_df["month"]).mean().mean()).reindex(range(1,13)).values
    yoy = xh["EMAE"].pct_change(12)
    g_base = float(yoy.dropna().tail(12).mean()) if yoy.dropna().shape[0] >= 6 else 0.015
    delta = variation_pp / 100.0
    g_min  = max(g_base - delta, -0.03)
    g_max  = min(g_base + delta,  0.05)
    last_year = xh.index.max().year
    anchor_year = last_year if len(xh.loc[str(last_year)]) == 12 else last_year - 1
    anchor_level = float(xh.loc[str(anchor_year), "EMAE"].mean())
    def targets(g): return {y: anchor_level * ((1.0 + g) ** (y - anchor_year)) for y in years}
    t_base, t_min, t_max = targets(g_base), targets(g_min), targets(g_max)
    def monthly_from_targets(t_dict):
        rows = []
        for yy in years:
            for m in range(1,13):
                rows.append({"indice_tiempo": pd.Timestamp(f"{yy}-{m:02d}-01"), "EMAE": t_dict[yy] * S[m-1]})
        return pd.DataFrame(rows).set_index("indice_tiempo").asfreq("MS")
    paths = {"base": monthly_from_targets(t_base),
             "min":  monthly_from_targets(t_min),
             "max":  monthly_from_targets(t_max)}
    def fc_monthly(exog_df):
        pred = fit_all.get_forecast(steps=exog_df.shape[0], exog=exog_df)
        mean = invert_transform(pred.predicted_mean.values, kind_y)
        out = pd.DataFrame({"Dem_IC_pred": mean}, index=exog_df.index)
        out["year"] = out.index.year
        return out
    monthly = {k: fc_monthly(v) for k, v in paths.items()}
    annual_sum = {k: v.groupby("year")["Dem_IC_pred"].sum().reindex(years) for k, v in monthly.items()}
    annual_df = pd.DataFrame({k: v.values for k, v in annual_sum.items()}, index=list(years))
    annual_df.index.name = "Year"
    monthly_out = pd.concat([
        df.assign(escenario=k).reset_index().rename(columns={"index":"indice_tiempo"})
        for k, df in monthly.items()
    ], ignore_index=True)
    meta = {"growth_rates": {"base": g_base, "min": g_min, "max": g_max},
            "variation_pp": variation_pp,
            "anchor_year": int(anchor_year),
            "anchor_level_EMAE": float(anchor_level),
            "note": "g_base = promedio del YoY mensual del EMAE (últimos 12m); min/máx = g_base ± variation_pp (pp), pisos/techos: −3%/+5%."}
    return annual_df, monthly_out, meta

# =============================================================================
# Residencial (OLS log con Temp^9) + SARIMA temperatura
# =============================================================================
def load_res_from_csv(csv_bytes: bytes) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = to_monthly_index(df, "indice_tiempo")
    cand_t = [c for c in df.columns if ("resid" in c.lower()) and ("dem" in c.lower())]
    dem_col = cand_t[0] if cand_t else df.select_dtypes(include=[np.number]).columns[0]
    cand_x = [c for c in df.columns if ("temp" in c.lower()) or ("temperatura" in c.lower())]
    if cand_x:
        temp_col = cand_x[0]
    else:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        temp_col = nums[1] if len(nums) > 1 else nums[0]
    df[dem_col]  = monthly_fill(df[dem_col].asfreq("MS"))
    df[temp_col] = monthly_fill(df[temp_col].asfreq("MS"))
    return df, dem_col, temp_col

def prepare_design_res(df: pd.DataFrame, dem_col: str, temp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["DEMANDA"] = pd.to_numeric(out[dem_col], errors="coerce").astype(float)
    out["temperatura"] = pd.to_numeric(out[temp_col], errors="coerce").astype(float)
    out["temperatura_9"] = out["temperatura"]**9
    out["timeIndex"] = np.arange(1, len(out)+1)
    out["Month"] = out.index.month
    dums = pd.get_dummies(out["Month"], prefix="Month")
    out = pd.concat([out, dums], axis=1)
    return out

def fit_ols_log_res(df: pd.DataFrame):
    df = df.copy()
    cutoff = int(np.floor(len(df)*0.9))
    train = df.iloc[:cutoff].copy()
    test  = df.iloc[cutoff:].copy()
    train["log_DEMANDA"] = np.log(train["DEMANDA"])
    test["log_DEMANDA"]  = np.log(test["DEMANDA"])
    month_terms = " + ".join([f"Month_{m}" for m in range(2,13) if f"Month_{m}" in train.columns])
    formula = f'log_DEMANDA ~ timeIndex + {month_terms} + temperatura_9'
    model = smf.ols(formula, data=train).fit()
    train["yhat_log"] = model.predict(train)
    test["yhat_log"]  = model.predict(test)
    train["yhat"] = np.exp(train["yhat_log"])
    test["yhat"]  = np.exp(test["yhat_log"])
    return model, train, test

def fit_sarima_temp(y: pd.Series, seasonal_periods: int = 12, maxiter: int = 200):
    tried = [((1,1,1),(0,1,1,seasonal_periods)),
             ((0,1,1),(1,1,1,seasonal_periods)),
             ((0,1,1),(0,1,1,seasonal_periods))]
    best, best_aic = None, np.inf
    for order, sorder in tried:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y, order=order, seasonal_order=sorder,
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = mod.fit(disp=False, maxiter=maxiter)
            if res.aic < best_aic:
                best, best_aic = res, res.aic
        except Exception:
            pass
    if best is None:
        mod = sm.tsa.statespace.SARIMAX(
            y, order=(0,1,1), seasonal_order=(0,1,1,seasonal_periods),
            enforce_stationarity=False, enforce_invertibility=False
        )
        best = mod.fit(disp=False, maxiter=maxiter)
    return best

def rolling_12_metrics_res(df: pd.DataFrame) -> Dict[str, float]:
    preds, trues = [], []
    last_n = 12
    if len(df) < 36:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "R2": np.nan}
    for t in range(last_n, 0, -1):
        split_end = df.index[-t-1]
        dtr = df.loc[:split_end].copy()
        dts = df.loc[[df.index[-t]]].copy()
        m, tr, _ = fit_ols_log_res(dtr)
        y_hat = float(np.exp(m.predict(dts)[0]))
        preds.append(y_hat); trues.append(float(dts["DEMANDA"].iloc[0]))
    return {"MAE": mae(trues, preds), "RMSE": rmse(trues, preds), "MAPE_%": mape(trues, preds), "R2": r2_score(trues, preds)}

def forecast_short_term_res(df: pd.DataFrame, model, alpha=0.20, history_months: int = 6):
    fit_t = fit_sarima_temp(df["temperatura"])
    horizon = next_12_from_last(df.index.max())
    pred_t = fit_t.get_forecast(steps=len(horizon))
    temp_fc = pred_t.predicted_mean.values
    fut = pd.DataFrame(index=horizon)
    fut["timeIndex"] = np.arange(df["timeIndex"].iloc[-1]+1, df["timeIndex"].iloc[-1]+1+len(horizon))
    fut["Month"] = fut.index.month
    dums_f = pd.get_dummies(fut["Month"], prefix="Month")
    for m in range(2,13):
        col = f"Month_{m}"
        fut[col] = dums_f[col] if col in dums_f.columns else 0
    fut["temperatura"]   = temp_fc
    fut["temperatura_9"] = fut["temperatura"]**9
    pr = model.get_prediction(fut).summary_frame(alpha=alpha)
    mean = np.exp(pr["mean"].values)
    lo   = np.exp(pr["obs_ci_lower"].values)
    hi   = np.exp(pr["obs_ci_upper"].values)
    hist_idx = pd.date_range(horizon[0] - pd.DateOffset(months=history_months),
                             periods=history_months, freq="MS")
    hist = df.loc[df.index.intersection(hist_idx), ["DEMANDA"]].reset_index().rename(
        columns={"index":"indice_tiempo","DEMANDA":"Dem_Res_observada"})
    fc = pd.DataFrame({
        "indice_tiempo": horizon,
        "Dem_Res_base": mean,
        "PI80_lo_base": lo,
        "PI80_hi_base": hi,
        "Temp_base": temp_fc
    })
    return hist, fc

def forecast_long_term_hot_cold(df: pd.DataFrame, model,
                                years=(2026,2027,2028,2029,2030,2031)):
    th = df["temperatura"].copy()
    tmp = pd.DataFrame({"Temp": th, "m": th.index.month})
    med = tmp.groupby("m")["Temp"].median()
    p10 = tmp.groupby("m")["Temp"].quantile(0.10)
    p90 = tmp.groupby("m")["Temp"].quantile(0.90)
    last_year = th.index.max().year
    anchor_year = last_year if len(th.loc[str(last_year)]) == 12 else last_year - 1
    anchor_level = float(th.loc[str(anchor_year)].mean())
    S_med = (med / med.mean()).reindex(range(1,13)).values
    d90 = (p90 - med).reindex(range(1,13)).values
    d10 = (p10 - med).reindex(range(1,13)).values

    def build_temp_path(kind: str) -> pd.Series:
        rows = []
        for yy in years:
            for m in range(1,13):
                base_val = anchor_level * S_med[m-1]
                if kind == "base":
                    val = base_val
                elif kind == "hot":
                    val = base_val + d90[m-1]
                else:
                    val = base_val + d10[m-1]
                rows.append({"indice_tiempo": pd.Timestamp(f"{yy}-{m:02d}-01"), "Temp": val})
        dfp = pd.DataFrame(rows).set_index("indice_tiempo").asfreq("MS")
        return dfp["Temp"]

    temp_paths = {"base": build_temp_path("base"),
                  "hot":  build_temp_path("hot"),
                  "cold": build_temp_path("cold")}

    def predict_from_temp_path(temp_series: pd.Series) -> pd.DataFrame:
        fut = pd.DataFrame(index=temp_series.index)
        fut["timeIndex"] = np.arange(df["timeIndex"].iloc[-1]+1, df["timeIndex"].iloc[-1]+1+len(fut))
        fut["Month"] = fut.index.month
        dums_f = pd.get_dummies(fut["Month"], prefix="Month")
        for m in range(2,13):
            col = f"Month_{m}"
            fut[col] = dums_f[col] if col in dums_f.columns else 0
        fut["temperatura"]   = temp_series.values
        fut["temperatura_9"] = fut["temperatura"]**9
        pr = model.get_prediction(fut).summary_frame(alpha=0.20)
        mean = np.exp(pr["mean"].values)
        out = pd.DataFrame({"Dem_Res_pred": mean}, index=fut.index)
        out["year"] = out.index.year
        return out

    monthly = {k: predict_from_temp_path(v) for k, v in temp_paths.items()}
    annual = {k: v.groupby("year")["Dem_Res_pred"].sum().reindex(years) for k, v in monthly.items()}

    annual_df = pd.DataFrame({
        "base":     annual["base"].values,
        "caliente": annual["hot"].values,
        "frio":     annual["cold"].values,
    }, index=list(years))
    annual_df.index.name = "Year"

    monthly_out = monthly["base"].reset_index().rename(columns={"index":"indice_tiempo"})
    meta = {"anchor_year": int(anchor_year),
            "anchor_level_Temp": float(anchor_level),
            "note": "Base = mediana mensual histórica anclada al último año completo; Caliente/Frío = p90/p10 mensuales (deltas sobre mediana)."}
    return annual_df, monthly_out, meta

# =============================================================================
# UI
# =============================================================================
st.title("Demanda — I+C & Residencial (Unificada)")

tab_ic, tab_res, tab_growth = st.tabs(["I+C", "Residencial", "Crecimiento 2026→2031"])

# -------------------- Pestaña I+C --------------------
with tab_ic:
    st.subheader("Demanda Industrial y Comercial — Corto y Largo Plazo")

    use_repo_ic = st.checkbox("Usar archivos del repo (/data)", value=True, key="use_repo_ic")
    variation_pp_ic = st.radio("Banda EMAE (pp)", [1,2,3], index=1, key="var_ic_body")

    csv_bytes  = repo_file_bytes(CSV_NAME)   if use_repo_ic else None
    emae_bytes = repo_file_bytes(EMAE_NAME)  if use_repo_ic else None

    if not use_repo_ic:
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            csv_ic = st.file_uploader("CSV histórico (Demanda I+C)", type=["csv"], key="csv_ic_body")
            if csv_ic is not None:
                csv_bytes = csv_ic.getvalue()
        with col_u2:
            xlsx_emae = st.file_uploader("EMAE (Excel)", type=["xlsx"], key="xlsx_emae_body")
            if xlsx_emae is not None:
                emae_bytes = xlsx_emae.getvalue()

    if (csv_bytes is None) or (emae_bytes is None):
        st.info("No encuentro **data/variables.csv** y/o **data/emae.xlsx**. Subí ambos archivos o colocalos en /data.")
    else:
        y_hist = load_dem_ic(csv_bytes)
        x_hist = load_emae(emae_bytes)

        # Corto plazo
        st.markdown("#### Corto plazo — histórico (6m) + próximos 12 meses")
        hist_df, fc_df, metrics_roll = forecast_short_term_ic(y_hist, x_hist, alpha=0.20, history_months=6)
        short_plot = fc_df[["indice_tiempo","Dem_IC_base","PI80_lo_base","PI80_hi_base"]].rename(
            columns={"PI80_lo_base":"Dem_IC_min","PI80_hi_base":"Dem_IC_max"})
        fig1, ax1 = plt.subplots()
        ax1.plot(hist_df["indice_tiempo"], hist_df["Dem_IC_observado"], linestyle=":", label="Histórico (6m)")
        ax1.plot(short_plot["indice_tiempo"], short_plot["Dem_IC_base"], label="Base")
        ax1.plot(short_plot["indice_tiempo"], short_plot["Dem_IC_min"], label="Min (PI80)")
        ax1.plot(short_plot["indice_tiempo"], short_plot["Dem_IC_max"], label="Max (PI80)")
        ax1.axvline(short_plot["indice_tiempo"].min(), color="gray", linestyle=":", linewidth=1)
        ax1.set_xlabel("Mes"); ax1.set_ylabel("Demanda I+C"); ax1.legend()
        st.pyplot(fig1)

        st.markdown("**Métricas (rolling 12)**")
        st.dataframe(pd.DataFrame(metrics_roll, index=["Valor"]).T.round(3), use_container_width=True, height=180)

        # Largo plazo
        st.markdown("#### Largo plazo — 2026–2031 (suma anual)")
        annual_ic_df, monthly_ic_out, meta_ic = long_horizon_annual_sum_ic(
            y_hist, x_hist, years=(2026,2027,2028,2029,2030,2031),
            alpha=0.20, variation_pp=int(variation_pp_ic)
        )
        years_vals = annual_ic_df.index.values
        w = 0.25
        fig2, ax2 = plt.subplots()
        ax2.bar(years_vals - w, annual_ic_df["base"].values, width=w, label="Base")
        ax2.bar(years_vals,       annual_ic_df["min"].values,  width=w, label="Min")
        ax2.bar(years_vals + w, annual_ic_df["max"].values,  width=w, label="Max")
        ax2.set_xlabel("Año"); ax2.set_ylabel("Suma anual"); ax2.legend()
        st.pyplot(fig2)
        with st.expander("Tabla anual I+C (suma)"):
            st.dataframe(annual_ic_df.reset_index(), use_container_width=True)

        # Guardar en sesión para la pestaña Crecimiento
        st.session_state["annual_ic_df"] = annual_ic_df

# -------------------- Pestaña Residencial --------------------
with tab_res:
    st.subheader("Demanda Residencial — OLS(log) + Temp^9 | SARIMA Temperatura")

    use_repo_res = st.checkbox("Usar archivo del repo (/data/variables.csv)", value=True, key="use_repo_res")
    csv_res_bytes = repo_file_bytes(CSV_NAME) if use_repo_res else None
    if not use_repo_res:
        csv_res = st.file_uploader("CSV histórico (Res + Temperatura)", type=["csv"], key="csv_res_body")
        if csv_res is not None:
            csv_res_bytes = csv_res.getvalue()

    if csv_res_bytes is None:
        st.info("No encuentro **data/variables.csv**. Subilo o colocalo en /data.")
    else:
        raw_df, dem_col, temp_col = load_res_from_csv(csv_res_bytes)
        df_res = prepare_design_res(raw_df, dem_col, temp_col)

        model, train, test = fit_ols_log_res(df_res)
        metrics_roll_res = rolling_12_metrics_res(df_res)
        r2_in = r2_score(train["DEMANDA"], train["yhat"])
        met_tbl = pd.DataFrame({
            "MAE":[metrics_roll_res["MAE"]],
            "RMSE":[metrics_roll_res["RMSE"]],
            "MAPE_%":[metrics_roll_res["MAPE_%"]],
            "R2_rolling12":[metrics_roll_res["R2"]],
            "R2_in_sample":[r2_in]
        }, index=["Valor"]).T.round(3)
        st.markdown("**Métricas (resumidas)**")
        st.dataframe(met_tbl, use_container_width=True, height=200)

        # Corto plazo
        st.markdown("#### Corto plazo — histórico (6m) + próximos 12 meses")
        hist_res, fc_res = forecast_short_term_res(df_res, model, alpha=0.20, history_months=6)
        fig3, ax3 = plt.subplots()
        ax3.plot(hist_res["indice_tiempo"], hist_res["Dem_Res_observada"], linestyle=":", label="Histórico (6m)")
        ax3.plot(fc_res["indice_tiempo"], fc_res["Dem_Res_base"], label="Base")
        ax3.plot(fc_res["indice_tiempo"], fc_res["PI80_lo_base"], linestyle="--", label="PI80 lo")
        ax3.plot(fc_res["indice_tiempo"], fc_res["PI80_hi_base"], linestyle="--", label="PI80 hi")
        ax3.axvline(fc_res["indice_tiempo"].min(), color="gray", linestyle=":", linewidth=1)
        ax3.set_xlabel("Mes"); ax3.set_ylabel("Demanda Residencial"); ax3.legend()
        st.pyplot(fig3)

        # Largo plazo — escenarios climáticos
        st.markdown("#### Largo plazo — 2026–2031 (suma anual): Base / Frío / Caliente")
        annual_res_df, monthly_res_out, meta_res = forecast_long_term_hot_cold(
            df_res, model, years=(2026,2027,2028,2029,2030,2031)
        )
        years_vals = annual_res_df.index.values
        w = 0.28
        fig4, ax4 = plt.subplots()
        ax4.bar(years_vals - w, annual_res_df["base"].values,     width=w, label="Base (típico)")
        ax4.bar(years_vals,       annual_res_df["frio"].values,   width=w, label="Frío (p10)")
        ax4.bar(years_vals + w, annual_res_df["caliente"].values, width=w, label="Caliente (p90)")
        ax4.set_xlabel("Año"); ax4.set_ylabel("Suma anual"); ax4.legend()
        st.pyplot(fig4)
        with st.expander("Tabla anual Residencial (suma)"):
            st.dataframe(annual_res_df.reset_index(), use_container_width=True)

        # Guardar en sesión para la pestaña Crecimiento
        st.session_state["annual_res_df"] = annual_res_df

# -------------------- Pestaña Crecimiento --------------------
with tab_growth:
    st.subheader("Crecimiento 2026→2031 (en %)")

    if ("annual_ic_df" not in st.session_state) or ("annual_res_df" not in st.session_state):
        st.info("Generá primero los escenarios en las pestañas **I+C** y **Residencial**.")
    else:
        annual_ic_df  = st.session_state["annual_ic_df"]
        annual_res_df = st.session_state["annual_res_df"]

        # I+C
        ic_g = {
            "I+C — Base": growth_2026_2031(annual_ic_df["base"]),
            "I+C — Min":  growth_2026_2031(annual_ic_df["min"]),
            "I+C — Max":  growth_2026_2031(annual_ic_df["max"]),
        }
        tbl_ic = (pd.DataFrame.from_dict(ic_g, orient="index", columns=["Crecimiento %"])
                  .reset_index().rename(columns={"index":"Escenario"}))

        # Residencial
        res_g = {
            "Residencial — Base":     growth_2026_2031(annual_res_df["base"]),
            "Residencial — Frío":     growth_2026_2031(annual_res_df["frio"]),
            "Residencial — Caliente": growth_2026_2031(annual_res_df["caliente"]),
        }
        tbl_res = (pd.DataFrame.from_dict(res_g, orient="index", columns=["Crecimiento %"])
                   .reset_index().rename(columns={"index":"Escenario"}))

        # Total (I+C + Res)
        tot_base = annual_ic_df["base"] + annual_res_df["base"]
        tot_min  = pd.DataFrame({
            "ic_min + res_frio":      annual_ic_df["min"] + annual_res_df["frio"],
            "ic_min + res_caliente":  annual_ic_df["min"] + annual_res_df["caliente"],
        }).min(axis=1)
        tot_max  = pd.DataFrame({
            "ic_max + res_frio":      annual_ic_df["max"] + annual_res_df["frio"],
            "ic_max + res_caliente":  annual_ic_df["max"] + annual_res_df["caliente"],
        }).max(axis=1)
        tot_g = {
            "Total — Base": growth_2026_2031(tot_base),
            "Total — Min":  growth_2026_2031(tot_min),
            "Total — Max":  growth_2026_2031(tot_max),
        }
        tbl_tot = (pd.DataFrame.from_dict(tot_g, orient="index", columns=["Crecimiento %"])
                   .reset_index().rename(columns={"index":"Escenario"}))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("I+C")
            st.dataframe(tbl_ic, use_container_width=True)
        with c2:
            st.subheader("Residencial")
            st.dataframe(tbl_res, use_container_width=True)
        with c3:
            st.subheader("Total (I+C + Res)")
            st.dataframe(tbl_tot, use_container_width=True)
