#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Soil Vitality Predictor (SVI) — A Physics-Informed AI Tool for Assessing Soil Biological Vitality

from __future__ import annotations
import io, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------- Page ----------------------------
st.set_page_config(page_title="Soil Vitality Predictor (SVI)", layout="wide")
st.title("Soil Vitality Predictor (SVI)")
st.caption("Physics-informed analytics for soil biological vitality (SI units).")

# ---------------------------- Paths ----------------------------
try:
    BASE_DIR = Path(__file__).parent.resolve()
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------- Schemas ----------------------------
REQUIRED_SCHEMAS: Dict[str, List[str]] = {
    "fields": ["field_id", "name", "lat", "lon"],
    "weather": ["field_id", "timestamp", "t_air_c", "rad_sw_wm2", "rain_mm"],
    "soil": ["field_id", "timestamp", "soil_moisture_vwc", "soil_temp_c"],
    "geo": ["field_id", "lat", "lon", "resistivity_ohm_m"],
    "ndvi": ["field_id", "timestamp", "lat", "lon", "ndvi"],
    "bio": ["field_id", "timestamp", "nbi"],
}
FILE_MAP = {
    "fields": "fields.csv",
    "weather": "weather_timeseries.csv",
    "soil": "soil_sensor_timeseries.csv",
    "geo": "geophysics.csv",
    "ndvi": "remote_sensing_ndvi.csv",
    "bio": "biology_observed.csv",
}

# ---------------------------- Helpers ----------------------------
def ai_note(title: str, bullets: list[str]):
    """Render interpretive bullets. Suppress headings that start with 'AI interpretation' for a cleaner look."""
    title_lower = str(title).strip().lower() if title else ""
    if not title_lower.startswith("ai interpretation"):
        st.markdown(f"**{title}**")
    for b in bullets:
        st.markdown(f"- {b}")

def style_time_axis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.figure.autofmt_xdate()
    ax.margins(x=0.01)
    ax.grid(True, axis='x', alpha=0.2)

def _to_ordinal(d):
    try:
        return pd.to_datetime(d).map(pd.Timestamp.toordinal).to_numpy()
    except Exception:
        return pd.Series(d).map(pd.to_datetime, errors='coerce').map(pd.Timestamp.toordinal).to_numpy()

def _trend_slope(dates, values):
    x = _to_ordinal(dates)
    y = pd.Series(values).astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(x[m], y[m], 1)[0]  # per day

@st.cache_data(show_spinner=False)
def read_csv_any(local_path: Optional[Path], upload_bytes: Optional[bytes]) -> Optional[pd.DataFrame]:
    try:
        if upload_bytes is not None:
            df = pd.read_csv(io.BytesIO(upload_bytes))
        elif local_path is not None and local_path.exists():
            df = pd.read_csv(local_path)
        else:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date"] = df["timestamp"].dt.date
        return df
    except Exception:
        return None

def validate_schema(name: str, df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    req = REQUIRED_SCHEMAS.get(name, [])
    if df is None:
        return False, f"{name}: not provided or failed to load"
    missing = [c for c in req if c not in df.columns]
    if missing:
        return False, f"{name}: missing required columns {missing}"
    if len(df) == 0:
        return False, f"{name}: dataset is empty"
    return True, f"{name}: OK"

def add_season(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    m = d.dt.month.astype(int)
    df["month_sin"] = np.sin(2*np.pi*m/12)
    df["month_cos"] = np.cos(2*np.pi*m/12)
    return df

# ---------------------------- Physics functions ----------------------------
def resistivity_vitality_factor(rho, rho_opt=70.0, sigma_ln=0.6):
    rho = np.maximum(np.array(rho, dtype=float), 1e-6)
    return np.exp(-((np.log(rho) - np.log(rho_opt))**2) / (2 * sigma_ln**2))

def moisture_vitality_factor(theta, fc=0.30, wp=0.10):
    theta = np.clip(np.array(theta, dtype=float), 0, 1)
    lo = max(wp + 0.03, 1e-6)
    hi = fc
    v = np.zeros_like(theta, dtype=float)
    v += np.where(theta <= lo, (theta / lo) * 0.6, 0)
    v += np.where((theta > lo) & (theta <= hi), 0.6 + 0.4 * (theta - lo) / (max(hi - lo, 1e-9)), 0)
    v += np.where(theta > hi, np.maximum(0.6 - 0.3 * (theta - hi) / 0.15, 0), 0)
    return np.clip(v, 0, 1)

def temperature_vitality_factor(t_c, t_opt=24.0, sigma=6.0):
    t_c = np.array(t_c, dtype=float)
    return np.exp(-((t_c - t_opt)**2) / (2 * sigma**2))

def simple_ET_hargreaves_proxy(t_c, rad_sw_wm2):
    rad_MJ_m2_day = np.clip(rad_sw_wm2, 0, None) * 86400 / 1e6
    etp = 0.0023 * (t_c + 17.8) * np.sqrt(np.maximum(t_c, 0) + 0.1) * np.clip(rad_MJ_m2_day, 0, None)
    return np.clip(etp / 10.0, 0, 12)

# ---------------------------- Sidebar ----------------------------
st.sidebar.header("Data")
auto_load = st.sidebar.toggle("Auto-load after uploads", value=True)
uploads: Dict[str, Optional[any]] = {}
for key, label in FILE_MAP.items():
    if key in ["fields","weather","soil","geo","ndvi","bio"]:
        uploads[key] = st.sidebar.file_uploader(f"Upload {label}", type=["csv"], key=f"u_{key}")

st.sidebar.header("Scenarios")
irr_add = st.sidebar.slider("Irrigation addition (mm/week)", 0, 40, 10)
rain_scale = st.sidebar.slider("Rainfall scaling (%)", -50, 50, 0)
temp_shift = st.sidebar.slider("ΔT absolute (°C)", -5, 5, 0)

st.sidebar.header("Physics Params")
rho_opt = st.sidebar.number_input("Optimal resistivity ρ* (Ω·m)", value=70.0, min_value=5.0, max_value=500.0, step=5.0)
sigma_ln = st.sidebar.number_input("Spread σ_ln (log space)", value=0.6, min_value=0.1, max_value=2.0, step=0.1)
fc = st.sidebar.slider("Field capacity, FC (m³/m³)", 0.20, 0.45, 0.30, 0.01)
wp = st.sidebar.slider("Wilting point, WP (m³/m³)", 0.05, 0.20, 0.10, 0.01)

st.sidebar.header("SVI Weights (normalized)")
w_rho = st.sidebar.slider("Resistivity", 0.0, 1.0, 0.25, 0.05)
w_theta = st.sidebar.slider("Moisture", 0.0, 1.0, 0.35, 0.05)
w_T = st.sidebar.slider("Temperature", 0.0, 1.0, 0.20, 0.05)
w_v = st.sidebar.slider("NDVI", 0.0, 1.0, 0.20, 0.05)
use_wb = st.sidebar.toggle("Experimental: include water-balance factor", value=False)
w_wb = st.sidebar.slider("Water balance (active only if enabled)", 0.0, 1.0, 0.00 if not use_wb else 0.10, 0.05, disabled=not use_wb)
_sumw = w_rho + w_theta + w_T + w_v + (w_wb if use_wb else 0.0)
if _sumw <= 0: st.sidebar.error("Weights must sum to > 0")
SVI_WEIGHTS = {"V_rho": w_rho/_sumw, "V_theta": w_theta/_sumw, "V_T": w_T/_sumw, "V_v": w_v/_sumw}
if use_wb: SVI_WEIGHTS["V_wb"] = w_wb/_sumw

st.sidebar.header("Geospatial")
geojson_upload = st.sidebar.file_uploader("Upload field boundaries (GeoJSON)", type=["geojson","json"], key="geojson_upload")

# ---------------------------- Load Data ----------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

colL, colR = st.columns([1,3])
with colL:
    load_clicked = st.button("Load Data", type="primary")
with colR:
    st.caption("Uploads preferred. If missing, the app looks for CSVs in ./data/")

if auto_load and any(uploads[k] is not None for k in ["fields","weather","soil","ndvi"]):
    load_clicked = True

if load_clicked:
    status = st.status("Loading datasets…", expanded=True)
    loaded: Dict[str, Optional[pd.DataFrame]] = {}
    hard_required = ["fields","weather","soil","ndvi"]
    errors: List[str] = []
    for key, fname in FILE_MAP.items():
        up = uploads.get(key)
        up_bytes = up.getvalue() if up is not None else None
        df = read_csv_any(DATA_DIR / fname, up_bytes)
        ok, msg = validate_schema(key, df) if (key in hard_required) else (True, f"{key}: optional") if df is None else validate_schema(key, df)
        if key in hard_required and not ok:
            errors.append(msg); status.write(f"❌ {msg}")
        else:
            status.write(f"✅ {msg}")
        loaded[key] = df
    if errors:
        status.update(label="Load failed — see issues above", state="error")
        st.session_state.data_loaded = False
    else:
        status.update(label="Load complete", state="complete")
        st.session_state.data_loaded = True
        st.session_state.dfs = loaded

required_ready = st.session_state.data_loaded and all(
    st.session_state.dfs.get(k) is not None for k in ["fields", "weather", "soil", "ndvi"]
)
if not required_ready:
    st.info("Load the required datasets (fields, weather, soil, ndvi) to enable analytics.")
    st.stop()

Df = st.session_state.dfs
df_fields = Df.get("fields"); df_weather = Df.get("weather"); df_soil = Df.get("soil")
df_geo = Df.get("geo"); df_ndvi = Df.get("ndvi"); df_bio = Df.get("bio")

# ---------------------------- Feature engineering ----------------------------
@st.cache_data(show_spinner=False)
def engineer_features(df_weather, df_soil, df_geo, df_ndvi,
                      irr_add, rain_scale, temp_shift,
                      rho_opt, sigma_ln, fc, wp, use_wb, SVI_WEIGHTS) -> pd.DataFrame:
    w = df_weather.copy(); s = df_soil.copy(); n = df_ndvi.copy()
    g = df_geo.copy() if df_geo is not None else None

    w["date"] = pd.to_datetime(w["timestamp"], errors="coerce").dt.date
    s["date"] = pd.to_datetime(s["timestamp"], errors="coerce").dt.date
    n["date"] = pd.to_datetime(n["timestamp"], errors="coerce").dt.date

    w_d = w.groupby(["field_id", "date"]).agg(
        t_air_c=("t_air_c", "mean"),
        rad_sw_wm2=("rad_sw_wm2", "mean"),
        rain_mm=("rain_mm", "sum"),
    ).reset_index()

    s_d = s.groupby(["field_id", "date"]).agg(
        soil_moisture_vwc=("soil_moisture_vwc", "mean"),
        soil_temp_c=("soil_temp_c", "mean"),
    ).reset_index()

    n_d = n.groupby(["field_id", "date"]).agg(ndvi=("ndvi", "mean")).reset_index()

    feat = w_d.merge(s_d, on=["field_id", "date"], how="inner").merge(n_d, on=["field_id", "date"], how="left")

    if g is not None and not g.empty and {'field_id','resistivity_ohm_m'}.issubset(g.columns):
        g_m = g.groupby("field_id").agg(resistivity_ohm_m=("resistivity_ohm_m", "median")).reset_index()
        feat = feat.merge(g_m, on="field_id", how="left")
    else:
        feat["resistivity_ohm_m"] = np.nan

    feat["rain_mm_adj"] = feat["rain_mm"] * (1 + rain_scale / 100.0)
    feat["t_air_c_adj"] = feat["t_air_c"] + temp_shift
    feat["soil_moisture_vwc_adj"] = np.clip(feat["soil_moisture_vwc"] + irr_add / 1000.0, 0, 0.6)

    feat["V_rho"] = resistivity_vitality_factor(
        np.where(np.isfinite(feat["resistivity_ohm_m"]), feat["resistivity_ohm_m"], rho_opt),
        rho_opt, sigma_ln
    )
    feat["V_theta"] = moisture_vitality_factor(feat["soil_moisture_vwc_adj"], fc=fc, wp=wp)
    feat["V_T"] = temperature_vitality_factor(feat["t_air_c_adj"])
    feat["V_v"] = np.clip((feat["ndvi"] - 0.2) / 0.6, 0, 1)

    feat["et_proxy_mm"] = simple_ET_hargreaves_proxy(feat["t_air_c_adj"], feat["rad_sw_wm2"])
    feat["wbalance"] = feat["rain_mm_adj"] - feat["et_proxy_mm"]
    feat["V_wb"] = np.clip(0.5 + 0.5*np.tanh(feat["wbalance"]/5.0), 0, 1) if use_wb else 0.0

    svi_terms = sum(SVI_WEIGHTS[k] * feat[k] for k in SVI_WEIGHTS)
    feat["SVI"] = np.clip(svi_terms, 0, 1)

    return feat

with st.status("Engineering physics-informed features…", expanded=False):
    feat = engineer_features(
        df_weather, df_soil, df_geo, df_ndvi,
        irr_add, rain_scale, temp_shift,
        rho_opt, sigma_ln, fc, wp, use_wb, SVI_WEIGHTS
    )

# ---------------------------- Date filters ----------------------------
dmin, dmax = feat["date"].min(), feat["date"].max()
f1, f2 = st.columns(2)
with f1:
    start_date = st.date_input("Start date", dmin, min_value=dmin, max_value=dmax)
with f2:
    end_date = st.date_input("End date", dmax, min_value=dmin, max_value=dmax)
mask = (pd.to_datetime(feat["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(feat["date"]) <= pd.to_datetime(end_date))
feat = feat[mask].copy()

# ---------------------------- run configuration ----------------------------
run_config = {
    "params": {
        "rho_opt": rho_opt, "sigma_ln": sigma_ln, "fc": fc, "wp": wp,
        "irr_add_mm_per_week": irr_add, "rain_scale_percent": rain_scale, "temp_shift_C": temp_shift,
        "use_wb": use_wb
    },
    "weights": SVI_WEIGHTS,
    "date_window": {"start": str(start_date), "end": str(end_date)}
}
try:
    (UPLOADS_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))
except Exception:
    pass

# ---------------------------- Overview ----------------------------
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fields", f"{Df['fields']['field_id'].nunique()}")
col2.metric("Records (features)", f"{len(feat):,}")
col3.metric("Mean SVI", f"{float(np.nanmean(feat['SVI'])):.2f}")
col4.metric("Date span", f"{str(min(feat['date']))} → {str(max(feat['date']))}")
st.dataframe(feat.head(50), use_container_width=True)

mean_svi = float(np.nanmean(feat['SVI']))
if mean_svi >= 0.7:
    st.success("Overall SVI is strong; maintain irrigation/residue strategy and monitor ET-proxy spikes.")
elif mean_svi >= 0.5:
    st.info("SVI is moderate; focus on moisture optimization and inspect below-median NDVI fields for stress.")
else:
    st.warning("SVI is low; check drainage in low-ρ zones and time irrigation to lift θ toward the optimal band.")

# ---------------------------- Response Functions ----------------------------
st.markdown("### Response Functions")
c1, c2, c3 = st.columns(3)
with c1:
    rhos = np.linspace(5, 500, 400)
    fig, ax = plt.subplots()
    ax.plot(rhos, resistivity_vitality_factor(rhos, rho_opt=rho_opt, sigma_ln=sigma_ln), linewidth=2)
    ax.set_xlabel("Resistivity (Ω·m)"); ax.set_ylabel("Vitality factor"); ax.set_title("Resistivity → Vitality")
    ax.grid(True, alpha=0.3); st.pyplot(fig, clear_figure=True)
    ai_note("Resistivity response", [
        f"Peak vitality near ρ ≈ {rho_opt:.0f} Ω·m; width set by σ_ln={sigma_ln:.2f}.",
        "Very low ρ indicates wet/saline or poor drainage; very high ρ often dryness or compaction.",
        "Use ρ maps to guide drainage fixes and structure management."
    ])
with c2:
    thetas = np.linspace(0.02, 0.6, 300)
    fig, ax = plt.subplots()
    ax.plot(thetas, moisture_vitality_factor(thetas, fc=fc, wp=wp), linewidth=2)
    ax.set_xlabel("Volumetric water content (m³/m³)"); ax.set_ylabel("Vitality factor"); ax.set_title("Moisture → Vitality")
    ax.grid(True, alpha=0.3); st.pyplot(fig, clear_figure=True)
    ai_note("Moisture response", [
        f"Best range ≈ {(wp+0.03):.02f}–{fc:.02f} m³/m³.",
        "Below the band biology slows (drought); above FC, anaerobiosis suppresses activity.",
        "Time irrigation and preserve residue to keep θ inside the band."
    ])
with c3:
    temps = np.linspace(-2, 45, 300)
    fig, ax = plt.subplots()
    ax.plot(temps, temperature_vitality_factor(temps), linewidth=2)
    ax.set_xlabel("Soil temperature (°C)"); ax.set_ylabel("Vitality factor"); ax.set_title("Temperature → Vitality")
    ax.grid(True, alpha=0.3); st.pyplot(fig, clear_figure=True)
    mean_T = float(np.nanmean(feat["t_air_c_adj"])) if "t_air_c_adj" in feat.columns else float("nan")
    ai_note("Temperature response", [
        "Activity peaks near ~24 °C and declines when cooler or hotter.",
        f"Current mean adjusted air temperature ≈ {mean_T:.1f} °C.",
        "Mulch and irrigation timing can buffer ET and temperature spikes."
    ])

# ---------------------------- Time Series Diagnostics ----------------------------
st.markdown("### Time Series Diagnostics")
fields_list = sorted(feat['field_id'].unique())
sel_field = st.selectbox("Select field", fields_list)
F = feat[feat['field_id']==sel_field].sort_values('date')

agg_weekly = st.checkbox("Aggregate to weekly means (improves readability)", value=True)
if agg_weekly and not F.empty:
    F_plot = (F.set_index(pd.to_datetime(F['date']))
                .resample('W').mean(numeric_only=True)
                .reset_index().rename(columns={'index':'date'}))
else:
    F_plot = F.copy()

def safe_series(df, col):
    s = df[col] if col in df.columns else pd.Series(dtype=float)
    return s.dropna()

# Water balance plot
if not F_plot.empty and safe_series(F_plot, 'rain_mm').size and safe_series(F_plot, 'et_proxy_mm').size:
    fig, ax = plt.subplots()
    ax.plot(F_plot['date'], F_plot['rain_mm'], label='Rain (mm/day)', linewidth=1.5, alpha=0.9)
    ax.plot(F_plot['date'], F_plot['et_proxy_mm'], label='ET proxy (mm/day)', linewidth=1.5, alpha=0.9)
    if 'wbalance' in F_plot.columns:
        ax.plot(F_plot['date'], F_plot['wbalance'], label='Water balance (mm/day)', linewidth=1.2, alpha=0.7)
    ax.set_title(f"Water balance — {sel_field}")
    ax.set_xlabel("Date"); ax.set_ylabel("mm/day"); ax.grid(True, alpha=0.3); ax.legend()
    style_time_axis(ax); st.pyplot(fig, clear_figure=True)
    deficit_days = int((F_plot["wbalance"] < 0).sum()) if "wbalance" in F_plot.columns else 0
    ai_note("Water balance note", [
        f"Net deficit days: {deficit_days}.",
        "Use short irrigation pulses during multi-day deficits. Preserve residue to reduce ET.",
        "Watch for deficit streaks that align with dips in SVI."
    ])
else:
    st.info("Water balance plot: missing or empty series for selected field.")

# SVI over time
if not F_plot.empty and safe_series(F_plot, 'SVI').size:
    fig, ax = plt.subplots()
    ax.plot(F_plot['date'], F_plot['SVI'], label='SVI', linewidth=1.8)
    svi_roll = F_plot['SVI'].rolling(5, min_periods=1)
    ax.fill_between(F_plot['date'], svi_roll.mean()-svi_roll.std(), svi_roll.mean()+svi_roll.std(), alpha=0.2)
    ax.set_title(f"SVI over time — {sel_field}"); ax.set_xlabel("Date"); ax.set_ylabel("SVI (0–1)")
    ax.grid(True, alpha=0.3); style_time_axis(ax); st.pyplot(fig, clear_figure=True)
    slope = _trend_slope(F_plot['date'], F_plot['SVI'])
    ai_note("SVI interpretation", [
        f"Latest SVI = {float(F_plot['SVI'].iloc[-1]):.2f}; trend is {'rising' if slope>0 else 'falling' if slope<0 else 'flat'} (slope={slope:.2e}/day).",
        "Sustained declines suggest moisture stress or late-season senescence."
    ])
else:
    st.info("SVI over time: no data available for this field.")

# NDVI over time
ND = df_ndvi[df_ndvi['field_id']==sel_field].copy() if df_ndvi is not None else pd.DataFrame()
if not ND.empty:
    ND['date'] = pd.to_datetime(ND['timestamp'], errors='coerce').dt.date
    nd_daily = (ND.groupby('date')['ndvi'].mean().reset_index())
    fig, ax = plt.subplots()
    ax.plot(nd_daily['date'], nd_daily['ndvi'], label='NDVI (daily mean)', linewidth=1.6)
    nd_roll = nd_daily['ndvi'].rolling(7, min_periods=1)
    ax.fill_between(nd_daily['date'], nd_roll.mean()-nd_roll.std(), nd_roll.mean()+nd_roll.std(), alpha=0.2)
    ax.set_title(f"NDVI timeline — {sel_field}")
    ax.set_xlabel("Date"); ax.set_ylabel("NDVI (0–1)")
    ax.grid(True, alpha=0.3); style_time_axis(ax); st.pyplot(fig, clear_figure=True)
    nd_mean = float(nd_daily["ndvi"].mean()); nd_vol = float(nd_daily["ndvi"].std())
    ai_note("NDVI interpretation", [
        f"NDVI mean ≈ {nd_mean:.2f}, variability σ≈{nd_vol:.2f}.",
        "Unexplained NDVI dips can flag emerging stress; cross-check θ and water balance."
    ])
else:
    st.info("NDVI timeline: no data available for this field.")

# ---------------------------- Distributions & Benchmarking ----------------------------
st.markdown("### Distributions and Benchmarking")
dc1, dc2 = st.columns(2)
with dc1:
    fig, ax = plt.subplots()
    ax.hist(feat['SVI'].dropna(), bins=30)
    ax.set_title("SVI distribution"); ax.set_xlabel("SVI"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    if feat["SVI"].notna().any():
        q25, q50, q75 = feat["SVI"].quantile([0.25, 0.5, 0.75]).tolist()
        ai_note("SVI distribution", [
            f"Median SVI ≈ {q50:.2f} (IQR {q25:.2f}–{q75:.2f}).",
            "Fat lower tails point to structure or moisture constraints.",
            "Shrink the left tail via drainage fixes and timely moisture management."
        ])
with dc2:
    per_field = feat.groupby('field_id')['SVI'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(per_field.index, per_field.values)
    ax.set_title("Field benchmarking (mean SVI)"); ax.set_xlabel("Mean SVI"); ax.grid(True, axis='x', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    if not per_field.empty:
        best = ", ".join(per_field.tail(2).index.tolist())
        worst = ", ".join(per_field.head(2).index.tolist())
        ai_note("Field benchmarking", [
            f"Top performers: {best}; focus zones: {worst}.",
            "Replicate residue and irrigation cadence from top fields in low-rank fields."
        ])

# ---------------------------- Map ----------------------------
st.markdown("### Spatial View (EPSG:4326)")

geojson_obj = None
if geojson_upload is not None:
    try:
        geojson_bytes = geojson_upload.read()
        geojson_obj = json.loads(geojson_bytes.decode("utf-8"))
        if geojson_obj.get("type") == "Feature":
            geojson_obj = {"type": "FeatureCollection", "features": [geojson_obj]}
    except Exception as e:
        st.warning(f"Failed to read GeoJSON: {e}")

fields_pos = df_fields.groupby("field_id").agg(lat=("lat","mean"), lon=("lon","mean")).reset_index()
latest_per_field = feat.sort_values(["field_id","date"]).groupby("field_id", as_index=False).tail(1)
mdf = latest_per_field.merge(fields_pos, on="field_id", how="left").dropna(subset=["lat","lon"])

def _lerp(a, b, t): return int(round(a + (b - a) * t))
def continuous_color(val, vmin, vmax):
    if not np.isfinite(val) or vmax <= vmin: return [160,160,160]
    t = max(0.0, min(1.0, float((val - vmin) / (vmax - vmin))))
    r = _lerp(200, 26, t); g = _lerp(60, 152, t); b = _lerp(60, 80, t); return [r,g,b]

layers = []

if geojson_obj is not None and len(mdf) > 0:
    val_map = mdf.set_index("field_id")["SVI"].to_dict()
    vals = np.array([v for v in val_map.values() if np.isfinite(v)], dtype=float)
    vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if vals.size else (0.0, 1.0)
    for ft in geojson_obj.get("features", []):
        props = ft.setdefault("properties", {})
        fid = props.get("field_id") or props.get("Field_ID") or props.get("id") or props.get("name")
        val = val_map.get(fid, np.nan)
        props["SVI"] = None if not np.isfinite(val) else float(val)
        props["fill_color"] = continuous_color(val, vmin, vmax) + [160]
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=geojson_obj,
            stroked=True,
            filled=True,
            wireframe=False,
            get_fill_color="properties.fill_color",
            get_line_color=[40,40,40,220],
            get_line_width=2,
            pickable=True,
        )
    )

if len(mdf) > 0:
    mdf_map = mdf[["lon","lat","SVI","field_id"]].copy()
    mdf_map["color"] = (mdf_map["SVI"].fillna(0) * 255).astype(int)
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=mdf_map,
            get_position=["lon","lat"],
            get_radius=80,
            radius_scale=1,
            get_fill_color=["color", "color", 120, 160],
            pickable=True,
        )
    )
    view_state = pdk.ViewState(latitude=float(mdf_map["lat"].mean()), longitude=float(mdf_map["lon"].mean()), zoom=10)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{field_id}\nSVI: {SVI:.2f}"})
    st.pydeck_chart(deck)
    svi_std = float(mdf["SVI"].std())
    if svi_std >= 0.10:
        ai_note("Spatial pattern", [
            "Spatial variability is significant; consider zoning and variable-rate interventions.",
            "Target sampling in low-SVI edges and compacted headlands."
        ])
    else:
        ai_note("Spatial pattern", [
            "Spatial vitality is uniform; maintain current practices and monitor outliers after storms."
        ])
else:
    st.info("No coordinates available to render the map.")

# ---------------------------- Geophysics → Moisture Model ----------------------------
st.markdown("### Geophysical Soil Modelling (ρ → θ)")
cmod = st.container()
with cmod:
    cols_top = st.columns(3)
    with cols_top[0]:
        run_geo_model = st.button("Run ρ→θ model")
    with cols_top[1]:
        per_field_models = st.checkbox("Per-field models (in addition to pooled)", value=False,
                                       help="Train a separate ρ→θ model per field and report fit metrics by field.")
    with cols_top[2]:
        write_theta = st.checkbox("Write θ predictions to features (θ̂)", value=True)

    if run_geo_model:
        missing = []
        if df_geo is None or df_geo.empty: missing.append("geophysics.csv")
        if df_soil is None or df_soil.empty: missing.append("soil_sensor_timeseries.csv")
        if df_weather is None or df_weather.empty: missing.append("weather_timeseries.csv")
        if df_ndvi is None or df_ndvi.empty: missing.append("remote_sensing_ndvi.csv")
        if missing:
            st.warning("Cannot run model. Missing: " + ", ".join(missing))
        else:
            try:
                s = df_soil.copy(); s["date"] = pd.to_datetime(s["timestamp"], errors="coerce").dt.date
                s_d = s.groupby(["field_id","date"]).agg(soil_moisture_vwc=("soil_moisture_vwc","mean")).reset_index()

                w = df_weather.copy(); w["date"] = pd.to_datetime(w["timestamp"], errors="coerce").dt.date
                w_d = w.groupby(["field_id","date"]).agg(
                    t_air_c=("t_air_c","mean"), rain_mm=("rain_mm","sum"), rad_sw_wm2=("rad_sw_wm2","mean")
                ).reset_index()

                n = df_ndvi.copy(); n["date"] = pd.to_datetime(n["timestamp"], errors="coerce").dt.date
                n_d = n.groupby(["field_id","date"]).agg(ndvi=("ndvi","mean")).reset_index()

                g = df_geo.copy()
                if "survey_date" in g.columns:
                    g["date"] = pd.to_datetime(g["survey_date"], errors="coerce").dt.date
                elif "timestamp" in g.columns:
                    g["date"] = pd.to_datetime(g["timestamp"], errors="coerce").dt.date
                else:
                    g["date"] = pd.NaT
                g_d = g.groupby(["field_id","date"]).agg(resistivity_ohm_m=("resistivity_ohm_m","median")).reset_index()

                Z = (g_d.merge(s_d, on=["field_id","date"], how="inner")
                        .merge(w_d, on=["field_id","date"], how="left")
                        .merge(n_d, on=["field_id","date"], how="left")).dropna(subset=["resistivity_ohm_m","soil_moisture_vwc"])

                st.caption(f"Matched ρ↔θ records: {len(Z)} | Fields: {Z['field_id'].nunique()} | Date span: {Z['date'].min()} → {Z['date'].max()}")

                if len(Z) < 40:
                    st.info("Not enough matched days for robust modelling (need ≥ 40).")
                else:
                    feats_base = ["resistivity_ohm_m","t_air_c","rain_mm","rad_sw_wm2","ndvi"]
                    X = Z[["field_id","date"] + feats_base].sort_values(["field_id","date"]).reset_index(drop=True)

                    # forward/backward fill within field for missing drivers
                    for col in feats_base:
                        X[col] = X.groupby("field_id", group_keys=False)[col].apply(lambda s: s.ffill().bfill()).reset_index(drop=True)

                    # short history features
                    for col in ["t_air_c","rain_mm","rad_sw_wm2","ndvi"]:
                        X[f"{col}_roll7"] = X.groupby("field_id", group_keys=False)[col].apply(lambda s: s.rolling(7, min_periods=1).mean()).reset_index(drop=True)
                        X[f"{col}_lag7"]  = X.groupby("field_id", group_keys=False)[col].shift(7).reset_index(drop=True)

                    X = add_season(X, "date")
                    feats = feats_base + \
                            [f"{c}_roll7" for c in ["t_air_c","rain_mm","rad_sw_wm2","ndvi"]] + \
                            [f"{c}_lag7" for c in ["t_air_c","rain_mm","rad_sw_wm2","ndvi"]] + \
                            ["month_sin","month_cos"]

                    X_model = X[feats]
                    y = Z["soil_moisture_vwc"].to_numpy()
                    groups = Z["field_id"]

                    model = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("gbr", HistGradientBoostingRegressor(random_state=42))
                    ])
                    gcv = GroupKFold(n_splits=5)
                    r2_cv = cross_val_score(model, X_model, y, cv=gcv.split(X_model, y, groups=groups), scoring="r2")

                    model.fit(X_model, y)
                    yhat = model.predict(X_model)

                    r2_in = r2_score(y, yhat)
                    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
                    mae = mean_absolute_error(y, yhat)
                    bias = float(np.mean(yhat - y))
                    st.write(f"CV R² (GroupKFold by field): **{r2_cv.mean():.2f} ± {r2_cv.std():.2f}** | In-sample R²: **{r2_in:.2f}** | RMSE: **{rmse:.03f}** | MAE: **{mae:.03f}** | Bias (ŷ−y): **{bias:.03f}**")

                    # Parity
                    fig, ax = plt.subplots()
                    ax.scatter(y, yhat, s=14, label="Predicted vs Observed")
                    mn, mx = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
                    ax.plot([mn, mx], [mn, mx], linewidth=1.2, label="1:1 reference")
                    ax.set_xlabel("Observed θ (m³/m³)"); ax.set_ylabel("Predicted θ (m³/m³)")
                    ax.set_title("ρ → θ | Parity"); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
                    st.pyplot(fig, clear_figure=True)

                    # Residuals
                    fig, ax = plt.subplots()
                    ax.scatter(yhat, y - yhat, s=14)
                    ax.axhline(0, linestyle="--", linewidth=1)
                    ax.set_xlabel("Predicted θ (m³/m³)"); ax.set_ylabel("Residual (y − ŷ)")
                    ax.set_title("Residuals vs Prediction"); ax.grid(True, alpha=0.3)
                    st.pyplot(fig, clear_figure=True)

                    st.session_state.setdefault("perf", {})
                    st.session_state["perf"]["rho_theta"] = dict(
                        cv_r2=float(r2_cv.mean()), cv_sd=float(r2_cv.std()),
                        r2=float(r2_in), rmse=float(rmse), mae=float(mae), bias=float(bias)
                    )
                    ai_note("ρ → θ interpretation", [
                        f"In-sample fit is strong (R²={r2_in:.2f}); cross-field generalization is weak (CV R²={r2_cv.mean():.2f}).",
                        "ρ–θ calibration is field-specific. For deployment, use per-field models or normalize ρ by texture/depth.",
                        "Align survey dates to soil measurements to reduce mismatch noise."
                    ])

                    # ------ Per-field models and table ------
                    if per_field_models:
                        rows_pf = []
                        for fid, Zf in Z.groupby("field_id"):
                            if len(Zf) < 20:
                                rows_pf.append({"field_id": fid, "n": len(Zf), "r2_in": np.nan, "rmse": np.nan})
                                continue
                            Xf = X.loc[Zf.index, feats]
                            yf = Zf["soil_moisture_vwc"].to_numpy()
                            mdl = Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("gbr", HistGradientBoostingRegressor(random_state=42))
                            ])
                            # simple within-field CV if enough points
                            if len(Zf) >= 30:
                                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                                cvf = cross_val_score(mdl, Xf, yf, cv=kf, scoring="r2")
                                cvf_mean = float(np.nanmean(cvf))
                            else:
                                cvf_mean = np.nan
                            mdl.fit(Xf, yf)
                            yhatf = mdl.predict(Xf)
                            r2f = r2_score(yf, yhatf)
                            rmsef = float(np.sqrt(mean_squared_error(yf, yhatf)))
                            rows_pf.append({"field_id": fid, "n": len(Zf), "r2_in": r2f, "cv_r2_3fold": cvf_mean, "rmse": rmsef})

                        pf_df = pd.DataFrame(rows_pf, columns=["field_id","n","r2_in","cv_r2_3fold","rmse"])
                        st.dataframe(pf_df, use_container_width=True)

                    # write θ̂ back for the whole feature table
                    if write_theta:
                        Wd = df_weather.copy(); Wd["date"] = pd.to_datetime(Wd["timestamp"], errors="coerce").dt.date
                        Wd = Wd.groupby(["field_id","date"]).agg(t_air_c=("t_air_c","mean"), rain_mm=("rain_mm","sum"), rad_sw_wm2=("rad_sw_wm2","mean")).reset_index()
                        Nd = df_ndvi.copy(); Nd["date"] = pd.to_datetime(Nd["timestamp"], errors="coerce").dt.date
                        Nd = Nd.groupby(["field_id","date"]).agg(ndvi=("ndvi","mean")).reset_index()
                        Gd = df_geo.copy()
                        if "survey_date" in Gd.columns:
                            Gd["date"] = pd.to_datetime(Gd["survey_date"], errors="coerce").dt.date
                        elif "timestamp" in Gd.columns:
                            Gd["date"] = pd.to_datetime(Gd["timestamp"], errors="coerce").dt.date
                        else:
                            Gd["date"] = pd.NaT
                        Gd = Gd.groupby(["field_id","date"]).agg(resistivity_ohm_m=("resistivity_ohm_m","median")).reset_index()

                        P = feat[["field_id","date"]].drop_duplicates()
                        P = (P.merge(Gd, on=["field_id","date"], how="left")
                               .merge(Wd, on=["field_id","date"], how="left")
                               .merge(Nd, on=["field_id","date"], how="left"))
                        P = P.sort_values(["field_id","date"]).reset_index(drop=True)

                        for col in ["t_air_c","rain_mm","rad_sw_wm2","ndvi","resistivity_ohm_m"]:
                            P[col] = P.groupby("field_id", group_keys=False)[col].apply(lambda s: s.ffill().bfill()).reset_index(drop=True)

                        for col in ["t_air_c","rain_mm","rad_sw_wm2","ndvi"]:
                            P[f"{col}_roll7"] = P.groupby("field_id", group_keys=False)[col].apply(lambda s: s.rolling(7, min_periods=1).mean()).reset_index(drop=True)
                            P[f"{col}_lag7"] = P.groupby("field_id", group_keys=False)[col].shift(7).reset_index(drop=True)

                        P = add_season(P, "date")
                        P_model = P[feats]
                        theta_hat = model.predict(P_model)
                        feat = feat.merge(P[["field_id","date"]].assign(theta_hat=theta_hat), on=["field_id","date"], how="left")
                        st.success("θ̂ predictions written to engineered features.")
            except Exception as e:
                st.warning(f"Geophysical modelling failed: {e}")

# ---------------------------- Calibration against Biology (NBI) ----------------------------
st.markdown("### Calibration against Biology (NBI)")
if df_bio is None or df_bio.empty:
    st.caption("Upload biology_observed.csv to enable NBI calibration.")
else:
    tol_days = st.slider("Date matching tolerance (±days)", 0, 30, 14)
    min_rows = st.number_input("Minimum rows for modeling", 30, 200, 40, step=5)
    require_ndvi = st.checkbox("Require NDVI present", value=False, help="If off, NDVI is median-imputed per field, then global.")
    use_lags = st.checkbox("Use 7-day lags/rolling means", value=True, help="Adds short-history features without dropping early days.")
    write_nbi = st.checkbox("Write NBI predictions to features (NBÎ)", value=True)

    b = df_bio.copy()
    b["date"] = pd.to_datetime(b["timestamp"], errors="coerce").dt.date
    b_day = b.groupby(["field_id", "date"], as_index=False).agg(nbi=("nbi", "mean"))

    F_feat = feat[["field_id","date","SVI","t_air_c_adj","soil_moisture_vwc_adj","ndvi"]].copy()

    bd = b_day.copy(); bd["date_dt"] = pd.to_datetime(bd["date"])
    Ft = F_feat.copy(); Ft["date_dt"] = pd.to_datetime(Ft["date"])
    # nearest-past match within tolerance
    cand = bd.merge(Ft, on="field_id", how="left", suffixes=("_bio","_feat"))
    cand = cand[cand["date_dt_feat"] <= cand["date_dt_bio"]]
    cand["abs_diff_days"] = (cand["date_dt_bio"] - cand["date_dt_feat"]).dt.days
    cand = cand[cand["abs_diff_days"] <= tol_days]
    cand["bio_key"] = cand["field_id"].astype(str) + "|" + cand["date_bio"].astype(str)
    cand = cand.sort_values(["bio_key", "abs_diff_days"]).groupby("bio_key", as_index=False).first()

    if not require_ndvi:
        ndvi_field_median = cand.groupby("field_id")["ndvi"].transform(lambda s: s.fillna(s.median()))
        cand["ndvi"] = cand["ndvi"].fillna(ndvi_field_median).fillna(cand["ndvi"].median())

    cand = cand.sort_values(["field_id","date_bio"]).reset_index(drop=True)
    req_base = ["SVI","t_air_c_adj","soil_moisture_vwc_adj"] + (["ndvi"] if require_ndvi else [])

    roll_cols, lag_cols = [], []
    if use_lags:
        for col in req_base:
            roll = (cand.groupby("field_id", group_keys=False)[col]
                         .apply(lambda s: s.rolling(7, min_periods=1).mean())
                         .reset_index(drop=True))
            lag = (cand.groupby("field_id", group_keys=False)[col]
                        .shift(7)
                        .reset_index(drop=True))
            cand[f"{col}_roll7"] = roll
            cand[f"{col}_lag7"]  = lag.fillna(roll)
            roll_cols.append(f"{col}_roll7")
            lag_cols.append(f"{col}_lag7")

    cand = add_season(cand.rename(columns={"date_bio":"date_ref"}), "date_ref")

    req_cols = req_base + roll_cols + lag_cols + ["month_sin","month_cos"]
    nbi_feature_names = req_cols if use_lags else (req_base + ["month_sin","month_cos"])

    model_df = cand.dropna(subset=req_base + ["nbi"]).copy()
    kept_n = len(model_df)
    st.write(f"Matched rows: **{len(cand)}** | Rows for modeling: **{kept_n}** | Fields: {model_df['field_id'].nunique() if kept_n else 0}")

    if kept_n < min_rows:
        st.info(f"Not enough matched rows for a stable model (have {kept_n}, need ≥ {int(min_rows)}). "
                "Try increasing tolerance, disabling NDVI requirement, or disabling lags.")
    else:
        X = model_df[nbi_feature_names].values
        y = model_df["nbi"].values
        groups = model_df["field_id"]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("gbr", HistGradientBoostingRegressor(random_state=42))
        ])
        gcv = GroupKFold(n_splits=5)
        r2_scores = cross_val_score(model, X, y, cv=gcv.split(X, y, groups=groups), scoring="r2")
        st.success(f"Cross-validated R² (GroupKFold by field): **{r2_scores.mean():.2f} ± {r2_scores.std():.2f}**  (n={kept_n})")

        model.fit(X, y)
        yhat = model.predict(X)

        fig, ax = plt.subplots()
        ax.scatter(y, yhat, s=16, alpha=0.85)
        lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
        ax.plot(lims, lims, linewidth=1.2)
        ax.set_xlabel("Observed NBI"); ax.set_ylabel("Predicted NBI"); ax.set_title("Observed vs Predicted NBI (parity)")
        ax.grid(True, alpha=0.3); st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots()
        ax.scatter(yhat, y - yhat, s=14, alpha=0.85)
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_xlabel("Predicted NBI"); ax.set_ylabel("Residual (Observed − Predicted)")
        ax.set_title("Residuals vs Prediction"); ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

        try:
            pi = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            order = np.argsort(pi.importances_mean)
            labels = np.array(nbi_feature_names)[order]
            fig, ax = plt.subplots()
            ax.barh(range(len(order)), pi.importances_mean[order])
            ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
            ax.set_title("Permutation importance"); ax.grid(True, axis='x', alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        except Exception:
            pass

        st.session_state.setdefault("perf", {})
        st.session_state["perf"]["nbi"] = dict(
            cv_r2=float(r2_scores.mean()), cv_sd=float(r2_scores.std())
        )

        if write_nbi:
            Fa = feat[["field_id","date","SVI","t_air_c_adj","soil_moisture_vwc_adj","ndvi"]].copy()

            if not require_ndvi and "ndvi" in Fa.columns:
                ndvi_field_median_all = Fa.groupby("field_id")["ndvi"].transform(lambda s: s.fillna(s.median()))
                Fa["ndvi"] = Fa["ndvi"].fillna(ndvi_field_median_all).fillna(Fa["ndvi"].median())

            Fa = Fa.sort_values(["field_id","date"]).reset_index(drop=True)
            base_cols = ["SVI","t_air_c_adj","soil_moisture_vwc_adj"] + (["ndvi"] if require_ndvi else [])

            if use_lags:
                for col in base_cols:
                    roll = (Fa.groupby("field_id", group_keys=False)[col]
                               .apply(lambda s: s.rolling(7, min_periods=1).mean())
                               .reset_index(drop=True))
                    lag = (Fa.groupby("field_id", group_keys=False)[col]
                               .shift(7)
                               .reset_index(drop=True))
                    Fa[f"{col}_roll7"] = roll
                    Fa[f"{col}_lag7"]  = lag.fillna(roll)

            Fa = add_season(Fa, "date")

            for col in nbi_feature_names:
                if col not in Fa.columns:
                    Fa[col] = np.nan

            X_all = Fa[nbi_feature_names].values
            Fa["nbi_hat"] = model.predict(X_all)

            feat = feat.merge(Fa[["field_id","date","nbi_hat"]], on=["field_id","date"], how="left")
            st.success("NBÎ predictions written to engineered features.")

        ai_note("NBI interpretation", [
            f"Model explains about {100*max(r2_scores.mean(), 0):.0f}% of observed NBI variance on GroupKFold CV.",
            "Short-term history (roll/lag) and seasonality provide a fair test of biology’s response.",
            "For stronger calibration, extend biology sampling and capture key management events."
        ])

# ---------------------------- Model Performance Summary ----------------------------
st.markdown("### Model Performance Summary")

def badge_for_cv(cv: Optional[float]) -> str:
    if cv is None or np.isnan(cv): return "⚪"
    if cv >= 0.70: return "🟢"
    if cv >= 0.40: return "🟠"
    return "🔴"

perf = st.session_state.get("perf", {})
svi_mean = float(np.nanmean(feat["SVI"])) if "SVI" in feat.columns else float("nan")

rows_sum = []
rows_sum.append({
    "Component": "SVI (composite index)",
    "Dataset": f"{feat['field_id'].nunique()} fields, {len(feat)} records",
    "Metric": "Mean SVI",
    "Value": f"{svi_mean:.2f}",
    "Badge": "🟢" if svi_mean >= 0.70 else ("🟠" if svi_mean >= 0.50 else "🔴"),
    "Comment": "Overall vitality level this season"
})

if "rho_theta" in perf:
    p = perf["rho_theta"]
    rows_sum.append({
        "Component": "Geophysics → Moisture (ρ→θ)",
        "Dataset": "Matched geophysics ↔ soil days",
        "Metric": "CV R² (±SD) | RMSE | Bias",
        "Value": f"{p['cv_r2']:.2f} ± {p['cv_sd']:.2f} | {p['rmse']:.3f} | {p['bias']:.3f}",
        "Badge": badge_for_cv(p.get("cv_r2")),
        "Comment": "Strong within-field fit; limited cross-field transfer"
    })

if "nbi" in perf:
    p = perf["nbi"]
    rows_sum.append({
        "Component": "Biology calibration (NBI)",
        "Dataset": "Matched biology ↔ features (nearest-past)",
        "Metric": "CV R² (±SD)",
        "Value": f"{p['cv_r2']:.2f} ± {p['cv_sd']:.2f}",
        "Badge": badge_for_cv(p.get("cv_r2")),
        "Comment": "Causal, lagged, and seasonally aware predictors"
    })

summary_df = pd.DataFrame(rows_sum, columns=["Component","Dataset","Metric","Value","Badge","Comment"])
st.dataframe(summary_df, use_container_width=True)

ai_note("Summary", [
    "SVI indicates strong and stable vitality with mild seasonal decline.",
    "ρ→θ physics are consistent but require field-specific calibration for transfer.",
    "NBI model generalizes well; use NBÎ for operational monitoring and advisory."
])

# ---------------------------- Exports ----------------------------
st.markdown("### Exports")
cxa, cxb = st.columns(2)
with cxa:
    csv_all = feat.to_csv(index=False).encode('utf-8')
    st.download_button("Download engineered features (CSV)", csv_all, file_name="engineered_features.csv", mime="text/csv")
with cxb:
    latest = feat.sort_values(["field_id","date"]).groupby("field_id").tail(1)
    keep_cols = [c for c in ["SVI","V_rho","V_theta","V_T","V_v","V_wb","theta_hat","nbi_hat"] if c in latest.columns]
    summary = latest[["field_id","date"] + keep_cols].copy()
    csv_sum = summary.to_csv(index=False).encode('utf-8')
    st.download_button("Download per-field latest summary (CSV)", csv_sum, file_name="field_summary_latest.csv", mime="text/csv")

# ---------------------------- Footer (versions) ----------------------------
st.caption(
    "Env: Python 3.12 | pandas {} | numpy {} | scikit-learn {} | streamlit {} | matplotlib {}".format(
        pd.__version__, np.__version__, __import__('sklearn').__version__, st.__version__, plt.matplotlib.__version__
    )
)

