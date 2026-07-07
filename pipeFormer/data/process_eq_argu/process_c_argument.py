
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["embed_from_excel", "embed_from_dataframe", "physics_normalize", "build_shape_vector"]

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        c2 = str(c).strip().replace("\n", "")
        if "水头" in c2 and ("m" in c2 or "M" in c2):
            ren[c] = "head_m"
        elif "效率" in c2:
            ren[c] = "eff"
        elif ("流量" in c2) and ("m3/h" in c2 or "m³/h" in c2 or "m3h" in c2 or "/h" in c2):
            ren[c] = "flow_m3h"
        elif ("转速" in c2) and ("RPM" in c2 or "rpm" in c2):
            ren[c] = "rpm"
    return df.rename(columns=ren)

def physics_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply affinity-law normalization (no diameter available):
      q_star = Q / N, h_star = H / N^2
    Efficiency is kept as-is (0..1).
    """
    df = df.copy()
    df = _rename_columns(df)
    need = [c for c in ["head_m", "flow_m3h", "rpm"] if c in df.columns]
    df = df.dropna(subset=need)
    rpm = df["rpm"].astype(float).replace(0, np.nan)
    df["q_star"] = df["flow_m3h"].astype(float) / rpm
    df["h_star"] = df["head_m"].astype(float) / (rpm**2)
    if "eff" in df.columns:
        if (df["eff"] > 1.5).any():
            df["eff"] = df["eff"] / 100.0
        df["eff"] = df["eff"].astype(float).clip(lower=0, upper=1)
    else:
        df["eff"] = np.nan
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["q_star", "h_star"])

def _smooth_on_grid(x, y, grid, bandwidth=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if bandwidth is None:
        bandwidth = 0.1 * (np.nanmax(x) - np.nanmin(x) + 1e-12)
    out = np.empty_like(grid, dtype=float)
    for i, g in enumerate(grid):
        d = np.abs(x - g)
        w = np.exp(- (d / (bandwidth + 1e-12))**2 )
        if w.sum() < 1e-9:
            out[i] = np.nan
        else:
            out[i] = np.nansum(w * y) / np.nansum(w)
    nans = np.isnan(out)
    if nans.any():
        not_nan_idx = np.where(~nans)[0]
        if len(not_nan_idx) >= 2:
            out[nans] = np.interp(np.where(nans)[0], not_nan_idx, out[~nans])
        else:
            out[nans] = np.nanmedian(out)
    return out

def build_shape_vector(dfn: pd.DataFrame, grid_size=32):
    q = dfn["q_star"].values
    h = dfn["h_star"].values
    e = dfn["eff"].values
    # 5th..95th percentile range
    q_lo, q_hi = np.nanpercentile(q, [5, 95]) if np.isfinite(q).any() else (0.0, 1.0)
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        q_lo, q_hi = np.nanmin(q), np.nanmax(q)
    q_grid = np.linspace(q_lo, q_hi, grid_size)
    h_smooth = _smooth_on_grid(q, h, q_grid)
    e_smooth = _smooth_on_grid(q, e, q_grid)
    # Shape-only scaling
    h_shape = h_smooth / (np.nanmax(h_smooth) + 1e-12)
    e_shape = np.clip(e_smooth, 0, 1)
    # Descriptors
    e_max = float(np.nanmax(e_shape))
    idx_max = int(np.nanargmax(e_shape)) if np.isfinite(e_shape).all() else int(np.argmax(np.nan_to_num(e_shape, nan=-1)))
    q_bep = float((q_grid[idx_max] - q_lo) / (q_hi - q_lo + 1e-12))
    thr = 0.9 * e_max
    above = np.where(e_shape >= thr)[0]
    band = float((above[-1] - above[0]) / (grid_size - 1)) if len(above) >= 2 else 0.0
    qn = (q_grid - q_lo) / (q_hi - q_lo + 1e-12)
    dhdq = np.gradient(h_shape, qn)
    slope_bep = float(dhdq[idx_max]) if np.isfinite(dhdq[idx_max]) else 0.0
    vec = np.concatenate([h_shape, e_shape, np.array([e_max, q_bep, band, slope_bep])])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, {"q_lo": float(q_lo), "q_hi": float(q_hi), "grid": q_grid}

def embed_from_dataframe(df: pd.DataFrame, grid_size=32) -> np.ndarray:
    dfn = physics_normalize(df)
    vec, _ = build_shape_vector(dfn, grid_size=grid_size)
    return vec

def _pca(X, k=8):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, Vt[:k], S[:k], X.mean(axis=0)

def embed_from_excel(excel_path: str, k=8, grid_size=32):
    xls = pd.ExcelFile(excel_path)
    sheets = [s for s in xls.sheet_names if s.startswith("C_")]
    vecs = []
    ids = []
    for s in sheets:
        df = pd.read_excel(excel_path, sheet_name=s).dropna(how="all").dropna(axis=1, how="all")
        df = df[[c for c in df.columns if not (isinstance(c, str) and "Unnamed" in c)]]
        if not any("转速" in str(c) for c in df.columns):
            continue
        vec = embed_from_dataframe(df, grid_size=grid_size)
        vecs.append(vec); ids.append(s)
    X = np.vstack(vecs)
    Xz = (X - X.mean(axis=0, keepdims=True))/(X.std(axis=0, keepdims=True)+1e-12)
    Z, Vt, S, mu = _pca(Xz, k=k)
    embed_df = pd.DataFrame(Z, columns=[f"pc{i+1}" for i in range(Z.shape[1])])
    embed_df.insert(0, "id", ids)
    return embed_df
