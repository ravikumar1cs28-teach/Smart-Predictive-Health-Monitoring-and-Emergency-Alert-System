"""Feature extraction for the three body-system modules (bone, muscle, skin)."""
import numpy as np
import pandas as pd

BONE_COLS = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
             "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]

EMG_WINDOW = 200   # samples per EMG window (0.2 s at 1000 Hz)
SKIN_WINDOW = 40   # samples per skin window (10 s at 4 Hz)


def _slope(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])


# ---------------- BONE ----------------
def clean_bone_df(df):
    """Normalise column names of the UCI/Kaggle Vertebral Column csv."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]
    return df


def bone_xy(df):
    df = clean_bone_df(df)
    X = df[BONE_COLS].astype(float).values
    y = (df["class"].astype(str).str.strip().str.lower() != "normal").astype(int).values
    return X, y


# ---------------- MUSCLE (EMG) ----------------
def emg_features(window):
    """window: array (n_samples, n_channels) -> feature vector (5 per channel)."""
    w = np.asarray(window, dtype=float)
    feats = []
    for c in range(w.shape[1]):
        s = w[:, c]
        rms = np.sqrt(np.mean(s ** 2))
        mav = np.mean(np.abs(s))
        wl = np.sum(np.abs(np.diff(s)))                       # waveform length
        zc = np.sum(s[:-1] * s[1:] < 0)                       # zero crossings
        d = np.diff(s)
        ssc = np.sum(d[:-1] * d[1:] < 0)                      # slope sign changes
        feats += [rms, mav, wl, zc, ssc]
    return np.array(feats)


def muscle_xy(df, window=EMG_WINDOW):
    """df columns: ch1..chN, label (0 healthy / 1 abnormal), optional subject."""
    ch = [c for c in df.columns if c.lower().startswith("ch")]
    X, y, g = [], [], []
    group_col = "subject" if "subject" in df.columns else None
    for key, part in (df.groupby(group_col) if group_col else [(0, df)]):
        for lab, sub in part.groupby("label"):
            arr = sub[ch].values
            for i in range(0, len(arr) - window + 1, window):
                X.append(emg_features(arr[i:i + window]))
                y.append(int(lab))
                g.append(key)
    return np.array(X), np.array(y), np.array(g)


# ---------------- SKIN (EDA / temperature / BVP) ----------------
def skin_features(eda, temp, bvp=None):
    # bvp is accepted but not used, so the ESP32 only needs a GSR and a temperature sensor
    f = [np.mean(eda), np.std(eda), _slope(eda), np.max(eda) - np.min(eda),
         np.mean(temp), _slope(temp)]
    return np.array(f, dtype=float)


def skin_xy(df, window=SKIN_WINDOW):
    """df columns: eda, temp, [bvp], label (0 no pain / 1 pain), optional subject."""
    has_bvp = "bvp" in df.columns
    X, y, g = [], [], []
    group_col = "subject" if "subject" in df.columns else None
    for key, part in (df.groupby(group_col) if group_col else [(0, df)]):
        for lab, sub in part.groupby("label"):
            eda, temp = sub["eda"].values, sub["temp"].values
            bvp = sub["bvp"].values if has_bvp else None
            for i in range(0, len(eda) - window + 1, window):
                sl = slice(i, i + window)
                X.append(skin_features(eda[sl], temp[sl], bvp[sl] if has_bvp else None))
                y.append(int(lab))
                g.append(key)
    return np.array(X), np.array(y), np.array(g)
