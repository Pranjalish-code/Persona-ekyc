# calibrate_thresholds.py
import json
import numpy as np
import pandas as pd

# ----------------------------
# Config (edit if you want)
# ----------------------------
CSV_PATH = "liveness_features.csv"     # your collected features CSV
LABEL_COL = "label"                   # expects "REAL"/"SPOOF"
SCORE_COL = "score"                   # final fused liveness score

TARGET_FAR = 0.01                     # 1% spoof accepted as live
REAL_UPPER_PCTL = 95                  # for "max allowed" gates on replay-ish features
REAL_LOWER_PCTL = 20                  # for "min required" gates on quality features

# Column names (from your collector script)
SCREEN_COL = "info.screen_fft_score"
DUP_COL = "info.dup_frame_rate"
QRPPG_COL = "info.Q_rppg"
SNR_COL = "info.snr_db"
CORR_COL = "info.corr_lr"

# Optional: you may have these too
PMOTION_COL = "comp.P_motion"
PSTATIC_COL = "comp.P_static"
PRPPG_COL = "comp.P_rppg"


def _as_float_series(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s


def roc_threshold_far(scores_real, scores_spoof, target_far=0.01):
    """
    Choose threshold such that FAR <= target_far.
    FAR = fraction of spoof scores >= threshold.
    """
    scores_spoof = np.asarray(scores_spoof, dtype=np.float64)
    scores_spoof = scores_spoof[np.isfinite(scores_spoof)]
    if len(scores_spoof) == 0:
        raise ValueError("No valid SPOOF scores found for ROC calibration.")

    # Threshold at (1 - FAR) quantile of spoof scores
    thr = float(np.quantile(scores_spoof, 1.0 - target_far))
    return thr


def pct(series, p):
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return None
    return float(np.percentile(series, p))


def main():
    df = pd.read_csv(CSV_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing '{LABEL_COL}' column in CSV.")
    if SCORE_COL not in df.columns:
        raise ValueError(f"Missing '{SCORE_COL}' column in CSV.")

    # Normalize labels
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper().str.strip()
    real = df[df[LABEL_COL] == "REAL"].copy()
    spoof = df[df[LABEL_COL] == "SPOOF"].copy()

    if len(real) == 0 or len(spoof) == 0:
        raise ValueError("Need both REAL and SPOOF samples in the CSV.")

    # Core score threshold (tau_live) from FAR target
    tau_live = roc_threshold_far(
        scores_real=real[SCORE_COL].values,
        scores_spoof=spoof[SCORE_COL].values,
        target_far=TARGET_FAR
    )

    # Compute gates from REAL percentiles
    thresholds = {
        "target_far": TARGET_FAR,
        "tau_live": float(tau_live),
    }

    # Screen FFT max (replay gate) - HIGH values are suspicious
    s_screen_r = _as_float_series(real, SCREEN_COL)
    if s_screen_r is not None:
        thresholds["screen_fft_max"] = pct(s_screen_r, REAL_UPPER_PCTL)

    # Duplicate frame rate max (replay/stutter gate) - HIGH suspicious
    s_dup_r = _as_float_series(real, DUP_COL)
    if s_dup_r is not None:
        thresholds["dup_rate_max"] = pct(s_dup_r, REAL_UPPER_PCTL)

    # rPPG quality minimum - LOW values mean unreliable rPPG
    s_qrppg_r = _as_float_series(real, QRPPG_COL)
    if s_qrppg_r is not None:
        thresholds["Q_rppg_min"] = pct(s_qrppg_r, REAL_LOWER_PCTL)

    # SNR minimum for trusting rPPG
    s_snr_r = _as_float_series(real, SNR_COL)
    if s_snr_r is not None:
        thresholds["snr_db_min"] = pct(s_snr_r, REAL_LOWER_PCTL)

    # L/R coherence minimum
    s_corr_r = _as_float_series(real, CORR_COL)
    if s_corr_r is not None:
        thresholds["corr_lr_min"] = pct(s_corr_r, REAL_LOWER_PCTL)

    # Optional “support thresholds” from REAL distribution (useful for gray-zone logic)
    for name, col, p in [
        ("P_motion_strong", PMOTION_COL, 50),   # median motion for "strong motion"
        ("P_static_strong", PSTATIC_COL, 50),
        ("P_rppg_strong", PRPPG_COL, 50),
    ]:
        s = _as_float_series(real, col)
        if s is not None:
            thresholds[name] = pct(s, p)

    # ----------------------------
    # Gray-zone lower bound
    # ----------------------------
    # A simple heuristic:
    # - Let gray_low be the 10th percentile of REAL scores (keeps most REAL),
    #   but not higher than tau_live.
    gray_low = pct(real[SCORE_COL].values, 10)
    if gray_low is not None:
        thresholds["gray_low"] = float(min(gray_low, tau_live))

    # Print quick diagnostics
    def far_at(thr):
        return float(np.mean(spoof[SCORE_COL].values >= thr))

    def frr_at(thr):
        return float(np.mean(real[SCORE_COL].values < thr))

    print("\n================ Calibration Summary ================")
    print(f"REAL samples : {len(real)}")
    print(f"SPOOF samples: {len(spoof)}\n")

    print(f"tau_live (FAR target={TARGET_FAR:.4f}): {tau_live:.6f}")
    print(f"Observed FAR @ tau_live: {far_at(tau_live):.4f}")
    print(f"Observed FRR @ tau_live: {frr_at(tau_live):.4f}")

    if "gray_low" in thresholds:
        gl = thresholds["gray_low"]
        print(f"\ngray_low: {gl:.6f}")
        print(f"Observed FAR @ gray_low: {far_at(gl):.4f}")
        print(f"Observed FRR @ gray_low: {frr_at(gl):.4f}")

    # Save thresholds to JSON
    out_path = "calibrated_thresholds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✅ Saved thresholds to: {out_path}")
    print("Thresholds:", json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
