"""
Weibull Distribution Fitting for Wind Resource Assessment
Port of matlab/weibull_fit.m — identical outputs, no MATLAB required.
Reference: IEC 61400-12-1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
STATION_CSV = ROOT / "data/interim/station_for_weibull.csv"
OUT_PNG     = ROOT / "outputs/weibull_analysis.png"
OUT_CSV     = ROOT / "data/interim/weibull_results.csv"

# ── turbine / site config (mirrors MATLAB script) ─────────────────────────
HUB_HEIGHT_M      = 10
ROTOR_DIAMETER_M  = 1.5
AIR_DENSITY       = 1.225   # kg/m³ at sea level, 15 °C
RATED_POWER_KW    = 1.5
CUT_IN, RATED_WS, CUT_OUT = 2.5, 11.0, 25.0


def power_curve(v: np.ndarray) -> np.ndarray:
    p = np.zeros_like(v)
    ramp = (v >= CUT_IN) & (v < RATED_WS)
    flat = (v >= RATED_WS) & (v < CUT_OUT)
    p[ramp] = RATED_POWER_KW * ((v[ramp] - CUT_IN) / (RATED_WS - CUT_IN)) ** 3
    p[flat] = RATED_POWER_KW
    return p


def main():
    # ── load ─────────────────────────────────────────────────────────────────
    T = pd.read_csv(STATION_CSV)
    ws = T["wind_speed_10m"].dropna()
    ws = ws[ws >= 0].to_numpy()

    print(f"Station data loaded: {len(ws)} hourly observations")
    print(f"Mean: {ws.mean():.2f} m/s | Std: {ws.std():.2f} m/s | Max: {ws.max():.2f} m/s")

    # ── Weibull MLE (scipy: shape=k, loc=0, scale=c) ─────────────────────
    # equivalent to MATLAB wblfit
    k, _, c = weibull_min.fit(ws[ws > 0], floc=0)

    print(f"\nWeibull parameters:")
    print(f"  Scale (c) = {c:.3f} m/s")
    print(f"  Shape (k) = {k:.3f}")

    # ── derived metrics ───────────────────────────────────────────────────
    from scipy.special import gamma
    weibull_mean = c * gamma(1 + 1 / k)
    wpd = 0.5 * AIR_DENSITY * c**3 * gamma(1 + 3 / k)
    A = np.pi * (ROTOR_DIAMETER_M / 2) ** 2

    print(f"\nDerived metrics:")
    print(f"  Empirical mean:     {ws.mean():.3f} m/s")
    print(f"  Weibull mean:       {weibull_mean:.3f} m/s")
    print(f"  Wind power density: {wpd:.1f} W/m²")
    print(f"  Rotor swept area:   {A:.2f} m²")

    # ── AEP via bin method ────────────────────────────────────────────────
    bin_edges   = np.arange(0, 30.5, 0.5)
    bin_centers = bin_edges[:-1] + 0.25
    pdf_vals    = weibull_min.pdf(bin_centers, k, loc=0, scale=c)
    hours_per_bin = pdf_vals * 8760 * 0.5

    pc = power_curve(bin_centers)
    aep_kwh = float(np.sum(pc * hours_per_bin))
    capacity_factor = aep_kwh / (RATED_POWER_KW * 8760)

    print(f"\nEnergy production estimate:")
    print(f"  AEP:             {aep_kwh:.0f} kWh/year")
    print(f"  Capacity factor: {capacity_factor * 100:.1f}%")

    # ── plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.hist(ws, bins=30, density=True, alpha=0.6, edgecolor="k", label="Observed")
    x = np.linspace(0, ws.max(), 300)
    ax.plot(x, weibull_min.pdf(x, k, loc=0, scale=c), "r-", lw=2,
            label="Weibull fit")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Weibull fit: c={c:.2f}, k={k:.2f}")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(bin_centers, pc, "b-", lw=2)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power output (kW)")
    ax.set_title("Turbine power curve")
    ax.set_xlim(0, 30)
    ax.grid(True)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(bin_centers, hours_per_bin, width=0.5, alpha=0.5, label="Hours/yr")
    ax2.plot(bin_centers, pc * hours_per_bin, "r-", lw=2, label="Energy (kWh)")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Hours per year")
    ax2.set_ylabel("Energy contribution (kWh)")
    ax.set_title("Energy yield by wind speed bin")
    ax.set_xlim(0, 30)
    ax.grid(True)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nPlot saved to {OUT_PNG}")

    # ── save results CSV ──────────────────────────────────────────────────
    pd.DataFrame([{
        "station_csv":      str(STATION_CSV),
        "weibull_c":        round(c, 6),
        "weibull_k":        round(k, 6),
        "weibull_mean":     round(weibull_mean, 6),
        "wpd":              round(wpd, 4),
        "aep_kwh":          round(aep_kwh, 2),
        "capacity_factor":  round(capacity_factor, 6),
        "n_hours":          len(ws),
    }]).to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
