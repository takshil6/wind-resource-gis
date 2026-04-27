"""Wind power calculations: power density, capacity factor, AEP.

Industry standard formulas:
- Wind power density (W/m²) = 0.5 * ρ * v³ * Γ(1 + 3/k)   [Weibull-aware]
- Simple form: 0.5 * ρ * v³                               [point-in-time]
- Capacity factor = AEP / (rated_power * 8760 hr)
"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from src.config import (
    AIR_DENSITY,
    HUB_HEIGHT_M,
    RATED_POWER_KW,
    CUT_IN_SPEED,
    RATED_SPEED,
    CUT_OUT_SPEED,
)


def power_density_simple(v: np.ndarray, rho: float = AIR_DENSITY) -> np.ndarray:
    """Instantaneous wind power density (W/m²) given mean wind speed.

    Note: this underestimates true energy because v³ averaged != (avg v)³.
    Use power_density_weibull for accurate annual estimates.
    """
    return 0.5 * rho * np.power(v, 3)


def power_density_weibull(
    c: np.ndarray, k: np.ndarray, rho: float = AIR_DENSITY
) -> np.ndarray:
    """Weibull-corrected power density (W/m²).

    Properly accounts for the fact that energy in higher-than-mean winds
    dominates the annual total. Typical correction factor: 1.5-2.5x simple.
    """
    return 0.5 * rho * np.power(c, 3) * gamma(1 + 3 / k)


def estimate_weibull_from_mean_std(
    mean_v: np.ndarray, std_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Weibull (c, k) from mean and std using Justus method.

    Standard wind industry shortcut when full hourly data isn't available.
    Reference: Justus & Mikhail (1976).
    """
    cv = np.where(mean_v > 0, std_v / mean_v, np.nan)  # coefficient of variation
    # Justus empirical formula: k ≈ (σ/μ)^(-1.086), valid for 1 <= k <= 10
    k = np.power(cv, -1.086)
    k = np.clip(k, 1.0, 10.0)
    c = mean_v / gamma(1 + 1 / k)
    return c, k


def power_curve(v: np.ndarray) -> np.ndarray:
    """Simple turbine power curve (cubic ramp) returning kW output.

    Uses Accelerate Wind-style 1.5 kW rooftop turbine spec from config.
    """
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= CUT_IN_SPEED) & (v < RATED_SPEED)
    rated = (v >= RATED_SPEED) & (v < CUT_OUT_SPEED)
    p[ramp] = RATED_POWER_KW * np.power(
        (v[ramp] - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED), 3
    )
    p[rated] = RATED_POWER_KW
    return p


def capacity_factor_from_weibull(
    c: np.ndarray, k: np.ndarray, n_bins: int = 60
) -> np.ndarray:
    """Capacity factor (0-1) using bin method over Weibull PDF.

    Vectorized over arbitrary-shape arrays of (c, k).
    """
    from scipy.stats import weibull_min

    c = np.asarray(c)
    k = np.asarray(k)
    out = np.zeros_like(c, dtype=float)

    bins = np.linspace(0, 30, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    pc = power_curve(centers)  # kW per bin

    flat_c = c.ravel()
    flat_k = k.ravel()
    flat_out = np.zeros_like(flat_c)

    for i in range(len(flat_c)):
        ci, ki = flat_c[i], flat_k[i]
        if not (np.isfinite(ci) and np.isfinite(ki)) or ci <= 0:
            flat_out[i] = np.nan
            continue
        pdf = weibull_min.pdf(centers, ki, scale=ci)
        hours = pdf * 8760 * width
        aep = (pc * hours).sum()  # kWh/yr
        flat_out[i] = aep / (RATED_POWER_KW * 8760)

    return flat_out.reshape(c.shape)