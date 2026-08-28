#!/usr/bin/env python3
"""Validate Mairan T DEMs against one official LOLA RDR track."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates


WEST, EAST, SOUTH, NORTH = 311.4, 311.8, 41.67, 41.93
MOON_RADIUS_M = 1_737_400.0
RECORD_BYTES = 256


def read_lola(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    records = raw.size // RECORD_BYTES
    if records == 0:
        raise ValueError("Empty LOLA file")
    usable_bytes = records * RECORD_BYTES
    raw = raw[:usable_bytes]
    longitude, latitude, elevation, detector = [], [], [], []
    for spot in range(5):
        base = 40 + 40 * spot
        lon_i = np.ndarray((records,), dtype="<i4", buffer=raw, offset=base, strides=(RECORD_BYTES,))
        lat_i = np.ndarray((records,), dtype="<i4", buffer=raw, offset=base + 4, strides=(RECORD_BYTES,))
        radius_i = np.ndarray((records,), dtype="<i4", buffer=raw, offset=base + 8, strides=(RECORD_BYTES,))
        flag_i = np.ndarray((records,), dtype="<u4", buffer=raw, offset=base + 36, strides=(RECORD_BYTES,))
        lon = lon_i.astype(np.float64) * 1.0e-7
        lon[lon < 0.0] += 360.0
        lat = lat_i.astype(np.float64) * 1.0e-7
        elev = radius_i.astype(np.float64) * 1.0e-3 - MOON_RADIUS_M
        valid = (
            (lon_i != np.iinfo(np.int32).min)
            & (lat_i != np.iinfo(np.int32).min)
            & (radius_i != -1)
            & ((flag_i & 0xFF) == 0)
            & (lon >= WEST)
            & (lon <= EAST)
            & (lat >= SOUTH)
            & (lat <= NORTH)
        )
        longitude.append(lon[valid])
        latitude.append(lat[valid])
        elevation.append(elev[valid])
        detector.append(np.full(int(valid.sum()), spot + 1, dtype=np.int16))
    return tuple(np.concatenate(values) for values in (longitude, latitude, elevation, detector))


def average_per_pixel(
    longitude: np.ndarray, latitude: np.ndarray, elevation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col = np.floor((longitude - WEST) * 300.0).astype(int)
    row = np.floor((NORTH - latitude) * 300.0).astype(int)
    key = row * 120 + col
    unique, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse)
    lon = np.bincount(inverse, weights=longitude) / count
    lat = np.bincount(inverse, weights=latitude) / count
    elev = np.bincount(inverse, weights=elevation) / count
    return lon, lat, elev, count


def read_dem(path: Path) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as source:
        dem = source.read(1).astype(np.float64)
        transform = source.transform
        nodata = source.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, transform


def sample_dem(dem: np.ndarray, transform: object, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    col = (lon - transform.c) / transform.a - 0.5
    row = (lat - transform.f) / transform.e - 0.5
    return map_coordinates(dem, [row, col], order=1, mode="constant", cval=np.nan)


def align_track(
    dem: np.ndarray,
    transform: object,
    lon: np.ndarray,
    lat: np.ndarray,
    lola: np.ndarray,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for north_shift in range(-15, 16):
        for east_shift in range(-15, 16):
            shifted_lon = lon + east_shift / 300.0
            shifted_lat = lat + north_shift / 300.0
            modeled = sample_dem(dem, transform, shifted_lon, shifted_lat)
            valid = np.isfinite(modeled) & np.isfinite(lola)
            if valid.sum() < 3:
                continue
            modeled_centered = modeled[valid] - np.mean(modeled[valid])
            lola_centered = lola[valid] - np.mean(lola[valid])
            residual = modeled_centered - lola_centered
            rmse = float(np.sqrt(np.mean(residual**2)))
            candidate = {
                "east_shift_px": east_shift,
                "north_shift_px": north_shift,
                "shift_norm_px": float(np.hypot(east_shift, north_shift)),
                "rmse_m": rmse,
                "bias_removed_m": float(np.mean(modeled[valid] - lola[valid])),
                "point_count": int(valid.sum()),
                "modeled": modeled,
                "valid": valid,
            }
            if best is None or rmse < float(best["rmse_m"]):
                best = candidate
    if best is None:
        raise RuntimeError("No valid DEM samples for this LOLA track")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rdr", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "05_validation" / "lola_092661826_validation"
    output.mkdir(parents=True, exist_ok=True)
    lon_raw, lat_raw, elev_raw, detector = read_lola(args.rdr.resolve())
    if elev_raw.size == 0:
        raise RuntimeError("This RDR has no quality-flag-zero LOLA points inside Mairan T")
    lon, lat, elev, count = average_per_pixel(lon_raw, lat_raw, elev_raw)

    paths = {
        "GLD100": root / "02_preprocessed" / "GLD100_MairanT_300ppd.tif",
        "Grumpe_AMSA": root / "04_results" / "mairan_t_amsa" / "MairanT_GRUMPE_AMSA_DEM.tif",
        "Horn_Poisson": root / "04_results" / "mairan_t_amsa" / "MairanT_Horn_Poisson_DEM.tif",
    }
    alignments: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        dem, transform = read_dem(path)
        alignments[name] = align_track(dem, transform, lon, lat, elev)

    metrics = {
        "rdr_product": args.rdr.stem,
        "raw_quality_flag_zero_points": int(elev_raw.size),
        "image_pixel_averaged_points": int(elev.size),
        "paper_rule": "each model independently searched +/-15 pixels; tracks with optimal shift norm >5 pixels are rejected",
        "models": {},
    }
    for name, result in alignments.items():
        metrics["models"][name] = {
            key: value
            for key, value in result.items()
            if key not in {"modeled", "valid"}
        }
        metrics["models"][name]["accepted_by_paper_shift_rule"] = bool(
            float(result["shift_norm_px"]) <= 5.0
        )
    (output / "lola_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output / "lola_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["longitude_east", "latitude_north", "elevation_m", "shots_in_pixel"])
        writer.writerows(zip(lon, lat, elev, count))

    order = np.argsort(lat)
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(lat[order], elev[order] - np.mean(elev), "k.-", label="LOLA (demeaned)")
    for name, result in alignments.items():
        modeled = np.asarray(result["modeled"])
        valid = np.asarray(result["valid"])
        centered = modeled - np.mean(modeled[valid])
        axis.plot(lat[order], centered[order], label=f"{name}, RMSE={result['rmse_m']:.1f} m")
    axis.set_xlabel("Latitude (degrees north)")
    axis.set_ylabel("Demeaned elevation (m)")
    axis.set_title("Mairan T: LOLA single-track validation")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(output / "lola_profile_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
