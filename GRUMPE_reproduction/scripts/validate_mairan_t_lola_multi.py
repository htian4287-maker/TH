#!/usr/bin/env python3
"""Track-wise LOLA validation for all downloaded Mairan T RDR products."""

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


def read_lola(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    records = raw.size // RECORD_BYTES
    raw = raw[: records * RECORD_BYTES]
    longitude, latitude, elevation = [], [], []
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
            (lon_i != np.iinfo(np.int32).min) & (lat_i != np.iinfo(np.int32).min)
            & (radius_i != -1) & ((flag_i & 0xFF) == 0)
            & (lon >= WEST) & (lon <= EAST) & (lat >= SOUTH) & (lat <= NORTH)
        )
        longitude.append(lon[valid]); latitude.append(lat[valid]); elevation.append(elev[valid])
    return tuple(np.concatenate(values) for values in (longitude, latitude, elevation))


def average_per_pixel(lon: np.ndarray, lat: np.ndarray, elev: np.ndarray):
    col = np.floor((lon - WEST) * 300.0).astype(int)
    row = np.floor((NORTH - lat) * 300.0).astype(int)
    key = row * 120 + col
    _, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse)
    return (
        np.bincount(inverse, weights=lon) / count,
        np.bincount(inverse, weights=lat) / count,
        np.bincount(inverse, weights=elev) / count,
    )


def read_dem(path: Path):
    with rasterio.open(path) as source:
        dem = source.read(1).astype(np.float64)
        transform, nodata = source.transform, source.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, transform


def sample(dem: np.ndarray, transform, lon: np.ndarray, lat: np.ndarray):
    col = (lon - transform.c) / transform.a - 0.5
    row = (lat - transform.f) / transform.e - 0.5
    return map_coordinates(dem, [row, col], order=1, mode="constant", cval=np.nan)


def align_track(dem: np.ndarray, transform, lon: np.ndarray, lat: np.ndarray, lola: np.ndarray):
    best = None
    for north_shift in range(-15, 16):
        for east_shift in range(-15, 16):
            modeled = sample(dem, transform, lon + east_shift / 300.0, lat + north_shift / 300.0)
            valid = np.isfinite(modeled) & np.isfinite(lola)
            if valid.sum() < 3:
                continue
            residual = (modeled[valid] - np.mean(modeled[valid])) - (lola[valid] - np.mean(lola[valid]))
            rmse = float(np.sqrt(np.mean(residual**2)))
            candidate = {
                "east_shift_px": east_shift, "north_shift_px": north_shift,
                "shift_norm_px": float(np.hypot(east_shift, north_shift)),
                "rmse_m": rmse, "point_count": int(valid.sum()),
                "squared_error_sum": float(np.sum(residual**2)),
            }
            if best is None or rmse < best["rmse_m"]:
                best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rdr-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "06_complete_reproduction" / "06_lola_multitrack"
    output.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "GLD100": root / "02_preprocessed" / "GLD100_MairanT_300ppd.tif",
        "old_PHCL_CG": root / "04_results" / "mairan_t_amsa" / "MairanT_GRUMPE_AMSA_DEM.tif",
        "new_PHCL_CG": root / "06_complete_reproduction" / "04_ablations" / "phcl_only_cg_dem.tif",
        "full_registered_AMSA": root / "06_complete_reproduction" / "04_ablations" / "registered_amsa_multiscale" / "final_dem.tif",
        "full_unregistered_AMSA": root / "06_complete_reproduction" / "04_ablations" / "unregistered_amsa_multiscale" / "final_dem.tif",
        "full_registered_IMSA": root / "06_complete_reproduction" / "04_ablations" / "registered_imsa_multiscale" / "final_dem.tif",
        "full_single_scale": root / "06_complete_reproduction" / "04_ablations" / "registered_amsa_single_scale" / "final_dem.tif",
        "literal_relaxation": root / "06_complete_reproduction" / "05_integration" / "paper_literal_relaxation_dem.tif",
    }
    models = {name: read_dem(path) for name, path in model_paths.items() if path.exists()}
    rows = []
    track_counts = []
    for path in sorted(args.rdr_dir.glob("*.dat")):
        lon, lat, elev = read_lola(path)
        if elev.size == 0:
            continue
        lon, lat, elev = average_per_pixel(lon, lat, elev)
        track_counts.append({"track": path.stem, "points": int(elev.size)})
        for name, (dem, transform) in models.items():
            result = align_track(dem, transform, lon, lat, elev)
            if result is None:
                continue
            rows.append({
                "track": path.stem, "model": name, **result,
                "accepted_shift_le_5px": bool(result["shift_norm_px"] <= 5.0),
            })
    if not rows:
        raise RuntimeError("No downloaded LOLA product crosses the Mairan T ROI")
    aggregate = {}
    for name in models:
        selected = [row for row in rows if row["model"] == name and row["accepted_shift_le_5px"]]
        points = sum(int(row["point_count"]) for row in selected)
        aggregate[name] = {
            "available_tracks": sum(row["model"] == name for row in rows),
            "accepted_tracks": len(selected),
            "accepted_points": points,
            "pooled_demeaned_rmse_m": float(np.sqrt(sum(row["squared_error_sum"] for row in selected) / points)) if points else None,
            "median_track_rmse_m": float(np.median([row["rmse_m"] for row in selected])) if selected else None,
            "mean_track_rmse_m": float(np.mean([row["rmse_m"] for row in selected])) if selected else None,
        }
    result = {
        "paper_rule": "Each model and track independently searches +/-15 px; shift norm >5 px rejected; vertical bias removed per track.",
        "downloaded_rdr_files": len(list(args.rdr_dir.glob("*.dat"))),
        "crossing_tracks": len(track_counts),
        "track_counts": track_counts,
        "aggregate": aggregate,
    }
    (output / "lola_multitrack_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "lola_track_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    names = list(aggregate)
    values = [aggregate[name]["pooled_demeaned_rmse_m"] or np.nan for name in names]
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.bar(np.arange(len(names)), values, color="#2563eb")
    axis.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axis.set_ylabel("Pooled de-meaned LOLA RMSE (m)")
    axis.set_title(f"Mairan T multi-track validation ({len(track_counts)} crossing tracks)")
    axis.grid(axis="y", alpha=0.3)
    fig.savefig(output / "lola_multitrack_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
