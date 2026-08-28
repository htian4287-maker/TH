#!/usr/bin/env python3
"""Prepare the three original M3 observations for the Mairan T experiment.

The script reads byte-range subsets of the PDS3 L1B LOC/OBS/RDN products,
converts the 1578.86 nm radiance band to bidirectional reflectance L/E, and
resamples reflectance plus per-pixel illumination/view geometry to the
300-pixel/degree simple-cylindrical grid used by Grumpe & Woehler (2014).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from metpy.interpolate import natural_neighbor_to_points
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator


SCENES = (
    "M3G20090209T072710",
    "M3G20090418T151350",
    "M3G20090612T060502",
)
BOUNDS = {"west": 311.4, "east": 311.8, "south": 41.67, "north": 41.93}
PIXELS_PER_DEGREE = 300
SAMPLES = 304
RDN_BANDS = 85
OBS_BANDS = 10
LOC_BANDS = 3
TARGET_WAVELENGTH_NM = 1578.86
MOON_RADIUS_M = 1_737_400.0

GLD = {
    "west": 269.99984782287,
    "north": 59.999966182861,
    "resolution": 0.0032977832301127183,
    "samples": 27291,
    "first_row": 5403,
    "last_row": 5635,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {folder / pattern}, got {matches}")
    return matches[0]


def parse_rows(path: Path) -> tuple[int, int, int]:
    match = re.search(r"_rows_(\d+)_(\d+)\.IMG$", path.name)
    if not match:
        raise ValueError(f"Cannot parse row range from {path}")
    first, last = map(int, match.groups())
    return first, last, last - first + 1


def parse_scene_mean_au(path: Path) -> float:
    text = path.read_text(encoding="ascii", errors="replace")
    match = re.search(
        r"To-Sun Path Length subtracted from Band 6 \(au\)\s*:\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Cannot find scene-mean solar distance in {path}")
    return float(match.group(1))


def parse_wavelengths(path: Path) -> np.ndarray:
    text = path.read_text(encoding="ascii", errors="replace")
    match = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, re.I | re.S)
    if not match:
        raise RuntimeError(f"Cannot find wavelengths in {path}")
    return np.asarray(
        [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", match.group(1))],
        dtype=np.float64,
    )


def azimuth_zenith_to_enu(azimuth_deg: np.ndarray, zenith_deg: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    zenith = np.deg2rad(zenith_deg)
    sin_zenith = np.sin(zenith)
    return np.stack(
        (
            np.sin(azimuth) * sin_zenith,
            np.cos(azimuth) * sin_zenith,
            np.cos(zenith),
        ),
        axis=-1,
    )


def target_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = round((BOUNDS["east"] - BOUNDS["west"]) * PIXELS_PER_DEGREE)
    height = round((BOUNDS["north"] - BOUNDS["south"]) * PIXELS_PER_DEGREE)
    resolution = 1.0 / PIXELS_PER_DEGREE
    longitudes = BOUNDS["west"] + (np.arange(width) + 0.5) * resolution
    latitudes = BOUNDS["north"] - (np.arange(height) + 0.5) * resolution
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    xi = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    return lon_grid, lat_grid, xi


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norm > 0.0, norm, np.nan)


def interpolate_scene(
    root: Path,
    scene: str,
    xi: np.ndarray,
    solar_wavelengths: np.ndarray,
    solar_irradiance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    folder = root / "01_raw" / scene
    loc_path = find_one(folder, f"{scene}_V03_LOC_rows_*.IMG")
    obs_path = find_one(folder, f"{scene}_V03_OBS_rows_*.IMG")
    rdn_path = find_one(folder, f"{scene}_V03_RDN_rows_*.IMG")
    first_row, last_row, rows = parse_rows(loc_path)
    if parse_rows(obs_path) != (first_row, last_row, rows) or parse_rows(rdn_path) != (
        first_row,
        last_row,
        rows,
    ):
        raise RuntimeError(f"LOC/OBS/RDN row ranges differ for {scene}")

    loc = np.fromfile(loc_path, dtype="<f8").reshape(rows, LOC_BANDS, SAMPLES)
    obs = np.fromfile(obs_path, dtype="<f4").reshape(rows, OBS_BANDS, SAMPLES)
    rdn = np.fromfile(rdn_path, dtype="<f4").reshape(rows, RDN_BANDS, SAMPLES)
    wavelengths = parse_wavelengths(folder / f"{scene}_V03_RDN.HDR")
    band_index = int(np.argmin(np.abs(wavelengths - TARGET_WAVELENGTH_NM)))
    wavelength = float(wavelengths[band_index])
    solar_index = int(np.argmin(np.abs(solar_wavelengths - wavelength)))
    irradiance = float(solar_irradiance[solar_index])
    scene_mean_au = parse_scene_mean_au(folder / f"{scene}_V03_OBS.HDR")

    longitude = loc[:, 0, :]
    latitude = loc[:, 1, :]
    radiance = rdn[:, band_index, :].astype(np.float64)
    distance_au = scene_mean_au + obs[:, 5, :].astype(np.float64)
    # The implemented Hapke equation returns bidirectional reflectance L/E,
    # hence no pi factor is applied (pi*L/E would instead be radiance factor I/F).
    reflectance = radiance * distance_au**2 / irradiance
    sun = azimuth_zenith_to_enu(obs[:, 0, :], obs[:, 1, :])
    view = azimuth_zenith_to_enu(obs[:, 2, :], obs[:, 3, :])

    margin = 0.12
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(reflectance)
        & (reflectance > 0.0)
        & (longitude >= BOUNDS["west"] - margin)
        & (longitude <= BOUNDS["east"] + margin)
        & (latitude >= BOUNDS["south"] - margin)
        & (latitude <= BOUNDS["north"] + margin)
        & np.all(np.isfinite(sun), axis=-1)
        & np.all(np.isfinite(view), axis=-1)
    )
    points = np.column_stack((longitude[valid], latitude[valid]))
    values = np.column_stack((reflectance[valid], sun[valid], view[valid]))
    # Exact duplicate geolocations can make Qhull degenerate. Preserve the first
    # occurrence after rounding well below the target grid spacing.
    _, unique_indices = np.unique(np.round(points, 9), axis=0, return_index=True)
    points = points[unique_indices]
    values = values[unique_indices]
    # MetPy's circumcenter calculation loses precision when operated directly
    # on longitudes near 311 degrees. Translation is geometry-preserving and
    # prevents the third scene from collapsing to nearly two constant values.
    local_origin = np.array(
        [0.5 * (BOUNDS["west"] + BOUNDS["east"]),
         0.5 * (BOUNDS["south"] + BOUNDS["north"])],
        dtype=np.float64,
    )
    interpolated = natural_neighbor_to_points(
        points - local_origin, values, xi - local_origin
    )
    shape = (round((BOUNDS["north"] - BOUNDS["south"]) * PIXELS_PER_DEGREE),
             round((BOUNDS["east"] - BOUNDS["west"]) * PIXELS_PER_DEGREE))
    interpolated = interpolated.reshape(*shape, 7)
    image = interpolated[..., 0]
    sun_grid = normalize_vectors(interpolated[..., 1:4])
    view_grid = normalize_vectors(interpolated[..., 4:7])
    valid_grid = (
        np.isfinite(image)
        & np.all(np.isfinite(sun_grid), axis=-1)
        & np.all(np.isfinite(view_grid), axis=-1)
    )
    image[~valid_grid] = np.nan
    sun_grid[~valid_grid] = np.nan
    view_grid[~valid_grid] = np.nan
    summary = {
        "scene": scene,
        "source_rows": [first_row, last_row],
        "source_points": int(points.shape[0]),
        "target_valid_pixels": int(valid_grid.sum()),
        "interpolation_coordinates": "east-longitude/latitude translated to local origin",
        "wavelength_nm": wavelength,
        "band_index_zero_based": band_index,
        "solar_irradiance_w_m2_um": irradiance,
        "scene_mean_solar_distance_au": scene_mean_au,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (loc_path, obs_path, rdn_path)},
    }
    return image, sun_grid, view_grid, summary


def interpolate_gld100(root: Path, lon_grid: np.ndarray, lat_grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    path = root / "01_raw" / "GLD100_rows_5403_5635.bin"
    rows = GLD["last_row"] - GLD["first_row"] + 1
    expected = rows * GLD["samples"] * 2
    if path.stat().st_size != expected:
        raise RuntimeError(f"GLD100 block has {path.stat().st_size} bytes, expected {expected}")
    block = np.fromfile(path, dtype="<i2").reshape(rows, GLD["samples"]).astype(np.float64)
    block[block == -32768] = np.nan
    row_ids = np.arange(GLD["first_row"], GLD["last_row"] + 1)
    latitudes = GLD["north"] - (row_ids + 0.5) * GLD["resolution"]
    columns = np.arange(GLD["samples"])
    longitudes = GLD["west"] + (columns + 0.5) * GLD["resolution"]
    col_mask = (longitudes >= BOUNDS["west"] - 0.1) & (longitudes <= BOUNDS["east"] + 0.1)
    subset = block[:, col_mask]
    subset_lons = longitudes[col_mask]
    interpolator = RegularGridInterpolator(
        (latitudes[::-1], subset_lons),
        subset[::-1, :],
        bounds_error=False,
        fill_value=np.nan,
    )
    dem = interpolator(np.column_stack((lat_grid.ravel(), lon_grid.ravel()))).reshape(lon_grid.shape)
    if not np.isfinite(dem).all():
        raise RuntimeError("GLD100 interpolation left invalid pixels inside Mairan T")
    return dem, {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_tif(path: Path, data: np.ndarray, *, nodata: float = -32768.0) -> None:
    array = np.asarray(data, dtype=np.float32)
    encoded = np.where(np.isfinite(array), array, nodata)
    transform = from_origin(
        BOUNDS["west"], BOUNDS["north"], 1.0 / PIXELS_PER_DEGREE, 1.0 / PIXELS_PER_DEGREE
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="+proj=longlat +R=1737400 +no_defs",
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(encoded, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "02_preprocessed"
    output.mkdir(parents=True, exist_ok=True)

    solar = np.loadtxt(root / "00_metadata" / "M3G20110224_RFL_SOLAR_SPEC.TAB")
    lon_grid, lat_grid, xi = target_grid()
    images, suns, views, scene_summaries = [], [], [], []
    for scene in SCENES:
        print(f"Natural-neighbor interpolation: {scene}", flush=True)
        image, sun, view, summary = interpolate_scene(
            root, scene, xi, solar[:, 0], solar[:, 1]
        )
        images.append(image)
        suns.append(sun)
        views.append(view)
        scene_summaries.append(summary)
        write_tif(output / f"{scene}_1579nm_bidirectional_reflectance.tif", image)
        for index, axis in enumerate("xyz"):
            write_tif(output / f"{scene}_sun_{axis}.tif", sun[..., index])
            write_tif(output / f"{scene}_view_{axis}.tif", view[..., index])

    image_stack = np.stack(images)
    sun_stack = np.stack(suns)
    view_stack = np.stack(views)
    dem, gld_summary = interpolate_gld100(root, lon_grid, lat_grid)
    common_mask = np.all(np.isfinite(image_stack), axis=0)
    if not common_mask.any():
        raise RuntimeError("The three M3 images have no common valid target pixels")
    write_tif(output / "GLD100_MairanT_300ppd.tif", dem)
    write_tif(output / "common_three_image_mask.tif", common_mask.astype(np.float32), nodata=-1.0)

    mean_latitude = 0.5 * (BOUNDS["south"] + BOUNDS["north"])
    pixel_size_x = MOON_RADIUS_M * np.cos(np.deg2rad(mean_latitude)) * np.deg2rad(1.0 / PIXELS_PER_DEGREE)
    pixel_size_y = MOON_RADIUS_M * np.deg2rad(1.0 / PIXELS_PER_DEGREE)
    np.savez_compressed(
        output / "mairan_t_m3_stack.npz",
        images=image_stack,
        sun_directions=sun_stack,
        view_directions=view_stack,
        initial_dem=dem,
        common_mask=common_mask,
        longitude=lon_grid,
        latitude=lat_grid,
        scene_ids=np.asarray(SCENES),
        pixel_size_x=np.asarray(pixel_size_x),
        pixel_size_y=np.asarray(pixel_size_y),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, scene in enumerate(SCENES):
        values = image_stack[index]
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, [1, 99])
        axes.flat[index].imshow(values, cmap="gray", vmin=vmin, vmax=vmax)
        axes.flat[index].set_title(f"{scene} 1579 nm L/E")
        axes.flat[index].axis("off")
    axes.flat[3].imshow(dem, cmap="terrain")
    axes.flat[3].contour(common_mask, levels=[0.5], colors="red", linewidths=0.7)
    axes.flat[3].set_title("GLD100; red = 3-image overlap")
    axes.flat[3].axis("off")
    fig.savefig(output / "preprocessing_quicklook.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": "Grumpe & Woehler (2014) Mairan T M3 reproduction",
        "bounds_east_longitude_degrees": BOUNDS,
        "grid_pixels_per_degree": PIXELS_PER_DEGREE,
        "shape": list(dem.shape),
        "pixel_size_m": {"x": pixel_size_x, "y": pixel_size_y},
        "reflectance_definition": "bidirectional reflectance L/E (no pi factor)",
        "common_three_image_pixels": int(common_mask.sum()),
        "common_fraction": float(common_mask.mean()),
        "scenes": scene_summaries,
        "gld100": gld_summary,
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
