from pathlib import Path
import numpy as np

root = Path('/mnt/e/M3_Photometry/experiment1_mairan_t')
stack = np.load(root / '02_preprocessed/mairan_t_m3_stack.npz')
for index, scene in enumerate(stack['scene_ids']):
    image = stack['images'][index]
    print(scene)
    print('  min/max', float(np.nanmin(image)), float(np.nanmax(image)))
    print('  pct', np.nanpercentile(image, [0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99]))

    folder = root / '01_raw' / str(scene)
    loc_path = next(folder.glob('*_LOC_rows_*.IMG'))
    rdn_path = next(folder.glob('*_RDN_rows_*.IMG'))
    rows = loc_path.stat().st_size // (3 * 304 * 8)
    loc = np.fromfile(loc_path, dtype='<f8').reshape(rows, 3, 304)
    rdn = np.fromfile(rdn_path, dtype='<f4').reshape(rows, 85, 304)
    roi = ((loc[:, 0] >= 311.4) & (loc[:, 0] <= 311.8)
           & (loc[:, 1] >= 41.67) & (loc[:, 1] <= 41.93))
    print('  loc lon range/unique', float(np.nanmin(loc[:, 0])), float(np.nanmax(loc[:, 0])),
          np.unique(loc[:, 0]).size)
    print('  loc lat range/unique', float(np.nanmin(loc[:, 1])), float(np.nanmax(loc[:, 1])),
          np.unique(loc[:, 1]).size)
    print('  exact roi rows/cols/count', np.flatnonzero(roi.any(axis=1))[[0, -1]],
          np.flatnonzero(roi.any(axis=0))[[0, -1]], int(roi.sum()))
    for band in (48, 49, 50):
        values = rdn[:, band][roi]
        print('  raw band', band + 1, 'count/unique', values.size, np.unique(values).size,
              'min/p50/max', float(np.nanmin(values)), float(np.nanmedian(values)),
              float(np.nanmax(values)))
    if str(scene).endswith('060502'):
        from metpy.interpolate import natural_neighbor_to_points
        from scipy.interpolate import griddata
        obs_path = next(folder.glob('*_OBS_rows_*.IMG'))
        obs = np.fromfile(obs_path, dtype='<f4').reshape(rows, 10, 304)
        distance = 1.017224633668 + obs[:, 5]
        refl = rdn[:, 49].astype(float) * distance**2 / 254.320526
        margin = 0.12
        use = (np.isfinite(refl) & (refl > 0) & (loc[:, 0] >= 311.4-margin)
               & (loc[:, 0] <= 311.8+margin) & (loc[:, 1] >= 41.67-margin)
               & (loc[:, 1] <= 41.93+margin))
        points = np.column_stack((loc[:, 0][use], loc[:, 1][use]))
        values = refl[use]
        _, idx = np.unique(np.round(points, 9), axis=0, return_index=True)
        points, values = points[idx], values[idx]
        lon = 311.4 + (np.arange(120)+.5)/300
        lat = 41.93 - (np.arange(78)+.5)/300
        xx, yy = np.meshgrid(lon, lat)
        xi = np.column_stack((xx.ravel(), yy.ravel()))
        for name, out in [('natural_scalar', natural_neighbor_to_points(points, values, xi)),
                          ('natural_centered', natural_neighbor_to_points(points-[311.6,41.8], values, xi-[311.6,41.8])),
                          ('linear', griddata(points, values, xi, method='linear'))]:
            print(' ', name, 'valid/unique/pct', int(np.isfinite(out).sum()),
                  np.unique(out[np.isfinite(out)]).size,
                  np.nanpercentile(out, [0,1,50,99,100]))
