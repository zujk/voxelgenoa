"""Regione Liguria open data for the AOI (geoservizi.regione.liguria.it, no registration):

  dtm.npy        DTM ed. 2023, 5 m (WCS M2056__L6952, EPSG:3003), bilinear onto the 1 m grid; NaN off the land
  ctr_edifici.json  CTR 1:5000 3D "Edificato" (WFS M1237:L3165, 2007-13): footprints whose z is the eave height a.s.l.
  ortho.npy      colour orthophoto 2022 (WMS M2555:L10401), fetched at 0.5 m and averaged to the 1 m grid, (D, W, 3)

Run: PYTHONPATH=scripts uv run python scripts/fetch_liguria.py
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
import tifffile
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
from aoi import D, RAW, W, lonlat_bounds, to_world, world_bounds  # noqa: E402

GS = "https://geoservizi.regione.liguria.it/geoserver"
UA = {"User-Agent": "genoa-voxel/0.1"}
OUT = Path(RAW) / "liguria"


def get(url, params, timeout=300):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as ex:  # noqa: BLE001
            print("retry", url, ex, file=sys.stderr)
    raise SystemExit(f"failed: {url}")


def grid_lonlat_to(crs):
    """world (EPSG:32632) coordinates of every 1 m cell centre, transformed to `crs`."""
    jj, ii = np.mgrid[0:D, 0:W].astype(np.float64)
    X, Y = to_world(ii + 0.5, jj + 0.5)
    return Transformer.from_crs(32632, crs, always_xy=True).transform(X, Y)


def dtm():
    out = OUT / "dtm.npy"
    if out.exists():
        return
    t = Transformer.from_crs(32632, 3003, always_xy=True)
    x0, y0, x1, y1 = world_bounds(150)
    xs, ys = zip(*[t.transform(x, y) for x in (x0, x1) for y in (y0, y1)])
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    r = get(f"{GS}/M2056/wcs", [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
                                ("coverageId", "M2056__L6952"), ("format", "image/tiff"),
                                ("subset", f"X({bx0},{bx1})"), ("subset", f"Y({by0},{by1})")])
    tif = tifffile.TiffFile(io.BytesIO(r.content))
    a = tif.pages[0].asarray().astype(np.float32)
    tags = tif.pages[0].tags
    m = tags["ModelTransformationTag"].value
    sx, sy, ox, oy = m[0], -m[5], m[3], m[7]
    a[a <= -32000] = np.nan
    a[a < -100] = np.nan
    print("dtm", a.shape, "pixel", sx, sy, "range", np.nanmin(a), np.nanmax(a))
    X, Y = grid_lonlat_to(3003)
    px, py = (X - ox) / sx - 0.5, (oy - Y) / sy - 0.5
    valid = map_coordinates(np.isfinite(a).astype(np.float32), [py, px], order=1, cval=0) > 0.99
    g = map_coordinates(np.nan_to_num(a, nan=0.0), [py, px], order=1, cval=0).astype(np.float32)
    g[~valid] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(out, g)


def edifici():
    out = OUT / "ctr_edifici.json"
    if out.exists():
        return
    s, w, n, e = lonlat_bounds(150)
    feats, start = [], 0
    while True:
        r = get(f"{GS}/M1237/ows", dict(service="WFS", version="2.0.0", request="GetFeature", typeNames="M1237:L3165",
                                       outputFormat="application/json", srsName="EPSG:32632", count=5000, startIndex=start,
                                       sortBy="ogr_fid", bbox=f"{s},{w},{n},{e},urn:ogc:def:crs:EPSG::4326"))
        f = r.json()["features"]
        feats += f
        print("edifici", len(feats))
        if len(f) < 5000:
            break
        start += 5000
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))


def ortho(res=0.5, chunk=1200):
    out = OUT / "ortho.npy"
    if out.exists():
        return
    x0, y0, x1, y1 = world_bounds(0)
    span = chunk * res
    nx, ny = int(np.ceil((x1 - x0) / span)), int(np.ceil((y1 - y0) / span))
    big = np.zeros((ny * chunk, nx * chunk, 3), np.uint8)
    for j in range(ny):
        for i in range(nx):
            bx, by = x0 + i * span, y1 - (j + 1) * span
            r = get(f"{GS}/M2555/wms", dict(service="WMS", version="1.3.0", request="GetMap", layers="L10401", styles="",
                                           crs="EPSG:32632", bbox=f"{bx},{by},{bx + span},{by + span}", width=chunk,
                                           height=chunk, format="image/jpeg"))
            big[j * chunk:(j + 1) * chunk, i * chunk:(i + 1) * chunk] = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        print("ortho row", j + 1, "/", ny)
    # the grid is unrotated (aoi.THETA = 0): row 0 is the north edge, as in the image
    k = int(round(1 / res))
    sm = big[:D * k, :W * k].reshape(D, k, W, k, 3).mean((1, 3))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(out, sm.astype(np.uint8))
    Image.fromarray(sm.astype(np.uint8)).resize((W // 4, D // 4)).save(OUT / "ortho_preview.jpg", quality=85)


if __name__ == "__main__":
    dtm()
    edifici()
    ortho()
