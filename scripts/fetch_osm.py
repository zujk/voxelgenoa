"""OSM for the AOI as Overpass JSON with full geometry, fetched in four chunks and merged (the public servers refuse
one big query) -> data/raw/osm.json. DEM: AWS Terrain Tiles (terrarium, zoom 15) -> data/raw/dem.npy on the 1 m grid."""
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from aoi import RAW, D, W, lonlat_bounds, to_grid, to_world  # noqa: E402
from pyproj import Transformer  # noqa: E402

SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
KEYS = "building|building:part|highway|railway|natural|landuse|leisure|waterway|man_made|bridge|amenity|water|aeroway"


def query(s, w, n, e):
    bb = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    sel = [f'way["{k}"]({bb});' for k in KEYS.split("|")] + [f'node["natural"="tree"]({bb});',
           f'node["highway"="street_lamp"]({bb});', f'relation["building"]({bb});', f'relation["natural"]({bb});',
           f'relation["landuse"]({bb});', f'relation["leisure"]({bb});', f'relation["type"="multipolygon"]["water"]({bb});']
    return "[out:json][timeout:300];(" + "".join(sel) + ");out body geom;"


def osm():
    out = Path(RAW) / "osm.json"
    if out.exists():
        return
    s, w, n, e = lonlat_bounds(100)
    elems, stamp = {}, ""
    for a in range(2):
        for b in range(2):
            box = (s + (n - s) * a / 2, w + (e - w) * b / 2, s + (n - s) * (a + 1) / 2, w + (e - w) * (b + 1) / 2)
            for attempt in range(9):
                url = SERVERS[attempt % len(SERVERS)]
                try:
                    r = requests.post(url, data={"data": query(*box)}, timeout=600, headers={"User-Agent": "genoa-voxel/0.1"})
                    r.raise_for_status()
                    j = r.json()
                    break
                except Exception as ex:  # noqa: BLE001
                    print("overpass", url, ex, file=sys.stderr)
                    time.sleep(10 * (attempt + 1))
            else:
                raise SystemExit("overpass failed")
            stamp = j["osm3s"]["timestamp_osm_base"]
            for el in j["elements"]:
                elems[(el["type"], el["id"])] = el
            print("chunk", a, b, len(j["elements"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"osm3s": {"timestamp_osm_base": stamp}, "elements": list(elems.values())}, open(out, "w"))
    print("osm elements", len(elems))


Z = 15


def dem():
    out = Path(RAW) / "dem.npy"
    if out.exists():
        return
    to_ll = Transformer.from_crs(32632, 4326, always_xy=True)
    s, w, n, e = lonlat_bounds(200)
    tx = lambda lon: (lon + 180) / 360 * 2 ** Z
    ty = lambda lat: (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2 ** Z
    x0, x1, y0, y1 = int(tx(w)), int(tx(e)), int(ty(n)), int(ty(s))
    big = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float32)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            r = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png", timeout=60)
            r.raise_for_status()
            a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.float32)
            big[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    from scipy.ndimage import map_coordinates
    jj, ii = np.mgrid[0:D, 0:W].astype(np.float64)
    X, Y = to_world(ii + 0.5, jj + 0.5)
    lon, lat = to_ll.transform(X, Y)
    px = (tx(lon) - x0) * 256 - 0.5
    py = (ty(lat) - y0) * 256 - 0.5
    dem = map_coordinates(big, [py, px], order=1, mode="nearest").astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, dem)
    print("dem", dem.min(), dem.max())


if __name__ == "__main__":
    osm()
    dem()
