"""Genoa voxel build: OSM + terrain tiles -> web/public/data.

  1. rasterise OSM onto the 1 m grid (aoi.py): ground height, surface class, buildings, sea
  2. place trees, bridge decks and pillars
  3. voxelise one quadtree node at a time (a 1 m window per node, so memory stays bounded) and write the tiles with
     tiles.py (same format as the Warsaw viewer reads)
  4. terrain.bin, zones.bin/json, meta.json, places.json

Run: uv run python scripts/build.py   (after scripts/fetch_osm.py)
"""
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union

sys.path.insert(0, str(Path(__file__).parent))
from aoi import D, I0, J0, RAW, W, to_grid  # noqa: E402
from osm import OSM  # noqa: E402
from landmarks import LABELS, VIEWS  # noqa: E402
import tiles  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "web/public/data"
ZOFF = 4            # voxel z of 0 m above sea level
SEA_BED = 2
T0 = time.time()


def note(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def rng_for(*key):
    return np.random.default_rng(int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16))


# ---------------------------------------------------------------- palette
PAL = [(0, 0, 0, 0)]
_idx = {PAL[0]: 0}
MAT = dict(SOLID=0, WALL=1, GLASS=2, LEAF=3, WATER=4, LAMP=5, METAL=6, GRASS=8, SOIL=10, ROOF=12, CONCRETE=13)


def pal(rgb, mat):
    k = (int(rgb[0]), int(rgb[1]), int(rgb[2]), mat)
    if k not in _idx:
        _idx[k] = len(PAL)
        PAL.append(k)
    return _idx[k]


def jitter(rgb, n, mat, amp=7, seed=0):
    r = np.random.default_rng(seed)
    return np.array([pal(np.clip(np.array(rgb) + r.integers(-amp, amp + 1, 3), 0, 255), mat) for _ in range(n)], np.uint16)


NV = 6  # shade variants per ground surface
(SC_SCRUB, SC_URBAN, SC_ROAD, SC_PAVED, SC_RAIL, SC_GRASS, SC_FOREST, SC_ROCK, SC_BEACH, SC_WATER) = range(10)
SURF_RGB = {SC_SCRUB: ((132, 138, 84), MAT["GRASS"]), SC_URBAN: ((166, 156, 144), MAT["SOLID"]),
            SC_ROAD: ((96, 96, 100), MAT["SOLID"]), SC_PAVED: ((186, 176, 160), MAT["SOLID"]),
            SC_RAIL: ((120, 108, 98), MAT["SOLID"]), SC_GRASS: ((124, 170, 88), MAT["GRASS"]),
            SC_FOREST: ((88, 124, 66), MAT["GRASS"]), SC_ROCK: ((150, 140, 126), MAT["SOLID"]),
            SC_BEACH: ((220, 204, 160), MAT["SOIL"]), SC_WATER: ((70, 126, 168), MAT["WATER"])}
SURF_PAL = np.stack([jitter(SURF_RGB[k][0], NV, SURF_RGB[k][1], 5 if k != SC_ROAD else 3, 100 + k) for k in range(10)])
SOIL_PAL = jitter((138, 112, 84), 4, MAT["SOIL"], 8, 7)
STONE_PAL = jitter((168, 160, 146), 4, MAT["SOLID"], 7, 15)    # the retaining walls of the terraced town
EARTH_PAL = jitter((96, 104, 64), 4, MAT["GRASS"], 7, 16)      # step faces under grass and scrub
STRIPE_PAL = pal((58, 60, 64), MAT["WALL"])                    # Genoese black-and-white banding
ROCK_PAL = jitter((126, 116, 106), 4, MAT["SOLID"], 8, 8)
SAND_PAL = pal((200, 182, 140), MAT["SOIL"])
SEA_PAL = pal((58, 112, 160), MAT["WATER"])
TRUNK_PAL = pal((98, 74, 50), MAT["SOLID"])
LEAF_PAL = np.concatenate([jitter(c, 5, MAT["LEAF"], 8, 30 + i) for i, c in enumerate(
    [(84, 128, 64), (96, 140, 70), (70, 112, 60), (110, 146, 74), (64, 100, 70)])])
DECK_PAL = pal((172, 168, 160), MAT["CONCRETE"])
DECK_TOP = jitter((92, 92, 98), 3, MAT["SOLID"], 3, 9)
PIER_PAL = pal((150, 146, 140), MAT["CONCRETE"])
# Genoese facades: ochre, yellow, salmon, rust, cream, grey, orange, faded red
WALLS = [((222, 178, 108), 5), ((232, 204, 128), 4), ((224, 152, 124), 5), ((206, 124, 104), 3), ((236, 222, 190), 5),
         ((208, 204, 196), 3), ((228, 150, 90), 3), ((188, 100, 86), 2), ((196, 176, 140), 3), ((178, 196, 170), 1),
         ((214, 168, 150), 3), ((240, 232, 214), 2)]
WALL_PAL = np.array([pal(np.clip(np.array(c) + d, 0, 255), MAT["WALL"]) for c, n in WALLS for d in range(-6, 7, 6)[:max(1, n // 2)]], np.uint16)
WALL_W = np.array([n / max(1, n // 2) for c, n in WALLS for _ in range(max(1, n // 2))])
WALL_W /= WALL_W.sum()
ROOF_TERRA = jitter((176, 98, 74), 4, MAT["ROOF"], 8, 11)
ROOF_SLATE = jitter((92, 96, 106), 4, MAT["ROOF"], 8, 12)
ROOF_FLAT = jitter((148, 144, 138), 4, MAT["ROOF"], 8, 13)
ROOF_MET = jitter((176, 184, 190), 3, MAT["ROOF"], 6, 14)
GLASS_PAL = pal((120, 150, 170), MAT["GLASS"])
GLASS_TINTS = [pal(c, MAT["GLASS"]) for c in ((92, 118, 138), (70, 92, 116), (104, 126, 124), (138, 150, 160), (84, 96, 108))]
ROOFQ_PAL = np.array([pal((r * 255 // 7, g * 255 // 7, b * 255 // 7), MAT["ROOF"])
                      for r in range(8) for g in range(8) for b in range(8)], np.uint16)
NAMED = {"white": (240, 240, 236), "yellow": (232, 204, 128), "red": (190, 100, 86), "grey": (180, 178, 172),
         "gray": (180, 178, 172), "orange": (228, 150, 90), "pink": (224, 160, 150), "beige": (222, 206, 172),
         "brown": (160, 120, 90), "green": (150, 180, 140), "cream": (236, 222, 190), "tan": (210, 180, 140),
         "salmon": (224, 152, 124), "ochre": (222, 178, 108), "black": (60, 60, 64), "blue": (140, 160, 190)}


def parse_colour(s):
    if not s:
        return None
    s = s.strip().lower()
    if s in NAMED:
        return NAMED[s]
    m = re.fullmatch(r"#?([0-9a-f]{6})", s)
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


# ---------------------------------------------------------------- rasterising
def fnum(v):
    if v is None:
        return None
    m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", str(v))
    return float(m.group(1).replace(",", ".")) if m else None


def paint(arr, geom, value, pad=2):
    """Fill geom (grid coords) into arr with value, honouring holes; works on a bbox crop."""
    if geom.is_empty:
        return None
    polys = [geom] if isinstance(geom, Polygon) else [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    out = None
    for p in polys:
        x0, y0, x1, y1 = p.bounds
        c0, r0, c1, r1 = max(int(x0) - pad, 0), max(int(y0) - pad, 0), min(int(x1) + pad + 1, W), min(int(y1) + pad + 1, D)
        if c0 >= c1 or r0 >= r1:
            continue
        im = Image.new("L", (c1 - c0, r1 - r0), 0)
        dr = ImageDraw.Draw(im)
        dr.polygon([(x - c0 - 0.5, y - r0 - 0.5) for x, y in p.exterior.coords], fill=1)
        for h in p.interiors:
            dr.polygon([(x - c0 - 0.5, y - r0 - 0.5) for x, y in h.coords], fill=0)
        m = np.asarray(im, bool)
        arr[r0:r1, c0:c1][m] = value
        out = (r0, r1, c0, c1, m) if out is None else out
    return out


def road_width(t):
    w = fnum(t.get("width"))
    if w:
        return min(w, 30)
    h = t.get("highway")
    base = {"motorway": 13, "motorway_link": 7, "trunk": 12, "trunk_link": 7, "primary": 10, "primary_link": 6,
            "secondary": 9, "secondary_link": 6, "tertiary": 8, "tertiary_link": 6, "residential": 6.5,
            "unclassified": 6, "living_street": 5, "service": 4, "pedestrian": 5, "footway": 2.2, "path": 1.8,
            "steps": 2.2, "cycleway": 2.5, "track": 3, "corridor": 2}.get(h)
    if base is None:
        return None
    lanes = fnum(t.get("lanes"))
    if lanes and h in ("motorway", "trunk", "primary", "secondary", "tertiary"):
        base = max(base, lanes * 3.3)
    return base


ROAD_RANK = {"footway": 0, "path": 0, "steps": 0, "cycleway": 0, "corridor": 0, "pedestrian": 1, "track": 1, "service": 2,
             "living_street": 3, "residential": 3, "unclassified": 3, "tertiary": 4, "tertiary_link": 4,
             "secondary": 5, "secondary_link": 5, "primary": 6, "primary_link": 6, "trunk": 7, "trunk_link": 7,
             "motorway": 8, "motorway_link": 8}
PAVED_ROADS = {"footway", "path", "steps", "cycleway", "pedestrian", "corridor", "track"}


def coast_sea(osm):
    """Sea mask from natural=coastline (land on the left of each way). Raster bool (D, W)."""
    lines = [g for _, t, g in osm.lines if t.get("natural") == "coastline"]
    sea = np.zeros((D, W), bool)
    if not lines:
        note("no coastline in the AOI")
        return sea
    frame = box(-200, -200, W + 200, D + 200)
    clipped = [l.intersection(frame) for l in lines]
    segs = []
    for g in clipped:
        segs += list(g.geoms) if hasattr(g, "geoms") else [g]
    segs = [s for s in segs if isinstance(s, LineString) and not s.is_empty]
    union = unary_union(segs + [frame.boundary])
    pieces = list(polygonize(union))
    for p in pieces:
        bd = p.boundary
        # a coast segment on the piece's boundary, with its direction: the sea is on its right (grid y points south)
        for s in segs:
            for (ax, ay), (bx, by) in zip(s.coords[:-1], s.coords[1:]):
                mx, my = (ax + bx) / 2, (ay + by) / 2
                if bd.distance(Point(mx, my)) < 1e-6 and math.hypot(bx - ax, by - ay) > 0.5:
                    ux, uy = bx - ax, by - ay
                    n = math.hypot(ux, uy)
                    probe = Point(mx - uy / n * 0.3, my + ux / n * 0.3)
                    if p.contains(probe):
                        paint(sea, p, True)
                    break
            else:
                continue
            break
    note(f"sea: {sea.mean() * 100:.1f}% of the map, {len(pieces)} coast pieces")
    return sea


def main():
    osm = OSM()
    dem = np.load(Path(RAW) / "dem.npy").astype(np.float32)
    lig = Path(RAW) / "liguria"
    if (lig / "dtm.npy").exists():  # Regione Liguria DTM 2023 (5 m); the terrain tiles only off it (the sea)
        dtm = np.load(lig / "dtm.npy")
        dem = np.where(np.isfinite(dtm), dtm, dem).astype(np.float32)
        note(f"DTM 2023 on {np.isfinite(dtm).mean() * 100:.0f}% of the map")
        del dtm
    ctrz = np.full((D, W), np.nan, np.float32)  # CTR eave height a.s.l. per cell
    if (lig / "ctr_edifici.json").exists():
        polys = []
        for f in json.load(open(lig / "ctr_edifici.json"))["features"]:
            if f["properties"].get("def_sotterr") == "Sotterraneo" or not f["geometry"]:
                continue
            gm = f["geometry"]
            for rings in ([gm["coordinates"]] if gm["type"] == "Polygon" else gm["coordinates"]):
                rs = []
                for ring in rings:
                    a_ = np.asarray(ring, np.float64)
                    i_, j_ = to_grid(a_[:, 0], a_[:, 1])
                    rs.append(list(zip(i_, j_)))
                z = float(np.mean(np.asarray(rings[0])[:, 2]))
                pg = Polygon(rs[0], rs[1:])
                if pg.is_valid and pg.area > 2:
                    polys.append((z, pg))
        for z, pg in sorted(polys, key=lambda a_: a_[0]):
            paint(ctrz, pg, z)
        note(f"CTR 3D buildings: {len(polys)} footprints, {np.isfinite(ctrz).mean() * 100:.1f}% of the map")
    ortho_f = canopy = None
    if (lig / "ortho.npy").exists():
        # tree canopy as the photo shows it: greener than the (yellowish) mean and darker, over 5 m
        o = np.load(lig / "ortho.npy").astype(np.float32)
        o /= o.reshape(-1, 3).mean(0)
        exg = ndimage.uniform_filter(o[..., 1] - (o[..., 0] + o[..., 2]) / 2, 5)
        canopy = (exg > 0.03) & (ndimage.uniform_filter(o.mean(-1), 5) < 1.0)
        del o, exg
  # 2022 orthophoto: roofs take its colour, quantised to 8 levels per channel
        ortho_f = ndimage.median_filter(np.load(lig / "ortho.npy"), size=(3, 3, 1)).astype(np.float32)
        # the 2022 flight has a yellow-green haze: grey-balance on the CTR roofs, then a little contrast
        on_roof = np.isfinite(ctrz)
        mean = ortho_f[on_roof].mean(0) if on_roof.any() else ortho_f.reshape(-1, 3).mean(0)
        ortho_f = ortho_f * (mean.mean() / mean)
        ortho_f = np.clip((ortho_f - mean.mean()) * 1.15 + mean.mean() * 0.95, 0, 255)
    note("osm", len(osm.areas), "areas", len(osm.lines), "lines", len(osm.nodes), "nodes")

    # ---------- surfaces
    surf = np.zeros((D, W), np.uint8)  # SC_SCRUB
    URBAN = {"residential", "commercial", "industrial", "retail", "railway", "garages", "construction", "military",
             "brownfield", "port", "education", "religious", "institutional"}
    for _, _, t, g in osm.areas:
        lu, nat, le = t.get("landuse"), t.get("natural"), t.get("leisure")
        if lu in URBAN or t.get("amenity") in ("school", "university", "hospital", "marketplace", "parking", "college"):
            paint(surf, g, SC_URBAN)
    for _, _, t, g in osm.areas:
        lu, nat, le = t.get("landuse"), t.get("natural"), t.get("leisure")
        if lu in ("grass", "meadow", "village_green", "recreation_ground", "cemetery", "allotments", "orchard", "vineyard", "flowerbed") \
                or le in ("park", "garden", "playground", "nature_reserve", "golf_course") or nat in ("grassland", "heath"):
            paint(surf, g, SC_GRASS)
    for _, _, t, g in osm.areas:
        if t.get("natural") in ("wood", "scrub") or t.get("landuse") == "forest":
            paint(surf, g, SC_FOREST if t.get("natural") != "scrub" else SC_SCRUB)
    for _, _, t, g in osm.areas:
        if t.get("natural") in ("bare_rock", "scree", "cliff", "rock"):
            paint(surf, g, SC_ROCK)
        elif t.get("natural") in ("beach", "sand"):
            paint(surf, g, SC_BEACH)
    for _, _, t, g in osm.areas:
        if t.get("natural") == "water" or "water" in t and t.get("natural") != "tree" or t.get("landuse") == "basin":
            paint(surf, g, SC_WATER)
    sea = coast_sea(osm)
    note("rasterised landuse")

    # waterways
    for _, t, g in osm.lines:
        ww = t.get("waterway")
        if ww in ("river", "canal", "stream", "drain") and t.get("tunnel") not in ("yes", "culvert") and t.get("covered") != "yes":
            w = {"river": 12, "canal": 5, "stream": 3, "drain": 2}[ww]
            paint(surf, g.buffer(w / 2, cap_style=2), SC_WATER)

    # roads, then bridges kept apart
    roads = []
    bridges = []
    for _, t, g in osm.lines:
        h = t.get("highway")
        if h is None or h in ("proposed", "construction", "bus_guideway", "raceway") or t.get("tunnel") in ("yes", "building_passage", "culvert") or t.get("covered") == "yes":
            continue
        if fnum(t.get("layer")) is not None and fnum(t.get("layer")) < 0:
            continue
        w = road_width(t)
        if not w:
            continue
        is_bridge = t.get("bridge") not in (None, "no") and h not in ("footway", "path", "steps")
        (bridges if is_bridge else roads).append((ROAD_RANK.get(h, 2), w, h, t, g))
    for rank, w, h, t, g in sorted(roads, key=lambda r: r[0]):
        paint(surf, g.buffer(w / 2, cap_style=2, join_style=2), SC_PAVED if h in PAVED_ROADS else SC_ROAD)
    for _, t, g in osm.lines:
        if t.get("railway") in ("rail", "light_rail", "tram", "subway", "narrow_gauge", "funicular") and t.get("tunnel") not in ("yes",) \
                and t.get("bridge") in (None, "no") and not (t.get("railway") == "subway"):
            paint(surf, g.buffer(1.8, cap_style=2), SC_RAIL)
    for _, _, t, g in osm.areas:  # plazas and pedestrian areas, parking
        if t.get("highway") in ("pedestrian", "footway") or t.get("area:highway") or t.get("amenity") == "parking" \
                or t.get("place") == "square":
            paint(surf, g, SC_PAVED if t.get("amenity") != "parking" else SC_ROAD)
    note("rasterised roads")

    # ---------- buildings
    blds = []
    for oid, _, t, g in osm.areas:
        if "building" in t or "building:part" in t:
            if t.get("building") == "no" or t.get("location") == "underground" or t.get("tunnel"):
                continue
            blds.append((oid, t, g))
    bid = np.zeros((D, W), np.int32)
    btop = np.zeros((D, W), np.int16)       # roof top z, absolute
    bwall = np.zeros((D, W), np.int16)      # wall top z (roof starts here)
    bwpal = np.zeros((D, W), np.uint16)
    brpal = np.zeros((D, W), np.uint16)
    # an outline filled by building:part polygons is drawn at its lowest part's height and the parts rise from it
    # (Simple 3D Buildings): its own height is the tallest part's, which would fill the whole complex
    from shapely.strtree import STRtree

    def part_h(t):
        lv = fnum(t.get("building:levels"))
        return fnum(t.get("height")) or (lv * 3.1 + 1.5 if lv else None)
    pidx = [n for n, b in enumerate(blds) if "building:part" in b[1] and "building" not in b[1]]
    ptree = STRtree([blds[n][2] for n in pidx]) if pidx else None
    n_out = 0
    for n, (oid, t, gm) in enumerate(blds):
        if "building" not in t or ptree is None:
            continue
        inside = [blds[pidx[q]] for q in ptree.query(gm) if gm.contains(blds[pidx[q]][2].representative_point())]
        hs = [h_ for h_ in (part_h(b[1]) for b in inside) if h_]
        if hs and unary_union([b[2] for b in inside]).area >= 0.8 * gm.area:
            t = dict(t, height=str(min(hs)))
            t.pop("building:levels", None)
            blds[n] = (oid, t, gm)
            n_out += 1
    note(f"{n_out} outlines lowered to their parts")
    order = sorted(blds, key=lambda b: (fnum(b[1].get("height")) or fnum(b[1].get("building:levels")) or 0))
    bstripe = np.zeros((D, W), bool)
    zones = []
    g0 = np.clip(np.round(dem), -5, 400).astype(np.int16) + ZOFF  # ground z, before sea handling
    ctr_n = [0]

    def put(k, r0, r1, c0, c1, m, t, key):
        """building k on the cells m of the window [r0:r1, c0:c1]: walls from the CTR eave heights where it has them,
        else from OSM; roof colours from the orthophoto; a hipped roof where the photo shows tiles or slate."""
        gloc = g0[r0:r1, c0:c1]
        base = int(np.percentile(gloc[m], 8))
        ys, xs = np.nonzero(m)
        area = float(m.sum())
        levels = fnum(t.get("building:levels"))
        h = fnum(t.get("height")) or (levels * 3.1 + 1.5 if levels else None)
        typ = t.get("building", "yes")
        if h is None:
            h = {"church": 20, "cathedral": 26, "chapel": 9, "industrial": 11, "warehouse": 10, "garage": 3, "garages": 3, "shed": 3,
                 "roof": 5, "hut": 3, "house": 9, "detached": 8, "service": 5, "kiosk": 3, "carport": 3, "cabin": 3,
                 "apartments": 20, "hospital": 22, "train_station": 14, "civic": 18, "public": 16}.get(typ, 15 if base < 60 else 10)
            if area < 40 and typ in ("yes", "house"):
                h = min(h, 6)
        h = max(float(h), 2.5)
        cz = ctrz[r0:r1, c0:c1]
        hasc = m & np.isfinite(cz) & (cz + ZOFF - gloc >= 2)
        ctr_top = float(cz[hasc].max()) + ZOFF - base if hasc.any() else 0.0
        # measured eaves; an OSM height tag wins only where the CTR is well below it (built after the survey)
        if hasc.sum() > 0.3 * area and ("height" not in t or ctr_top >= 0.85 * h):
            zc = float(np.median(cz[hasc]))
            wl = np.where(hasc, np.round(np.nan_to_num(cz, nan=zc)) + ZOFF, round(zc) + ZOFF)
            h = zc + ZOFF - base
            ctr_n[0] += 1
        else:
            wl = np.full(gloc.shape, base + int(round(h)))
        wl = np.maximum(wl, gloc + 3).astype(np.int16)
        roof_shape = t.get("roof:shape")
        oc = ortho_f[r0:r1, c0:c1][m].mean(0) if ortho_f is not None else None
        if roof_shape is not None:
            pitched = roof_shape in ("gabled", "hipped", "pyramidal", "half-hipped", "gambrel", "skillion", "mansard")
        elif oc is not None:  # terracotta tiles or dark slate (ardesia) are pitched; light grey or beige are terraces
            terra = oc[0] > oc[1] + 14 and oc[0] > oc[2] + 24
            slate = oc.mean() < 118 and oc.max() - oc.min() < 22
            pitched = (terra or slate) and area < 2500
        else:
            pitched = area < 220 and h < 14
        rr = np.zeros_like(gloc, np.int16)
        if pitched:
            dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
            rise = min(max(0.15 * math.sqrt(area), 1.5), 4.5)
            rr = np.where(m, np.minimum(np.round(dist * 0.5), round(rise)), 0).astype(np.int16)
        rng = rng_for(key)
        col = parse_colour(t.get("building:colour"))
        wp = pal(col, MAT["WALL"]) if col else int(rng.choice(WALL_PAL, p=WALL_W))
        # curtain walls: glass or mirror facades, and the office towers (the plastered palette is for the old town)
        fm = (t.get("building:facade:material") or t.get("building:material") or "").lower()
        office = typ in ("commercial", "office") or t.get("building:part") in ("commercial", "office")
        if fm in ("glass", "mirror") or (office and h >= 45 and fm not in ("concrete", "stone", "brick", "plaster")):
            wp = pal(col, MAT["GLASS"]) if col else int(rng.choice(GLASS_TINTS))
        rc = parse_colour(t.get("roof:colour"))
        if rc:
            rp = np.full(gloc.shape, pal(rc, MAT["ROOF"]), np.uint16)
        elif roof_shape in ("dome", "onion"):
            rp = np.full(gloc.shape, int(rng.choice(ROOF_MET)), np.uint16)
        elif oc is not None:  # the building's own mean colour, with some of each cell's (vents, terraces, panels)
            mix = 0.6 * oc + 0.4 * ortho_f[r0:r1, c0:c1]
            lum = mix.mean(-1, keepdims=True)
            chroma = mix - lum  # near-greys snap to grey: the 8-level palette would tint them pink or green
            chroma = np.where(np.abs(chroma).max(-1, keepdims=True) < 18, 0, chroma * 0.85)
            mix = np.clip(np.round((lum + chroma) / 255 * 7), 0, 7).astype(np.uint16)
            rp = ROOFQ_PAL[mix[..., 0] * 64 + mix[..., 1] * 8 + mix[..., 2]]
        else:
            rp = np.full(gloc.shape, int(rng.choice(ROOF_TERRA if pitched else ROOF_FLAT)), np.uint16)
        bw, bt, bp, rp_ = bwall[r0:r1, c0:c1], btop[r0:r1, c0:c1], bwpal[r0:r1, c0:c1], brpal[r0:r1, c0:c1]
        bw[m] = wl[m]
        bt[m] = (wl + rr + 1)[m]
        bp[m] = wp
        rp_[m] = rp[m]
        zones.append([int(xs.mean()) + c0, int(ys.mean()) + r0, k, c0 + int(xs.min()), r0 + int(ys.min()),
                      c0 + int(xs.max()) + 1, r0 + int(ys.max()) + 1, int(area * (6 + h / 3))])

    k = 0
    for k, (oid, t, g) in enumerate(order, 1):
        if k >= 60000:
            break
        r = paint(bid, g, k)
        if r is None:
            continue
        r0, r1, c0, c1, _ = r
        m = bid[r0:r1, c0:c1] == k
        if m.any():
            put(k, r0, r1, c0, c1, m, t, oid)
    # buildings the CTR has and OSM lacks (courtyard wings, sheds, port buildings)
    extra = np.isfinite(ctrz) & (bid == 0) & (ctrz + ZOFF - g0 >= 2.5)
    extra = ndimage.binary_opening(extra, iterations=1)
    lab, _ = ndimage.label(extra)
    added = 0
    for li, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == li
        if m.sum() < 15 or k >= 65000:
            continue
        k += 1
        bid[sl][m] = k
        put(k, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, m, {"building": "yes"}, ("ctr", li))
        added += 1
    note(f"CTR heights on {ctr_n[0]} buildings, {added} CTR-only buildings added")
    # landmark colours (landmarks.py), by footprint after every building is down: OSM often has several overlapping
    # polygons for one monument (outline, parts, the church area), and whichever came last would otherwise win
    shapes = {}  # landmarks built by hand after the decks: name -> (label, centre x, y, top z, top cells' x, y)
    for L in LABELS:
        if not any(k in L for k in ("wall", "roof", "stripes", "mat", "top", "spire", "shape", "lantern")):
            continue
        ids = L.get("parts") or ([L["osm"]] if L.get("osm") else None)
        hits = [(gm.area, gm) for i, _, t, gm in osm.areas
                if (i in ids if ids else bool(L.get("q")) and re.search(L["q"], t.get("name", ""), re.I))]
        if not hits:
            note(f"landmark {L['name']}: no footprint for its colours")
            continue
        fp = np.zeros((D, W), bool)
        if L.get("parts"):  # the whole complex: slivers of other polygons and CTR-only wings between the parts too
            paint(fp, unary_union([gm for _, gm in hits]).convex_hull, True)
            own = np.zeros((D, W), bool)  # `top` raises the parts themselves only
            for _, gm in hits:
                paint(own, gm, True)
        else:
            paint(fp, max(hits, key=lambda h: h[0])[1], True)
            own = fp
        ys_, xs_ = np.nonzero(fp)
        if not len(ys_):
            continue
        r0, r1, c0, c1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
        m = fp[r0:r1, c0:c1] & (bid[r0:r1, c0:c1] > 0)
        bw_, bt_ = bwall[r0:r1, c0:c1], btop[r0:r1, c0:c1]
        mo = m & own[r0:r1, c0:c1]
        if "top" in L and mo.any():  # height from a better source than OSM and the CTR (landmarks.py)
            base = int(np.percentile(g0[r0:r1, c0:c1][mo], 8))
            bw_[mo] = base + L["top"]
            bt_[mo] = base + L["top"] + 1
        if L.get("spire") and m.any():  # the Matitone's pencil point: a pyramid over the whole footprint
            dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
            bt_[m] = (bw_ + 1 + np.round(dist * L["spire"] / dist.max()))[m]
        if "wall" in L:
            bwpal[r0:r1, c0:c1][m] = pal(L["wall"], MAT[L.get("mat", "WALL")])  # SOLID: no windows drawn
        if "roof" in L:
            brpal[r0:r1, c0:c1][m] = pal(L["roof"], MAT["ROOF"])
        bstripe[r0:r1, c0:c1][m] = bool(L.get("stripes"))
        if L.get("shape") or L.get("lantern"):
            top_ = int(bt_[m].max()) if m.any() else 0
            ty_, tx_ = np.nonzero(m & (bt_ == top_)) if m.any() else (ys_ - r0, xs_ - c0)
            shapes[L["name"]] = (L, float(xs_.mean()), float(ys_.mean()), top_, float(tx_.mean()) + c0, float(ty_.mean()) + r0)
        if L.get("shape"):  # the footprint gives way to voxels built with the decks
            f_ = fp[r0:r1, c0:c1]
            bid[r0:r1, c0:c1][f_] = 0
            bw_[f_] = 0
            bt_[f_] = 0
    # re-number: ids in bid are positions in `order` (some blds fully overwritten leave unused ids; zones keyed by id)
    ids = [z[2] for z in zones]
    remap = np.zeros(65536, np.uint16)
    for n, i in enumerate(ids, 1):
        remap[i] = n
    bid = remap[np.clip(bid, 0, 65535)].astype(np.uint16)
    for z in zones:
        z[2] = 1
    note(f"buildings: {len(zones)}")

    # ---------- ground
    sea &= (bid == 0)
    sea &= ~np.isin(surf, (SC_ROAD, SC_PAVED, SC_RAIL)) | (dem < 0.3)
    land_elev = np.maximum(dem, 0.4)
    g = np.where(sea, ZOFF, np.clip(np.round(land_elev), 0, 250).astype(np.int16) + ZOFF).astype(np.int16)
    g = np.clip(g, 0, 250)
    # rock on steep slopes outside the city: the hillsides show their stone, as in the real cuts
    gy, gx = np.gradient(ndimage.uniform_filter(g.astype(np.float32), 5))
    steep = np.hypot(gx, gy) > 1.1
    surf = np.where(steep & np.isin(surf, (SC_SCRUB, SC_FOREST, SC_GRASS)) & (bid == 0), SC_ROCK, surf).astype(np.uint8)
    surf = np.where(sea, SC_WATER, surf).astype(np.uint8)
    # building footprints sit on a level base: raise the terrain under the footprint up to its lowest ground
    # (nothing to do: walls grow from the local ground)

    # ---------- trees
    tz = []
    for _, t, p in osm.nodes_with("natural", ["tree"]):
        tz.append((p.x, p.y, fnum(t.get("height")) or 8, 3.5, 1))
    rnd = rng_for("trees")
    # scattered trees only where the orthophoto shows canopy: OSM grass covers bare lawns and paved squares too.
    # In town, canopy over courtyards and squares with no landuse also gets trees
    sp = {SC_FOREST: (5, 1.0, 10.5, 3.6), SC_SCRUB: (8, 0.45, 4.5, 2.2), SC_GRASS: (8, 0.7, 8.0, 3.4),
          SC_URBAN: (9, 0.6, 8.0, 3.2), SC_PAVED: (9, 0.6, 8.0, 3.2)}
    if canopy is None:
        sp = {k: v for k, v in sp.items() if k not in (SC_URBAN, SC_PAVED)}
    for cls, (step, prob, h, r) in sp.items():
        ys, xs = np.mgrid[0:D:step, 0:W:step]
        x = xs + rnd.uniform(0, step, xs.shape)
        y = ys + rnd.uniform(0, step, ys.shape)
        xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, D - 1)
        ok = (surf[yi, xi] == cls) & (bid[yi, xi] == 0) & (rnd.random(xs.shape) < prob)
        if canopy is not None:
            ok &= canopy[yi, xi]
        for a, b in zip(x[ok], y[ok]):
            tz.append((a, b, h + rnd.uniform(-1.5, 3), r + rnd.uniform(-0.6, 0.9), 0))
    tx_ = np.array([t[0] for t in tz]); ty_ = np.array([t[1] for t in tz])
    thh = np.array([t[2] for t in tz]); trr = np.array([t[3] for t in tz])
    xi, yi = np.clip(tx_.astype(int), 0, W - 1), np.clip(ty_.astype(int), 0, D - 1)
    keep = (bid[yi, xi] == 0) & ~np.isin(surf[yi, xi], (SC_ROAD, SC_WATER, SC_RAIL)) & ~sea[yi, xi]
    trees = np.stack([tx_[keep], ty_[keep], g[yi, xi][keep].astype(float), thh[keep], np.clip(trr[keep], 1.5, 6.5)], 1)
    note(f"trees: {len(trees)}")

    # ---------- bridge decks
    deck = [[], [], [], []]
    pillars = []
    from scipy.spatial import cKDTree
    for rank, w, h, t, line in bridges:
        L = line.length
        if L < 8:
            continue
        n = max(int(L), 2)
        s = np.linspace(0, L, n)
        pts = np.array([line.interpolate(v).coords[0] for v in s])
        gz = g[np.clip(pts[:, 1].astype(int), 0, D - 1), np.clip(pts[:, 0].astype(int), 0, W - 1)].astype(float)
        layer = fnum(t.get("layer")) or 1
        lin = gz[0] + (gz[-1] - gz[0]) * (s / L)
        dz = np.maximum(lin, gz + 4 + 2 * max(layer - 1, 0)) + 1
        k = min(max(n // 6, 3), 41) | 1
        dz = ndimage.uniform_filter1d(dz, k, mode="nearest")
        dz = np.maximum(dz, gz + 3.5)
        poly = line.buffer(w / 2, cap_style=2)
        mask = np.zeros((D, W), bool)
        r = paint(mask, poly, True)
        if r is None:
            continue
        r0, r1, c0, c1, _ = r
        yy, xx = np.nonzero(mask[r0:r1, c0:c1])
        yy, xx = yy + r0, xx + c0
        _, near = cKDTree(pts).query(np.stack([xx + 0.5, yy + 0.5], 1))
        zz = np.round(dz[near]).astype(int)
        for off, p in ((0, DECK_TOP[(xx + yy) % len(DECK_TOP)]), (-1, DECK_PAL)):
            deck[0].append(xx); deck[1].append(yy); deck[2].append(zz + off)
            deck[3].append(np.full(len(xx), p, np.uint16) if np.isscalar(p) else p.astype(np.uint16))
        # pillars every 24 m where the ground is far below the deck
        for i in range(12, n - 6, 24):
            if dz[i] - gz[i] > 5:
                pillars.append((pts[i, 0], pts[i, 1], gz[i], dz[i]))
    # hand-built landmarks (landmarks.py `shape` / `lantern`), as free voxels like the decks
    for name, (L, cx, cy, top, tx, ty) in shapes.items():
        gz = int(g[int(cy), int(cx)])
        vx = []
        if L.get("shape") == "sphere":  # Biosfera: a glass ball of 20 m on a short white stand
            R = 10
            oz, oy, ox = np.mgrid[-R - 1:R + 2, -R - 1:R + 2, -R - 1:R + 2]
            d = np.sqrt(ox ** 2 + oy ** 2 + oz ** 2)
            sh = (d <= R + 0.5) & (d > R - 0.7)
            zc = gz + 2 + R
            vx.append((ox[sh] + int(cx), oy[sh] + int(cy), oz[sh] + zc, pal((150, 178, 188), MAT["GLASS"])))
            ring = np.abs(np.hypot(ox[0], oy[0]) - 4) < 0.8
            for z in range(gz, gz + 3):
                vx.append((ox[0][ring] + int(cx), oy[0][ring] + int(cy), np.full(int(ring.sum()), z), pal((236, 236, 232), MAT["SOLID"])))
        elif L.get("shape") == "bigo":  # Il Bigo: white tubular arms fanning out of the quay, the tallest 40 m
            white = pal((240, 240, 236), MAT["SOLID"])
            for az, reach, hgt, thick in ((200, 18, 40, 2), (250, 22, 30, 2), (150, 14, 34, 2), (300, 24, 24, 1),
                                          (100, 20, 22, 1), (20, 16, 18, 1), (340, 20, 28, 1)):
                n = int(math.hypot(reach, hgt) * 2)
                tt = np.linspace(0, 1, n)
                x = cx + math.cos(math.radians(az)) * reach * tt ** 1.3
                y = cy + math.sin(math.radians(az)) * reach * tt ** 1.3
                z = gz + hgt * tt
                for ddx in range(thick):
                    for ddy in range(thick):
                        vx.append((np.round(x).astype(int) + ddx, np.round(y).astype(int) + ddy, np.round(z).astype(int), white))
            # the round panoramic lift, halfway up the tallest arm
            a, b = np.mgrid[-3:4, -3:4]
            disc = a ** 2 + b ** 2 <= 10
            f = 0.5 ** 1.3
            px, py = cx + math.cos(math.radians(200)) * 18 * f, cy + math.sin(math.radians(200)) * 18 * f
            for z, c_ in ((gz + 20, white), (gz + 21, pal((150, 178, 188), MAT["GLASS"])), (gz + 22, white)):
                vx.append((a[disc] + int(px), b[disc] + int(py), np.full(int(disc.sum()), z), c_))
        if L.get("lantern"):  # Lanterna: the glazed lamp room and its cap on the upper tower
            a, b = np.mgrid[-1:2, -1:2]
            for z in range(top, top + 4):
                vx.append((a.ravel() + int(tx), b.ravel() + int(ty), np.full(9, z), pal((200, 210, 200), MAT["GLASS"])))
            dark = pal((70, 74, 78), MAT["SOLID"])
            vx.append((a.ravel() + int(tx), b.ravel() + int(ty), np.full(9, top + 4), dark))
            vx.append((np.array([int(tx)]), np.array([int(ty)]), np.array([top + 5]), dark))
        for x_, y_, z_, p_ in vx:
            ok = (x_ >= 0) & (x_ < W) & (y_ >= 0) & (y_ < D)
            deck[0].append(x_[ok]); deck[1].append(y_[ok]); deck[2].append(z_[ok])
            deck[3].append(np.full(int(ok.sum()), p_, np.uint16))
        note(f"landmark {name}: {sum(len(v[0]) for v in vx)} voxels by hand")
    deck = [np.concatenate(a) if a else np.zeros(0, int) for a in deck]
    note(f"bridge decks: {len(deck[0])} voxels, {len(pillars)} pillars")

    # ---------- ground arrays
    del dem
    rr_ = rng_for("ground")
    shade = rr_.integers(0, NV, (D, W)).astype(np.uint8)
    spal = SURF_PAL[surf, shade]
    soil = SOIL_PAL[rr_.integers(0, 4, (D, W))]
    rock = ROCK_PAL[(np.arange(W)[None, :] + np.arange(D)[:, None]) % 4]
    # the faces of terrain steps: rock, green earth under vegetation, stone walls in town, soil on beaches
    stone = STONE_PAL[rr_.integers(0, 4, (D, W))]
    earth = EARTH_PAL[rr_.integers(0, 4, (D, W))]
    deep = np.select([surf == SC_ROCK, np.isin(surf, (SC_SCRUB, SC_GRASS, SC_FOREST)), surf == SC_BEACH],
                     [rock, earth, soil], stone).astype(np.uint16)

    # ---------- exports: terrain / zones / meta (before the long tile pass)
    OUT.mkdir(parents=True, exist_ok=True)
    tw, th = -(-W // 4), -(-D // 4)
    hd = np.zeros((th * 4, tw * 4), np.uint16)
    hd[:D, :W] = np.clip(g, 0, 6000) * 10
    hd[D:, :] = hd[D - 1:D, :]
    hd[:, W:] = hd[:, W - 1:W]
    colors = np.array(PAL, np.uint8)[:, :3]
    cw, ch = W // 2, D // 2
    rgb = colors[np.where(bid > 0, brpal, spal)][:ch * 2:2, :cw * 2:2]
    with gzip.open(OUT / "terrain.bin", "wb", 9) as f:
        f.write(hd[::4, ::4][:th, :tw].astype("<u2").tobytes() + rgb.tobytes())
    with gzip.open(OUT / "zones.bin", "wb", 6) as f:
        f.write(np.zeros((D, W), "<u2").tobytes() + bid.astype("<u2").tobytes())
    json.dump({"blocks": [], "buildings": zones}, open(OUT / "zones.json", "w"))
    note("terrain, zones")

    # ---------- tiles
    keep_ids = ()
    fol_ids = list(LEAF_PAL) + [TRUNK_PAL]
    deck_x, deck_y, deck_z, deck_p = deck
    vol_cache = {}

    def window(a, c, size):
        """1 m volume (H, size, size) over grid rows a.., cols c..; ground (size, size) uint8"""
        ra, rb, ca, cb = max(a, 0), min(a + size, D), max(c, 0), min(c + size, W)
        gw = np.zeros((size, size), np.int16)
        if ra >= rb or ca >= cb:
            return None, gw.astype(np.uint8)
        sl = (slice(ra, rb), slice(ca, cb))
        gl, bt, bw = g[sl], btop[sl], bwall[sl]
        # trees and decks in the window
        tm = (trees[:, 0] >= ca - 7) & (trees[:, 0] < cb + 7) & (trees[:, 1] >= ra - 7) & (trees[:, 1] < rb + 7)
        tr = trees[tm]
        top = int(max(gl.max() + 3, bt.max() if bt.size else 0, (tr[:, 2] + tr[:, 3] + 3).max() if len(tr) else 0))
        dm = (deck_x >= ca) & (deck_x < cb) & (deck_y >= ra) & (deck_y < rb)
        if dm.any():
            top = max(top, int(deck_z[dm].max()) + 2)
        H = min(top + 1, 400)
        vol = np.zeros((H, size, size), np.uint16)
        v = vol[:, ra - a:rb - a, ca - c:cb - c]
        sp_, dp_, sea_, bp_, rp_, bs_ = spal[sl], deep[sl], sea[sl], bwpal[sl], brpal[sl], bstripe[sl]
        striped = bs_.any()
        isb = bid[sl] > 0
        zlo = max(int(gl.min()) - 4, 0)
        v[:max(zlo, 0)] = np.where(sea_ | True, dp_, 0)[None] if zlo > 0 else 0
        for z in range(zlo if zlo > 0 else 0, min(H, int(max(gl.max(), bt.max())) + 1)):
            s_ = v[z]
            m_soil = z < gl - 3
            s_[m_soil] = dp_[m_soil]
            m_top = (z >= gl - 3) & (z < gl)
            s_[m_top & ~sea_] = sp_[m_top & ~sea_]
            if z < gl.max() + 1:  # sea: bed up to SEA_BED, water above to the quay level
                ms = sea_ & (z < gl)
                s_[ms & (z < SEA_BED)] = SAND_PAL
                s_[ms & (z >= SEA_BED)] = SEA_PAL
            mb = isb & (z >= gl) & (z < bw)
            s_[mb] = bp_[mb]
            if striped and (z // 2) % 2:  # 2 m bands: 1 m ones blur to grey from any distance
                s_[mb & bs_] = STRIPE_PAL
            mr = isb & (z >= bw) & (z < bt)
            s_[mr] = rp_[mr]
            # footings: a building on a slope is solid right down to the ground
            mf = isb & (z < gl) & (z >= gl - 3)
            s_[mf] = bp_[mf]
        gw[ra - a:rb - a, ca - c:cb - c] = np.where(sea_, gl - 0, gl)
        # decks
        if dm.any():
            xs, ys, zs, ps = deck_x[dm] - c, deck_y[dm] - a, deck_z[dm], deck_p[dm]
            ok = (zs >= 0) & (zs < H)
            vol[zs[ok], ys[ok], xs[ok]] = ps[ok]
        for px, py, pg, pd in pillars:
            if ca - 2 <= px < cb + 2 and ra - 2 <= py < rb + 2:
                for ddx in (0, 1):
                    for ddy in (0, 1):
                        xx, yy = int(px) - c + ddx, int(py) - a + ddy
                        if 0 <= xx < size and 0 <= yy < size:
                            vol[max(int(pg), 0):min(int(pd) - 1, H), yy, xx] = PIER_PAL
        # trees
        if len(tr):
            tx0, ty0, tg, th_, trad = tr[:, 0] - c, tr[:, 1] - a, tr[:, 2], tr[:, 3], tr[:, 4]
            rnd2 = rng_for("tv", a, c)
            for R in np.unique(np.round(trad).astype(int)):
                sel = np.round(trad).astype(int) == R
                n = int(sel.sum())
                oz, oy, ox = np.mgrid[-R:R + 1, -R:R + 1, -R:R + 1]
                inside = (ox ** 2 + oy ** 2 + (oz / 0.9) ** 2) <= (R + 0.3) ** 2
                oz, oy, ox = oz[inside], oy[inside], ox[inside]
                bx, by = tx0[sel].astype(int), ty0[sel].astype(int)
                crown_h = np.minimum(th_[sel] * 0.62, th_[sel] - R * 0.9)
                cz = (tg[sel] + np.maximum(th_[sel] - R * 0.9, 1.5)).astype(int)
                col = LEAF_PAL[rnd2.integers(0, len(LEAF_PAL), n)]
                Z = cz[:, None] + oz[None]
                Y = by[:, None] + oy[None]
                X = bx[:, None] + ox[None]
                P = np.broadcast_to(col[:, None], Z.shape)
                ok = (Z >= 0) & (Z < H) & (Y >= 0) & (Y < size) & (X >= 0) & (X < size)
                tgt = vol[Z[ok], Y[ok], X[ok]]
                free = (tgt == 0)
                zz, yy, xx = Z[ok][free], Y[ok][free], X[ok][free]
                vol[zz, yy, xx] = P[ok][free]
                for k in range(n):  # trunks
                    x_, y_ = bx[k], by[k]
                    if 0 <= x_ < size and 0 <= y_ < size:
                        z0 = int(tg[sel][k])
                        z1 = min(int(cz[k]), H)
                        col_ = vol[z0:z1, y_, x_]
                        col_[col_ == 0] = TRUNK_PAL
        return vol, np.clip(gw, 0, 255).astype(np.uint8)

    info = dict(N=tiles.N, levels=[], bytes=0)
    skip_tiles = os.environ.get("SKIP_TILES") == "1" and (OUT / "meta.json").exists()
    if skip_tiles:
        info = json.load(open(OUT / "meta.json", encoding="utf-8"))["tiles"]
        note("SKIP_TILES: tiles left as they are")
    N = tiles.N
    is_fol = np.zeros(65536, bool)
    is_fol[fol_ids] = True
    for s in (() if skip_tiles else tiles.LEVELS):
        span = N * s
        nx, ny = -(-W // span), -(-D // span)
        (OUT / "tiles" / str(s)).mkdir(parents=True, exist_ok=True)
        if s == 1:
            (OUT / "tiles" / "1f").mkdir(parents=True, exist_ok=True)
        tot = 0
        for ty in range(ny):
            for tx in range(nx):
                a, c = ty * span - s, tx * span - s
                size = span + 2 * s
                vol, g1 = window(a, c, size)
                if vol is None:
                    vol = np.zeros((1, size, size), np.uint16)
                win = vol if s == 1 else tiles.downsample(vol, s)
                gw = g1[::s, ::s]
                Hs = win.shape[0]
                top = int(np.nonzero(win.reshape(Hs, -1).any(1))[0].max()) + 1 if win.any() else 1
                win = win[:top]
                if s == 1:
                    f = is_fol[win]
                    fol = np.where(f, win, 0).astype(np.uint16)
                    win = np.where(f, 0, win).astype(np.uint16)
                    fc, fr = tiles.column_runs(fol)
                    with gzip.GzipFile(OUT / "tiles" / "1f" / f"{tx}_{ty}.bin", "wb", compresslevel=9, mtime=0) as fh:
                        fh.write(fc.tobytes() + gw.tobytes() + fr.tobytes())
                counts, runs = tiles.column_runs(win)
                p = OUT / "tiles" / str(s) / f"{tx}_{ty}.bin"
                with gzip.GzipFile(p, "wb", compresslevel=9, mtime=0) as fh:
                    fh.write(counts.tobytes() + gw.tobytes() + runs.tobytes())
                tot += p.stat().st_size
        info["levels"].append(dict(s=s, nx=nx, ny=ny))
        info["bytes"] += tot
        note(f"tiles level {s}: {nx}x{ny} nodes, {tot / 2 ** 20:.1f} MB gz")

    # ---------- meta
    H = int(min(g.max() + 120, 400))
    lamps = []
    for _, t, p in osm.nodes_with("highway", ["street_lamp"]):
        lamps.append([int(p.x), int(p.y), int(g[min(int(p.y), D - 1), min(int(p.x), W - 1)])])
    for rank, w, h, t, line in roads:
        if rank >= 3 and h not in PAVED_ROADS:
            for d_ in np.arange(10, line.length, 38):
                q = line.interpolate(d_)
                if 0 <= q.x < W and 0 <= q.y < D and bid[int(q.y), int(q.x)] == 0:
                    lamps.append([int(q.x), int(q.y), int(g[int(q.y), int(q.x)])])
    labels, views, places = [], {}, []
    gat = lambda x, y: int(g[min(max(int(y), 0), D - 1), min(max(int(x), 0), W - 1)])
    from pyproj import Transformer
    ll = Transformer.from_crs(4326, 32632, always_xy=True)
    def resolve(L):
        """grid position and OSM height tag of a label's object (see landmarks.py)"""
        if L.get("osm") or L.get("q"):
            def ok(i, t):
                if L.get("osm"):
                    return i == L["osm"]
                return re.search(L["q"], t.get("name", ""), re.I) or re.search(L["q"], t.get("name:en", "") or "\0", re.I)
            hits = [(gm.area, gm.centroid, t) for i, _, t, gm in osm.areas if ok(i, t)]
            if not hits:
                hits = [(gm.length, gm.interpolate(0.5, normalized=True), t) for i, t, gm in osm.lines if ok(i, t)]
            if not hits:
                hits = [(0, gm, t) for i, t, gm in osm.nodes if ok(i, t)]
            if hits:
                _, c, t = max(hits, key=lambda h: h[0])
                return c.x, c.y, fnum(t.get("height"))
            note(f"label {L['name']}: no OSM match, at lon/lat")
        if "lon" not in L:
            return None
        x, y = (float(v) for v in to_grid(*ll.transform(L["lon"], L["lat"])))
        return x, y, None

    at = {}
    for L in LABELS:
        r = resolve(L)
        if r is None or not (0 <= r[0] < W and 0 <= r[1] < D):
            note(f"label {L['name']}: off the map")
            continue
        x, y, oh = r
        gz = gat(x, y)
        r0, r1, c0, c1 = max(int(y) - 8, 0), min(int(y) + 9, D), max(int(x) - 8, 0), min(int(x) + 9, W)
        top = max(int(btop[r0:r1, c0:c1].max()), gz)  # the label floats above the roof under it
        h = L.get("height") or oh
        if L.get("shape") and h:  # hand-built: no roof under the label
            top = max(top, gz + int(h))
        at[L["name"]] = (x, y, gz, top)
        labels.append({k: v for k, v in dict(name=L["name"], en=L.get("en"), kind=L["kind"], height=round(h) if h else None,
                                          year=L.get("year"), alt=L.get("alt"), x=int(x), y=int(y), z=top + 10, site=L.get("site"),
                                          far=L.get("far", 3500), g=gz, R=L.get("R", 25)).items() if v is not None})
        places.append([L["name"], L["kind"], int(x), int(y), L.get("R", 60) * 3, L.get("alt", "")])
    for name, v in VIEWS.items():
        if v.get("at") in at:
            ex, ny_, tg, _ = at[v["at"]]
        else:
            ex, ny_ = (float(q) for q in to_grid(*ll.transform(v["lon"], v["lat"])))
            tg = gat(ex, ny_)
        pos = [ex + v["dx"], tg + v["up"], ny_ + v["dy"]]
        views[name] = {"label": v["label"], "target": [ex, tg + v.get("tz", 15), ny_], "pos": pos}
    meta = dict(W=W, D=D, H=H, base=ZOFF, tiles=info, center=[I0, J0], theta_deg=0.0, crs="EPSG:32632", trams=[], jets=[],
                palette=[list(p) for p in PAL], lamps=lamps, floods=[], labels=labels, views=views,
                sources={"osm": osm.timestamp, "terrain": "Regione Liguria DTM 2023 (5 m); sea: AWS Terrain Tiles",
                         "buildings": "Regione Liguria CTR 1:5000 3D (2007-13) eave heights", "roofs": "Regione Liguria ortofoto 2022"})
    json.dump(meta, open(OUT / "meta.json", "w"), separators=(",", ":"))
    json.dump({"places": places}, open(OUT / "places.json", "w"))

    # street names: anchors every ~160 m along each named street, with its direction (the viewer writes them near
    # the camera, along the street)
    by_name = {}
    for _, t, gm in osm.lines:
        n = t.get("name")
        if n and t.get("highway") and t.get("tunnel") not in ("yes", "building_passage") and t.get("covered") != "yes":
            by_name.setdefault(n, []).append(gm)
    streets = []
    for n, gs in by_name.items():
        mg = linemerge(gs)
        for ln in getattr(mg, "geoms", [mg]):
            L = ln.length
            if L < 25:
                continue
            k = max(1, int(L // 160))
            for i in range(k):
                s_ = (i + 0.5) * L / k
                p, a, b = ln.interpolate(s_), ln.interpolate(max(s_ - 6, 0)), ln.interpolate(min(s_ + 6, L))
                if 0 <= p.x < W and 0 <= p.y < D:
                    streets.append([n, round(p.x, 1), round(p.y, 1), gat(p.x, p.y), round(math.atan2(b.y - a.y, b.x - a.x), 2)])
    json.dump({"streets": streets}, open(OUT / "streets.json", "w", encoding="utf-8"), ensure_ascii=False)
    note(f"street names: {len(by_name)} streets, {len(streets)} anchors")
    note(f"meta: {len(PAL)} palette entries, {len(labels)} labels, {len(lamps)} lamps; done")


if __name__ == "__main__":
    main()
