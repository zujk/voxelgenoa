# Genova in voxels

A 4800 × 2800 m diorama of central Genoa at 1 m voxels: the Lanterna and Ponte Parodi in the west, the Porto Antico,
the centro storico and Piazza De Ferrari, Brignole and the Foce in the east, and the hills up to Castelletto and
Righi behind. The viewer is a fork of [stared/warsaw-vibe-voxel](https://github.com/stared/warsaw-vibe-voxel) (MIT),
in the style of MagicaVoxel renders (ephtracy). Time of day, date and weather are sliders.

```sh
cd web && npm install && npm run dev     # http://localhost:5173  (?view=lanterna&hour=19&tilt=0&ui=0)
```

The viewer streams a quadtree of tiles (`web/public/data/tiles/{1,2,4,8}/x_y.bin`, 128 × 128 cells of 1, 2, 4 or 8 m),
meshed in workers; see `scripts/tiles.py` for the format and `web/src/stream.ts` for the streaming.

## Data (scripts/)

The Warsaw original fuses GUGiK LiDAR. Liguria's 1 m LiDAR is only served as rendered images (MASE WMS), and the 50 cm
coastal survey (ISPRA Virgeo) needs an account, so this build uses open data that needs no registration:

| Source | Used for |
|---|---|
| **Regione Liguria CTR 1:5000 3D, "Edificato"** (WFS `M1237:L3165`, surveyed 2007–13) | Building heights: each footprint carries its eave height above sea level, so wall height = eave − ground, per volume. Buildings the CTR has and OSM lacks are added |
| **Regione Liguria DTM ed. 2023** (WCS `M2056__L6952`, 5 m) | Ground relief, bilinear to 1 m |
| **Regione Liguria colour orthophoto 2022** (WMS `M2555:L10401`, fetched at 0.5 m) | Roof colours, cell by cell; whether a roof is pitched (tiles or slate) or a terrace |
| **OpenStreetMap** (Geofabrik extract for north-west Italy, or Overpass) | Footprints, tags (height, colours, roof shape where present), roads, bridges, rails, parks and woods, the coastline, trees and street lamps |
| **AWS Terrain Tiles** (terrarium z15) | Fallback relief where the DTM has no data (sea, edges) |
| **Open-Meteo** (ERA5) | Typical weather by day of the year (`scripts/climate.py`) |

`scripts/fetch_liguria.py` downloads the three regional layers into `data/raw/liguria/`.

What that still costs against the Warsaw build:
- The CTR gives the eave, not the ridge: pitched roofs are a generated hip (up to 6 m) on footprints whose orthophoto
  colour reads as tiles or slate. There are no measured roof shapes.
- The CTR is 2007–13: anything built or demolished since then follows OSM (an OSM `height` tag wins over the CTR).
- Trees are OSM nodes plus a seeded scatter over mapped woods, scrub and parks, not a measured canopy.
- Facade colours are drawn from a Genoese palette (ochre, salmon, yellow, rust, cream) by a hash of the OSM id unless
  OSM has `building:colour`.
- Bridge decks (the Sopraelevata, viaducts) are interpolated between the ground at each end; pillars every 24 m.
- No windows map, foliage seasons, river gauge or traffic: those layers need data this build does not have.

## Rebuild the data

```sh
uv sync
curl -L -o data/raw/geofabrik/nord-ovest.osm.pbf https://download.geofabrik.de/europe/italy/nord-ovest-latest.osm.pbf
sh scripts/build_data.sh          # OSM cut, terrain, then tiles into web/public/data
```

Curated labels and camera views are in `scripts/landmarks.py`. A label snaps to the OSM building whose name matches
its regex. The same entries can restyle their building: wall and roof colours, glass curtain walls, the cathedral's
bands, a height from a better source (Torre MSC, 100 m), the Matitone's point, and hand-built voxels for the Bigo,
the Biosfera and the Lanterna's lamp room. Elsewhere office towers over 45 m and OSM glass/mirror facades get glass,
and the CTR's measured heights win over OSM `height` tags unless the CTR is well below them (newer buildings).

## Sources and licence

Code: MIT (© 2026 Piotr Migdał for the original viewer). Generated data in `web/public/data` derives from
OpenStreetMap, © OpenStreetMap contributors, under the [ODbL](https://opendatacommons.org/licenses/odbl/).
Buildings, terrain and orthophoto from the [Regione Liguria geoportal](https://geoportal.regione.liguria.it/) open
services; fallback terrain from [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/); weather from
[Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
