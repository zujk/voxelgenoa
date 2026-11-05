#!/bin/sh
# Full data pipeline: OSM (Geofabrik pbf cut to the AOI, or Overpass) -> terrain -> tiles -> web/public/data
set -e
cd "$(dirname "$0")/.."
P=data/raw/geofabrik/nord-ovest.osm.pbf
if [ ! -s data/raw/osm.json ]; then
  if [ -s $P ]; then
    BB=$(PYTHONPATH=scripts uv run python -c "from aoi import lonlat_bounds; print('%.5f,%.5f,%.5f,%.5f' % lonlat_bounds(100))")
    uv run --with osmium python scripts/osm_pbf.py $P $BB data/raw/osm.json
  else
    uv run python scripts/fetch_osm.py
  fi
fi
uv run python scripts/fetch_osm.py   # skips what exists; fetches the fallback terrain
PYTHONPATH=scripts uv run python scripts/fetch_liguria.py   # DTM 2023, CTR 3D buildings, 2022 orthophoto
PYTHONPATH=scripts uv run python scripts/climate.py
PYTHONPATH=scripts uv run python scripts/build.py
