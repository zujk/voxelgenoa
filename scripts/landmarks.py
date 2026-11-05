"""Genoa labels and camera views.

A label sits on its OSM object: `osm` (a way or relation id) or else `q`, a case-insensitive regex on the name,
searched over the whole map (the largest matching area wins, then the longest line, then a node). `lon`/`lat` only
for what OSM has no object for. Heights come from the OSM `height` tag unless given here.
wall / roof: RGB for the building under the label; stripes: black-and-white banded walls; mat: "SOLID" for walls
without windows (towers, gates), "GLASS" for curtain walls. parts: several OSM footprints styled together (towers on
a podium); top: height in metres where OSM and the CTR are wrong; spire: a pointed top this many metres high;
shape: built by hand ("bigo", "sphere"), the footprint cleared; lantern: the lamp room on top.
kind: tower, church, palace, station, museum, fountain, gate, quarter, port, bridge, castle ...
"""
LABELS = [
    dict(name="Lanterna", en="Genoa's lighthouse", kind="tower", osm=19515124, wall=(238, 232, 218), mat="SOLID", roof=(150, 152, 150), lantern=True, year=1543, alt="lighthouse faro", far=9000, R=30, site="https://www.lanternadigenova.it"),
    dict(name="Palazzo Ducale", en="Doge's Palace", kind="palace", q=r"^palazzo ducale$", wall=(226, 216, 196), year=1339, alt="doge palace", far=5000, R=60),
    dict(name="Cattedrale di San Lorenzo", en="Genoa Cathedral", kind="church", q=r"^cattedrale di san lorenzo", wall=(236, 232, 222), mat="SOLID", stripes=True, height=40, year=1118, alt="duomo cathedral", far=5000, R=45),
    dict(name="Piazza De Ferrari", en="the main square, with its bronze fountain", kind="fountain", q=r"^piazza de ferrari$", alt="fontana fountain", far=4000, R=40),
    dict(name="Teatro Carlo Felice", en="Opera house", kind="theatre", q=r"^teatro carlo felice", year=1828, alt="opera", far=3000, R=45),
    dict(name="Acquario di Genova", en="Genoa Aquarium", kind="museum", q=r"^acquario di genova$", year=1992, alt="aquarium", far=5000, R=60, site="https://acquariodigenova.it"),
    dict(name="Il Bigo", en="Renzo Piano's harbour crane", kind="tower", osm=22969165, shape="bigo", height=40, year=1992, alt="bigo piano", far=4500, R=30),
    dict(name="Palazzo San Giorgio", en="Palace of the Bank of St George", kind="palace", q=r"^palazzo san giorgio$", wall=(206, 128, 96), year=1260, alt="san giorgio", far=3500, R=45),
    dict(name="Galata Museo del Mare", en="Maritime museum", kind="museum", osm=207920143, year=2004, alt="galata sea museum", far=3500, R=50),
    dict(name="Stazione Principe", en="Genova Piazza Principe station", kind="station", osm=18462349, alt="railway principe", far=5000, R=70),
    dict(name="Stazione Brignole", en="Genova Brignole station", kind="station", osm=988663262, alt="railway brignole", far=5000, R=70),
    dict(name="Basilica di Carignano", en="Santa Maria Assunta in Carignano", kind="church", osm=610430306, wall=(222, 206, 186), roof=(120, 124, 128), height=60, year=1552, alt="carignano", far=4500, R=45),
    dict(name="Porta Soprana", en="Medieval city gate", kind="gate", q=r"^porta soprana", wall=(178, 160, 136), mat="SOLID", year=1155, alt="gate colombo", far=2500, R=30),
    dict(name="Ponte Monumentale", en="Monumental bridge on Via XX Settembre", kind="bridge", q=r"^ponte monumentale", year=1895, alt="bridge xx settembre", far=3000, R=45),
    dict(name="Torre Piacentini", en="1940 skyscraper on Piazza Dante", kind="tower", osm=23515467, wall=(214, 206, 192), year=1940, alt="grattacielo piacentini dante", far=6000, R=40),
    dict(name="Torre degli Embriaci", en="Medieval tower", kind="tower", osm=227715388, wall=(164, 146, 124), mat="SOLID", alt="embriaci", far=2000, R=15),
    dict(name="Palazzo Reale", en="Royal Palace, Via Balbi", kind="palace", q=r"palazzo reale", alt="royal palace via balbi", far=3500, R=50),
    dict(name="Palazzo Rosso", en="Palazzo Rosso, Strada Nuova", kind="palace", q=r"^palazzo rosso$", wall=(176, 74, 62), alt="via garibaldi", far=2500, R=40),
    dict(name="Castello D'Albertis", en="Neo-Gothic castle above the harbour", kind="castle", osm=154545927, year=1892, alt="castle museo culture del mondo", far=3500, R=35),
    dict(name="Magazzini del Cotone", en="Old Port cotton warehouses", kind="museum", osm=725294304, alt="porto antico", far=3500, R=60),
    dict(name="Loggia dei Mercanti", en="Merchants' loggia, Piazza Banchi", kind="palace", q=r"^loggia della mercanzia", alt="banchi", far=2000, R=30),
    dict(name="Ponte Parodi", en="Ponte Parodi pier, being rebuilt", kind="port", osm=945131264, alt="parodi pier", far=3500, R=60),
    dict(name="Matitone", en="The 'big pencil' office tower", kind="tower", osm=23515380, wall=(112, 116, 122), roof=(64, 66, 72), spire=14, year=1992, alt="san benigno pencil", far=6000, R=40),
    dict(name="Torre MSC", en="MSC headquarters, San Benigno (2014)", kind="tower", osm=321110558, parts=[566741290, 566741295, 566741312], top=100, wall=(70, 92, 120), mat="GLASS", year=2014, alt="msc san benigno", far=6000, R=40),
    dict(name="World Trade Center", en="WTC Genova, San Benigno", kind="tower", osm=245309014, wall=(78, 98, 112), mat="GLASS", year=1992, alt="wtc san benigno", far=6000, R=35),
    dict(name="Corte Lambruschini", en="Office and residential towers by Brignole", kind="tower", osm=13861109, parts=[13861109, 43175920, 245394201, 15512852], wall=(86, 98, 108), mat="GLASS", year=1990, alt="lambruschini brignole", far=5000, R=50),
    dict(name="Biosfera", en="Renzo Piano's glass sphere", kind="museum", osm=23500617, shape="sphere", height=22, year=2001, alt="biosphere bolla piano", far=3500, R=20),
    dict(name="Torre San Vincenzo", en="Office tower by Brignole", kind="tower", osm=23602076, alt="san vincenzo", far=4000, R=30),
    dict(name="Sopraelevata", en="Elevated road along the waterfront (Strada Aldo Moro)", kind="bridge", q=r"^(strada )?(sopraelevata|aldo moro)", alt="elevated highway aldo moro", far=3000, R=60),
    dict(name="Castelletto", en="Spianata Castelletto lookout", kind="quarter", q=r"spianata castelletto|belvedere.*montaldo", lon=8.9288, lat=44.4118, alt="belvedere montaldo lift", far=3000, R=40),
]

# camera on a label (`at`, its name) or at lon/lat: target at the ground + tz, camera offset dx east, dy south,
# up above the target's ground (metres)
VIEWS = {
    "overview": dict(label="Overview", lon=8.9300, lat=44.4090, dx=-700, dy=1500, up=650, tz=20),
    "ferrari": dict(label="Piazza De Ferrari", at="Piazza De Ferrari", dx=180, dy=380, up=260, tz=5),
    "lanterna": dict(label="Lanterna", at="Lanterna", dx=260, dy=260, up=110, tz=45),
    "porto": dict(label="Porto Antico", at="Acquario di Genova", dx=-150, dy=600, up=240, tz=10),
    "centro": dict(label="Centro storico", at="Cattedrale di San Lorenzo", dx=-250, dy=420, up=230, tz=20),
    "brignole": dict(label="Brignole", at="Stazione Brignole", dx=350, dy=450, up=200, tz=20),
    "principe": dict(label="Principe", at="Stazione Principe", dx=-300, dy=450, up=220, tz=20),
    "piacentini": dict(label="Piazza Dante", at="Torre Piacentini", dx=300, dy=330, up=160, tz=60),
    "sanbenigno": dict(label="San Benigno towers", at="Torre MSC", dx=330, dy=380, up=170, tz=55),
    "skyline": dict(label="Skyline from the sea", at="Palazzo Ducale", dx=-200, dy=1600, up=110, tz=50),
    "city": dict(label="Whole map", lon=8.9190, lat=44.4100, dx=0, dy=2200, up=2900, tz=20),
}
