from nlp.geotagger.service import GeoEntity
from pipelines.news.places import dedupe_places


def test_collapses_same_city_by_city_id():
    places = [
        GeoEntity(text="Madrid", type="city", city_id=10, city_name="Madrid"),
        GeoEntity(text="MADRID", type="city", city_id=10, city_name="Madrid"),
    ]
    out = dedupe_places(places)
    assert len(out) == 1
    assert out[0].text == "Madrid"          # first kept


def test_collapses_region_by_geonames_id():
    places = [
        GeoEntity(text="Cataluña", type="region", name="Cataluña", geonames_id=3336901),
        GeoEntity(text="Catalunya", type="region", name="Cataluña", geonames_id=3336901),
    ]
    assert len(dedupe_places(places)) == 1


def test_distinct_unresolved_streets_stay_separate():
    # two streets the b4c API never resolved (no edge_ids), same imputed city, different text
    places = [
        GeoEntity(text="Calle Mayor", type="street", city_id=10, edge_ids=[]),
        GeoEntity(text="Calle Real", type="street", city_id=10, edge_ids=[]),
    ]
    assert len(dedupe_places(places)) == 2


def test_same_street_by_edge_ids_collapses_regardless_of_order():
    places = [
        GeoEntity(text="Gran Vía", type="street", city_id=10, edge_ids=[3, 1, 2]),
        GeoEntity(text="la Gran Via", type="street", city_id=10, edge_ids=[1, 2, 3]),
    ]
    assert len(dedupe_places(places)) == 1


def test_location_collapses_by_rounded_coords_when_unresolved():
    places = [
        GeoEntity(text="el parque", type="location", lat=40.41680, lon=-3.70380),
        GeoEntity(text="el Parque", type="location", lat=40.41683, lon=-3.70379),
    ]
    assert len(dedupe_places(places)) == 1


def test_preserves_order_of_distinct_places():
    places = [
        GeoEntity(text="Madrid", type="city", city_id=10),
        GeoEntity(text="Sevilla", type="city", city_id=20),
        GeoEntity(text="Madrid", type="city", city_id=10),
    ]
    out = dedupe_places(places)
    assert [p.text for p in out] == ["Madrid", "Sevilla"]
