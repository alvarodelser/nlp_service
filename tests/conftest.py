import pytest


@pytest.fixture
def madrid_text():
    return (
        "El Ayuntamiento de Madrid ha aprobado la ampliación del carril bici "
        "en la Gran Vía. Las obras comenzarán en junio con un presupuesto de "
        "2,4 millones de euros."
    )


@pytest.fixture
def barcelona_text():
    return (
        "Barcelona supera a Madrid en kilómetros de carril bici según un estudio "
        "reciente. La capital catalana lidera el ranking nacional de movilidad "
        "sostenible, seguida de cerca por Sevilla."
    )


@pytest.fixture
def national_text():
    return (
        "El Ministerio de Transportes ha presentado la nueva Estrategia Nacional "
        "de Movilidad Ciclista 2026-2030 ante el Congreso. El plan dotará con "
        "2.000 millones a municipios de todo España."
    )


@pytest.fixture
def regional_text():
    return (
        "La Comunidad de Madrid ha aprobado un plan para conectar los municipios "
        "del corredor del Henares mediante una red ciclista interurbana de 120 km. "
        "La inversión beneficiará a Alcalá de Henares, Torrejón de Ardoz y Coslada."
    )
