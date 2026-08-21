"""Reverse geocoding via Nominatim (OpenStreetMap) — gratuito, nessuna
API key, richiede solo uno User-Agent identificativo per la policy
d'uso: https://operations.osmfoundation.org/policies/nominatim/
(max ~1 richiesta/secondo, va bene per un bot personale con traffico
basso; per volumi più alti andrebbe sostituita con un'istanza
self-hosted o un provider a pagamento).
"""

from __future__ import annotations

import requests

_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"


def reverse_geocode(lat: float, lon: float, timeout: float = 10.0) -> str:
    """Risolve coordinate geografiche nel nome della località
    (tipicamente un comune). Solleva ValueError se Nominatim non
    restituisce un'informazione di comune per quelle coordinate."""
    resp = requests.get(
        _ENDPOINT,
        params={
            "format": "json",
            "lat": lat,
            "lon": lon,
            "zoom": 10,
            "addressdetails": 1,
        },
        headers={"User-Agent": "Bordeus/1.0 (mattemin.dev@gmail.com)"},
        timeout=timeout,
    )
    resp.raise_for_status()
    address = resp.json().get("address", {})

    # Ordine di affidabilità: a seconda della zona (urbana/rurale)
    # Nominatim valorizza campi diversi per indicare il comune.
    for field in ("city", "town", "village", "municipality"):
        value = address.get(field)
        if value:
            return value

    raise ValueError("nessuna informazione di comune trovata per le coordinate date")
