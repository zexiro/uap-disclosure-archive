"""Unified sighting schema.

Every record from every source — war.gov, NUFORC, MUFON, Blue Book, news,
Reddit — is normalized into the shape below before it lands in
raw/sightings/sightings.json.

`provenance` is the load-bearing field: it lets the UI show clear
"verified by Pentagon" vs "unverified civilian eyewitness" badges, and
lets the correlation pass match official records against civilian noise
without conflating them.
"""
from __future__ import annotations

from typing import Literal, TypedDict

# Verification ladder. Treat in this order of trust when surfacing
# duplicates / correlations.
Provenance = Literal[
    "official_us_military",   # war.gov, AARO, DoD, USAF Blue Book
    "official_us_civilian",   # FBI, NARA, FAA — non-military govt
    "official_foreign",       # UK MoD, French GEIPAN, etc.
    "civilian_witness",       # NUFORC, MUFON — first-person reports
    "media_unverified",       # Reddit, Twitter/X, news, blogs, YouTube
]

VerificationStatus = Literal["official", "unverified"]

# How precise is occurred_at? "year" means we only know "1952"; "minute"
# means we have a real timestamp. Used by the correlation pass to widen
# the time window for low-precision records.
DatePrecision = Literal[
    "minute", "hour", "day", "month", "year", "decade", "unknown"
]


class Location(TypedDict, total=False):
    name: str          # original free-text, e.g. "Phoenix, AZ" or "Oak Ridge, TN"
    lat: float
    lng: float
    country: str       # ISO-2 lowercase ("us", "gb", "fr"), or empty


class MediaRef(TypedDict, total=False):
    type: Literal["photo", "video", "audio", "document"]
    url: str           # remote URL
    local_path: str    # on-disk relative path, when downloaded
    caption: str


class Sighting(TypedDict, total=False):
    # --- identity ---
    id: str                     # "<source>:<source_id>"
    source: str                 # "wargov", "nuforc", "mufon", "blue_book", ...
    source_id: str              # opaque, source-local
    source_url: str             # canonical link back to origin
    fetched_at: str             # ISO date the record was pulled

    # --- provenance ---
    provenance: Provenance
    verification_status: VerificationStatus

    # --- the sighting ---
    title: str
    summary: str                # short blurb (1-3 sentences)
    text: str                   # long-form witness narrative if available

    occurred_at: str            # ISO 8601, or empty if unknown
    occurred_at_precision: DatePrecision

    location: Location

    shape: str                  # NUFORC/MUFON: triangle, disc, light, ...
    duration_seconds: int       # seconds, if normalisable

    media: list[MediaRef]

    # --- escape hatch ---
    raw: dict                   # source-specific extras for debugging


PROVENANCE_FOR_SOURCE: dict[str, Provenance] = {
    "wargov": "official_us_military",
    "blue_book": "official_us_military",
    "aaro": "official_us_military",
    "fbi_vault": "official_us_civilian",
    "nara": "official_us_civilian",
    "uk_mod": "official_foreign",
    "geipan": "official_foreign",
    "nuforc": "civilian_witness",
    "mufon": "civilian_witness",
    "reddit": "media_unverified",
    "news": "media_unverified",
}


def verification_for(provenance: Provenance) -> VerificationStatus:
    return "official" if provenance.startswith("official_") else "unverified"
