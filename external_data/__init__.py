"""Real-world reference datasets, kept separate from this project's own sensor pipeline.

Unlike ``synthetic/`` (fake data shaped exactly like our ESP32 payload) and
``backend/`` (real data from our own hardware, once it exists), this package
holds *other people's* real, published data - useful for sanity-checking
that "detect a leak from a time series" generalises, but never a
drop-in stand-in for our vibration/soil-moisture sensor, which is a
different measurement entirely.
"""
