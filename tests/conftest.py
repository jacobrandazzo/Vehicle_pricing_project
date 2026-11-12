"""Shared test setup: made-up listings, a fitted model, and guards that keep tests offline.

pytest loads this file automatically before running any test in this folder.
Fixtures defined here can be used by any test just by naming them as arguments.
"""

import os
import socket

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Draw plots off-screen: tests never open windows, and work without a display.
matplotlib.use("Agg")

# Use a fake key so importing vehicle_pricing.agent never picks up the real one
# from .env (load_dotenv doesn't override variables that are already set).
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"

from vehicle_pricing.data import vehicle_age  # noqa: E402  (must come after the env var)
from vehicle_pricing.model import fit_model  # noqa: E402

COLUMNS = ["manufacturer", "model", "year", "odometer", "price"]


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """Runs around every test: fail loudly if a test tries the network or reads a CSV.

    Blocking socket connections means no Claude API call can get out, and
    blocking pandas.read_csv means load_data() can't read the real data file.
    Connections to this machine itself (127.0.0.1) are allowed: on Windows,
    asyncio opens one to build its event loop, which Streamlit's AppTest needs.
    """
    real_connect = socket.socket.connect

    def no_network(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(sock, address, *args, **kwargs)
        raise RuntimeError("Tests must not use the network (no Claude API calls).")

    def no_csv(*args, **kwargs):
        raise RuntimeError("Tests must not load the real data file; use a fixture.")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(pd, "read_csv", no_csv)


@pytest.fixture
def listings():
    """About 20 made-up listings across several makes, shaped like cleaned data."""
    rows = [
        ("toyota", "camry", 2015, 90_000, 13_500),
        ("toyota", "camry", 2018, 45_000, 19_900),
        ("toyota", "corolla", 2012, 130_000, 7_200),
        ("toyota", "tacoma", 2016, 120_000, 26_000),
        ("toyota", "rav4", 2019, 30_000, 24_500),
        ("toyota", "prius", 2010, 180_000, 5_400),
        ("honda", "civic", 2014, 110_000, 9_800),
        ("honda", "civic", 2019, 25_000, 18_700),
        ("honda", "accord", 2016, 75_000, 15_200),
        ("honda", "cr-v", 2017, 60_000, 19_400),
        ("honda", "odyssey", 2011, 160_000, 6_900),
        ("ford", "f-150", 2015, 140_000, 17_500),
        ("ford", "f-150", 2020, 35_000, 34_000),
        ("ford", "focus", 2013, 95_000, 5_900),
        ("ford", "escape", 2017, 70_000, 13_800),
        ("ford", "mustang", 2016, 50_000, 21_000),
        ("chevrolet", "silverado 1500", 2014, 150_000, 16_500),
        ("chevrolet", "malibu", 2018, 55_000, 14_300),
        ("chevrolet", "equinox", 2015, 100_000, 9_900),
        ("chevrolet", "camaro", 2012, 85_000, 15_000),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def camry_listings():
    """44 made-up Camry listings (model years 2010-2020) posted in April 2021.

    Price falls as mileage rises and as the car gets older. The model needs at
    least 30 listings (MIN_LISTINGS), so this is bigger than `listings`. A fixed
    random seed makes the "noise" the same on every run.
    """
    rng = np.random.default_rng(seed=0)
    n = 44
    year = np.repeat(np.arange(2010, 2021), 4)  # 4 listings per model year
    posting_date = pd.Series(
        pd.Timestamp("2021-04-01", tz="UTC") + pd.to_timedelta(np.arange(n) % 30, unit="D")
    )
    age = vehicle_age(posting_date, year)
    odometer = rng.uniform(10_000, 200_000, size=n)
    price = (
        30_000
        * np.exp(-odometer / 150_000)
        * np.exp(-0.08 * age)
        * rng.uniform(0.85, 1.15, size=n)
    )
    return pd.DataFrame({
        "manufacturer": "toyota",
        "model": "camry",
        "year": year,
        "odometer": odometer,
        "price": price.round(),
        "posting_date": posting_date,
        "age": age,
    })


@pytest.fixture
def camry_model(camry_listings):
    """The price model fitted to `camry_listings` (a fixture that uses another fixture)."""
    return fit_model(camry_listings)
