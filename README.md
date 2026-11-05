# Vehicle price prediction

Estimates a used car's price from its mileage, using Craigslist listings. It fits a curve of price against odometer and gives a typical price plus a 95% range.

## Setup (once)

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/)). Open a terminal in this folder and run:

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This creates a `venv` folder holding a private copy of Python with the libraries the script needs, so nothing else on your computer is changed.

If Windows PowerShell refuses to run `activate` with a "running scripts is disabled" error, skip activating and run the script with `venv\Scripts\python vehicle_pricing.py` instead.

## Running

With the venv activated:
```
python vehicle_pricing.py
```

It prints the estimate and saves two plots next to the script: `fit.png` (listings with the fitted curve) and `residuals.png`.

In VS Code, open this folder. If it asks, select the `venv` interpreter, then press Run.

To try a different car, edit the settings near the bottom of `vehicle_pricing.py` (`manufacturer`, `year_range`, `condition`, and so on).

## Files

- `vehicle_pricing.py`: the model and an example at the bottom
- `vehicles_slim.csv.gz`: the listings data (compressed, about 11 MB)
- `requirements.txt`: the libraries to install
