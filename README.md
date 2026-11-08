# Vehicle price prediction

Estimates a used car's price from its mileage, using Craigslist listings. It fits a curve of price against odometer and gives a typical price plus a 95% range.

You describe a vehicle in plain English in the terminal. Claude picks out the details, then the price model runs and shows plots of the listings it used.

## Setup (once)

You need Python 3.12 or newer ([python.org](https://www.python.org/downloads/)) and an Anthropic API key ([console.anthropic.com](https://console.anthropic.com/)). Open a terminal in this folder and run:

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -e .
copy .env.example .env
```

**Mac / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

This creates a `venv` folder holding a private copy of Python with the libraries the project needs, so nothing else on your computer is changed.

Then open `.env` and paste your key after `ANTHROPIC_API_KEY=`. Git ignores `.env`, so your key stays out of the repo. Never put the key in `.env.example` or in the code.

If Windows PowerShell refuses to run `activate` with a "running scripts is disabled" error, skip activating and start commands with `venv\Scripts\python` instead, for example `venv\Scripts\python main.py`.

## Running

With the venv activated:
```
python main.py
```

Describe a vehicle, for example:

```
> 2015 Toyota Camry, 90,000 miles
```

It prints the estimated price and its 95% range, then opens two plot windows: the listings with the fitted curve (your vehicle marked on it), and the residuals. Close both windows to ask again; typing `quit` exits.

A description needs at least a make or model. Without a mileage you get the plots but no price. Follow-ups change the last vehicle: "what about 120k miles?", "make it a 2018", or "only manual ones". Describing a different vehicle ("now a Honda Civic") starts fresh.

Each description sends one request to the Anthropic API, which is billed to your API key.

In VS Code, open this folder. If it asks, select the `venv` interpreter, then open `main.py` and press Run.

## Testing the model without the API

To check the data and model without calling Claude:
```
python make_baseline.py --offline
```

This runs three fixed queries through filtering, fitting and prediction and writes the results to `baseline_offline.txt`. It doesn't use your API key. If you change the code and the file changes, the results changed.

## Files

- `main.py`: the terminal app, and the file you run
- `src/vehicle_pricing/agent.py`: asks Claude to turn a description into a `VehicleQuery`
- `src/vehicle_pricing/schemas.py`: the `VehicleQuery` and `PriceResult` data shapes
- `src/vehicle_pricing/data.py`: loads and cleans the listings, and filters them for a query
- `src/vehicle_pricing/model.py`: fits the price curve and predicts a price with its range
- `src/vehicle_pricing/plots.py`: the fit and residual plots
- `make_baseline.py`: writes `baseline.txt` and `baseline_offline.txt`, reference results for checking changes
- `data/vehicles_slim.csv.gz`: the listings data (compressed, about 11 MB)
- `pyproject.toml`, `requirements.txt`: package settings and the libraries `pip install -e .` installs
- `.env.example`: a template for your `.env` file

`vehicles.csv`, the full 1.4 GB dataset, isn't in the repo and isn't needed to run the project.
