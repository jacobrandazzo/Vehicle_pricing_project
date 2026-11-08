# Vehicle price prediction

Estimates a used car's price from its mileage, using Craigslist listings. It fits a curve of price against odometer and gives a typical price plus a 95% range.

You ask questions in plain English in the terminal. A Claude agent picks out the details, runs the price model, and can save plots of the data it used.

## Setup (once)

You need Python 3.12 or newer ([python.org](https://www.python.org/downloads/)) and an Anthropic API key ([console.anthropic.com](https://console.anthropic.com/)). Open a terminal in this folder and run:

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

**Mac / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

Describe a vehicle and ask for a price or a plot, for example:

```
> What's a 2015 Toyota Camry with 90,000 miles worth?
> Now plot the listings
```

A price estimate needs at least a manufacturer and a mileage. A plot needs at least a manufacturer. Follow-up questions work, and typing `quit` exits.

Each question sends requests to the Anthropic API, which is billed to your API key.

Plots are saved in the `plots` folder as `fit.png` (listings with the fitted curve) and `residuals.png`. Each new plot replaces the previous one of that type.

In VS Code, open this folder. If it asks, select the `venv` interpreter, then open `main.py` and press Run.

## Testing the model without the API

To run the model and plots without calling Claude:
```
python tools.py
```

This prints a sample Toyota estimate and plot details, and saves both plots. It doesn't use your API key.

## Files

- `main.py`: the terminal chat, and the file you run
- `agent.py`: sends your questions to Claude and runs the tools it asks for
- `tools.py`: connects the agent to the price model and saves plots
- `legacy_vehicle_pricing.py`: the price model and plotting functions
- `src/vehicle_pricing/data.py`: loads and cleans the listings, and filters them for a query
- `data/vehicles_slim.csv.gz`: the listings data (compressed, about 11 MB)
- `requirements.txt`: the libraries to install
- `.env.example`: a template for your `.env` file

`vehicles.csv`, the full 1.4 GB dataset, isn't in the repo and isn't needed to run the project.
