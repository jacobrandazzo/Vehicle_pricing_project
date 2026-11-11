"""Validated data shapes for a vehicle price query and its result."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Limits shared by the fields below.
EARLIEST_MODEL_YEAR = 1900
LATEST_MODEL_YEAR = date.today().year + 1  # next year's models are already on sale
MAX_ODOMETER_MILES = 1_000_000
MIN_LISTINGS = 30  # the price model refuses to fit on fewer listings than this
YEAR_WINDOW = 5  # with only a model year, compare against listings this many years either side

# The exact values used in the listings data.
Condition = Literal["excellent", "fair", "good", "like new", "new", "salvage"]
Fuel = Literal["diesel", "electric", "gas", "hybrid", "other"]
Transmission = Literal["automatic", "manual", "other"]
Cylinders = Literal[
    "3 cylinders",
    "4 cylinders",
    "5 cylinders",
    "6 cylinders",
    "8 cylinders",
    "10 cylinders",
    "12 cylinders",
    "other",
]
Drive = Literal["4wd", "fwd", "rwd"]


class VehicleQuery(BaseModel):
    """The vehicle to price, and the filters that choose comparison listings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manufacturer: str = Field(
        min_length=1,
        max_length=50,
        description="Make, e.g. 'toyota'. Text label, no units.",
    )
    odometer: float | None = Field(
        default=None,
        gt=0,
        le=MAX_ODOMETER_MILES,
        description=(
            "Mileage of the vehicle being priced, in miles. "
            "Required for a price estimate; leave unset for plots."
        ),
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Model name, e.g. 'camry'. Text label, no units.",
    )
    year: int | None = Field(
        default=None,
        ge=EARLIEST_MODEL_YEAR,
        le=LATEST_MODEL_YEAR,
        description="Model year of the vehicle being priced, in calendar years.",
    )
    year_min: int | None = Field(
        default=None,
        ge=EARLIEST_MODEL_YEAR,
        le=LATEST_MODEL_YEAR,
        description="Earliest model year of comparison listings, in calendar years.",
    )
    year_max: int | None = Field(
        default=None,
        ge=EARLIEST_MODEL_YEAR,
        le=LATEST_MODEL_YEAR,
        description="Latest model year of comparison listings, in calendar years.",
    )
    condition: Condition | None = Field(
        default=None,
        description="Listed condition, e.g. 'good'. Category, no units.",
    )
    fuel: Fuel | None = Field(
        default=None,
        description="Fuel type, e.g. 'gas'. Category, no units.",
    )
    transmission: Transmission | None = Field(
        default=None,
        description="Transmission type, e.g. 'automatic'. Category, no units.",
    )
    cylinders: Cylinders | None = Field(
        default=None,
        description="Engine size as a count of cylinders, e.g. '6 cylinders'.",
    )
    drive: Drive | None = Field(
        default=None,
        description="Drivetrain, e.g. '4wd'. Category, no units.",
    )

    @field_validator(
        "manufacturer",
        "model",
        "condition",
        "fuel",
        "transmission",
        "cylinders",
        "drive",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        """Trim spaces and lowercase text, so 'Toyota ' and 'toyota' match."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def check_year_order(self) -> Self:
        """Reject a year range that runs backwards, or that leaves out the car's own year."""
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            raise ValueError("year_min cannot be greater than year_max")
        if self.year is not None:
            if self.year_min is not None and self.year < self.year_min:
                raise ValueError("year cannot be earlier than year_min")
            if self.year_max is not None and self.year > self.year_max:
                raise ValueError("year cannot be later than year_max")
        return self


class PriceResult(BaseModel):
    """A predicted price, its prediction interval, and what produced them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_price: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Predicted median price, in US dollars.",
    )
    lower_bound: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Lower end of the prediction interval, in US dollars.",
    )
    upper_bound: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Upper end of the prediction interval, in US dollars.",
    )
    n_listings: int = Field(
        ge=MIN_LISTINGS,
        description="Number of comparison listings used to fit the model, as a count.",
    )
    extrapolated: bool = Field(
        description=(
            "True if the query's mileage is outside the mileage range of the "
            "comparison listings, so the estimate is less reliable. True/false, no units."
        ),
    )
    query: VehicleQuery = Field(
        description="The query that produced this result. Units are set per field.",
    )

    @model_validator(mode="after")
    def check_query_has_odometer(self) -> Self:
        """A price is always for a specific mileage, so the query must have one."""
        if self.query.odometer is None:
            raise ValueError("query.odometer must be set for a price result")
        return self

    @model_validator(mode="after")
    def check_price_order(self) -> Self:
        """Require lower_bound <= predicted_price <= upper_bound."""
        if not self.lower_bound <= self.predicted_price <= self.upper_bound:
            raise ValueError(
                "Prices must satisfy lower_bound <= predicted_price <= upper_bound"
            )
        return self
