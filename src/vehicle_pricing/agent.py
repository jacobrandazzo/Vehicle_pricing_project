"""Turn a plain-English vehicle description into a validated VehicleQuery using Claude."""

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from vehicle_pricing.schemas import VehicleQuery

__all__ = ["ParseError", "parse_description"]

load_dotenv()  # lets a local .env supply ANTHROPIC_API_KEY; real environment variables win

# Created once and reused; reads ANTHROPIC_API_KEY from the environment.
_client = anthropic.Anthropic()

_MODEL = "claude-haiku-5-5"
_MAX_TOKENS = 300  # one small tool call; nothing else is generated
_TOOL_NAME = "record_vehicle_query"

_TOOL = {
    "name": _TOOL_NAME,
    "description": (
        "Record the vehicle described by the user as structured fields. "
        "Set only fields the description states or clearly implies; leave the rest out."
    ),
    "input_schema": VehicleQuery.model_json_schema(),
}

_SYSTEM_PROMPT = """You extract used-vehicle details from a short description.
- manufacturer is the make in lowercase. If only the model is given, use its usual make (e.g. "Camry" -> "toyota").
- model is the model name in lowercase, e.g. "civic", "f-150".
- A single model year sets both year_min and year_max to that year. A range like "1996-2000" sets year_min=1996 and year_max=2000.
- odometer is the mileage in miles as a number: "90,000 miles" and "90k" are both 90000.
- Leave out any field the description doesn't mention."""


class ParseError(Exception):
    """The description couldn't be turned into a valid VehicleQuery. str(e) is safe to show users."""


def parse_description(text: str) -> VehicleQuery:
    """Ask Claude to read a vehicle description and return it as a validated VehicleQuery.

    Raises ParseError with a short, user-friendly message if Claude doesn't return
    the fields, or returns fields that fail validation. API failures (network,
    authentication, rate limits) are raised as the SDK's own exceptions.
    """
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": _TOOL_NAME},  # always answer with this tool
        messages=[{"role": "user", "content": text}],
    )
    print(response.usage)  # temporary: shows token counts for each call

    if response.stop_reason == "refusal":
        raise ParseError("Sorry, I can't help with that description.")
    if response.stop_reason == "max_tokens":
        raise ParseError("That description was too long to read. Try a shorter one.")

    tool_call = next(
        (block for block in response.content
         if block.type == "tool_use" and block.name == _TOOL_NAME),
        None,
    )
    if tool_call is None:
        raise ParseError(
            "Sorry, I couldn't read that description. "
            "Try something like '2015 Toyota Camry, 90,000 miles'."
        )

    try:
        return VehicleQuery.model_validate(tool_call.input)
    except ValidationError as e:
        raise ParseError(_friendly_message(e)) from e


def _friendly_message(error: ValidationError) -> str:
    """A one-line explanation of what was wrong with the parsed fields."""
    problems = error.errors()
    if any(p["type"] == "missing" and p["loc"] == ("manufacturer",) for p in problems):
        return "Please include the vehicle's make or model, e.g. 'Toyota Camry'."

    fields = sorted({str(p["loc"][0]) for p in problems if p["loc"]})
    if fields:
        names = ", ".join(field.replace("_", " ") for field in fields)
        return f"Some details didn't look right ({names}). Please check them and try again."
    # Model-level checks (e.g. year_min > year_max) have no field name.
    return f"Some details didn't look right: {problems[0]['msg'].removeprefix('Value error, ')}."
