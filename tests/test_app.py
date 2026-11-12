"""The Streamlit app answers a description, keeps the last car for follow-ups, and fails politely.

Streamlit's AppTest runs app.py in this process without a browser. Claude and the
real listings are replaced: load_data returns the made-up Camry listings, and
parse_description is a stand-in that records how it was called.
"""

import re
from pathlib import Path
from unittest.mock import Mock

import anthropic
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from vehicle_pricing.agent import ParseError
from vehicle_pricing.schemas import VehicleQuery

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CAMRY_2015 = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)

# Markdown text has its dollar signs escaped (\$), or Streamlit would read $...$ as maths.
PRICE_LINE = re.compile(
    r"Estimated price: \\\$[\d,]+ \(95% range \\\$[\d,]+ to \\\$[\d,]+\), "
    r"based on \d+ listings \(2010–2020 model years\)"
)


@pytest.fixture
def vehicles(camry_listings):
    """The Camry listings plus the two columns the model reads (as in test_pipeline.py)."""
    return camry_listings.assign(title_status="clean", drive=np.nan)


@pytest.fixture
def parse(monkeypatch):
    """The stand-in for parse_description: a Mock that returns the 2015 Camry by default.

    Tests change what it does with `parse.side_effect = ...`, and read how it
    was called from `parse.call_args_list`.
    """
    mock = Mock(side_effect=lambda text, previous=None: CAMRY_2015)
    monkeypatch.setattr("vehicle_pricing.agent.parse_description", mock)
    return mock


@pytest.fixture
def app(monkeypatch, vehicles, parse):
    """app.py loaded and run once, with the fake listings and the fake parser."""
    monkeypatch.setattr("vehicle_pricing.data.load_data", lambda: vehicles)
    # cache_resource keeps its result for the whole process, so a cached copy
    # from an earlier test would otherwise be reused.
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    yield at
    st.cache_resource.clear()


def texts(at):
    """Every markdown text in the main area (not the sidebar), in order."""
    return [element.value for element in at.main.markdown]


def send(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, [e.value for e in at.exception]


def test_description_gets_a_price_line_and_charts(app):
    assert not app.exception

    send(app, "2015 Camry, 90,000 miles")

    assert any(PRICE_LINE.search(t) for t in texts(app))
    assert len(app.get("plotly_chart")) == 1  # the residuals chart waits inside its expander
    assert [e.label for e in app.expander] == ["Residuals and method"]
    assert any("Built by Jacob Randazzo" in t for t in (m.value for m in app.sidebar.markdown))


def test_follow_up_gets_the_last_query_as_previous(app, parse):
    send(app, "2015 Camry, 90,000 miles")
    send(app, "what about 120k miles?")

    first, second = parse.call_args_list
    assert first.args == ("2015 Camry, 90,000 miles", None)
    assert second.args == ("what about 120k miles?", CAMRY_2015)


def test_a_description_that_cannot_be_read_is_explained_and_keeps_the_last_car(app, parse):
    send(app, "2015 Camry, 90,000 miles")
    parse.side_effect = ParseError("Please include the vehicle's make or model.")

    send(app, "asdf")

    assert "Sorry, I couldn't use that description. Please include" in texts(app)[-1]
    assert app.session_state["previous"] == CAMRY_2015
    assert app.get("plotly_chart") == []  # no charts under an error message


def test_a_claude_api_error_is_not_shown_to_the_user(app, parse):
    parse.side_effect = anthropic.APIError("secret internal detail", request=None, body=None)

    send(app, "2015 Camry, 90,000 miles")

    last = texts(app)[-1]
    assert "isn't responding right now" in last
    assert "secret internal detail" not in " ".join(texts(app))
    assert app.session_state["previous"] is None


def test_no_matching_listings_gets_a_friendly_message(app, parse):
    parse.side_effect = lambda text, previous=None: VehicleQuery(
        manufacturer="toyota", model="camry", year=1981, odometer=90_000
    )

    send(app, "1981 Camry")

    assert "couldn't find any listings like that" in texts(app)[-1]
    assert app.get("plotly_chart") == []


def test_an_example_button_acts_like_typing_it(app, parse):
    example = app.button[0].label

    app.button[0].click().run()

    assert not app.exception
    assert parse.call_args_list[0].args == (example, None)
    assert any(PRICE_LINE.search(t) for t in texts(app))
    assert [m["role"] for m in app.session_state["messages"]] == ["user", "assistant"]
