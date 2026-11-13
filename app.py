"""The website: describe a used car in plain English, get its price, a 95% range and its depreciation.

Run it with `streamlit run app.py`. Streamlit runs this whole script from top
to bottom every time the user does anything (sends a message, clicks a
button), so anything that must survive between runs lives in a cache or in
st.session_state.
"""

import logging
import os

import anthropic
import pandas as pd
import streamlit as st

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import UnknownManufacturerError, comparison_years, load_data
from vehicle_pricing.limits import (
    MAX_CALLS_PER_SESSION,
    MAX_DESCRIPTION_CHARS,
    DailyCounter,
)
from vehicle_pricing.model import NoAgeSpreadError
from vehicle_pricing.pipeline import (
    Estimate,
    TooFewListingsError,
    WarningCode,
    describe_years,
    estimate,
    warning_message,
)
from vehicle_pricing.plotly_charts import depreciation_figure, residuals_figure

log = logging.getLogger(__name__)

EXAMPLES = (
    "2015 Toyota Camry, 90,000 miles",
    "2016 Toyota Tacoma, 120k miles",
    "1996-2000 Honda Civic, 200,000 miles",
)

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Used Car Price Estimator", layout="wide")


@st.cache_resource(show_spinner="Loading about 320,000 listings (the first visit takes a moment)...")
def get_vehicles() -> pd.DataFrame:
    """Load and clean the listings once per server; every visitor and rerun shares the result.

    cache_resource hands back the very same DataFrame each time, with no
    copy. cache_data would copy it on every rerun, which is slow and wasteful
    for a table this size. The price is that the DataFrame is shared, so
    nothing may change it. estimate() only reads it.
    """
    return load_data()


@st.cache_resource
def get_daily_counter() -> DailyCounter:
    """One counter for the whole server: every visitor's session gets this same object.

    A server restart (a redeploy, or the host putting the app to sleep) starts
    a new counter at 0, which can only loosen the cap. The spending limit in
    the Anthropic Console is the real backstop.
    """
    return DailyCounter()


def calls_left() -> int:
    """How many Claude calls this visit has left."""
    return MAX_CALLS_PER_SESSION - st.session_state.calls_used


def use_a_call(text: str) -> str | None:
    """Check the limits and, if they all pass, count one Claude call.

    Returns the message to show instead if a limit stops the call, else None.
    Called just before parse_description, and counts whether or not the call
    then succeeds: a failed call costs tokens too. The session limit is checked
    before the daily one, so a visitor who is out of calls doesn't use up the
    day's.
    """
    if len(text) > MAX_DESCRIPTION_CHARS:  # also checked here: buttons and tests bypass max_chars
        return f"Please keep the description under {MAX_DESCRIPTION_CHARS} characters."
    if calls_left() <= 0:
        return (f"You've used all {MAX_CALLS_PER_SESSION} descriptions for this visit. "
                "Thanks for trying it!")
    if not get_daily_counter().try_use():
        return "The app has reached today's limit. Please come back tomorrow."
    st.session_state.calls_used += 1
    return None


def configure_api_key() -> bool:
    """Say whether a Claude API key is available: Streamlit secrets first, then the environment.

    st.secrets reads .streamlit/secrets.toml locally, or the Secrets settings on
    Streamlit Community Cloud. A key found there is copied into the environment,
    where the Anthropic client reads it, so it wins over a key from .env
    (which agent.py's load_dotenv() put in the environment). The key itself is
    never shown or logged.
    """
    try:
        secret = st.secrets.get("ANTHROPIC_API_KEY")
    except FileNotFoundError:  # no secrets file at all: fine, use the environment
        secret = None
    except Exception as e:  # a broken secrets file: log only the error's type, not its text
        log.warning("Could not read Streamlit secrets (%s); using the environment", type(e).__name__)
        secret = None
    if secret:
        os.environ["ANTHROPIC_API_KEY"] = str(secret)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def plain(text: str) -> str:
    """Escape dollar signs: Streamlit's markdown reads text between two $ as LaTeX maths."""
    return text.replace("$", r"\$")


def price_summary(est: Estimate) -> str:
    """The price line, worded like the terminal app's (main.py)."""
    years = describe_years(*comparison_years(est.query))
    dropped = est.n_dropped
    plural = "" if dropped == 1 else "s"
    dropped_note = f" (dropped {dropped:,} unusual price{plural})" if dropped else ""

    result = est.result
    if result is None:
        return (
            "No mileage given, so no price estimate. "
            f"Showing {est.model.n:,} listings ({years}){dropped_note}."
        )
    return (
        f"Estimated price: ${result.predicted_price:,.0f} "
        f"(95% range ${result.lower_bound:,.0f} to ${result.upper_bound:,.0f}), "
        f"based on {result.n_listings:,} listings ({years}){dropped_note}."
    )


def reply_to(text: str) -> dict:
    """Work out the answer to one description, as an assistant message to store and show.

    A message is a dict: role, text, and the warnings (code, sentence) to show
    under it. Every failure becomes a friendly message here, never a traceback.
    Only a fully successful estimate replaces `previous`, so a failed message
    leaves the last car in place for the next follow-up. The estimate is also
    kept as `latest_estimate` for the charts; it is cleared first, so after a
    failure no charts are left under an error message.
    """
    st.session_state.latest_estimate = None

    def failure(message: str) -> dict:
        return {"role": "assistant", "text": message, "warnings": []}

    refusal = use_a_call(text)
    if refusal is not None:
        return failure(refusal)

    try:
        query = parse_description(text, st.session_state.previous)
        est = estimate(vehicles, query)
    except ParseError as e:
        return failure(f"Sorry, I couldn't use that description. {e}")
    except anthropic.APIError:  # network, rate limit, bad key: log it, don't show it
        log.exception("Claude API call failed")
        return failure(
            "The description service isn't responding right now. Please try again in a minute."
        )
    except (UnknownManufacturerError, TooFewListingsError, NoAgeSpreadError) as e:  # worded for visitors
        return failure(str(e))
    except Exception:
        log.exception("Unexpected error while answering %r", text)
        return failure("Something went wrong on my side. Please try again.")

    st.session_state.previous = query
    st.session_state.latest_estimate = est
    return {
        "role": "assistant",
        "text": price_summary(est),
        "warnings": [(code, warning_message(code, est)) for code in est.warnings],
    }


def show_body(message: dict) -> None:
    """Draw a message's text, then each of its warnings as a coloured box."""
    st.markdown(plain(message["text"]))
    for code, sentence in message.get("warnings", []):
        box = st.warning if code == WarningCode.EXTRAPOLATED else st.info
        box(plain(sentence))


def show_message(message: dict) -> None:
    """Draw one chat bubble for a message."""
    with st.chat_message(message["role"]):
        show_body(message)


def show_charts(est: Estimate) -> None:
    """The depreciation chart, and the residuals chart inside a collapsed expander.

    Only the latest answer gets charts: they are the heavy part of a rerun.
    theme=None keeps each figure's own colours instead of Streamlit's restyling.
    """
    title = " ".join(part for part in (est.query.manufacturer, est.query.model) if part).title()
    st.plotly_chart(
        depreciation_figure(est.listings, est.curves, title=title),
        theme=None,
        alt=f"Price against mileage and against model year for {title}, with the car marked",
    )

    # An expander's body runs on every rerun even while collapsed. on_change="rerun"
    # makes `.open` say whether the user has opened it, so the chart is built only then.
    details = st.expander("Residuals and method", on_change="rerun")
    with details:
        st.write(
            "A residual is a listing's asking price minus the price the model expects for "
            "its mileage and age, in dollars. A model that fits well leaves dots scattered "
            "evenly around zero, with no curve or fan shape. The shaded band is the 95% "
            "range for a car of your car's age, so most dots should sit inside it."
        )
        if details.open:
            st.plotly_chart(
                residuals_figure(est.listings, est.model, est.curves,
                                 title=f"{title}: residuals vs. odometer"),
                theme=None,
                alt=f"Residuals against odometer for {title}, with the 95% range shaded",
            )


def show_about() -> None:
    """The sidebar: how the estimate is made, where the data comes from, and who built it."""
    with st.sidebar:
        st.header("About")
        st.markdown(
            "- **Claude (Haiku)** turns your description into a structured query.\n"
            "- The app compares it with similar listings of the same make and model, "
            "within ±5 model years (or the years you give).\n"
            "- A **Box-Cox multiple regression** prices the car from its mileage, age, "
            "clean title and 4WD.\n"
            "- Outlier prices are dropped before the fit.\n"
            "- In 5-fold cross-validation, the median error was **10.6%**, and **94.9%** of "
            "real prices fell inside the 95% range."
        )

        st.subheader("Data")
        st.markdown(
            "The [Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) "
            "by Austin Reese on Kaggle (version 10): Craigslist listings from April–May 2021."
        )

        st.subheader("Limitations")
        st.markdown(
            "- Prices are in 2021 dollars.\n"
            '- Model names like "camry le" aren\'t merged into "camry" yet.'
        )

        st.divider()
        st.markdown(
            "[GitHub repository](https://github.com/jacobrandazzo/Vehicle_pricing_project)  \n"
            "Built by Jacob Randazzo · "
            "[LinkedIn](https://www.linkedin.com/in/jacob-randazzo-3a1595430)"
        )


def use_example(text: str) -> None:
    """Button callback: park the example as the next message. It runs just before the rerun."""
    st.session_state.pending = text


# session_state survives reruns; setdefault only sets a value the first time.
st.session_state.setdefault("messages", [])  # every message so far, oldest first
st.session_state.setdefault("previous", None)  # the last successful VehicleQuery
st.session_state.setdefault("latest_estimate", None)  # its Estimate, for the charts; None after a failure
# Claude calls this visit has made. A page refresh starts a new session at 0,
# which is why there is also a daily cap shared by everyone.
st.session_state.setdefault("calls_used", 0)

st.title("Used Car Price Estimator")
st.write(
    "Describe a used car in plain English and get its expected price, a 95% range, "
    "and how it depreciates, from 2021 Craigslist listings."
)

show_about()  # before the data loads, so the sidebar is there while the spinner runs
vehicles = get_vehicles()

has_key = configure_api_key()
if not has_key:
    st.info("Text input is unavailable right now.")
can_send = has_key and calls_left() > 0

for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(example, on_click=use_example, args=(example,), width="stretch",
                  disabled=not can_send)

for message in st.session_state.messages:
    show_message(message)

# chat_input returns the text on the run where the user sends it, else None.
# An example button parked its text in `pending`, so a click acts like typing.
if not has_key:
    placeholder = "Text input is unavailable right now"
elif calls_left() <= 0:
    placeholder = f"You've used all {MAX_CALLS_PER_SESSION} descriptions for this visit"
else:
    placeholder = "Describe a car, e.g. '2015 Toyota Camry, 90,000 miles'"
text = st.chat_input(placeholder, max_chars=MAX_DESCRIPTION_CHARS, disabled=not can_send)
text = text or st.session_state.pop("pending", None)

if text and has_key:  # checked here too: a disabled widget only stops the browser
    user_message = {"role": "user", "text": text}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.chat_message("assistant"):  # one bubble: spinner first, then the answer in its place
        with st.spinner("Working out the price..."):
            answer = reply_to(text)
        show_body(answer)
    st.session_state.messages.append(answer)
    # The input above was drawn before this call was counted. If that was the
    # last one, rerun at once so the page shows the input disabled.
    if calls_left() <= 0:
        st.rerun()

# Last on the page, so the charts sit under the newest answer.
if st.session_state.latest_estimate is not None:
    show_charts(st.session_state.latest_estimate)

if has_key:  # the last element, so it sits just above the chat input
    st.caption(f"{max(calls_left(), 0)} of {MAX_CALLS_PER_SESSION} descriptions left")
