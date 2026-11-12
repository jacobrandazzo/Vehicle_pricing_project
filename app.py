"""The website: describe a used car in plain English, get its price, a 95% range and its depreciation.

Run it with `streamlit run app.py`. Streamlit runs this whole script from top
to bottom every time the user does anything (sends a message, clicks a
button), so anything that must survive between runs lives in a cache or in
st.session_state.
"""

import logging

import anthropic
import pandas as pd
import streamlit as st

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import comparison_years, load_data
from vehicle_pricing.pipeline import (
    Estimate,
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
    except ValueError as e:  # unknown make, or too few listings: the wording is user-safe
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

st.title("Used Car Price Estimator")
st.write(
    "Describe a used car in plain English and get its expected price, a 95% range, "
    "and how it depreciates, from 2021 Craigslist listings."
)

show_about()  # before the data loads, so the sidebar is there while the spinner runs
vehicles = get_vehicles()

for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(example, on_click=use_example, args=(example,), width="stretch")

for message in st.session_state.messages:
    show_message(message)

# chat_input returns the text on the run where the user sends it, else None.
# An example button parked its text in `pending`, so a click acts like typing.
text = st.chat_input("Describe a car, e.g. '2015 Toyota Camry, 90,000 miles'")
text = text or st.session_state.pop("pending", None)

if text:
    user_message = {"role": "user", "text": text}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.chat_message("assistant"):  # one bubble: spinner first, then the answer in its place
        with st.spinner("Working out the price..."):
            answer = reply_to(text)
        show_body(answer)
    st.session_state.messages.append(answer)

# Last on the page, so the charts sit under the newest answer.
if st.session_state.latest_estimate is not None:
    show_charts(st.session_state.latest_estimate)
