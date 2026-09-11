import streamlit as st

from core.client_context import render_client_selector
from core.costs import LINES, apify_rate_usd, client_costs

st.set_page_config(page_title="Costs", page_icon="💰", layout="wide")
st.title("💰 Costs")

client = render_client_selector()
client_id = client["client_id"]

costs = client_costs(client_id)


def _fmt(usd):
    return "—" if usd is None else f"${usd:,.4f}"


st.subheader(client["name"])

cols = st.columns(4)
cols[0].metric("Reels analysed", f"{costs['reels_analysed']:,}")
cols[1].metric("Pieces generated", f"{costs['pieces_generated']:,}")
cols[2].metric(
    "Total" + ("" if costs["total_is_complete"] else " (partial)"),
    _fmt(costs["total_usd"]),
)
cols[3].metric("Cost per piece", _fmt(costs["cost_per_piece"]))

st.divider()

rows = []
for key, label in LINES:
    line = costs["lines"][key]
    rows.append(
        {
            "Line": label,
            "Cost": _fmt(line["usd"]),
            "Complete": "yes" if line["known"] else "no",
        }
    )
st.table(rows)

if not costs["total_is_complete"]:
    n = costs["unpriced_scrape_jobs"]
    if apify_rate_usd() is None:
        st.warning(
            f"{n} scrape job(s) have no recorded cost because no Apify rate is set. "
            "The total below understates what this client actually costs.\n\n"
            "Set `APIFY_USD_PER_RESULT` to your plan's effective cost per result. "
            "There is no default: Apify prices in credits and the rate depends on "
            "your plan, so a guessed number would be confidently wrong."
        )
    else:
        st.info(
            f"{n} scrape job(s) ran before a rate was configured, so their cost is "
            "unknown. New scrapes are priced from here on."
        )

if costs["pieces_generated"] == 0:
    st.caption(
        "Cost per piece appears once content has been generated. It is the number "
        "worth comparing against a client's existing tool subscriptions."
    )
