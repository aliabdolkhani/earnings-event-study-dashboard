import streamlit as st
import pandas as pd
import altair as alt

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
events = pd.read_csv("events.csv", parse_dates=["Ann Date"])
meta   = pd.read_csv("meta.csv",   parse_dates=["Ann Date"])
ar     = pd.read_csv("ar.csv",     index_col=0)  # index: "TICKER | YYYY-MM-DD"

# ─── 2. Build event_id & helper mappings ─────────────────────────────────────
events["event_id"] = events["Ticker"] + " | " + events["Ann Date"].dt.date.astype(str)
meta  ["event_id"] = meta  ["Ticker"] + " | " + meta  ["Ann Date"].dt.date.astype(str)

# ─── 3. Streamlit UI ─────────────────────────────────────────────────────────
st.title("Earnings-Announcement: Stock-Centric Dashboard")

# 3.1 Ticker selector
ticker_list = sorted(events["Ticker"].unique())
selected    = st.selectbox("Select a ticker", ticker_list)

# 3.2 Filter for the selected ticker
ev = events[events["Ticker"] == selected].sort_values("Ann Date")
md = meta  [meta  ["Ticker"] == selected].set_index("event_id")
ar_sub = ar.loc[ev["event_id"].values]  # AR rows for this ticker

# ─── 4. Top-line Metrics ──────────────────────────────────────────────────────
# Compute two-day CAR and pick last event
car_m1p1     = ar_sub[["-1","0","1"]].sum(axis=1)
avg_car      = car_m1p1.mean()
last_event   = ev.iloc[-1]
last_eid     = last_event["event_id"]
last_surprise = md.loc[last_eid, "Surprise"]
last_car      = car_m1p1.loc[last_eid]
n_events      = len(ev)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg CAR(−1,+1)", f"{avg_car:.2%}")
col2.metric("Last Surprise",     f"{last_surprise:.1%}")
col3.metric("Last CAR(−1,+1)",   f"{last_car:.2%}")
col4.metric("# of Events",       f"{n_events}")

# ─── 5. Earnings History Table ───────────────────────────────────────────────
st.subheader("Earnings Events History")
df_table = ev.copy().set_index("event_id")
# add CAR columns
df_table["CAR(0,0)"]   = ar_sub["0"].values
df_table["CAR(−1,+1)"] = car_m1p1.values
df_table["CAR(−5,+5)"] = ar_sub.loc[:, [str(i) for i in range(-5,6)]].sum(axis=1).values

st.dataframe(
    df_table[["Ann Date","Surprise","CAR(0,0)","CAR(−1,+1)","CAR(−5,+5)"]]
        .sort_values("Ann Date", ascending=False)
        .style.format({
            "Surprise":    "{:.1%}",
            "CAR(0,0)":    "{:.1%}",
            "CAR(−1,+1)":  "{:.1%}",
            "CAR(−5,+5)":  "{:.1%}"
        }),
    height=300
)

# ─── 6. Latest AR Curve ──────────────────────────────────────────────────────
st.subheader("Latest Abnormal-Return Curve (−5…+5 days)")
ar_latest = ar_sub.loc[last_eid]
df_chart = pd.DataFrame({
    "Day": [int(d) for d in ar_latest.index],
    "AR":  ar_latest.values
})
st.line_chart(df_chart.set_index("Day"))

# ─── 7. Surprise vs. CAR Scatter ─────────────────────────────────────────────
st.subheader("Surprise vs. CAR(−1,+1) Across Events")
df_sc = pd.DataFrame({
    "Surprise": md["Surprise"].loc[car_m1p1.index],
    "CAR":      car_m1p1
}).reset_index().rename(columns={"index":"Event"})
chart = alt.Chart(df_sc).mark_circle(size=60).encode(
    x=alt.X("Surprise", title="EPS Surprise"),
    y=alt.Y("CAR",      title="CAR(−1,+1)"),
    tooltip=["Event","Surprise","CAR"]
).interactive()
st.altair_chart(chart, use_container_width=True)
