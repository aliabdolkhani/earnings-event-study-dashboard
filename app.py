import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# 1. Load Core Data
events = pd.read_csv("events.csv", parse_dates=["Ann Date"])
meta   = pd.read_csv("meta.csv",   parse_dates=["Ann Date"])
ar     = pd.read_csv("ar.csv",     index_col=0)

# Clean Surprise column
meta["Surprise"] = pd.to_numeric(meta["Surprise"], errors="coerce")
meta["Surprise"] = meta["Surprise"].replace([np.inf, -np.inf], np.nan)

# 2. Build event_id & reindex meta
events["event_id"] = events["Ticker"] + " | " + events["Ann Date"].dt.date.astype(str)
meta["event_id"]   = meta["Ticker"]   + " | " + meta["Ann Date"].dt.date.astype(str)
meta = meta.set_index("event_id")

# 3. Prepare CAR windows
car0  = ar["0"]
car1  = ar[["-1","0","1"]].sum(axis=1)
car11 = ar.loc[:, [str(i) for i in range(-5,6)]].sum(axis=1)
windows = {
    "CAR(0,0)"   : car0,
    "CAR(-1,+1)" : car1,
    "CAR(-5,+5)" : car11
}

# 4. Streamlit UI Setup
st.title("Earnings-Announcement Dashboard")
st.markdown(
    "_Interactive earnings-announcement event-study. "
    "Select a stock and CAR window to explore its earnings-driven returns._"
)

# 4.1 User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose a ticker", sorted(events["Ticker"].unique()))
with col2:
    window = st.selectbox("Choose CAR window", list(windows.keys()))

# 4.2 Filter Data for Selected Ticker
car_series = windows[window]
ev     = events[events["Ticker"] == ticker].sort_values("Ann Date")
ar_sub = ar.loc[ev["event_id"].values]
md     = meta.loc[ev["event_id"].values]

# 4.3 Key Metrics Display
st.subheader("Key Metrics")
st.markdown("_Average & most recent abnormal returns around earnings._")
st.markdown("Use these metrics to see typical earnings impact and how the latest event compares.")
car_win       = car_series.loc[ev["event_id"]]
avg_car       = car_win.mean()
last_eid      = ev.iloc[-1]["event_id"]
last_surprise = md.loc[last_eid, "Surprise"]
last_car      = car_win.loc[last_eid]
n_events      = len(ev)
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Avg {window}",  f"{avg_car:.2%}")
k2.metric("Last Surprise",  f"{last_surprise:.1%}")
k3.metric(f"Last {window}", f"{last_car:.2%}")
k4.metric("# of Events",    f"{n_events}")

# 5. Ranking Section
st.subheader(f"Ranking: Average {window} by Ticker")
st.markdown("Compare tickers by their average CAR to identify top and bottom performers.")
df_rank = (
    pd.DataFrame({"event_id": car_series.index, "CAR": car_series.values})
      .merge(events[["event_id","Ticker"]], on="event_id")
      .groupby("Ticker", as_index=False)["CAR"].mean()
      .sort_values("CAR", ascending=False)
)
st.dataframe(
    df_rank.rename(columns={"CAR": f"Avg {window}"})
           .style.format({f"Avg {window}": "{:.1%}"})
)

# 6. Earnings History Table
st.subheader("Earnings History")
st.markdown("Review each past announcement’s date, surprise, and CAR in one table.")
df_table = ev[["event_id","Ann Date"]].copy()
df_table["Surprise"]   = df_table["event_id"].map(md["Surprise"])
df_table["CAR(0,0)"]   = ar_sub["0"].values
df_table["CAR(-1,+1)"] = car_series.loc[ev["event_id"]].values
df_table["CAR(-5,+5)"] = ar_sub.loc[:, [str(i) for i in range(-5,6)]].sum(axis=1).values
st.dataframe(
    df_table
      .sort_values("Ann Date", ascending=False)
      .rename(columns={"Ann Date":"Date"})
      .style.format({
         "Surprise"   : "{:.1%}",
         "CAR(0,0)"   : "{:.1%}",
         "CAR(-1,+1)" : "{:.1%}",
         "CAR(-5,+5)" : "{:.1%}"
      }),
    height=300
)

# 7. Latest AR Curve
st.subheader("Latest AR Curve")
st.markdown("Visualize the abnormal return trajectory around the most recent earnings date.")
full_ar = ar_sub.loc[last_eid].astype(float)
if window == "CAR(0,0)":
    days = ["0"]
elif window == "CAR(-1,+1)":
    days = ["-1","0","1"]
else:
    days = [str(i) for i in range(-5,6)]
df_curve = (
    full_ar.loc[days]
           .rename_axis("Day")
           .reset_index(name="AR")
           .assign(Day=lambda d: d.Day.astype(int))
           .set_index("Day")
)
st.line_chart(df_curve)

# 8. Surprise vs. CAR Scatter
st.subheader("Surprise vs. Return")
st.markdown("Inspect how EPS surprise correlates with the selected CAR window.")
df_sc = pd.DataFrame({
    "Surprise": md["Surprise"].values,
    "CAR"     : car_win.values,
    "Date"    : ev["Ann Date"].dt.date.values
})
scatter = alt.Chart(df_sc).mark_circle(size=60).encode(
    x=alt.X("Surprise", title="EPS Surprise"),
    y=alt.Y("CAR",      title=window),
    tooltip=["Date","Surprise","CAR"]
).interactive()
st.altair_chart(scatter, use_container_width=True)

# 9. Forecast Next Event CAR (using exported CSV only)
st.subheader("Forecast Next Event CAR")
st.markdown("Select a future event to view its predicted CAR and 95% confidence interval.")

# Load only the predictions CSV and build event_id
upcoming = pd.read_csv("upcoming_predictions.csv", parse_dates=["Ann Date"])
upcoming["event_id"] = upcoming["Ticker"] + " | " + upcoming["Ann Date"].dt.date.astype(str)

# User selects which future event to display
choice = st.selectbox("Pick an upcoming event", upcoming["event_id"])
sel    = upcoming.set_index("event_id").loc[choice]

pred_car = sel["Pred_CAR_1"]
ci_lo    = sel["CI_lower"]
ci_hi    = sel["CI_upper"]

# Show the precomputed values
st.metric(
    label = f"Predicted {window} for {choice}",
    value = f"{pred_car:.2%}",
    delta = None
)
st.write(f"95% CI: [{ci_lo:.2%}, {ci_hi:.2%}]")
