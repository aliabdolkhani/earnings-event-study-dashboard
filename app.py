import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# 1. Load Data
events = pd.read_csv("events.csv", parse_dates=["Ann Date"])
meta   = pd.read_csv("meta.csv",   parse_dates=["Ann Date"])
ar     = pd.read_csv("ar.csv",     index_col=0)  # index = "TICKER | YYYY-MM-DD"

# ─── Ensure Surprise is numeric for regression and drop infinite values ───
meta["Surprise"] = pd.to_numeric(meta["Surprise"], errors="coerce")
meta["Surprise"] = meta["Surprise"].replace([np.inf, -np.inf], np.nan)

# 2. Build event_id & reindex meta
events["event_id"] = (
    events["Ticker"] + " | " +
    events["Ann Date"].dt.date.astype(str)
)
meta["event_id"] = (
    meta["Ticker"] + " | " +
    meta["Ann Date"].dt.date.astype(str)
)
meta = meta.set_index("event_id")

# 3. Prepare CAR windows
car0  = ar["0"]
car1  = ar[["-1","0","1"]].sum(axis=1)
car11 = ar.loc[:, [str(i) for i in range(-5,6)]].sum(axis=1)
windows = {
    "CAR(0,0)":    car0,
    "CAR(-1,+1)":  car1,
    "CAR(-5,+5)":  car11
}

# 4. Streamlit UI
st.title("Earnings-Announcement")
st.markdown(
    "_Interactive earnings-announcement event-study. "
    "Select a stock and CAR window to explore its earnings-driven returns._"
)

# 4.1 Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose a ticker", sorted(events["Ticker"].unique()))
with col2:
    window = st.selectbox("Choose CAR window", list(windows.keys()))

# 4.2 Filter for selected ticker
ev     = events[events["Ticker"] == ticker].sort_values("Ann Date")
ar_sub = ar.loc[ev["event_id"].values]
md     = meta.loc[ev["event_id"].values]  # Surprise indexed by event_id
car_series = windows[window]

# 5. Key Metrics
st.subheader("Key Metrics")
st.markdown("_Average & most recent abnormal returns around earnings._")

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

# 6. Earnings History Table
st.subheader("Earnings History")
st.markdown("_Full list of announcement dates, surprises, and CARs._")

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

# 7. Latest AR Curve (window-sensitive)
st.subheader("Latest AR Curve")
st.markdown(f"_Abnormal returns around {last_eid.split(' | ')[1]} for {window}._")

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
st.markdown(f"_Each event’s EPS surprise versus its {window}._")

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

# 9. Build & fit cross-sectional model for selected window
st.subheader("Forecast Next Event CAR")
upcoming = pd.read_csv("upcoming_surprises.csv", parse_dates=["Ann Date"])
upcoming["Surprise"] = pd.to_numeric(upcoming["Surprise"], errors="coerce")
# use CSV’s built-in event_id

# prepare regression target for chosen window
target = windows[window].rename("CARw")
df_temp  = pd.DataFrame({
    "event_id": target.index,
    "CARw":     target.values
})
df_pred = (
    df_temp
      .merge(meta[["Surprise"]].reset_index(), on="event_id", how="left")
      .dropna(subset=["Surprise","CARw"])
)

if df_pred.empty:
    st.warning("No historical Surprise+CAR data to fit model.")
    gamma0, gamma1 = 0.0, 0.0
else:
    x, y = df_pred["Surprise"].values, df_pred["CARw"].values
    xm, ym = x.mean(), y.mean()
    gamma1 = ((x - xm)*(y - ym)).sum() / ((x - xm)**2).sum()
    gamma0 = ym - gamma1*xm

# upcoming-event prediction
choice    = st.selectbox("Pick an upcoming event", upcoming["event_id"])
next_surp = upcoming.set_index("event_id").loc[choice, "Surprise"]
pred_car  = gamma0 + gamma1 * next_surp
st.metric(f"Predicted {window} for {choice}", f"{pred_car:.2%}")