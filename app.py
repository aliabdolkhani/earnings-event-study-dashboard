import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# Load precomputed CSVs
events = pd.read_csv("events.csv", parse_dates=["Ann Date"])
meta   = pd.read_csv("meta.csv", parse_dates=["Ann Date"])
ar     = pd.read_csv("ar.csv", index_col=0)      # event_id in index
ranking= pd.read_csv("ranking.csv", index_col="Ticker")

# Prepare CAR series
car0    = ar["0"]
car1    = ar[["-1","0","1"]].sum(axis=1);   car1.name = "CAR"
car11   = ar.loc[:, [str(i) for i in range(-5,6)]].sum(axis=1); car11.name = "CAR11"

# Build event_id for merges
events["event_id"] = events["Ticker"] + " | " + events["Ann Date"].dt.date.astype(str)
meta  ["event_id"] = meta  ["Ticker"] + " | " + meta  ["Ann Date"].dt.date.astype(str)

# Sidebar: choose window
window = st.sidebar.selectbox(
    "Choose CAR window",
    ["CAR(0,0)", "CAR(-1,+1)", "CAR(-5,+5)"],
    index=1
)
car_series = {"CAR(0,0)": car0, "CAR(-1,+1)": car1, "CAR(-5,+5)": car11}[window]

st.title("Earnings-Announcement Event-Study Dashboard")

# 1. Global KPIs
st.header("Key Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg CAR(0,0)", f"{car0.mean():.2%}")
col2.metric("Avg CAR(-1,+1)", f"{car1.mean():.2%}")
col3.metric("Avg CAR(-5,+5)", f"{car11.mean():.2%}")

# 2. Upcoming Earnings Calendar
st.header("Upcoming Earnings (Next 7 Days)")
today = datetime.now().date()
future = events[(events["Ann Date"].dt.date >= today) & 
                (events["Ann Date"].dt.date <= today + timedelta(days=7))]
df_cal = (
    future
    .merge(meta[["event_id","Surprise"]], on="event_id")
    .merge(car1.rename("Last CAR"), left_on="event_id", right_index=True)
    .sort_values("Ann Date")
)[["Ann Date","Ticker","Surprise","Last CAR"]]
st.dataframe(df_cal.style.format({"Surprise":"{:.1%}","Last CAR":"{:.1%}"}), height=200)

# 3. Live Stock Ranking
st.header("Stock Ranking by Avg CAR(-1,+1)")
st.dataframe(ranking.style.format("{:.2%}"), height=250)

# 4. Event-Window Viewer
st.header("Event-Window Viewer")
selected = st.selectbox("Select Ticker", events["Ticker"].unique())
last_date = events[events["Ticker"]==selected]["Ann Date"].max().date()
event_id = f"{selected} | {last_date}"
ar_series = ar.loc[event_id].astype(float)
chart_df = pd.DataFrame({
    "Day": [int(d) for d in ar_series.index],
    "AR": ar_series.values
})
st.line_chart(chart_df.set_index("Day"))

# 5. Surprise vs. CAR Scatter
st.header("Surprise vs. CAR(-1,+1) Scatter")
df_scatter = (
    events.merge(meta[["event_id","Surprise"]], on="event_id")
          .merge(car1.rename("CAR"), left_on="event_id", right_index=True)
).dropna(subset=["Surprise","CAR"])
chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
    x=alt.X("Surprise", title="EPS Surprise"),
    y=alt.Y("CAR", title="CAR(-1,+1)"),
    color="Surprise",
    tooltip=["Ticker","Ann Date","Surprise","CAR"]
).interactive()
st.altair_chart(chart, use_container_width=True)

# 6. Decile Performance Widget
st.header("Average CAR by Surprise Decile")
df_plot = df_scatter.copy()
df_plot["Decile"] = pd.qcut(df_plot["Surprise"], 10, labels=False) + 1
decile_perf = df_plot.groupby("Decile")["CAR"].mean().reset_index()
bar = alt.Chart(decile_perf).mark_bar().encode(
    x=alt.X("Decile:O", title="Surprise Decile"),
    y=alt.Y("CAR", title=f"Avg {window}")
)
st.altair_chart(bar, use_container_width=True)
