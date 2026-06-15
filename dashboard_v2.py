
import sqlite3
import time
import pandas as pd
import streamlit as st
from datetime import date

DB_FILE = "vehicle_counts.db"

st.set_page_config(
    page_title="Vehicle Counter Dashboard",
    page_icon="",
    layout="wide"
)
st.title("Universitas Santo Borromeus Vehicle Counter Dashboard | V1.1")


def get_conn():
    """Open a fresh, WAL-mode connection with timeout."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  
    return conn

def load_daily_counts():
    try:
        conn = get_conn()
        df   = pd.read_sql("SELECT * FROM counts ORDER BY date DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "cars", "motorcycles"])

def load_crossings(filter_date=None, filter_class=None):
    try:
        conn  = get_conn()
        query = "SELECT * FROM crossings"
        conds, params = [], []
        if filter_date:
            conds.append("date = ?"); params.append(str(filter_date))
        if filter_class and filter_class != "All":
            conds.append("class = ?"); params.append(filter_class)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY id DESC"
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(
            columns=["id","date","time","track_id","class","custom_name"])

def save_name_changes(edited_df, original_df):
    conn    = get_conn()
    changed = 0
    for _, row in edited_df.iterrows():
        orig = original_df.loc[original_df["id"] == row["id"], "custom_name"]
        if orig.empty:
            continue
        orig_val = orig.values[0]
        new_val  = row["custom_name"]
        if str(orig_val or "") != str(new_val or ""):
            conn.execute(
                "UPDATE crossings SET custom_name = ? WHERE id = ?",
                (new_val if new_val else None, int(row["id"]))
            )
            changed += 1
    conn.commit()
    conn.close()
    return changed


# ─────────────────────────────────────────
st.sidebar.header(" Dashboard Settings")


auto_refresh = st.sidebar.toggle(" Auto Refresh", value=True)
interval     = st.sidebar.slider(
    "Refresh interval (seconds)", min_value=2, max_value=30, value=5
)


st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: **{time.strftime('%H:%M:%S')}**")


tab1, tab2 = st.tabs([" Daily Summary", " Vehicle Log"])


with tab1:
    df_counts = load_daily_counts()
    today     = str(date.today())
    today_row = df_counts[df_counts["date"] == today]

    car_today  = int(today_row["cars"].values[0])        if not today_row.empty else 0
    moto_today = int(today_row["motorcycles"].values[0]) if not today_row.empty else 0

    st.subheader(f" Today  {today}")
    c1, c2, c3 = st.columns(3)
    c1.metric(" Cars",          car_today)
    c2.metric(" Motorcycles",  moto_today)
    c3.metric(" Total",         car_today + moto_today)

    st.divider()

    if not df_counts.empty:
        st.subheader(" Daily Vehicle History")
        chart_df = df_counts.set_index("date")[["cars","motorcycles"]].sort_index()
        st.bar_chart(chart_df, color=["#00C8FF","#FF6400"])

        st.subheader(" Full Daily Log")
        df_counts["total"] = df_counts["cars"] + df_counts["motorcycles"]
        st.dataframe(df_counts.rename(columns={
            "date":"Date","cars":"Cars",
            "motorcycles":"Motorcycles","total":"Total"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Run vehicle_counter.py to start counting.")


with tab2:
    st.subheader("🚗 Vehicle Crossing Log")
    st.caption(
        "Every vehicle that crossed the line is listed here. "
        "Click ** Custom Name** to rename any vehicle, then press **Save Changes**."
    )

    col1, col2 = st.columns([2, 2])
    with col1:
        sel_date  = st.date_input("Filter by date", value=None)
    with col2:
        sel_class = st.selectbox("Filter by type", ["All","Car","Motorcycle"])

    df_cross = load_crossings(
        filter_date  = sel_date  if sel_date  else None,
        filter_class = sel_class if sel_class != "All" else None
    )

    if df_cross.empty:
        st.info("No crossings recorded yet, or no results match your filter.")
    else:
        df_cross["display"] = df_cross.apply(
            lambda r: r["custom_name"]
                      if pd.notna(r["custom_name"]) and r["custom_name"] != ""
                      else f"{r['class']} #{r['track_id']}",
            axis=1
        )

        n_cars  = (df_cross["class"] == "Car").sum()
        n_motos = (df_cross["class"] == "Motorcycle").sum()
        m1, m2, m3 = st.columns(3)
        m1.metric(" Cars in log",         n_cars)
        m2.metric(" Motorcycles in log", n_motos)
        m3.metric(" Total events",         len(df_cross))

        st.divider()
        df_original = df_cross.copy()

        edited = st.data_editor(
            df_cross[["id","date","time","track_id","class","custom_name","display"]],
            column_config={
                "id":          st.column_config.NumberColumn("ID",          disabled=True, width="small"),
                "date":        st.column_config.TextColumn("Date",          disabled=True, width="small"),
                "time":        st.column_config.TextColumn("Time",          disabled=True, width="small"),
                "track_id":    st.column_config.NumberColumn("Track ID",    disabled=True, width="small"),
                "class":       st.column_config.TextColumn("Type",          disabled=True, width="small"),
                "display":     st.column_config.TextColumn("Auto Label",    disabled=True, width="medium"),
                "custom_name": st.column_config.TextColumn(
                    " Custom Name", disabled=False, width="medium",
                    help="Click to type a custom name for this vehicle"
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="crossing_editor"
        )

        if st.button(" Save Changes", type="primary"):
            n = save_name_changes(edited, df_original)
            if n > 0:
                st.success(f" {n} name(s) updated!")
                st.rerun()
            else:
                st.info("No changes detected.")


if auto_refresh:
    time.sleep(interval)
    st.rerun()
