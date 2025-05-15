import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta, datetime


dfstock = None  # グローバル変数として定義

def filter_japanese_stocks(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["銘柄コード"].notna()].copy()


def prepare_dropdown_options(df: pd.DataFrame) -> list[str]:
    df["銘柄コード"] = df["銘柄コード"].astype(str).str.replace(".0", "", regex=False)
    opts = df[["銘柄コード", "銘柄"]].drop_duplicates().sort_values("銘柄コード")
    return opts.apply(lambda r: f"{r['銘柄コード']} : {r['銘柄']}", axis=1).tolist()


def extract_trade_history(df: pd.DataFrame, selected_code: str) -> pd.DataFrame:
    code = selected_code.split(":")[0].strip()
    trades = df[df["銘柄コード"].astype(str).str.startswith(code)]
    trades = trades[["約定日", "取引", "約定数量", "約定単価"]].copy()
    trades["約定日"] = pd.to_datetime(trades["約定日"], errors="coerce")
    trades = trades.dropna(subset=["約定日", "約定単価"])
    return trades.sort_values("約定日")


def fetch_stock_data(code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    #グローバル変数を定義
    global dfstock
    ticker = yf.Ticker(f"{code}.T")
    # 実際の株価を取得（分割・配当非調整）
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    df.index = pd.to_datetime(df.index)
    dfstock = df.copy()  # グローバル変数に代入

    return df



def adjust_trades_for_splits(trades: pd.DataFrame, splits_ser: pd.Series) -> pd.DataFrame:
    if splits_ser.empty:
        trades["調整後単価"] = trades["約定単価"]
        trades["調整後数量"] = trades["約定数量"]
        return trades

    splits_ser.index = pd.to_datetime(splits_ser.index).tz_localize(None)  # ← tz-naive に統一
    splits_ser = splits_ser.sort_index()

    adj_prices, adj_qty = [], []
    for _, row in trades.iterrows():
        factor = splits_ser[splits_ser.index > row["約定日"]].prod() or 1.0
        adj_prices.append(row["約定単価"] / factor)
        adj_qty.append(row["約定数量"] * factor)

    trades["調整後単価"] = adj_prices
    trades["調整後数量"] = adj_qty
    return trades



def build_trade_chart(price_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="株価（ローソク足）",
        )
    )

    min_size, max_size = 10, 18
    min_qty, max_qty = trades_df["調整後数量"].min(), trades_df["調整後数量"].max()

    def scale_size(q):
        if max_qty == min_qty:
            return (min_size + max_size) / 2
        return min_size + (q - min_qty) / (max_qty - min_qty) * (max_size - min_size)

    for _, row in trades_df.iterrows():
        color = "orange" if "買" in row["取引"] else "lightblue"
        fig.add_trace(
            go.Scatter(
                x=[row["約定日"]],
                y=[row["調整後単価"]],
                mode="markers",
                marker=dict(color=color, size=scale_size(row["調整後数量"])),
                name=f"{row['約定日'].date()} {row['取引']} ({row['約定数量']}株)",
                hovertemplate=(
                    f"日付: {row['約定日'].date()}<br>"
                    f"調整後単価: {row['調整後単価']:.2f}<br>"
                    f"数量: {row['調整後数量']:.0f}<br>"
                    f"取引: {row['取引']}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="株価チャート（分割調整済み,配当を含めない）＋売買ポイント",
        xaxis_title="日付",
        yaxis_title="価格",
        template="plotly_white",
        height=700  # ← この行を追加
    )

    return fig


def visualize_trades(df: pd.DataFrame):
    df_jp = filter_japanese_stocks(df)
    options = prepare_dropdown_options(df_jp)
    selected = st.selectbox("銘柄を選択", options)

    trade_df = extract_trade_history(df_jp, selected)
    if trade_df.empty:
        st.warning("この銘柄の売買履歴が見つかりませんでした。")
        return

    code = selected.split(":")[0].strip()
    min_date = trade_df["約定日"].min() - timedelta(days=30)
    max_date = datetime.today()

    try:
        price_df = fetch_stock_data(code, min_date, max_date)
    except Exception as e:
        st.error(f"⚠️ 株価データ取得エラー: {e}")
        return
    if price_df.empty:
        st.error("株価データの取得に失敗しました。")
        return

    splits = yf.Ticker(f"{code}.T").splits
    trade_df = adjust_trades_for_splits(trade_df, splits)

    fig = build_trade_chart(price_df, trade_df)
    fig.update_xaxes(range=[price_df.index.min(), price_df.index.max()], tickformat="%Y-%m-%d")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("取引履歴", expanded=False):
        st.dataframe(trade_df)
    with st.expander("株価詳細データ", expanded=False):
        st.dataframe(dfstock)
