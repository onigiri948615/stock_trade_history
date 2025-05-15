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

    # --- ローソク足 ---
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

    # --- 平均約定単価ラインを生成 ---
    trades_df = trades_df.copy()
    trades_df["sort_order"] = trades_df["取引"].apply(lambda x: 0 if "買" in x else 1)
    trades_df = trades_df.sort_values(by=["約定日", "sort_order"]).drop(columns="sort_order")

    avg_lines = []
    holding_qty, avg_price = 0, 0
    last_date = None

    for _, row in trades_df.iterrows():
        date, qty, price = row["約定日"], row["調整後数量"], row["調整後単価"]

        if "買" in row["取引"]:
            total_cost = avg_price * holding_qty + price * qty
            holding_qty += qty
            avg_price = 0 if holding_qty == 0 else total_cost / holding_qty
            avg_lines.append((date, round(avg_price, 1)))

        else:  # 売り
            avg_lines.append((date, round(avg_price, 1)))  # 売却直前まで維持
            holding_qty -= qty
            if holding_qty <= 0:                       # 全部売却
                holding_qty, avg_price = 0, 0
                avg_lines.append((date, 0))            # 当日を 0 で区切り
        last_date = date

    # 保有継続ならチャート終端まで水平維持
    end_date = price_df.index.max().tz_localize(None)
    if holding_qty > 0 and last_date is not None:
        avg_lines.append((end_date, round(avg_price, 1)))

    # --- 0 をセパレータとして区間分割し、非ゼロのみ描画 ---
    avg_df = pd.DataFrame(avg_lines, columns=["日付", "平均単価"]).drop_duplicates()

    segments, seg = [], []
    for _, r in avg_df.iterrows():
        if r["平均単価"] == 0:
            if len(seg) >= 2:
                segments.append(pd.DataFrame(seg))
            seg = []  # リセット
        else:
            seg.append(r)
    if len(seg) >= 2:      # 末尾セグメント
        segments.append(pd.DataFrame(seg))

    # 最初のセグメントだけ legend を表示
    for i, seg_df in enumerate(segments):
        fig.add_trace(
            go.Scatter(
                x=seg_df["日付"],
                y=seg_df["平均単価"],
                mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                line_shape="hv",
                name="平均約定単価(試作)" if i == 0 else None,   # ← 最初だけ名前を出す
                hovertemplate="日付: %{x}<br>平均単価: %{y:.1f}円<extra></extra>",
                showlegend=(i == 0)
            )
        )


    # --- 売買ポイントのマーカー ---
    min_size, max_size = 10, 18
    min_qty, max_qty = trades_df["調整後数量"].min(), trades_df["調整後数量"].max()

    def scale_size(q):
        return (min_size + max_size) / 2 if max_qty == min_qty else \
               min_size + (q - min_qty) / (max_qty - min_qty) * (max_size - min_size)

    # --- 売買ポイント（まとめて描画） ---
    buy_points = trades_df[trades_df["取引"].str.contains("買")].copy()
    sell_points = trades_df[trades_df["取引"].str.contains("売")].copy()

    fig.add_trace(
        go.Scatter(
            x=buy_points["約定日"],
            y=buy_points["調整後単価"],
            mode="markers",
            name="買",
            marker=dict(
                color="#FFA500",
                size=buy_points["調整後数量"].apply(scale_size),
                line=dict(color="black", width=1.5)
            ),
            hovertemplate=(
                "日付: %{x|%Y-%m-%d}<br>" +
                "調整後単価: %{y:.2f}円<br>" +
                "数量: %{customdata[0]:.0f}<br>" +
                "取引: %{customdata[1]}<extra></extra>"
            ),
            customdata=buy_points[["調整後数量", "取引"]].values,
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sell_points["約定日"],
            y=sell_points["調整後単価"],
            mode="markers",
            name="売",
            marker=dict(
                color="#0000FF",
                size=sell_points["調整後数量"].apply(scale_size),
                line=dict(color="black", width=1.5)
            ),
            hovertemplate=(
                "日付: %{x|%Y-%m-%d}<br>" +
                "調整後単価: %{y:.2f}円<br>" +
                "数量: %{customdata[0]:.0f}<br>" +
                "取引: %{customdata[1]}<extra></extra>"
            ),
            customdata=sell_points[["調整後数量", "取引"]].values,
            showlegend=True
        )
    )


    fig.update_layout(
        title="株価チャート（分割調整済み, 配当を含めない）＋売買ポイント",
        xaxis_title="日付",
        yaxis_title="価格",
        template="plotly_white",
        height=700
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
