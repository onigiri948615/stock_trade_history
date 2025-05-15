import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

def filter_japanese_stocks(df):
    return df[df['銘柄コード'].notna()].copy()

def prepare_dropdown_options(df):
    df['銘柄コード'] = df['銘柄コード'].astype(str).str.replace('.0', '', regex=False)
    options = df[['銘柄コード', '銘柄']].drop_duplicates()
    options = options.sort_values('銘柄コード')
    return options.apply(lambda row: f"{row['銘柄コード']} : {row['銘柄']}", axis=1).tolist()

def extract_trade_history(df, selected_code):
    code = selected_code.split(':')[0].strip()
    df_filtered = df[df['銘柄コード'].astype(str).str.startswith(code)]
    
    # 👇 「約定単価」も含める
    df_filtered = df_filtered[['約定日', '取引', '約定数量', '約定単価']].copy()
    
    df_filtered['約定日'] = pd.to_datetime(df_filtered['約定日'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['約定日', '約定単価'])
    return df_filtered.sort_values('約定日')


def fetch_stock_data(code, start_date, end_date):
    ticker = f"{code}.T"

    data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
    print(data)
    # 🔧 インデックスを明示的に日付として解釈させる
    data.index = pd.to_datetime(data.index)
    print(f"[DEBUG] yfinance取得データ件数: {len(data)}")
    return data


def plot_trades_with_prices(price_df, trades_df):
    fig = go.Figure()

    # 株価の折れ線グラフ（終値）
    fig.add_trace(go.Candlestick(
        x=price_df.index,
        open=price_df['Open'],
        high=price_df['High'],
        low=price_df['Low'],
        close=price_df['Close'],
        name='株価（ローソク足）'
    ))

    # スケーリング設定（サイズはそのまま）
    min_size = 6
    max_size = 16
    min_qty = trades_df['約定数量'].min()
    max_qty = trades_df['約定数量'].max()

    def scale_size(qty):
        if max_qty == min_qty:
            return (min_size + max_size) / 2
        else:
            return min_size + (qty - min_qty) / (max_qty - min_qty) * (max_size - min_size)

    for idx, row in trades_df.iterrows():
        color = 'orange' if '買' in row['取引'] else 'lightblue'
        matched_date = row['約定日']
        matched_price = row['約定単価']  # ← 🔥 ここで約定単価を使う！
        size = scale_size(row['約定数量'])

        legend_name = f"{matched_date.date()} {row['取引']} ({row['約定数量']}株)"

        fig.add_trace(go.Scatter(
            x=[matched_date],
            y=[matched_price],  # ← 🔥 Y座標を約定単価に変更！
            mode='markers',
            marker=dict(color=color, size=size),
            name=legend_name,
            hovertemplate=(
                f"日付: {matched_date.date()}<br>"
                f"約定単価: {matched_price:.2f}<br>"
                f"数量: {row['約定数量']}<br>"
                f"取引: {row['取引']}<extra></extra>"
            )
        ))

    fig.update_layout(
        title='株価チャート（終値＋売買ポイント）',
        xaxis_title='日付',
        yaxis_title='価格',
        template='plotly_white'
    )

    return fig



from datetime import datetime

# --- 中略 ---

def visualize_trades(df):
    df_jp = filter_japanese_stocks(df)

    options = prepare_dropdown_options(df_jp)
    selected = st.selectbox("銘柄を選択", options)

    trade_df = extract_trade_history(df_jp, selected)

    if trade_df.empty:
        st.warning("この銘柄の売買履歴が見つかりませんでした。")
        return

    min_date = trade_df['約定日'].min() - timedelta(days=30)
    max_date = datetime.today()

    code = selected.split(':')[0].strip()

    try:
        price_df = fetch_stock_data(code, min_date, max_date)
        price_df.index = pd.to_datetime(price_df.index).normalize()
    except Exception as e:
        st.error(f"⚠️ 株価データ取得エラー: {e}")
        return

    if price_df.empty:
        st.error("株価データの取得に失敗しました。")
        return

    fig = plot_trades_with_prices(price_df, trade_df)
    fig.update_xaxes(
        range=[price_df.index.min(), price_df.index.max()],
        tickformat="%Y-%m-%d"
    )
    st.plotly_chart(fig)
    st.dataframe(price_df)

    

