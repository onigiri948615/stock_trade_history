import streamlit as st
import pandas as pd
import io
from stock_trade_visualizer import visualize_trades


st.title("日本株取引履歴可視化サイト")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")

if uploaded_file is not None:
    # 一旦バイトを読み込む
    content = uploaded_file.read()
    
    # テキストとして行ごとに読み込み（文字コード自動対応）
    for enc in ["utf-8", "cp932", "shift_jis"]:
        try:
            lines = content.decode(enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("⚠️ 文字コードが判別できませんでした。")
        st.stop()

    # '約定日' を含む行番号を探す
    header_row_index = None
    for i, line in enumerate(lines):
        if '約定日' in line:
            header_row_index = i
            break

    if header_row_index is None:
        st.error("⚠️ '約定日' を含む行が見つかりませんでした。")
        st.stop()

    # ヘッダー行からDataFrameを作成
    df = pd.read_csv(io.StringIO("\n".join(lines)), skiprows=header_row_index)

    st.success(f"✅ '約定日' を含む行 {header_row_index + 1} 行目から読み込みました。")


    visualize_trades(df)
    st.dataframe(df)