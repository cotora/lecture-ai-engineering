# app.py
import data  # データモジュール
import database  # データベースモジュール
import llm  # LLMモジュール
import metrics  # 評価指標モジュール
import streamlit as st
import torch
import ui  # UIモジュール
from config import MODEL_NAME
from transformers import pipeline

# --- アプリケーション設定 ---
st.set_page_config(
    page_title="Gemma Chatbot", layout="wide", initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown(
    """
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()


# LLMモデルのロード（キャッシュを利用）
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with st.spinner(f"モデルをロード中... ({device}を使用)"):
            pipe = pipeline(
                "text-generation",
                model=MODEL_NAME,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device,
            )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error(
            "GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。"
        )
        return None


pipe = llm.load_model()

# --- Streamlit アプリケーション ---
# ヘッダーセクション
st.markdown(
    """
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #1E88E5;'>🤖 Gemma 2 Chatbot</h1>
    <p style='color: #666;'>Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- サイドバー ---
with st.sidebar:
    st.markdown(
        """
    <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
        <h2 style='color: #1E88E5;'>📱 ナビゲーション</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # セッション状態を使用して選択ページを保持
    if "page" not in st.session_state:
        st.session_state.page = "チャット"

    # ページ選択UI
    page = st.radio(
        "ページ選択",
        ["チャット", "履歴閲覧", "サンプルデータ管理"],
        key="page_selector",
        index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(
            st.session_state.page
        ),
        on_change=lambda: setattr(
            st.session_state, "page", st.session_state.page_selector
        ),
        label_visibility="collapsed",
    )

    # ページごとのアイコンと説明
    page_info = {
        "チャット": {"icon": "💬", "desc": "Gemmaモデルとチャット"},
        "履歴閲覧": {"icon": "📚", "desc": "過去の会話履歴を確認"},
        "サンプルデータ管理": {"icon": "📊", "desc": "サンプルデータの管理"},
    }

    st.markdown(
        f"""
    <div style='padding: 1rem; margin-top: 1rem; background-color: #e8f0fe; border-radius: 10px;'>
        <h3 style='color: #1E88E5;'>{page_info[page]["icon"]} {page}</h3>
        <p style='color: #666;'>{page_info[page]["desc"]}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # フッター
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666;'>開発者: [Your Name]</p>
        <p style='color: #999; font-size: 0.8em;'>Version 1.0.0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# --- メインコンテンツ ---
# ページコンテンツをコンテナで囲む
with st.container():
    if st.session_state.page == "チャット":
        if pipe:
            with st.spinner("チャットを準備中..."):
                ui.display_chat_page(pipe)
        else:
            st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
    elif st.session_state.page == "履歴閲覧":
        ui.display_history_page()
    elif st.session_state.page == "サンプルデータ管理":
        ui.display_data_page()
