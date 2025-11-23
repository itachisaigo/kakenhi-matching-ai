import streamlit as st
import google.generativeai as genai
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import os

# ===== 設定 =====
# APIキーの取得 (Streamlit Secrets または 環境変数)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("APIキーが設定されていません。Streamlit CloudのSecretsを設定してください。")
    st.stop()

genai.configure(api_key=api_key)

# 埋め込み用モデル（JSONを作ったときと同じモデルを指定する）
EMBEDDING_MODEL_ID = "gemini-embedding-001"
# アドバイス生成用モデル（文章が作れるモデルを指定する）
GENERATION_MODEL_ID = "gemini-1.5-flash"

# ===== データの読み込みと前処理 =====
@st.cache_data
def load_and_process_data():
    # JSONの読み込み
    try:
        with open("academic_embeddings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None, None, None
        
    words = [d["word"] for d in data]
    embeddings = np.array([d["vector"] for d in data])
    
    # 2次元マップ用に次元圧縮 (PCA) を事前に計算しておく
    # データ数が少ない場合の対策
    n_samples = len(embeddings)
    n_components = 2
    if n_samples < 2:
         return words, embeddings, None, None

    pca = PCA(n_components=n_components)
    coords_2d = pca.fit_transform(embeddings)
    
    return words, embeddings, coords_2d, pca

words, embeddings, base_coords_2d, pca_model = load_and_process_data()

if words is None:
    st.error("埋め込みデータファイル(academic_embeddings.json)が見つかりません。")
    st.stop()

# ===== UI構築 =====
st.set_page_config(page_title="科研費マッチングAI", layout="wide")
st.title("科研費・審査区分マッチングAI 🎓")
st.markdown("あなたの研究テーマを入力すると、AIが最適な審査区分を推薦し、研究の立ち位置を可視化します。")

# 入力フォーム
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area("研究タイトルまたは要旨", height=150, 
                         placeholder="例：〇〇の△△における✕✕の解明")

    if st.button("分析する 🔍", type="primary"):
        if not query:
            st.warning("テキストを入力してください。")
        else:
            with st.spinner("AIが分析中..."):
                try:
                    # 1. 入力テキストをベクトル化
                    # GenerativeModelではなく、モジュール関数を直接呼び出します
                    result = genai.embed_content(model=EMBEDDING_MODEL_ID, content=query)
                    query_vec = np.array(result['embedding'])

                    # 2. 類似度計算
                    sims = cosine_similarity([query_vec], embeddings)[0]
                    
                    # 3. ランキング作成
                    top_n = 5
                    top_indices = sims.argsort()[::-1][:top_n]
                    top_scores = sims[top_indices]

                    # --- 結果表示エリア ---
                    st.divider()
                    
                    # A. ニッチ度判定ロジック
                    score_1st = top_scores[0]
                    score_2nd = top_scores[1]
                    diff = score_1st - score_2nd
                    
                    st.subheader("📊 分析結果")
                    
                    # 判定メッセージ
                    if score_1st < 0.6: 
                        st.info("💡 **非常に新規性が高い、または学際的なテーマのようです。**\n\nどの区分にも完全には当てはまらない可能性があります。複合領域での申請も検討してみてください。")
                    elif diff > 0.05: 
                        st.success("🎯 **王道のテーマです！**\n\n1位の区分が非常に強くマッチしています。迷わずこの区分で良いでしょう。")
                    else: 
                        st.warning("⚖️ **境界領域のテーマです。**\n\n1位と2位のスコアが近いです。どちらのコミュニティで評価されたいか、戦略的に選ぶ必要があります。")

                    # B. ランキング表示
                    st.write("#### おすすめの審査区分")
                    for i, idx in enumerate(top_indices):
                        score = sims[idx]
                        category = words[idx]
                        st.write(f"**{i+1}. {category}** (一致度: {score:.3f})")
                        st.progress(min(float(score), 1.0))
                    
                    # C. キーワードアドバイス
                    st.write("#### 💡 申請書作成アドバイス")
                    target_cat = words[top_indices[0]]
                    advice_prompt = f"""
                    以下の研究テーマを、科研費の審査区分「{target_cat}」に申請しようとしています。
                    この区分で採択されやすくするために、含めるべきキーワードや、強調すべき観点を3点以内で簡潔にアドバイスしてください。
                    
                    研究テーマ: {query}
                    """
                    
                    # アドバイス生成（ここでは文章生成モデルを使用）
                    try:
                        model_gen = genai.GenerativeModel(GENERATION_MODEL_ID) 
                        advice_resp = model_gen.generate_content(advice_prompt)
                        st.info(advice_resp.text)
                    except Exception as e:
                        st.warning(f"アドバイス生成中にエラーが発生しました: {e}")

                    # D. 可視化 (Plotly)
                    if pca_model is not None:
                        st.write("#### 🗺 学問の地図")
                        
                        # ユーザーのクエリを同じPCAモデルで2次元に落とす
                        user_coord = pca_model.transform([query_vec])[0]
                        
                        # 散布図の作成
                        fig = go.Figure()

                        # 全体の点 (グレー)
                        fig.add_trace(go.Scatter(
                            x=base_coords_2d[:, 0],
                            y=base_coords_2d[:, 1],
                            mode='markers',
                            text=words,
                            marker=dict(size=8, color='lightgray', opacity=0.5),
                            name='その他の区分',
                            hoverinfo='text'
                        ))

                        # 上位5つの点 (青)
                        top_coords = base_coords_2d[top_indices]
                        top_words = [words[i] for i in top_indices]
                        fig.add_trace(go.Scatter(
                            x=top_coords[:, 0],
                            y=top_coords[:, 1],
                            mode='markers+text',
                            text=top_words,
                            textposition="top center",
                            marker=dict(size=12, color='blue', opacity=0.8),
                            name='候補の区分',
                            hoverinfo='text'
                        ))

                        # ユーザーの点 (赤)
                        fig.add_trace(go.Scatter(
                            x=[user_coord[0]],
                            y=[user_coord[1]],
                            mode='markers+text',
                            text=["★あなたの研究"],
                            textposition="bottom center",
                            marker=dict(size=18, color='red', symbol='star'),
                            name='あなたの研究',
                            hoverinfo='text'
                        ))

                        fig.update_layout(
                            height=600,
                            plot_bgcolor='white',
                            hovermode='closest',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            margin=dict(l=0, r=0, t=0, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"詳細エラー: {e}")

with col2:
    st.info("""
    **使い方**
    1. 研究タイトルや要旨を入力します。
    2. 「分析する」ボタンを押します。
    3. AIが最適な審査区分を判定します。
    
    **見方**
    - **一致度**: AIが計算した類似度です。
    - **地図**: 全400区分の中での立ち位置です。
    """)