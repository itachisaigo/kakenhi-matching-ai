import streamlit as st
import google.generativeai as genai
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

# ===== 設定 =====
# Streamlit Cloudなどでデプロイする場合、APIキーは環境変数やSecrets機能で管理します
# ローカルで動かす場合は .env などを使ってください
api_key = st.secrets["GEMINI_API_KEY"] 
genai.configure(api_key=api_key)
MODEL_ID = "models/text-embedding-004"

# ===== データの読み込み（キャッシュ化して高速化） =====
@st.cache_data
def load_data():
    with open("academic_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    words = [d["word"] for d in data]
    # JSONから読み込むとリストになるのでnumpy配列に戻す
    embeddings = np.array([d["vector"] for d in data])
    return words, embeddings

try:
    words, embeddings = load_data()
except FileNotFoundError:
    st.error("埋め込みデータファイル(academic_embeddings.json)が見つかりません。")
    st.stop()

# ===== UI構築 =====
st.title("科研費・審査区分マッチングAI 🎓")
st.markdown("研究タイトルや要旨を入力すると、AIが**最も近い審査区分（小区分）**を推薦します。")

query = st.text_area("研究タイトルまたは要旨を入力してください", height=150, 
                     placeholder="例：サルTE野における顔・表面質感・形状の情報表現と、その神経メカニズムの解明...")

if st.button("審査区分を探す 🔍"):
    if not query:
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("AIが学問の地図を検索中..."):
            try:
                # 1. 入力テキストをベクトル化
                result = genai.GenerativeModel(MODEL_ID).embed_content(content=query)
                query_vec = np.array(result['embedding'])

                # 2. 類似度計算
                sims = cosine_similarity([query_vec], embeddings)[0]
                
                # 3. ランキング作成
                top_n = 5
                top_indices = sims.argsort()[::-1][:top_n]

                st.subheader("おすすめの審査区分ベスト5")
                
                for i, idx in enumerate(top_indices):
                    score = sims[idx]
                    category = words[idx]
                    
                    # スコアに応じたバーを表示
                    st.write(f"**{i+1}. {category}** (一致度: {score:.3f})")
                    st.progress(min(float(score), 1.0))
                
                st.success("検索完了！この区分で申請書を書いてみましょう。")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")