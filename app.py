import streamlit as st
import numpy as np
import cv2
import joblib

from sklearn.preprocessing import StandardScaler
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# ---------------------------
# モデル読み込み
# ---------------------------
modelffs = load_model("ffs_model.keras")

# 前処理読み込み
ffselect = joblib.load("ffselect.pkl")
sescaler = joblib.load("sescaler.pkl")
y_scaler = joblib.load("yscaler.pkl")

embedder = FaceNet()
detector = MTCNN()

# ---------------------------
# 顔検出
# ---------------------------
def detect_face(img):

    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x,y,w,h = results[0]['box']

    face = img[y:y+h, x:x+w]

    face = cv2.resize(face,(160,160))

    return face

# ---------------------------
# embedding
# ---------------------------
def get_embedding(face):

    emb = embedder.embeddings([face])[0]

    return emb

# ---------------------------
# 推定
# ---------------------------
def predict(img):

    face = detect_face(img)

    if face is None:
        return None,None

    # ① embedding
    emb = get_embedding(face)   # (512,)

    # ② PCA（学習済み）
    emb = ffselect.transform([emb])  # (1,32)
    
    # ③ Scaler（学習済み）
    emb = sescaler.transform(emb)
    

    # ④ 推論
    pred = modelffs(emb, training=False).numpy()
    pred = y_scaler.inverse_transform(pred)
    
    return pred[0]

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="顔→パラメータ推定→FF推定")

st.title("顔画像 → 2パラメータ推定 → FF推定AI")

uploaded_file = st.file_uploader("撮影 or アップロード", type=["jpg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="入力画像")

    result= predict(img)

    if result is None:
        st.error("顔が検出できませんでした")
    else:
#        st.image(face, caption="検出された顔")

        st.success(f"ParameterFW: {result[0]:.4f}")
        st.success(f"ParameterFL: {result[1]:.4f}")
        st.success(f"ParameterPC1: {result[2]:.4f}")
        st.success(f"ParameterPC2: {result[3]:.4f}")
        st.success(f"ParameterFF1: {result[4]:.4f}")
        st.success(f"ParameterFF2: {result[5]:.4f}")
        st.success(f"ParameterFF3: {result[6]:.4f}")
        st.success(f"ParameterFF4: {result[7]:.4f}")
        st.success(f"ParameterFF5: {result[8]:.4f}")
        st.success(f"ParameterFF6: {result[9]:.4f}")
        st.success(f"ParameterFF7: {result[10]:.4f}")

#        st.success(f"ParameterFWFL: {result[0],result[1]:7.1f}")
#        st.success(f"ParameterPC1PC2: {result[2],result[3]:10.3f}")
#        st.success(f"ParameterFF: {result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11]:7.1f}")
