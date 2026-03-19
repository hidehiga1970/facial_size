import streamlit as st
import numpy as np
import cv2

from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# ---------------------------
# モデル読み込み
# ---------------------------
model = load_model("face_parameter_model.keras")

mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")
pca_components = np.load("pca_model.npy")

embedder = FaceNet()
detector = MTCNN()

# ---------------------------
# PCA適用
# ---------------------------
def apply_pca(x):
    return np.dot(x, pca_components.T)

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

    emb = get_embedding(face)

    emb = apply_pca(emb)

    emb = (emb - mean) / scale

    emb = np.expand_dims(emb,axis=0)

    pred = model.predict(emb)

    return pred[0], face

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="顔→パラメータ推定")

st.title("顔画像 → 2パラメータ推定AI")

uploaded_file = st.file_uploader("撮影 or アップロード", type=["jpg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="入力画像")

    result, face = predict(img)

    if result is None:
        st.error("顔が検出できませんでした")
    else:
        st.image(face, caption="検出された顔")

        st.success(f"Parameter1: {result[0]:.4f}")
        st.success(f"Parameter2: {result[1]:.4f}")