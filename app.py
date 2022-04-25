
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import plotly.graph_objects as go


st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("画像認識アプリ")
st.write("ResNetベース")

st.write("")

img_file = st.file_uploader("画像を選択", type=["png", "jpg"])

if img_file is not None:
    with st.spinner("判定中"):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        results = predict(img)

        st.subheader("判定結果")
        n_top = 5
        for result in results[:n_top]:
            st.write(str(round(result[1]*100, 2)) + "%の確率で" + result[0] + "です。")

        pie_labels = [result[0] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs =  [result[1] for result in results[:n_top]]
        pie_probs.append(sum([result[1] for result in results[n_top:]]))
        fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_probs,
                                    direction='clockwise', hole=0.5)])
        st.plotly_chart(fig)
