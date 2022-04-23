
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("CIFAR10ベース")

st.sidebar.write("")

img_file = st.sidebar.file_uploader("画像を選択", type=["png", "jpg"])

if img_file is not None:
    with st.spinner("推定中"):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        results = predict(img)

        st.subheader("判定結果")
        n_top = 3
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs =  [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops = {"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}

        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)
