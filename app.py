import streamlit as st
import numpy as np
import cv2
from stego_dct import SteganografiDCT 
from psnr import psnr

st.set_page_config(page_title="Steganografi DCT", layout="centered")
st.title("Steganografi DCT — Encoder, Decoder, PSNR Checker")

model = SteganografiDCT()

st.sidebar.header("Navigasi")
mode = st.sidebar.radio("Pilih Mode:", ["Encode", "Decode", "Psnr Checker"])

if mode == "Encode":
    st.header("Encode Pesan ke Dalam Gambar")

    pesan = st.text_input("Masukkan Pesan Rahasia:")
    uploaded = st.file_uploader("Upload Gambar Cover", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption="Gambar Cover")

    if st.button("Encode Pesan"):
        if uploaded is None:
            st.error("Upload gambar terlebih dahulu!")
        elif pesan.strip() == "":
            st.error("Pesan tidak boleh kosong!")
        else:
            try:
                stego_img = model.encode(pesan, img)

                st.success("Pesan berhasil disisipkan!")

                stego_rgb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)

                st.image(stego_rgb, caption="Gambar Stego")

                # Convert ke file downloadable
                _, buffer = cv2.imencode(".png", stego_img)
                st.download_button(
                    label="Download Stego Image",
                    data=buffer.tobytes(),
                    file_name="stego_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Terjadi error: {e}")

elif mode == "Decode":
    st.header("Decode Pesan dari Gambar Stego")

    uploaded = st.file_uploader("Upload Gambar Stego", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption="Gambar Stego")

    if st.button("Decode Pesan"):
        if uploaded is None:
            st.error("Upload gambar terlebih dahulu!")
        else:
            try:
                pesan = model.decode(img)
                st.success("Pesan berhasil diekstraksi!")
                st.code(pesan, language="text")

            except Exception as e:
                st.error(f"Terjadi error: {e}")
else:
    st.header("PSNR Checker")
    uploaded1 = st.file_uploader("Upload Gambar Asli", type=["jpg", "jpeg", "png"], key="img1")
    uploaded2 = st.file_uploader("Upload Gambar Stego", type=["jpg", "jpeg", "png"], key="img2")

    if st.button("Hitung PSNR"):
        if uploaded1 is None or uploaded2 is None:
            st.error("Upload kedua gambar terlebih dahulu!")
        else:
            file_bytes1 = np.asarray(bytearray(uploaded1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

            if img1.shape != img2.shape:
                st.error("Gambar harus memiliki dimensi yang sama!")
            else:
                psnr_value = psnr(img1, img2)
                st.success(f"PSNR antara kedua gambar: {psnr_value:.2f} dB")