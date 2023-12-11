import streamlit as st
import os
l = os.listdir(r"plots")

root = "plots"
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: radial-gradient(circle, #FFD700, #FF69B4, #87CEEB, #FFD700, #98FB98, #FFA07A, #ADD8E6, #00FF00, #FF6347, #8A2BE2);
        background-size: 200% 200%;
        animation: radialGradientAnimation 10s infinite linear;
    }}
        [data-testid="stSidebar"] {{
        background: radial-gradient(circle, #87CEEB, #FF69B4); /* Radial gradient for sidebar */
        background-size: cover;
        background-position: top left;
        background-attachment: local;
    }}
    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.7);
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}

    @keyframes radialGradientAnimation {{
        0% {{
            background-position: 0% 0%;
        }}
        100% {{
            background-position: 100% 100%;
        }}
    }}

    @keyframes linearGradientAnimation {{
        0% {{
            background-position: 0% 0%;
        }}
        100% {{
            background-position: 100% 100%;
        }}
    }}
    </style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
for i in l:
    first_image_path = os.path.join(root, i)
    st.subheader(i.split(".")[0])
    st.image(first_image_path)
