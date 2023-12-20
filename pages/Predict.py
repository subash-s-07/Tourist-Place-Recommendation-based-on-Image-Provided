import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
from io import BytesIO

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(45deg, #FFD700, #FF69B4, #87CEEB, #FFD700, #98FB98, #FFA07A, #ADD8E6, #00FF00, #FF6347, #8A2BE2);

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


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from transformers import pipeline
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
word2vec_model = Word2Vec.load(r"Word2vecTourism.model")
spot_names_saved = pd.read_csv(r"spot_names.csv", header=None, index_col=False)
spot_names_saved = spot_names_saved.iloc[:, 0]
spot_caption_vectors_saved = np.load(r"spot_caption_vectors.npy")
df = pd.read_csv(r"processed_dataset.csv", header=None)
df_link = pd.read_excel(r"links.xlsx",header=None)
def display_images_with_slider(recommended_places, images):
    gradient_style = (
        "background: linear-gradient(to right, #b4e5ff, #ffd3b6); "
        "padding: 20px; border-radius: 10px; box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);"
        "height:600px;"
        "width:800px;"
    )
    st.title(recommended_places)
    #images = ["https://www.mumbaiindians.com/static-assets/waf-images/d1/85/18/21-9/1200-675/icOTMY7y37.jpeg", "https://images.thequint.com/thequint%2F2021-04%2F15d4fcf5-7c0e-481e-9dc6-37e444c58fef%2FIPL21M8_55.JPG?rect=0%2C0%2C3872%2C2178&auto=format%2Ccompress&fmt=webp&width=720","https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_1280,q_70/lsci/db/PICTURES/CMS/346400/346456.4.jpg","https://c.ndtvimg.com/2023-03/7v2ldgro_rahul-dravid_625x300_13_March_23.jpg?im=FeatureCrop,algorithm=dnn,width=806,height=605"]
    image_html = ''
    for url in range(len(images)):
        image_html += f'<a href={images[url]} target = "_blank"><img src="{images[url]}" style="width:500px; height:500px; object-fit: cover; display: block;margin-right: 40px;"></a>'

    st.markdown(f"""<div style="{gradient_style}"><div style="display: flex; overflow-x: scroll; padding: 10px 0;width:750px; height:750px; ">{image_html}</div></div>""", unsafe_allow_html=True )
    st.markdown("<br><hr>", unsafe_allow_html=True)
def display_image(recommend_places,input_caption):
    image_links=[]
    r=[]
    for place in recommended_places:
            filtered_df = df[df[0] == place]
            link =df_link[df_link[0] == place]
            captions_to_compare=filtered_df.iloc[0, 1:].dropna().tolist()
            place_link=link.iloc[0, 1:].dropna().tolist()
            all_captions = [input_caption] + captions_to_compare

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_captions)

            knn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
            knn_model.fit(tfidf_matrix)

            input_vector = tfidf_vectorizer.transform([input_caption])

            distances, indices = knn_model.kneighbors(input_vector)
            image=[]
            for i, index in enumerate(indices.flatten()[1:]):
                similarity_score = 1 - distances.flatten()[i + 1]
                recommended_caption = all_captions[index]
                original_index = filtered_df.index[0]
                image.append(place_link[index-1])
            image_links.append(image)
            r.append(place)
    for i in range(len(r)):
        display_images_with_slider(r[i], image_links[i])
            
def preprocess_caption(caption):
    caption = caption.lower()
    words = caption.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

def recommend_tourist_spots(input_caption, word2vec_model, spot_caption_vectors, spot_names, top_n=5):
    input_caption = preprocess_caption(input_caption)
    input_caption_vectors = [word2vec_model.wv[word] for word in input_caption if word in word2vec_model.wv]
    if input_caption_vectors:
        input_caption_vector = np.mean(input_caption_vectors, axis=0)
        similarities = cosine_similarity(input_caption_vector.reshape(1, -1), spot_caption_vectors)
        top_indices = similarities.argsort()[0][-top_n:][::-1]
        recommendations = [(spot_names.iloc[i], similarities[0][i]) for i in top_indices]
    else:
        recommendations = [("No relevant spots found", 0.0)]
    return recommendations
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from io import BytesIO

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Assuming you've already loaded these variables from your previous code
model_name = "local_model"
word2vec_model = Word2Vec.load(r"Word2vecTourism.model")
spot_names_saved = pd.read_csv(r"spot_names.csv", header=None, index_col=False)
spot_names_saved = spot_names_saved.iloc[:, 0]
spot_caption_vectors_saved = np.load(r"spot_caption_vectors.npy")
df = pd.read_csv(r"processed_dataset.csv", header=None)
df_link = pd.read_excel(r"links.xlsx", header=None)

# Function to display images with a slider
def display_images_with_slider(recommended_places, images):
    gradient_style = (
        "background: linear-gradient(to right, #b4e5ff, #ffd3b6); "
        "padding: 20px; border-radius: 10px; box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);"
        "height:600px;"
        "width:800px;"
        "transition: all 0.3s ease;"
    )
    st.title(recommended_places)
    image_html = ''
    for url in range(len(images)):
        image_html += f'<a href={images[url]} target="_blank"><img src="{images[url]}" style="width:500px; height:500px; object-fit: cover; display: block;margin-right: 40px; border-radius: 10px;"></a>'

    st.markdown(
        f"""<div style="{gradient_style}"><div style="display: flex; overflow-x: scroll; padding: 10px 0;width:750px; height:750px; ">{image_html}</div></div>""",
        unsafe_allow_html=True
    )
    st.markdown("<br><hr>", unsafe_allow_html=True)
def display_image(recommend_places, input_caption):
    image_links = []
    r = []
    for place in recommend_places:
        filtered_df = df[df[0] == place]
        link = df_link[df_link[0] == place]
        captions_to_compare = filtered_df.iloc[0, 1:].dropna().tolist()
        place_link = link.iloc[0, 1:].dropna().tolist()
        all_captions = [input_caption] + captions_to_compare

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_captions)

        knn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
        knn_model.fit(tfidf_matrix)

        input_vector = tfidf_vectorizer.transform([input_caption])

        distances, indices = knn_model.kneighbors(input_vector)
        image = []
        for i, index in enumerate(indices.flatten()[1:]):
            similarity_score = 1 - distances.flatten()[i + 1]
            recommended_caption = all_captions[index]
            original_index = filtered_df.index[0]
            image.append(place_link[index - 1])
        image_links.append(image)
        r.append(place)

    for i in range(len(r)):
        display_images_with_slider(r[i], image_links[i])

# Function to preprocess caption
def preprocess_caption(caption):
    caption = caption.lower()
    words = caption.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Function to recommend tourist spots
def recommend_tourist_spots(input_caption, word2vec_model, spot_caption_vectors, spot_names, top_n=5):
    input_caption = preprocess_caption(input_caption)
    input_caption_vectors = [word2vec_model.wv[word] for word in input_caption if word in word2vec_model.wv]
    if input_caption_vectors:
        input_caption_vector = np.mean(input_caption_vectors, axis=0)
        similarities = cosine_similarity(input_caption_vector.reshape(1, -1), spot_caption_vectors)
        top_indices = similarities.argsort()[0][-top_n:][::-1]
        recommendations = [(spot_names.iloc[i], similarities[0][i]) for i in top_indices]
    else:
        recommendations = [("No relevant spots found", 0.0)]
    return recommendations

# Streamlit UI
st.title("Image Captioning and Tourist Spot Recommendations")
tabs = st.tabs(["Upload Images and Predict", "Input Caption and Predict"])

with tabs[0]:
    st.header("Upload Images")
    uploaded_files = st.file_uploader("Choose one image...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files is not None and len(uploaded_files) >= 1:
        try:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).resize((150, 150))
                caption = pipe(image)[0]['generated_text']
                st.image(image, caption=caption, use_column_width=True)
                st.markdown(f"<div style='color: #008080;'>Generated Caption: {caption}</div>", unsafe_allow_html=True)
                recommendations = recommend_tourist_spots(caption, word2vec_model, spot_caption_vectors_saved, spot_names_saved)
                st.subheader("Top recommended tourist spots:")
                for spot, similarity in recommendations:
                    st.markdown(f"<div style='color: pink;font-size:30px;'>- {spot}<span style='color: #008080;'> (Similarity: {similarity:.2f}<span>)</div>", unsafe_allow_html=True)
                recommended_places = [spot_name for spot_name, _ in recommendations]
                display_image(recommended_places, caption)
                
        except Exception as e:
            st.error(f"<div style='color: red;'>Error processing the images: {str(e)}</div>", unsafe_allow_html=True)

with tabs[1]:
    input_caption = st.text_input("Enter a caption:")
    if st.button("Generate Recommendations"):
        recommendations = recommend_tourist_spots(input_caption, word2vec_model, spot_caption_vectors_saved, spot_names_saved)
        st.subheader("Top recommended tourist spots:")
        for spot, similarity in recommendations:
            st.markdown(f"<div style='color: pink;font-size:30px;'>- {spot}<span style='color: #008080;'> (Similarity: {similarity:.2f}<span>)</div>", unsafe_allow_html=True)
        recommended_places = [spot_name for spot_name, _ in recommendations]
        display_image(recommended_places, input_caption)