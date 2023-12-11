import streamlit as st
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(45deg, #FFD700, #FF69B4, #87CEEB, #FFD700, #98FB98, #FFA07A, #ADD8E6, #00FF00, #FF6347, #8A2BE2);
        background-size: 500% 500%;
        animation: gradientAnimation 10s infinite linear;
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

    @keyframes gradientAnimation {{
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
st.write("<h1 style='text-align: center;color:  #FFA500;'>Tourist Place Recommender</h1><br><br>", unsafe_allow_html=True)
st.image("https://www.un.org/sites/un2.un.org/files/events_tourism_day.jpg", use_column_width=True)

st.markdown(
    """
    <style>
        .hi {
            color: #FFA500;
            text-align: center;
        }
        .reportview-container {
            background: linear-gradient(to right, #FFD700, #87CEEB, #FFD700);
            color: #2E4053;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to right, #FFD700, #87CEEB, #FFD700);
            color: #2E4053;
        }
        .dataframe {
            background: linear-gradient(to right, #FFD700, #87CEEB, #FFD700);
            color: #2E4053;
            font-size: 14px; /* Adjusted font size */
        }
        .st-cy {
            color: #00FFFF;
        }
        .st-dz {
            color: #FF4500;
        }
.custom-container {
    background: linear-gradient(to right, #FFD700, #87CEEB, #FFD700); /* Gradient background */
    padding: 10px; /* Adjusted padding */
    border-radius: 10px;
    margin-top: 10px; /* Adjusted margin top */
    font-size: 18px; /* Adjusted font size */
    border: 0.2px solid #2E4053; /* Dark slate gray border */
}

.description-container {
    background: linear-gradient(to right, #FFD700, #87CEEB, #FFD700); /* Gradient background */
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 18px;
    border: 0.2px solid #2E4053; /* Dark slate gray border */
}

        p {
            font-size: 18px; /* Adjusted font size */
            color: #2E4053;
        }
        h1 {
            color: #FFA500;
            text-align: center;
        }
        strong {
            color: #FFA500;
        }
    </style>
    """, 
    unsafe_allow_html=True
)


with st.container():
    st.markdown(
        """
        <div class="description-container">
        <h1 class="hi">About Project</h1>
        <p>
        With the increasing popularity of travel and tourism, there is a growing demand for personalized suggestions on where to go.<br><br>
        This is designed to elevate travel planning through intelligent and personalized tourist spot recommendations by giving suggestions that suit your taste.
        Utilizing advanced image and text analysis technologies, the platform allows users to receive tailored suggestions based on either uploaded photos or provided captions.
        </p>
        """
    ,unsafe_allow_html=True)

st.markdown("""<div class="custom-container">
    <h1>Dataset</h1>
    <p>The dataset is a compilation of U.S. tourist spots sourced from TripAdvisor.com, featuring a diverse array of popular destinations. Each entry in the dataset includes essential information about these tourist spots and is enriched with over 60 image URLs captured from TripAdvisor.com.
        This extensive collection not only provides insights into the preferences and recommendations of a wide-ranging user base but also offers a visual exploration of each location.<br><br>
        Therefore,making it an ideal resource in exploring and understanding the vibrant landscape of U.S. tourism.</p>
</div>""",unsafe_allow_html=True)

st.markdown("""<div class="custom-container">
    <h1>Process</h1>
    <p><strong>INPUT :</strong> Upload an image or Enter a textual description of a tourist spot.</p>
    <p><strong>Caption Retrieval :</strong> Utilized the Blip Hugging Face model to extract descriptive captions for images in the dataset, enhancing understanding.</p>
    <p><strong>Caption Preprocessing :</strong> Employed various Natural Language Toolkit (NLTK) processes to preprocess and refine captions, optimizing them for subsequent analysis.</p>
    <p><strong>Word Embedding (Word2Vec):</strong> Applied Word2Vec, a neural network model, to transform preprocessed captions into vector representations, enabling semantic understanding and similarity calculations.</p>
    <p><strong>Cosine Similarity: </strong>Utilized cosine similarity on Word2Vec embeddings to identify tourist spots with captions similar to user input, enhancing personalized recommendations.</p>
    <p><strong>K-Nearest Neighbors (KNN):</strong> Applied KNN algorithm to identify the k-nearest captions among the suggested tourist spots, refining recommendations based on textual similarity.</p>
    <p><strong>Image Recommended: </strong> Displayed images corresponding to the refined tourist spot recommendations, providing users with visual insights and aiding in decision-making.</p>
</div>""",unsafe_allow_html=True)
	
