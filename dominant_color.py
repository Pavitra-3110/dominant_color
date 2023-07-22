import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



st.title("Dominant Color Extraction")
st.subheader("Uplaod An Image")
img = st.file_uploader("Choose an image")
if img is not None:
    st.header("Original Image")
    st.image(img)

    
    n_clusters=  int(st.number_input('Pick a number', 1,1000))
    print(type(img))
    img = plt.imread(img)
   

    #KMeans code
    n = img.shape[0]*img.shape[1]
    # flattening an image
    all_pixels = img.reshape((n, 3))
    model  = KMeans(n_clusters)
    model.fit(all_pixels)
    centers = model.cluster_centers_.astype('uint8')
    new_img = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
       group_idx = model.labels_[i]
       new_img[i] = centers[group_idx]
    new_img = new_img.reshape(*img.shape)

    st.header("Modified Image")
    st.image(new_img)