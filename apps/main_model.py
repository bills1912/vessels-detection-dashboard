import time
import torch
import numpy as np
import streamlit as st

def app():
    st.write("## Ship Imagery Prediction")
    st.write("### Model evaluation:")
    eval_col1, eval_col2, eval_col3, eval_col4, eval_col5 = st.columns(spec=5)
    eval_col1.metric("Precision", "89.52%")
    eval_col2.metric("Recall", "83.54%")
    eval_col3.metric("F1-Score", "86.43%")
    eval_col4.metric("mAP 0.5", "85.39%")
    eval_col5.metric("mAP 0.5:0.95", "62.63%")

    uploaded_file = st.file_uploader("Choose a ship imagery")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image to predict')
    st.write("If you don't have any satellit imagery data, you can choose the sample data form the table below:")
    folder_path = st.text_input("Image path",
                                help="This field the image path field that the model will predict\
                                the object inside the image that we have uploaded",
                                placeholder="Copy the path of image to this field")

    prediction = st.button("Predict")
    if prediction:
        ship_model = torch.hub.load('ultralytics/yolov5', 'custom', path="apps/model/main_model.pt", force_reload=True)
        ship_model.conf = 0.6
        ship_model.iou = 0.55
        results = ship_model(f"{folder_path}")
        with st.spinner("Loading..."):
            time.sleep(3.5)
            st.success("Done!")
        st.image(np.squeeze(results.render()))
        results.print()
