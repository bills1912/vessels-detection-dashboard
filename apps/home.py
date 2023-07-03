import streamlit as st

def app():
    st.write("# Welcome to Ship Detection Application! :satellite:")
    st.markdown(
            """
            This application is build based on YOLOv5 with extral large model. User just
            upload an image, and press the 'Predict' button to make a prediction base on
            a training model before **(the explanation of the result from the detection and tutorial how to use the model
            is on the bottom of this page. Scroll down or [click here!](#explanation-of-the-ship-detection-result))**.

            ### For more information, please visit:

            - Check out [my github](https://github.com/bills1912)
            - Jump into YOLOv5 [documentation](https://docs.ultralytics.com/)

        """
        )
    
    st.markdown(
        """
        ## Tutorial: How to Use Ship Detection Model
        Here is the step by step how to use the model on this dashboard:
        - first, **prepare the satellite imagery image** that you want to use. If you don't have the image, you can use this sample image, by clicking the **"Download Image"** \
          on the end of this dashboard usage explanation or you can use the table that will be provided in the menu on the sidebar\
          that will be selected by clicking the field of the table **(and you can skip step 3, 4, and 5 in this tutorial)**;
        - then, **choose the detection model** that you want to use **(on the side bar)**, **First Detection Model** to use the model that is built\
          using **ShipRSImageNet dataset** or **Second Detection Model** to use the model that is built using **Tanjung Priok Port satellite imagery\
          dataset**;
        - to upload your image, **click the "Browse File"** button, then upload your image;
        - after the image is uploaded, **right click the image** and then **copy the image address** by clicking **"Copy image address"** button;
        - then **paste the image address** on the box below the image;
        - finally, **click the "Predict"** button to start the detection of the object inside your image. Wait untill the result appear.
        
        ## Explanation of The Ship Detection Result
        <p align="center">
             <img src="https://huggingface.co/spaces/billsar1912/YOLOv5x6-marine-vessels-detection/resolve/main/apps/image/result.png" alt="Example of the result"/>
        </p>

        Here is the explanation of the result from the example on the image above:\

        - **box**, indicate the object that the model can detect;
        - **label of the box**, indicate the name of the object that the model detect;
        - **number beside the label**, indicate how much the confidence of the model detect the object;
        """, unsafe_allow_html=True
    )
    with open("apps/image/sample.jpg", "rb") as file:
        st.download_button(
            label="Download Sample Image",
            data=file,
            file_name="sample.jpg",
            mime="image/jpg"
        )