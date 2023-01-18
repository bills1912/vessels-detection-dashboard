import time
import torch
import cv2
import urllib
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

def app():
    st.write("## Shiploads Estimation")
    st.write("### Model evaluation:")
    eval_col1, eval_col2, eval_col3, eval_col4, eval_col5 = st.columns(spec=5)
    eval_col1.metric("Precision", "99.03%")
    eval_col2.metric("Recall", "98.39%")
    eval_col3.metric("F1-Score", "98.71%")
    eval_col4.metric("mAP 0.5", "98.96%")
    eval_col5.metric("mAP 0.5:0.95", "69.61%")

    uploaded_file = st.file_uploader("Choose a ship imagery")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image to predict')

    st.write("If you don't have any satellit imagery data, you can choose the sample data form the table below:")
    fine_tuning_sample = pd.read_csv('https://raw.githubusercontent.com/bills1912/vessels-detection-dashboard/main/apps/sample_data/fine_tuning_sample.csv',
    sep=';')
    gb = GridOptionsBuilder.from_dataframe(fine_tuning_sample)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_auto_height(autoHeight = True)
    gb.configure_selection('multiple', use_checkbox=False) #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        fine_tuning_sample,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        theme='dark', #Add theme color to the table
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=True
    )

    selected = grid_response['selected_rows']
    if uploaded_file:
        folder_path = st.text_input("Image path",
                                    help="This field the image path field that the model will predict\
                                    the object inside the image that we have uploaded",
                                    placeholder="Copy the path of image to this field")
    elif selected:
        folder_path = st.text_input("Image path", value=f"{selected[0]['Sample Data']}",
                                    help="This field the image path field that the model will predict\
                                    the object inside the image that we have uploaded",
                                    placeholder="Copy the path of image to this field")

    prediction = st.button("Predict")
    if prediction:
        request = urllib.request.urlopen(folder_path)
        arr = np.asarray(bytearray(request.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ship_model = torch.hub.load('ultralytics/yolov5', 'custom', path="apps/model/fine_tune.pt", force_reload=True)
        ship_model.conf = 0.6
        ship_model.iou = 0.55
        results = ship_model(f"{folder_path}")
        data_vis = []
        with st.spinner("Loading..."):
            time.sleep(3.5)
            st.success("Done!")

            # Special for visualization
            for j in results.xyxy[0].cpu().numpy():

                data_vis.append(j)
                for idx in range(len(data_vis)):
                    x = int(data_vis[idx][0])
                    y = int(data_vis[idx][1])
                    w = int(data_vis[idx][2])
                    h = int(data_vis[idx][3])
                    ship_class = int(data_vis[idx][5])
                # cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 6)
                if ship_class == 0:
                    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 6)
                    cv2.putText(img, (f'Merchant {idx+1}'), (x-60, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif ship_class == 1:
                    cv2.rectangle(img, (x, y), (w, h), (255, 61, 0), 6)
                    cv2.putText(img, (f'Other Ship {idx+1}'), (x-60, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 61, 0), 3)
                else:
                    cv2.rectangle(img, (x, y), (w, h), (30, 144, 255), 6)
                    cv2.putText(img, (f'Warship {idx+1}'), (x-60, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 3)
            st.image(img)
        # results.print()

        # Shiploads estimation
        st.write('Here is the estimation of the ship that has been detected from the image above:')
        est_data = pd.DataFrame()
        ship = []
        lth = []
        wdh = []
        ar = []
        ship_load = []
        dh, dw, _ = img.shape
        print(dw, dh)

        initial_resolution = 0.30
        data_est = []

        for l in results.xyxy[0].cpu().numpy():

            data_est.append(l)
            for idx in range(len(data_est)):
                x = int(data_est[idx][0])
                y = int(data_est[idx][1])
                w = int(data_est[idx][2])
                h = int(data_est[idx][3])
                ship_class = int(data_est[idx][5])

            length = (w - x)*initial_resolution
            width = (h - y)*initial_resolution
            area = length * width
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 6)
            if ship_class == 0:
                ship.append(f"Merchant {idx+1}")
                if length > width:
                    lth.append(length)
                    wdh.append(width)
                else:
                    lth.append(width)
                    wdh.append(length)
                ar.append(area)
                if (length * width < 545):
                    ship_load.append("< 400 imperial tons")
                elif (length * width >= 545) and (length * width <= 1831.2):
                    ship_load.append("400 imperial tons - 5000 imperial tons")
                elif (length * width > 1831.2) and (length * width <= 5980):
                    ship_load.append("10000 imperial tons - 40000 imperial tons")
                elif (length * width > 5980) and (length * width <= 9237.8):
                    ship_load.append("50000 imperial tons - 150000 imperial tons")
                elif (length * width > 9237.8):
                    ship_load.append("> 150000 imperial tons")
        est_data['Ship'] = ship
        est_data['Overall Length (m)'] = lth
        est_data['Width (m)'] = wdh
        est_data['Area (m^2)'] = ar
        est_data['Shipload Estimation (m^2)'] = ship_load
        
        hide_table_row_index = """
                    <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                    </style>
                    """
        center_align = """
                    <style>
                        thead tr th:first-child {text_align: center}
                        tbody th {text_align: center}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.markdown(center_align, unsafe_allow_html=True)

        # Display a static table
        st.table(est_data)


