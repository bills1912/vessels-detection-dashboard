import streamlit as st
import streamlit as st
from multiapp import MultiApp
from apps import (
    home,
    main_model,
    fine_tune
)
st.set_page_config(
        page_title="Ship Detection using YOLOv5 Medium Model",
        page_icon=":ship:",
        layout="wide"
    )

apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)
apps.add_app("YOLOv5x6 Model", main_model.app)
apps.add_app("Fine-Tuning Model", fine_tune.app)

# The main app
apps.run()