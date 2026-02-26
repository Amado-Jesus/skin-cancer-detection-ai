from PIL import Image
import streamlit as st
import torch
from utils import *




st.title("Sistema Inteligente para la Detección Temprana de Cáncer de Piel")
st.write("Sube una imagen del melanoma  para obtener la predicción del modelo.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png","webp"]
)



@st.cache_resource
def load_model():
    
    model = CNN()
    model.load_state_dict(
    torch.load("skin_cancer_residualCNN.pth", map_location=torch.device("cpu"))
)
    
   
   
    return model

model = load_model()

# ---------------------------
# PREDICCIÓN
# ---------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
   
    

    fig = predict(
    img=image,
    model=model,
    transforms=transforms,
    device='cpu'
)


    st.subheader("Resultado")
    st.pyplot(fig)
   
