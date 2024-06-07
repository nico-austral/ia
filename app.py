import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from keras.models import load_model
import streamlit as st
import folium
from streamlit_folium import st_folium

# Ruta absoluta al directorio que contiene el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo y las etiquetas una vez al inicio
model_path = os.path.join(script_dir, "keras_model.h5")
model = load_model(model_path, compile=False)

labels_path = os.path.join(script_dir, "labels.txt")
class_names = open(labels_path, "r").readlines()

# Función para clasificar el residuo
def classify_waste(img):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    return class_name

# Aplicación Streamlit
st.set_page_config(page_title="Clasificador de Residuos", page_icon=":recycle:", layout="centered")

# Agregar CSS personalizado
st.markdown("""
    <style>
    
    .stApp {
        background-color: #ffffff;
        padding-left: 50px;
        padding-right: 50px;
    }
    .stButton>button {
        color: white;
        background-color: #6c757d;
        border-radius: 10px;
    }
    .stFileUploader label {
        color: #495057;
        font-weight: bold;
    }
    .stImage {
        text-align: center;
    }
    .css-1wa3eu0 {
        border: 2px solid #007BFF !important; /* Blue border color */
        border-radius: 10px;
    }

    .selectbox{
         border-color: #000000
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Boti recicla :robot_face:", divider='rainbow')

st.header("Clasifica tu residuo y descubre dónde puedes tirarlo :recycle: :round_pushpin:", divider='rainbow')

# Cargar imagen
st.subheader('Subí una foto de tu resiudo')
input_img = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if input_img:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(input_img, caption='Tu residuo', use_column_width=True)
        
    with col2:
        st.subheader("Resultado de la clasificación")
        image_file = Image.open(input_img)
        label = classify_waste(image_file)
        labels_dict = {
            "0 trash": "basura",
            "1 battery": "batería / electrónico",
            "2 metal": "metal",
            "3 paper": "papel",
            "4 glass": "vidrio",
            "5 plastic": "plástico",
            "6 cardboard": "cartón",
            "7 biological": "biológico"
        }
        st.success(f"Tu residuo es **{labels_dict.get(label, 'No clasificado')}**")

# Cargar el archivo de puntos verdes y las comunas
excel_file_path = os.path.join(script_dir, "puntos-verdes.xlsx")
df = pd.read_excel(excel_file_path)
df['Coordinates'] = df['WKT'].apply(lambda wkt: tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', wkt)[::-1])))

# Mapear nombres de comunas a números para mostrar en el menú desplegable
comunas_map = {
    "Comuna 1": 1,
    "Comuna 2": 2,
    "Comuna 3": 3,
    "Comuna 4": 4,
    "Comuna 5": 5,
    "Comuna 6": 6,
    "Comuna 7": 7,
    "Comuna 8": 8,
    "Comuna 9": 9,
    "Comuna 10": 10,
    "Comuna 11": 11,
    "Comuna 12": 12,
    "Comuna 13": 13,
    "Comuna 14": 14,
    "Comuna 15": 15
}

# Crear menú desplegable para seleccionar comuna
st.subheader("Selecciona tu comuna")
comuna_nombre = st.selectbox("",list(comunas_map.keys()))
if comuna_nombre:
    comuna_numero = comunas_map[comuna_nombre]
    filtered_df = df[df['comuna'] == f'COMUNA {comuna_numero}']
    if filtered_df.empty:
        st.warning("No hay puntos de reciclaje en esta comuna")
    else:
        st.subheader("Mapa de sitios de reciclaje")
        map_center = filtered_df['Coordinates'].apply(pd.Series).mean().tolist()
        m = folium.Map(location=map_center, zoom_start=12)
        for _, row in filtered_df.iterrows():
            folium.Marker(row['Coordinates'], popup=f"<b>Materiales:</b> {row['materiales']}").add_to(m)
        st_folium(m, width=700, height=500)

st.balloons()

# Sección de suscripción al newsletter
st.header("Suscríbete a nuestro newsletter")
email = st.text_input("Ingresa tu correo electrónico")
if st.button("Suscribirse"):
    if email:
        # Guardar el email en un archivo de texto
        subscribers_file_path = os.path.join(script_dir, "subscribers.txt")
        with open(subscribers_file_path, "a") as f:
            f.write(email + "\n")
        st.success(f"¡Gracias por suscribirte, {email}!")
    else:
        st.error("Por favor, ingresa un correo electrónico válido.")