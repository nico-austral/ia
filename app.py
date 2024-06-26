import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from keras.models import load_model
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import base64
import sqlite3

# Configuración de la página
st.set_page_config(page_title="Clasificador de Residuos", page_icon="BotiRecicla.png", layout="wide")

# Ruta absoluta al directorio que contiene el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Función para convertir imagen a base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convertir la imagen del robot a base64
image_path = os.path.join(script_dir, "BotiRecicla.png")  # Cambia esto a la ruta adecuada
image_base64 = get_base64_of_bin_file(image_path)

# Cargar el modelo y las etiquetas una vez al inicio
@st.cache_resource
def load_model_and_labels():
    model_path = os.path.join(script_dir, "keras_model.h5")
    model = load_model(model_path, compile=False)
    
    labels_path = os.path.join(script_dir, "labels.txt")
    class_names = open(labels_path, "r").readlines()
    
    return model, class_names

model, class_names = load_model_and_labels()

# Función para clasificar el residuo (sin caché)
def classify_waste(_img):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = _img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    class_name = class_names[index].strip()
    return class_name, confidence

# Cargar el archivo de puntos verdes y barrios
@st.cache_data
def load_recycling_points():
    excel_file_path = os.path.join(script_dir, "puntos-verdes.xlsx")
    df = pd.read_excel(excel_file_path)
    df['Coordinates'] = df['WKT'].apply(lambda wkt: tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', wkt)[::-1])))
    df = df.drop_duplicates(subset=['Coordinates'])
    return df

df = load_recycling_points()

# Agregar CSS personalizado
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    .stApp {
        background-color: white;  /* Fondo verde claro */
        padding-left: 50px;
        padding-right: 50px;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #6c757d;
        border-radius: 10px;
        font-family: 'Poppins', sans-serif;
    }
    .stFileUploader label {
        color: #495057;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
    }
    .stImage {
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    .css-1wa3eu0 {
        border: 2px solid #007BFF !important; /* Blue border color */
        border-radius: 10px;
        font-family: 'Poppins', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        font-family: 'Poppins', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .stTabs [data-baseweb="tab"]:nth-child(3) {
        margin-left: auto;
    }
    .stTabs [data-baseweb="tab"]:nth-child(3)::before {
        content: '\\1F5DD'; /* Icono de llave */
        margin-right: 8px;
    }
    h1, h2, h3, h4, h5, h6, .stHeader, .stText {
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        font-weight: bold;
        font-size: 50px;
        margin-bottom: 5px;  /* Espacio entre título y subtítulo */
        text-align: center;  /* Centrar el título */
    }
    .byline {
      
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        font-size: 19px;
        text-align: center;  /* Alinear al centro */
        margin-top: 0;  /* Remover margen superior */
    }
    video::-webkit-media-controls {
        display: none !important;
    }
    .stCameraInput {
        display: none; /* Ocultar inicialmente */
    }
    .camera-button {
        font-size: 20px;
        color: #6c757d;
        cursor: pointer;
    }
    .camera-button:hover {
        color: #007BFF;
    }
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .logo-spin {
        animation: rotate 2s linear infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Clasificación de Residuos", "Información y Suscripción", "Gestión de Suscriptores"])

# Estado de la cámara
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

with tab1:
    st.markdown(f'''
    <div style="text-align: center;">
        <h1><img id="logo" src="data:image/png;base64,{image_base64}" width="50" onclick="spinLogo()"/> Boti Recicla</h1>
        <div class="byline">By NeuralDios</div>
    </div>
    ''', unsafe_allow_html=True)
    st.header("Clasifica tu residuo y descubre dónde puedes tirarlo :recycle: :round_pushpin:")

    # Cargar imagen
    st.subheader('Subí una foto de tu residuo o toma una foto con tu dispositivo')
    input_img = st.file_uploader("Selecciona una imagen de tu residuo", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

    if st.button("📷 Tomar una foto de tu residuo"):
        st.session_state['camera_active'] = not st.session_state['camera_active']

    if st.session_state['camera_active']:
        img_data = st.camera_input("Toma una foto de tu residuo")
    else:
        img_data = None

    if input_img or img_data:
        if input_img:
            image = Image.open(input_img)
        else:
            image = Image.open(img_data)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Tu residuo', use_column_width=True)
        with col2:
            st.subheader("Resultado de la clasificación")
            label, confidence = classify_waste(image)
            if confidence < 0.99:
                st.warning("La clasificación no es segura. Por favor, sube otra foto.")
            else:
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
                residuo_clasificado = labels_dict.get(label, 'No clasificado')
                st.success(f"Tu residuo es *{residuo_clasificado}*")

                st.subheader("Selecciona tu barrio")
                barrios_disponibles = df['barrio'].unique()
                barrio_nombre = st.selectbox("Selecciona un barrio", barrios_disponibles, label_visibility="collapsed")

                if barrio_nombre:
                    filtered_df = df[(df['barrio'].str.contains(barrio_nombre, case=False)) & (df['materiales'].str.contains(residuo_clasificado, case=False))]
                    if filtered_df.empty:
                        st.warning("No hay puntos de reciclaje para este tipo de residuo en este barrio")
                    else:
                        st.subheader("Mapa de sitios de reciclaje")
                        map_center = filtered_df['Coordinates'].apply(pd.Series).mean().tolist()
                        m = folium.Map(location=map_center, zoom_start=14)

                        bounds = [[row['Coordinates'][0], row['Coordinates'][1]] for _, row in filtered_df.iterrows()]
                        m.fit_bounds(bounds)

                        marker_cluster = MarkerCluster().add_to(m)
                        for _, row in filtered_df.iterrows():
                            folium.Marker(
                                row['Coordinates'],
                                popup=f"<b>Dirección:</b> {row['direccion']}<br><b>Materiales:</b> {row['materiales']}"
                            ).add_to(marker_cluster)
                        st_folium(m, width=1200, height=800)

with tab2:
    st.header("Información sobre reciclaje en la ciudad")
    st.write("""
        La Ciudad de Buenos Aires cuenta con múltiples iniciativas para fomentar el reciclaje y la correcta disposición de residuos. 
        Puedes encontrar más información sobre los distintos puntos verdes y las campañas de reciclaje visitando el [sitio oficial de reciclaje](https://buenosaires.gob.ar/inicio/tramites-y-servicios/5).

        Además, puedes suscribirte a nuestro newsletter para recibir noticias y actualizaciones sobre reciclaje y sustentabilidad.
    """)
    
    st.subheader("Suscríbete a nuestro newsletter")
    email = st.text_input("Ingresa tu correo electrónico")

    def is_valid_email(email):
        return re.match(r"[^@]+@[^@]+\.[^@]+", email)

    # Conectar a la base de datos SQLite
    conn = sqlite3.connect(os.path.join(script_dir, 'subscribers.db'))
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS subscribers (email TEXT PRIMARY KEY)''')
    conn.commit()

    if st.button("Suscribirse"):
        if email and is_valid_email(email):
            try:
                cursor.execute("INSERT INTO subscribers (email) VALUES (?)", (email,))
                conn.commit()
                st.success(f"¡Gracias por suscribirte, {email}!")
            except sqlite3.IntegrityError:
                st.warning("Este correo ya está suscrito.")
        else:
            st.error("Por favor, ingresa un correo electrónico válido.")
    
    st.subheader("Descubrí cómo se trabaja dentro de un centro de reciclaje")
    st.video("https://www.youtube.com/watch?v=pTw0_R6dUkg")

# Contraseña para acceder a la gestión de suscriptores
admin_password = "tu_contraseña_secreta"

with tab3:
    st.header("Gestión de Suscriptores")

    # Pedir la contraseña al usuario
    password = st.text_input("Introduce la contraseña para acceder", type="password")
    if st.button("Enviar"):
        if password == "neuraldios":
            # Conectar a la base de datos SQLite
            conn = sqlite3.connect(os.path.join(script_dir, 'subscribers.db'))
            cursor = conn.cursor()

            def get_all_subscribers():
                cursor.execute("SELECT * FROM subscribers")
                rows = cursor.fetchall()
                return rows
            
            subscribers = get_all_subscribers()
            
            st.subheader("Lista de Suscriptores")
            if subscribers:
                df_subscribers = pd.DataFrame(subscribers, columns=["Email"])
                st.dataframe(df_subscribers)

                csv = df_subscribers.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings
                href = f'<a href="data:file/csv;base64,{b64}" download="subscribers.csv">Descargar CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("No hay suscriptores registrados.")

            # Cerrar la conexión a la base de datos
            conn.close()
        else:
            st.error("Contraseña incorrecta. Por favor, inténtalo de nuevo.")

# Agregar script de JavaScript para controlar la cámara y la animación del logo
st.markdown("""
    <script>
    function toggleCamera() {
        var cameraInput = document.querySelector('.stCameraInput');
        if (cameraInput.style.display === "none" || cameraInput.style.display === "") {
            cameraInput.style.display = "block";
        } else {
            cameraInput.style.display = "none";
        }
    }
    </script>
    """, unsafe_allow_html=True)
