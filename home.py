import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import base64
import streamlit.components.v1 as components

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# Suprimir advertencias ValueWarning
warnings.simplefilter("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Configuración de Streamlit
st.set_page_config(page_title="Crypto LSTM Modelo Predicción", page_icon="img/icono-page.png", layout="wide")

theme_plotly = None

#"""" codigo de particulas que se agregan en le background""""
particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    background-color: #191970;    
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#fffc33"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#fffc33",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""
globe_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Globe Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      body {
        overflow: hidden;
        height: 100%;
        margin: 0;
        background-color: #1817ed; /* Fondo azul */
      }
      #canvas-globe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-globe"></div>       

    <!-- Scripts de Three.js y Vanta.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.globe.min.js"></script>

    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.GLOBE({
          el: "#canvas-globe", // El elemento donde se renderiza la animación
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xd1ff3f, // Color verde amarillento
          backgroundColor: 0x1817ed // Fondo azul
        });
      });
    </script>
  </body>
</html>
"""
waves_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Waves Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      html, body {
        height: 100%;
        margin: 0;
        overflow: hidden;
      }
      #canvas-dots {
        position: absolute;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-waves"></div>       
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.waves.min.js"></script>
    
    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.WAVES({
          el: "#canvas-waves", // Especificar el contenedor donde debe renderizarse
           mouseControls: true,
           touchControls: true,
           gyroControls: false,
           minHeight: 200.00,
           minWidth: 200.00,
           scale: 1.00,
           scaleMobile: 1.00,
           color: 0x15159b
        });
      });
    </script>
  </body>
</html>
"""

#""" imagen de background"""
def add_local_background_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stApp{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )
add_local_background_image("img/fondo.jpg")

#""" imagen de sidebar"""
def add_local_sidebar_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stSidebar{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )

add_local_sidebar_image("img/fondo1.jpg")

# Agregar imágenes
# ---- animación de inicio de pagina----
with st.container():
    #st.write("---")
    left, right = st.columns(2, gap='small', vertical_alignment="center")
    with left:
        components.html(waves_js, height=150,scrolling=False)
    with right:
       components.html(particles_js, height=150,scrolling=False) 
    #st.write("---")    
#-------------- animacion con css de los botones  ------------------------
with open('asset/styles.css') as f:
        css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Título principal
st.button("Modelo de pronóstico LSTM", key="pulse", use_container_width=True)
st.write('###')
st.warning("este modelo hace un pedido de cotizaciones a Yahoo Finance, lo cual puede fallar la respuesta de los datos, abriendo la posiblidad de dar un mensaje de error. Si se produce este evento por favor vuelta a cargar la página o refreque la misma. Otro dato no menor es que este modelo trabajo con periodo de analisis de 60 dias y el variar los meses historicos y dias a predecir puede modificar sensiblente la respueta de precios proyectado o de predicción")

with st.sidebar:
    components.html(globe_js, height=150, scrolling=False)
    st.button("Parámetros del modelo", key="topulse", use_container_width=True)
    crypto_symbol = st.sidebar.text_input("Simbolo Cryptomoneda", "BTC-USD", help="debe ingresar el dato en Mayusculas").upper()
    crypto_symbol = crypto_symbol.upper()
    periodo = st.sidebar.number_input("Meses historico para calcular", min_value=1, max_value=12, value=6, step=1, help="valor debe oscilar entre 1/12")
    prediction_ahead = st.sidebar.number_input("Días a Predecir Precio", min_value=1, max_value=30, value=15, step=1, help="valor debe oscilar entre 1/30")
    st.write("###")

if st.sidebar.button("Predecir", key="predecir", use_container_width=True):
    
    with st.status("Generando los datos...", expanded=True) as status:
        st.write("Buscando datos...")
        time.sleep(2)
        st.write("Deplegando modelo LSTM.")
        time.sleep(1)
        st.write("Generando los Calculos...")
        time.sleep(1)
        status.update(
            label="Modelo LSTM completado!", state="complete", expanded=True
    )
        
    # Paso 1: Obtener datos de criptomonedas para el año    
    df_data = yf.download(crypto_symbol, period=f'{periodo}mo', interval='1d')
    # Eliminar cualquier valor nulo antes de realizar el escalado
    df_data = df_data[['Close']].dropna() 
    # Escalado de los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_data)
    # Parámetros de la red LSTM   
    train_size = int(len(scaled_data) * 0.8)    
    train_data = scaled_data[:train_size]
    #test_data = scaled_data[train_size:]
    test_size = int(len(scaled_data) * 0.2) 
    test_data = scaled_data[:test_size]
    
    # Función para crear los datasets
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    # Dividir los datos en conjunto de entrenamiento y prueba
    time_step = 60
    X_train, y_train = create_dataset(scaled_data[:train_size], time_step)
    X_test, y_test = create_dataset(scaled_data[train_size-time_step:], time_step) 
    # Reshape para el modelo LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Crear el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=5, verbose=0)

    # Predicción sobre los datos de prueba
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)   
 
    # transformo invirtiendo predicciones en valores actuales
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train.reshape(-1,1))
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    
    # Pronóstico para los días proyectados    
    last_60_days = scaled_data[-time_step:]
    future_input = last_60_days.reshape(1, time_step, 1)
    future_forecast = []
    for _ in range(prediction_ahead):  # Predicción para los próximos días
        next_pred = model.predict(future_input)[0, 0]
        future_forecast.append(next_pred)           
        next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0) 
        future_input = next_input.reshape(1, time_step, 1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1))
    # ultimo cierre y ultimo precio pedicho
    latest_close_price = float(df_data['Close'].iloc[-1])
    last_predicted_price = float(future_forecast[-1])
   
    col1, col2= st.columns(2, border=True, vertical_alignment="center")        
    st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
    with col1:                       
            st.subheader(f"""Precio de Cierre de : {crypto_symbol}""")
            st.button(f" -- U$D {latest_close_price:,.2f}  --", key="inpulse") 
    with col2:
            st.subheader(f"""Precio proyectado a {prediction_ahead} Dia/s """)
            st.button(f" -- U$D  {last_predicted_price:,.2f}  --", key="toinpulse")     
    st.markdown("</div>", unsafe_allow_html=True) 

   
    # Graficar los resultados
    plt.figure(figsize=(14, 5))
    plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
    plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')

    # Datos de entrenamiento/prueba y predicciones
    train_range = df_data.index[time_step:train_size]
    test_range = df_data.index[train_size:train_size + len(test_predictions)]
    plt.plot(train_range, train_predictions[:len(train_range)], label='Train Predictions', color='green')
    plt.plot(test_range, test_predictions[:len(test_range)], label='Test Predictions', color='orange')
      
    future_index = pd.date_range(start=df_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]    
    plt.plot(future_index, future_forecast, label=f'{prediction_ahead}-Day Forecast', color='red')

    plt.title(f'{crypto_symbol} Predicciones del modelo LSTM')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()

    # Mostrar el gráfico
    st.subheader(f"""Predicción del Modelo LSTM para : {crypto_symbol} """)
    st.pyplot(plt)

# --------------- footer -----------------------------
st.write("---")
with st.container():
  #st.write("---")
  st.write("&copy; - derechos reservados -  2024 -  Walter Gómez - FullStack Developer - Data Science - Business Intelligence")
  #st.write("##")
  left, right = st.columns(2, gap='medium', vertical_alignment="bottom")
  with left:
    #st.write('##')
    st.link_button("Mi LinkedIn", "https://www.linkedin.com/in/walter-gomez-fullstack-developer-datascience-businessintelligence-finanzas-python/",use_container_width=True)
  with right: 
     #st.write('##') 
    st.link_button("Mi Porfolio", "https://walter-portfolio-animado.netlify.app/", use_container_width=True)
      