import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import time

class ImageDownloader:

    def __init__(self, download_folder="downloaded_images"):
        self.download_folder = download_folder
    
class Multiprocessing:
    def __init__(self):
        self.image_downloader = ImageDownloader()
        self.max_images_to_display = 10  # Límite de imágenes para mostrar
        self.selected_filter = None
        self.filter_options = {
            "Class1": np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]),
            "Class2": np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
            "Class3": np.array([[0, -1, 0], [0, 3, 0], [0, -3, 0], [0, 1, 0], [0, 0, 0]]),
            "Square 3x3": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]),
            "Edge 3x3": np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]),
            "Square 5x5": np.array([
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],  
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]
                ]),
            "Edge 5x5": np.array([
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]),
            "Sobel Vertical": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
            "Sobel Horizontal": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "Laplace": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "Prewitt Vertical": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "Prewitt Horizontal": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        }

    def get_all_image_paths(self, base_folder):
        image_paths = []
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def filter_image_stats(self, image):
        # Calcular estadísticas
        min_val, max_val, _, _ = cv2.minMaxLoc(image)  # Ignora las ubicaciones
        mean_val = image.mean()
        std_dev = image.std()
        return min_val, max_val, mean_val, std_dev

    def show_image_stats(self, image):
        if len(image.shape) == 2:  # Imagen en escala de grises
            min_val, max_val, mean_val, std_dev = self.filter_image_stats(image)
            st.text(f"Dimensiones: {image.shape[0]}x{image.shape[1]}")
            st.text(f"Valor Mínimo: {min_val}")
            st.text(f"Valor Máximo: {max_val}")
            st.text(f"Valor Medio: {mean_val:.2f}")
            st.text(f"Desviación Estándar: {std_dev:.2f}")
        elif len(image.shape) == 3:  # Imagen en color
            st.text(f"Dimensiones: {image.shape[0]}x{image.shape[1]}")

    def filter_image(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = self.filter_options.get(self.selected_filter)
        
        if kernel is not None:
            filtered_image = cv2.filter2D(gray_image, -1, kernel)
            return image_path, filtered_image
        else:
            return image_path, None
        
    def filter_image_sequentially(self, image_path, kernel):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if kernel is not None:
            filtered_image = cv2.filter2D(gray_image, -1, kernel)
        else:
            filtered_image = gray_image  # Si no hay filtro, muestra la imagen en escala de grises
        return filtered_image

    def apply_filter_multiprocessing(self, image_paths):
        start_time = time.time()  # Inicia el cronómetro para el filtrado multiprocessing
        with Pool(processes=self.num_processes) as pool:
            pool.map(self.filter_image, image_paths)
        end_time = time.time()  # Detiene el cronómetro
        return end_time - start_time

    def apply_filter_sequentially(self, image_paths, kernel):
        start_time = time.time()  # Inicia el cronómetro para el filtrado secuencial
        for image_path in image_paths:
            self.filter_image_sequentially(image_path, kernel)
        end_time = time.time()  # Detiene el cronómetro
        return end_time - start_time

    def show_images(self, folder):
        all_images = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))

        all_images = all_images[:self.max_images_to_display]

        num_columns = 4 
        for image_path in all_images:
            cols = st.columns(num_columns)
            original = cv2.imread(image_path, cv2.IMREAD_COLOR)
            grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            filtered = self.filter_image(image_path)[1] if self.selected_filter else None
            sequential_filtered = self.filter_image_sequentially(image_path, self.filter_options.get(self.selected_filter))


            with cols[0]:  # Columna para la imagen original
                st.image(original, caption="Original", use_column_width=True)
                self.show_image_stats(original)  # Mostrar estadísticas de la imagen original

            with cols[1]:  # Columna para la imagen en escala de grises
                st.image(grayscale, caption="Gris", use_column_width=True, channels='GRAY')
                self.show_image_stats(grayscale)  # Mostrar estadísticas de la imagen en escala de grises

            if filtered is not None:
                with cols[2]:  # Columna para la imagen filtrada
                    st.image(filtered, caption=f"Filtrada (Multiprocessing) {self.selected_filter}", use_column_width=True)
                    self.show_image_stats(filtered)  # Mostrar estadísticas de la imagen filtrada
            if sequential_filtered is not None:
                with cols[3]:  # Columna para la imagen filtrada
                    st.image(sequential_filtered, caption=f"Filtrada (Sequential) {self.selected_filter}", use_column_width=True)
                    self.show_image_stats(filtered)  # Mostrar estadísticas de la imagen filtrada

    def multiprocessig(self):
        st.write("Aplicar Filtros con Multiprocessing")

        # Selección desplegable para filtros
        self.selected_filter = st.selectbox("Elegir filtro para aplicar", options=list(self.filter_options))
        self.num_processes = st.number_input("Número de procesos", min_value=1, max_value=16, value=4)
        if st.button("Aplicar Filtro Seleccionado"):
            all_images = self.get_all_image_paths(self.image_downloader.download_folder)
            kernel = self.filter_options.get(self.selected_filter)

            # Aplica filtros y mide el tiempo
            sequential_time = self.apply_filter_sequentially(all_images, kernel)
            multiprocessing_time = self.apply_filter_multiprocessing(all_images)

            # Muestra las imágenes y los tiempos
            self.show_images(self.image_downloader.download_folder)
            st.write(f"Tiempo de filtrado secuencial: {sequential_time:.2f} segundos")
            st.write(f"Tiempo de filtrado multiprocessing: {multiprocessing_time:.2f} segundos")

            # Cálculo de la aceleración
            if sequential_time > 0:  # Evitar división por cero
                acceleration = sequential_time / multiprocessing_time
            else:
                acceleration = 0

            # Desglose de tiempos
            times = [sequential_time, multiprocessing_time]
            labels = ["Secuencial", "Multiprocessing"]
            st.write(f"Aceleración: {acceleration:.2f}x")

            # Gráfico de barras para los tiempos
            fig, ax = plt.subplots()
            ax.bar(labels, times, color=['blue', 'green'])
            ax.set_ylabel('Tiempo (s)')
            ax.set_title('Comparación de tiempos de filtrado')
            st.pyplot(fig)

if __name__ == "__main__":
    app = Multiprocessing()  # Crear una instancia de la clase Multiprocessing
    app.multiprocessig()