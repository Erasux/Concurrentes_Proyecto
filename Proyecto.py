import streamlit as st
from bing_image_downloader import downloader
import threading
import os
import shutil
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from multi import Multiprocessing
from MPI import MPI4
from codigoCpy import codigoCpy
from pycudaEx import PyCudaEx


class ImageDownloader:
    """Clase para manejar la descarga de imágenes utilizando múltiples hilos."""

    def __init__(self, download_folder="downloaded_images"):
        self.download_folder = download_folder

    def download_images(self, thread_id, keyword, num_images, max_total_images):
        modified_keyword = f"{keyword} {thread_id}"
        thread_folder = f"{self.download_folder}/thread_{thread_id}"
        os.makedirs(thread_folder, exist_ok=True)

        total_downloaded = sum([len(files) for r, d, files in os.walk(self.download_folder)])
        if total_downloaded >= max_total_images:
            return

        attempts = 0
        while attempts < 5 and total_downloaded < max_total_images:
            attempts += 1
            downloader.download(modified_keyword, limit=num_images, output_dir=thread_folder, adult_filter_off=True, force_replace=False, timeout=60)
            total_downloaded = sum([len(files) for r, d, files in os.walk(self.download_folder)])


    def clean_download_folder(self):
        """Limpia el directorio de descarga."""
        if os.path.exists(self.download_folder):
            shutil.rmtree(self.download_folder)
        os.makedirs(self.download_folder)

class StreamlitApp:
    def __init__(self):
        self.image_downloader = ImageDownloader()
        self.num_threads = 10
        self.images_per_thread = 1000
        self.max_images_to_display = 10 
        self.max_total_images = (self.images_per_thread*self.num_threads )  # Límite de imágenes para mostrar
        self.selected_filter = None

    def show_images(self, folder):
            all_images = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Verifica que sea una imagen
                        all_images.append(os.path.join(root, file))

            all_images = all_images[:self.max_images_to_display]  # Limitar a 10 imágenes

            num_columns = 3  # Para mostrar imagen original y dos filtros
            num_rows = len(all_images)

            filtered_images = self.apply_filter(all_images) if self.selected_filter else [(img, None, None) for img in all_images]

            for row in range(num_rows):
                cols = st.columns(num_columns)
                result = filtered_images[row]
                original = result[0]
                grayscale = cv2.cvtColor(cv2.imread(original, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
                filtered = result[1] if len(result) > 1 else None
                
                with cols[0]:  # Columna para la imagen original
                    st.image(original, caption="Original", use_column_width=True)
                with cols[1]:  # Columna para la imagen en escala de grises
                    st.image(grayscale, caption="Grayscale", use_column_width=True, channels='GRAY')
                if filtered is not None:
                    with cols[2]: # Columna para la imagen filtrada
                        st.image(filtered, caption=f"{self.selected_filter}", use_column_width=True)

    def download_images_threaded(self, keyword):
        thread_pool = ThreadPool(self.num_threads)
        args = [(i, keyword, self.images_per_thread, self.max_total_images) for i in range(self.num_threads)]
        thread_pool.starmap(self.image_downloader.download_images, args)
        thread_pool.close()
        thread_pool.join()

    def main(self):
        st.sidebar.header("Opciones de la Barra Lateral")
        selected_option = st.sidebar.selectbox("Selecciona una página", ["Página Principal","C","MPI4PY","Multiprocessing", "PyCUDA"])

        if selected_option == "Página Principal":
            self.vista_principal()
        if selected_option == "C":
            codigo_cpy_instance = codigoCpy()  # Crear una instancia de codigoCpy
            codigo_cpy_instance.main()
        elif selected_option == "Multiprocessing":
            multiprocessing_instance = Multiprocessing()  # Crear una instancia de Multiprocessing
            multiprocessing_instance.multiprocessig()
        if selected_option == "MPI4PY":
            codigo_mpi = MPI4()  # Crear una instancia de MPI4PY
            codigo_mpi.main()

    def vista_principal(self):
        st.title('Aplicativo filtros UCALDAS')

        keyword = st.text_input("¿Qué imágenes buscaremos hoy?")

        if st.button("Descargar imágenes"):
            self.image_downloader.clean_download_folder()
            self.download_images_threaded(keyword)
            st.success("Descarga completada")
            self.show_images(self.image_downloader.download_folder)

if __name__ == "__main__":
    app = StreamlitApp()
    app.main()
