import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import cv2
import json
import subprocess
import random
import time
import tempfile

class MPI4:
    def __init__(self):
        self.download_folder = "downloaded_images"
        self.filtered_folder = "filtered_images"
        self.selected_filter = None
        self.filter_options = {
            "Class1": [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
            "Class2": ([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
            "Class3": ([[0, -1, 0], [0, 3, 0], [0, -3, 0], [0, 1, 0], [0, 0, 0]]),
            "Square 3x3": ([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]),
            "Edge 3x3": ([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]),
            "Square 5x5": ([
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],  
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]
                ]),
            "Edge 5x5": ([
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]),
            "Sobel Vertical": ([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
            "Sobel Horizontal": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "Laplace": ([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "Prewitt Vertical": ([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "Prewitt Horizontal": ([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        }

    def filter_image_stats(self, image):
        # Calcular estadísticas
        min_val, max_val, _, _ = cv2.minMaxLoc(image)  # Ignora las ubicaciones
        mean_val = image.mean()
        std_dev = image.std()
        return min_val, max_val, mean_val, std_dev

    def show_image_stats(self, image):
        if len(image.shape) == 2:  # Imagen en escala de grises
            min_val, max_val, mean_val, std_dev = self.filter_image_stats(image)
            st.text(f"Dim: {image.shape[0]}x{image.shape[1]}")
            st.text(f"Valor Mínimo: {min_val}")
            st.text(f"Valor Máximo: {max_val}")
            st.text(f"Valor Medio: {mean_val:.2f}")
            st.text(f"Des Estándar: {std_dev:.2f}")
        elif len(image.shape) == 3:  # Imagen en color
            st.text(f"Dimensiones: {image.shape[0]}x{image.shape[1]}")

    def get_kernel(self, filter_name):
        return self.filter_options.get(filter_name, None)

    def get_all_image_paths(self, base_folder):
        image_paths = []
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def clear_directory(self, directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            # Eliminar archivos
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Error al eliminar archivo {file_path}: {e}")

            # Eliminar subdirectorios
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    print(f"Error al eliminar directorio {dir_path}: {e}")

    def filter_image_sequentially(self, image_path, kernel):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if kernel is not None:
            # Asegurarse de que el kernel sea un array de NumPy con el tipo de dato correcto
            kernel = np.array(kernel, dtype=np.float32)
            filtered_image = cv2.filter2D(gray_image, -1, kernel)
        else:
            filtered_image = gray_image  # Si no hay filtro, muestra la imagen en escala de grises

        return filtered_image

    def apply_filter_sequentially(self, image_paths, kernel):
        start_time = time.time()  # Inicia el cronómetro para el filtrado secuencial
        for image_path in image_paths:
            self.filter_image_sequentially(image_path, kernel)
        end_time = time.time()  # Detiene el cronómetro
        return end_time - start_time

    def show_images(self, folder):
        all_images = self.get_all_image_paths(folder)
        if len(all_images) > 10:
            selected_images = random.sample(all_images, 10)
        else:
            selected_images = all_images

        for image_path in selected_images:
            cols = st.columns(4)  # 4 columnas

            original = cv2.imread(image_path, cv2.IMREAD_COLOR)
            grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            sequential_filtered = self.filter_image_sequentially(image_path, self.get_kernel(self.selected_filter))

            # Obtiene el path relativo de la imagen en la carpeta de origen
            rel_path = os.path.relpath(image_path, self.download_folder)

            # Construye el path de la imagen filtrada en la carpeta de imágenes filtradas
            filtered_image_path = os.path.join(self.filtered_folder, rel_path)
            filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR) if os.path.exists(filtered_image_path) else None

            with cols[0]:  # Columna para la imagen original
                st.image(original, caption="Original", use_column_width=True)
                self.show_image_stats(original)  # Mostrar estadísticas de la imagen original
            with cols[1]:  # Columna para la imagen en escala de grises
                st.image(grayscale, caption="Grayscale", use_column_width=True, channels='GRAY')
                self.show_image_stats(grayscale)  # Mostrar estadísticas de la imagen en escala de grises
            if filtered_image is not None:
                with cols[2]:  # Columna para la imagen filtrada desde 'filtered_images'
                    st.image(filtered_image, caption="Filtered (MPI)", use_column_width=True)
                    self.show_image_stats(sequential_filtered)  # Mostrar estadísticas de la imagen filtrada
            with cols[3]:  # Columna para la imagen filtrada de manera secuencial
                st.image(sequential_filtered, caption="Filtered (Secuencial)", use_column_width=True)
                self.show_image_stats(sequential_filtered)  # Mostrar estadísticas de la imagen filtrada


    def main(self):
        st.write("Aplicar Filtros con MPI")

        self.selected_filter = st.selectbox("Elegir filtro para aplicar", options=list(self.filter_options.keys()))
        num_processes = st.number_input("Número de procesos", min_value=1, max_value=16, value=4)

        if st.button("Aplicar Filtro Seleccionado"):
            self.clear_directory(self.filtered_folder)
            image_paths = self.get_all_image_paths(self.download_folder)
            kernel = self.get_kernel(self.selected_filter)

            start_time_mpi = time.time()

            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                json.dump({
                    "image_paths": image_paths,
                    "kernel": kernel,  # Ya es una lista, no se necesita .tolist()
                    "filtered_folder": self.filtered_folder,
                    "download_folder": self.download_folder,
                    "num_processes": num_processes
                }, temp_file)
                temp_file_path = temp_file.name

            # Llamar a mpi_filter.py con la ruta del archivo temporal
            subprocess.run(["mpiexec", "-n", str(num_processes), "python", "mpi_filter.py", temp_file_path])


            end_time_mpi = time.time()
            mpi_time = end_time_mpi - start_time_mpi

            sequential_time = self.apply_filter_sequentially(image_paths, kernel)

            self.show_images(self.download_folder)
            st.write(f"Tiempo de filtrado MPI: {mpi_time:.2f} segundos")
            st.write(f"Tiempo de filtrado secuencial: {sequential_time:.2f} segundos")

            if sequential_time > 0:
                acceleration = sequential_time / mpi_time
            else:
                acceleration = 0

            times = [sequential_time, mpi_time]
            labels = ["Secuencial", "MPI"]
            st.write(f"Aceleración: {acceleration:.2f}x")

            fig, ax = plt.subplots()
            ax.bar(labels, times, color=['blue', 'green'])
            ax.set_ylabel('Tiempo (s)')
            ax.set_title('Comparación de tiempos de filtrado')
            st.pyplot(fig)

if __name__ == "__main__":
    app = MPI4()
    app.main()
