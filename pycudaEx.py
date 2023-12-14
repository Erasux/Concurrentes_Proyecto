from bing_image_downloader import downloader
import streamlit as st
import os
import shutil
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class ImageDownloader:
    """Clase para manejar la descarga de imágenes utilizando múltiples hilos."""

    def __init__(self, download_folder="downloaded_images"):
        self.download_folder = download_folder
    
class PyCudaEx:
    def __init__(self):
        self.image_downloader = ImageDownloader()
        self.num_threads = 10
        self.images_per_thread = 3
        self.max_images_to_display = 10  # Límite de imágenes para mostrar
        self.selected_filter = None
        self.filter_options = {
            "Ninguno": None,
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

    def apply_filter_cuda(self, image_paths):
        # Definir el filtro
        filter_kernel = np.array([0, 1, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        kernel_width = 3  # Ancho del kernel

        # Compilar el kernel de CUDA
        mod = SourceModule("""
        __global__ void filter_kernel(float *output, float *input, int width, int height, float *kernel, int kernelWidth) {
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int row = blockIdx.y * blockDim.y + threadIdx.y;

            if (col < width && row < height) {
                float sum = 0.0;
                int start_col = col - (kernelWidth / 2);
                int start_row = row - (kernelWidth / 2);

                for (int i = 0; i < kernelWidth; i++) {
                    for (int j = 0; j < kernelWidth; j++) {
                        int cur_row = start_row + i;
                        int cur_col = start_col + j;
                        if (cur_row > -1 && cur_row < height && cur_col > -1 && cur_col < width) {
                            sum += input[cur_row * width + cur_col] * kernel[i * kernelWidth + j];
                        }
                    }
                }

                output[row * width + col] = sum;
            }
        }
        """)

        filter_kernel_func = mod.get_function("filter_kernel")

        results = []

        # Inicializar CUDA
        cuda.init()
        device = cuda.Device(0)  # Seleccionar el primer dispositivo CUDA

        for image_path in image_paths:
            # Crear un contexto CUDA
            ctx = device.make_context()

            try:
                # Cargar y preparar la imagen
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                image_gpu = cuda.mem_alloc(image.nbytes)
                cuda.memcpy_htod(image_gpu, image)

                # Preparar el filtro para CUDA
                filter_gpu = cuda.mem_alloc(filter_kernel.nbytes)
                cuda.memcpy_htod(filter_gpu, filter_kernel)

                # Preparar la salida
                output = np.empty_like(image)
                output_gpu = cuda.mem_alloc(output.nbytes)

                # Configurar dimensiones de la cuadrícula y el bloque
                block_size = (16, 16, 1)  # Ejemplo, ajustar según sea necesario
                grid_size = (int(np.ceil(image.shape[1] / block_size[0])), int(np.ceil(image.shape[0] / block_size[1])), 1)

                # Aplicar el filtro
                filter_kernel_func(output_gpu, image_gpu, np.int32(image.shape[1]), np.int32(image.shape[0]), filter_gpu, np.int32(kernel_width), block=block_size, grid=grid_size)

                # Recuperar la imagen filtrada
                cuda.memcpy_dtoh(output, output_gpu)

                results.append((image_path, output))

            finally:
                # Liberar memoria y destruir el contexto CUDA
                image_gpu.free()
                filter_gpu.free()
                output_gpu.free()
                ctx.pop()
                ctx.detach()

        return results


    def show_images(self, folder):
        all_images = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))

        all_images = all_images[:self.max_images_to_display]

        num_columns = 3
        for image_path in all_images:
            cols = st.columns(num_columns)
            original = cv2.imread(image_path, cv2.IMREAD_COLOR)
            grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            filtered = self.filter_image(image_path)[1] if self.selected_filter else None

            with cols[0]:  # Columna para la imagen original
                st.image(original, caption="Original", use_column_width=True)
                self.show_image_stats(original)  # Mostrar estadísticas de la imagen original

            with cols[1]:  # Columna para la imagen en escala de grises
                st.image(grayscale, caption="Grayscale", use_column_width=True, channels='GRAY')
                self.show_image_stats(grayscale)  # Mostrar estadísticas de la imagen en escala de grises

            if filtered is not None:
                with cols[2]:  # Columna para la imagen filtrada
                    st.image(filtered, caption=f"{self.selected_filter}", use_column_width=True)
                    self.show_image_stats(filtered)  # Mostrar estadísticas de la imagen filtrada

    def pycudaEx(self):
        st.write("Aplicar Filtros con PyCUDA")


        # Selección desplegable para filtros
        self.selected_filter = st.selectbox("Elegir filtro para aplicar", options=list(self.filter_options))

        if st.button("Aplicar Filtro Seleccionado"):
            self.show_images(self.image_downloader.download_folder)

if __name__ == "__main__":
    app = PyCudaEx()  # Crear una instancia de la clase PyCuda
    app.pycudaEx()