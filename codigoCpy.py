import streamlit as st
import subprocess
import os
from PIL import Image

class codigoCpy:
    @staticmethod
    def show_filtered_images(directory):
        for filename in os.listdir(directory):
            if filename.startswith("filtered_"):
                image_path = os.path.join(directory, filename)
                image = Image.open(image_path)
                st.image(image, caption=filename)

    @staticmethod
    def run_cpp_code(filter_name):
        subprocess.run(["./codigoC", filter_name])

    @staticmethod
    def main():
        filter_options = [
            "Ninguno",
            "Class1",
            "Class2",
            "Class3",
            "Square 3x3",
            "Edge 3x3",
            "Square 5x5",
            "Edge 5x5",
            "Sobel Vertical",
            "Sobel Horizontal",
            "Laplace",
            "Prewitt Vertical",
            "Prewitt Horizontal"
        ]

        st.write("Aplicar Filtros con C")
        selected_filter = st.selectbox("Elegir filtro para aplicar", options=filter_options)

        if st.button('Aplicar Filtro y Mostrar Imágenes'):
            codigoCpy.run_cpp_code(selected_filter)
            filtered_images_dir = './filtered_images'
            codigoCpy.show_filtered_images(filtered_images_dir)

if __name__ == "__main__":
    codigoCpy.main()