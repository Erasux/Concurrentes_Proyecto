# mpi_filter.py
from mpi4py import MPI
import cv2
import numpy as np
import json
import sys
import os

def apply_filter_to_image(image_path, kernel, input_folder, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered_image = cv2.filter2D(image, -1, kernel)
    print(f"Procesando imagen: {image_path}")
    # Determinar el subdirectorio relativo y el nombre del archivo
    relative_path = os.path.relpath(image_path, input_folder)
    output_path = os.path.join(output_folder, relative_path)

    # Crear el subdirectorio en 'filtered_images' si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar la imagen filtrada en el subdirectorio correcto
    cv2.imwrite(output_path, filtered_image)
    print(f"Imagen guardada: {output_path}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Cambios principales aquí
    if rank == 0:
        if len(sys.argv) != 2:
            print("Uso: python mpi_filter.py <temp_file>")
            sys.exit(1)

        temp_file = sys.argv[1]

        # Leer los datos del archivo temporal
        with open(temp_file, 'r') as file:
            data = json.load(file)
            image_paths = data['image_paths']
            kernel = np.array(data['kernel'])  # Asegúrate de convertir el kernel a numpy array
            output_folder = data['filtered_folder']
            download_folder = data['download_folder']  # Si es necesario
            num_processes = data['num_processes']

        # Dividir las rutas de las imágenes en chunks para MPI
        chunks = [image_paths[i::num_processes] for i in range(num_processes)]
    else:
        # Inicializa variables para los otros procesos
        chunks = None
        kernel = None
        output_folder = None
        download_folder = None 

    # Distribuir los datos a los diferentes procesos
    chunk = comm.scatter(chunks, root=0)
    kernel = comm.bcast(kernel, root=0)
    output_folder = comm.bcast(output_folder, root=0)
    download_folder = comm.bcast(download_folder, root=0) 

    # Aplicar el filtro a las imágenes asignadas a este proceso
    for image_path in chunk:
        apply_filter_to_image(image_path, kernel, download_folder, output_folder)

    # Esperar a que todos los procesos terminen
    comm.Barrier()

if __name__ == "__main__":
    main()