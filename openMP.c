#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <omp.h>
#include <stdlib.h>

IplImage* applyFilter(const char* imagePath, const CvMat* kernel) {
    IplImage* image = cvLoadImage(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image) {
        fprintf(stderr, "Error al abrir la imagen: %s\n", imagePath);
        return NULL;
    }

    IplImage* filteredImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
    cvFilter2D(image, filteredImage, kernel, cvPoint(-1, -1));

    cvReleaseImage(&image);
    return filteredImage;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <nombre_filtro>\n", argv[0]);
        return 1;
    }

    const char* filterName = argv[1];
    CvMat* kernel = NULL;

    // Definición de filtros (ejemplo con Class1)
    // Repetir para otros filtros...
    if (strcmp(filterName, "Class1") == 0) {
        float data[] = {0, 1, 0, 0, -1, 0, 0, 0, 0};
        kernel = cvCreateMat(3, 3, CV_32F);
        memcpy(kernel->data.fl, data, 9 * sizeof(float));
    }
    // ...

    if (kernel == NULL) {
        fprintf(stderr, "Filtro no reconocido. Usando filtro por defecto.\n");
        kernel = cvCreateMat(3, 3, CV_32F);
        cvSetIdentity(kernel, cvScalarAll(1.0));
    }

    const char* inputDir = "./downloaded_images";
    const char* outputDir = "./filtered_images";

    DIR* dir;
    struct dirent* ent;
    char** filenames = NULL;
    int fileCount = 0;

    if ((dir = opendir(inputDir)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                filenames = (char**) realloc(filenames, (fileCount + 1) * sizeof(char*));
                filenames[fileCount] = strdup(ent->d_name);
                fileCount++;
            }
        }
        closedir(dir);

        #pragma omp parallel for
        for (int i = 0; i < fileCount; i++) {
            char inputPath[256];
            char outputPath[256];
            snprintf(inputPath, sizeof(inputPath), "%s/%s", inputDir, filenames[i]);
            snprintf(outputPath, sizeof(outputPath), "%s/filtered_%s", outputDir, filenames[i]);

            IplImage* filteredImage = applyFilter(inputPath, kernel);
            if (filteredImage) {
                cvSaveImage(outputPath, filteredImage, NULL);
                cvReleaseImage(&filteredImage);
            }

            free(filenames[i]);
        }

        free(filenames);
    } else {
        perror("No se pudo abrir el directorio");
        return 1;
    }

    cvReleaseMat(&kernel);
    return 0;
}
