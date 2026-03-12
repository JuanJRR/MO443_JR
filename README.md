# MO443 - Introduction to Digital Image Processing

This repository contains the practical activities, laboratories and projects developed for the MO443 discipline, taught by Prof. Hélio Pedrini at the Instituto de Computação (IC-UNICAMP) during the first semester of 2026.

## Development Environment (Setup)

To ensure that the image processing algorithms (convolutions, filtering, morphology, etc.) run with adequate performance and compatibility with Google Colab, this repository uses the Docker configuration previously set up in:

```text
Ubuntu 24.04 with NVIDIA GPU support (CUDA 12.8)
  |-- Python 3.12
    |-- NumPy
    |-- SciPy
    |-- Matplotlib
    |-- OpenCV
    |-- Scikit-Image
    |-- Scikit-Learn
    |-- Pandas
```

## Project structure

Each practical assignment (TP) includes the algorithm, the implementation, the tests, and a concise report.

```text
/MO443
├── data/               # Test images (Lena, Baboon, etc.)
├── src/
│   ├── tp1_realce/     # Histograms and gray transformations
│   ├── tp2_filtros/    # Convolution, spatial filtering, and frequency
│   └── tpn_segmento/   # ... 
├── notebooks/          # Rapid prototyping of filters and transformations
├── reports/            # PDF reports of each activity 
└── pyproject.toml      # Dependency management (scikit-image, opencv, etc.)
```

## Installation & Getting Started

1. Prerequisites
    * Install Docker and Docker Compose.
    * Install NVIDIA Container Toolkit (for GPU acceleration).
    * Visual Studio Code with the Dev Containers extension.

2. Launching the Environment
    * Clone this repository to your local machine.
    * Create a .env file in the root directory.
    * Compile the container.

## References

* Pedrini, H. & Schwartz, W.R. Análise de Imagens Digitais: Princípios, Algoritmos e Aplicações.
* Gonzalez, R.C. & Woods, R.E. Digital Image Processing.

## Author

Juan Jose Rodriguez Rodriguez. PhD student, Institute of Computing (IC). State University of Campinas (UNICAMP).
