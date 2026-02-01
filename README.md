# ⚡ GPU-Accelerated Image Convolution Using CUDA and Python

This project implements high-performance 2D image convolution using three backends:
1. **CPU (C implementation)**
2. **CUDA GPU executable**
3. **Python-callable CUDA shared library (`.so`)**

The goal is to compare performance across CPU, raw CUDA, and Python+CUDA, and demonstrate the speedups achieved through GPU acceleration.

## 🚀 Features
- Custom CUDA kernels for 2D convolution  
- Shared library (`libconv.so`) callable from Python using `ctypes`  
- CPU baseline implementation for comparison  
- Benchmarks on multiple images and filters (3×3, 5×5, 7×7)  
- End-to-end pipeline for loading images, running convolution, and saving outputs  

## 🖥️ Environment
This project was developed and executed entirely on **Google Colab** using the **NVIDIA T4 GPU** runtime.

Images were uploaded from the project’s `data/` folder directly into the Colab session using:
- Colab’s file upload tool  
- Or by mounting Google Drive  

## 📌 How to Run
1. Open the notebook in Google Colab  
2. Enable GPU:  
   **Runtime → Change runtime type → GPU (T4)**  
3. Upload images from your `data/` folder into the Colab environment  
4. Compile the CUDA code and build the shared library  
5. Run the Python wrapper to execute GPU convolution  

## 📤 Output
- Convolved images saved in the notebook environment  
- Timing results for CPU, CUDA, and Python+CUDA  
- Demonstrated speedups of **50×–170×** depending on image and filter size

