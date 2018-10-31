# CnnSV-typer
Calling genotype of deletions using deep learning
## Introduction
CnnSV-typer, an novel approach taking the gene sequence images as input and calling the variation genotype from next generation sequencing data through the deep learning network. CnnSV-typer, firstly searches the text records from the BAM files according to the candidate variation information from VCF files, and then, visualizes them as variation images, finally, classifies the genotype of the images. Gene sequence images are not only capable of making up for the disadvantages of interpreting BAM files, VCF and other documents, but also provide feature-rich input for subsequent deep learning networks. CnnSV-typer also implements techniques for working with noisy training data. The experimental results on low and high coverage of massive data indicate CnnSV-typer surpasses other existing methods with higher precision, wider range of detectable deletion length, and better performance on both simulated and real data. 
![WorkFlow](https://github.com/BRF123/Cnn-typer/blob/master/workflow.png)
## Requirements
  * python 3.6, numpy, Matplotlib
  * Cuda 8.0, Cudnn, pycuda
  * TensorFlow
  * Pysam
  * PIL

## Installation
### Tools
  bash Anaconda3-4.3.1-Linux-x86_64.sh <br/>
  
### Cuda & cudnn
   Installation tutorial can be downloaded from the official website
    
### TensorFlow
* pip install tensorflow-gpu

### pysam
* pip install pysam

## Usage
### Data
BAM file & VCF file <br/>
First provide the bam files and vcf files, then extract the breakpoints of heterozygous deletion and homozygous deletiond from VCF files respectively, and then the non deletion regions are extracted by yourself.<br/>
### Generation Images of Candidates
Run the following program in the custom path <br/> 
* python 1.breakpoints_png_2.py
* python 2.breakpoints_png_1.py
* python 3.breakpoints_png_0.py

### Compress Images Using CUDA
* python 4.compress_png_cuda.py

### Split Data Set into Training Set and Test Set
* python 1.split_train_test.py img_path all_path/all_txt train_path/train_txt test_path/test_txt reset

### Training CNN
* python 2.cuda_normalization.py image_label_list file_dir xdata_file xlabel_file train/test
* python 31.train_cnn.py xdata_file xlabel_file
* python X_train 
  X_train_right = pickle.load( open(sys.argv[2], 'rb'))
    train_label = pickle.load( open(sys.argv[2], 'rb'))
    noise = sys.argv[3]
    noise_label = make_noise(train_label,noise,200)
    list_lr=float(sys.argv[4])
    list_batch=int(sys.argv[5])
    list_epoch=int(sys.argv[6])
    X_test = pickle.load( open(sys.argv[7], 'rb'))
    #X_test_right = pickle.load( open(sys.argv[9], 'rb'))
    test_label = pickle.load( open(sys.argv[8], 'rb'))

### Using a trained network for calling genotype of deletions

### Extracting deletion information from test results
* python extract_breakpoint.py

### Generating VCF File
* python generate_final_vcf.py
