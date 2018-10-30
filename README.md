# CnnSV-typer
Calling genotype of deletions using deep learning
## Introduction
CnnSV-typer, an novel approach taking the gene sequence images as input and calling the variation genotype from next generation sequencing data through the deep learning network. CnnSV-typer, firstly searches the text records from the BAM files according to the candidate variation information from VCF files, and then, visualizes them as variation images, finally, classifies the genotype of the images. Gene sequence images are not only capable of making up for the disadvantages of interpreting BAM files, VCF and other documents, but also provide feature-rich input for subsequent deep learning networks. CnnSV-typer also implements techniques for working with noisy training data. The experimental results on low and high coverage of massive data indicate CnnSV-typer surpasses other existing methods with higher precision, wider range of detectable deletion length, and better performance on both simulated and real data. 
![WorkFlow](https://github.com/BRF123/Cnn-typer/blob/master/workflow.png)
## Requirements
  * python 3.6, Jupyter Notebook, numpy, scipy, pandas, Matplotlib
  * Cuda 8.0, Cudnn
  * TensorFlow
  * Digits
  * Pysam

## Installation
### Tools
  bash Anaconda3-4.3.1-Linux-x86_64.sh <br/>
### Jupyter Notebook
  * pip install jupyter
  * Configure <br/>
    jupyter notebook --generate-config <br/>
    Create a ciphertext password:
    from notebook.auth import passwd
  * Modify the default configuration file <br/>
    c.NotebookApp.ip='*' <br/>
	  c.NotebookApp.password = u'sha:...' <br/>
	  c.NotebookApp.open_browser = False  <br/>
	  c.NotebookApp.port =8888            <br/>
	  c.NotebookApp.notebook_dir = u’/home/...’ <br/>
  
### Cuda & cudnn
   Installation tutorial can be downloaded from the official website
    
### TensorFlow
* pip install tensorflow-gpu

### Digits
    cd ~
    git clone https://github.com/NVIDIA/DIGITS.git digits
    cd digits
    sudo apt-get install graphviz gunicorn
    for req in $(cat requirements.txt); do sudo pip install $req; done 
    pip install -r ~/digits/requirements.txt 
    ./digits-devserver

### pysam
* pip install pysam

## Usage
### Data
BAM file & VCF file <br/>
First provide the bam files and vcf files for program<br/>
### Generation Candidates
Run Generate_Deletion_Image.py and Generate_Non_Deletion_Image.py in the custom path <br/> 
* python Generate_Deletion_Image.py --del_length <br/>
* python Generate_Non_Deletion_Image.py --del_length <br/>

### Geerationg Images Path
Generate the path of all pictures for training the network
* python my_file_travel.py

### Using Digits training CNN
Send all the generated pictures to the network training
* Using the CNN architecture in CNN_Source.py 

### Using a trained network for calling deletion
Generating whole genome pictures
* python Whole_genome_Image.py

### Extracting deletion information from test results
* python extract_breakpoint.py

### Generating VCF File
* python generate_final_vcf.py
