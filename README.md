# Emap2sec


<p align="center">
<img src="https://user-images.githubusercontent.com/50850224/192007403-f0e42715-e79d-4f37-9456-5f4e4b2f1882.jpeg" width="300">
</p>

Emap2sec is a computational tool using deep learning that can accurately identify protein secondary structures, alpha helices, beta sheets, others (coils/turns), in cryo-Electron Microscopy (EM) maps of medium to low resolution.

Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, & Daisuke Kihara, Protein secondary structure detection in intermediate-resolution cryo-EM maps using deep learning. Nature Methods. (2019)

## Online Platform
**All the functions in this github are available here. Related instructions are included in the website.**
### 1. Server: https://em.kiharalab.org/algorithm/emap2sec
### 2. Colab: https://bit.ly/emap2sec  or https://github.com/kiharalab/Emap2sec/blob/master/Emap2sec.ipynb
### 3. CodeOcean: https://codeocean.com/capsule/4439990

## Version Updates  

7/1/2021 - Unused dependencies have been removed to optimize data generation scripts

## Pre-required software

- Python 3.6 : https://www.python.org/downloads/
- tensorflow 1.15 : pip/conda install tensorflow==1.15
- scikit-learn 0.24.2 : pip/conda install scikit-learn==0.24.2  
- pandas 1.1.5 : pip/conda install pandas==1.1.5  
- numpy 1.16.4 : pip/conda install numpy==1.16.4
- gcc 4.8+
- EMAN2 : https://blake.bcm.edu/emanwiki/EMAN2/Install/
- Pymol (optional, for visualiztion) : https://pymol.org/2/  


## Input file generation  
Generate the input file called [your_map_id]_dataset from your map file by following these 3 steps.  

  ### 1) Trimmap generation  

<b>data_generate/map2train [sample_mrc] [options] > [output_trimmap_filename]</b>  
<b>INPUTS</b>:  
map2train expects sample_mrc to be a valid filename. Supported file formats are Situs, CCP4, and MRC2000. 
Input may be gzipped. Format is deduced from FILE's extension.  

<b>SAMPLE INPUTS</b>:  
  A sample mrc file can be found in data/ folder  
    
<b>OPTIONS:(Options marked with asterisk (*) are to be used only for benchmark purposes i.e., when you've the 
underlying crystal structure available)</b>:  
-c, --contour The level of isosurface to generate density values for.  
You can use a value of 0 for simulated maps and the author recommended contour level for experimental EM maps.
default=0.0  

-g, --gzip Set this option to force reading input as gzipped.  
You can use gzip to compress a very large EM map and input the compressed file by setting this option.  


-P* PDBFILE Input a PDB file to use C-Alpha (CA) atom position.  

-r* [float] This option assigns true secondary structures labels to the generated voxels with the closest CA 
atom that's within a sphere of radius r.  
These true labels can be compared to the secondary structures assigned by Emap2sec for benchmarking.  
default=3.0  

-sstep [integer] This option sets the stride size of the sliding cube used for input data generation.  
We recommend using a value of 4 that slides the cube by 4Å in each direction.   
Decreasing this value by 1 produces 8 times more data (increase by a factor of 2 in each direction)   
and thus slows the running time down by 8 times so please be mindful lowering this value.  
default=4  

-vw [integer] This option sets the dimensions of sliding cube used for input data generation.  
The size of the cube is calculated as 2*vw+1.  
We recommend using a value of 5 for this option that generates input cube of size 11*11*11.  
Please be mindful while increasing this option as it increases the portion of an EM map a single cube covers.  
Increasing this value also increases running time.    
default=5 (->11x11x11)  

-gnorm  Set this option to normalize density values of the sliding cube, used for input data generation,   
by global maximum density value. Set this option as -gnorm.  
default=true  

-lnorm  Set this option to normalize density values of the sliding cube, used for input data generation,   
by local maximum density value. Set this option as -lnorm.  
We recommend using -gnorm option.  
default=false  

-h, --help, -?, /? Displays the list of above options.  
  
<b>USAGE:</b>  
./data_generate/map2train data/1733.mrc -c 2.75 > data/trimmap

### 2) Input dataset file generation  
This program is used to generate input dataset file from the trimmap file generated in step 1.  
This program is a python script and it works with both python2 and python3.    

<b>python data_generate/dataset.py [sample_trimmap] [output_dataset_file]</b>  
<b>INPUTS:</b>  
Trimmap file.  

<b>OUTPUT:</b>  
Specify a name for input dataset file in place of [output_dataset_file].  

<b>USAGE:</b>  
python data_generate/dataset.py data/trimmap data/protein_dataset  
  
## Emap2sec SS identification (Phase1 and Phase2)  
  Run Emap2sec program for identification of secondary structures.  
Use emap2sec/Emap2sec.py when your input is an experimental EM map.
The models for Phase 1 and 2 can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12584153.svg)](https://doi.org/10.5281/zenodo.12584153).
This program requires trained models which can downloaded from here : http://dragon.bio.purdue.edu/Emap2sec_models     
Upon download please create a models/ directory in the current working directory and add emap2sec_models_exp1/ and 	emap2sec_models_exp2/ to that directory.  
This program is a python script and it works with both python2 and python3.  

<b>python emap2sec/Emap2sec.py [dataset_location_file]</b>  
<b>INPUT:</b>  
This program takes input as a file that contains location of input dataset.  
It also allows you to test multiple files at a time. File locations are to be "\n" delimited.  

<b>OUTPUT:</b>  
This program writes two output files, one for each phase, which contain output predictions along with 
the probability value for each prediction.  
Sample output files are provided in the github link in Downloads tab and are named as outputP1_0 for Phase1
and outputP2_0 for Phase2.  
Only the output of Phase2 is needed for the visualization step.  

<b>OPTIONS:</b>  
--prefix : File name prefix for output files [OPTIONAL]. Useful to include the output file path or to differentiate between several parallel executions using the same files as input without them overwriting other's results. This is useful because a process that has already created the result file and ties to read it might interfere with another process overwriting that file at the same time. Default: outputP1_<dataset_filename> and outputP2_<dataset_filename>."  

<b>USAGE:</b>  
First run : echo [location of protein_dataset file] > dataset_location_file to save the location of your 
protein dataset file in dataset_location_file. You can write the location for a different dataset input file for each line of the dataset_location_file and they will be processed in batch with the same Emap2sec.py's execution.
You can then run emap2sec/Emap2sec.py as shown below.  

echo data/protein_dataset > dataset_location_file  
or  
echo $'data/protein_dataset_1\ndata/protein_dataset_2\ndata/protein_dataset_3' > dataset_location_file  

and then  
 
python emap2sec/Emap2sec.py dataset_location_file  
or  
python emap2sec/Emap2sec.py dataset_location_file --prefix results/

## SS Visualization   
  Visualize the secondary structure assignments made in the previous step using the output file of Emap2sec SS identification program. Below program generates a PDB file containing secondary structure assignments. You can visualize these pdb structures using a molecular visualization tool such as pymol  
<b>visual/Visual.pl ../data/trimmap ../results/outputP2_0 -p > out_fin.pdb</b>  

<b>INPUT:</b>  
This program takes as inputs, the trimmap file generated in step 1 of input file generation
and output file of Emap2sec SS identification.  
You can visualize Phase1 or Phase2 output by using the appropriate output file.  

<b>OUTPUT:</b>  
This program outputs a pdb file that contains secondary structure assignments.  
A sample output file is provided in the github link in Downloads tab.   

<b>OPTIONS:</b>  
-p : Show predicted data (Predicted secondary structures)  
-n : Show native data (True secondary structures)[OPTIONAL - Use in case you've the crystal structure
information available]  

<b>USAGE:</b>  
Visual/Visual.pl data/trimmap outputP2_protein_dataset -p > out_fin.pdb  

Upon pymol installation, from pymol download directory you can run the below code from command line,  
<b>pymol out_fin.pdb</b>  
or
Open Pymol GUI and load visual.pdb.

Then run <b>run pymol_script.py</b> from the pymol command line. This gives you the final clean secondary structure visualization.
