# Emap2sec
Emap2sec is a computational tool using deep learning that can accurately identify protein secondary structures, alpha helices, beta sheets, others (coils/turns), in cryo-Electron Microscopy (EM) maps of medium to low resolution.

Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, & Daisuke Kihara, Protein secondary structure detection in intermediate-resolution cryo-EM maps using deep learning. Nature Methods. (2019)


## Pre-required software

Python 3 : https://www.python.org/downloads/  
tensorflow : pip/conda install tensorflow  
EMAN2 : https://blake.bcm.edu/emanwiki/EMAN2/Install/  
STRIDE : http://webclu.bio.wzw.tum.de/stride/install.html  
Pymol{for visualiztion} : https://pymol.org/2/  


## Input file generation  
Generate the input file called [your_map_id]_dataset from your map file by following these 3 steps.  

  ## 1) Trimmap generation  

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
We recommend using a value of 2 that slides the cube by 2Å in each direction.   
Decreasing this value to 1 produces 8 times more data (increase by a factor of 2 in each direction)   
and thus slows the running time down by 8 times so please be mindful lowering this value.  
default=2  

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
./map2train ../data/1733.mrc -c 2.75 > ../data/trimmap


## 2) [OPTIONAL] STRIDE File generation  
Use this step to generate STRIDE file in case you have a solved crystal structure for your map. This step is for verification purposes only.  
STRIDE is a secondary structure assignment program that takes a PDB file as input.  

<b>./stride -f[output_stride_filename] [sample_pdb]</b>   
<b>INPUT:</b>  
Specify the name of your pdb file in place of [sample_pdb].  
<b>OUTPUT:</b>    
Specify a name for output STRIDE file after -f option without space.  
<b>USAGE:</b>  
./stride -fprotein.stride protein.pdb  

## 3) Input dataset file generation  
This program is used to generate input dataset file from the trimmap file generated in step 1.  
This program is a python script and it works with both python2 and python3.  
STRIDE file is an optional input for this program. Provide it for benchmarking purposes only.  

<b>python data_generate/dataset.py [sample_trimmap] {sample_stride} [input_dataset_file] [ID]</b>  
<b>INPUTS:</b>  
Inputs to this script are trimmap, an optional STRIDE file, and ID is a unique identifier of a map such as 
SCOPe ID, EMID, etc.  
<b>OUTPUT:</b>    
Specify a name for input dataset file in place of [input_dataset_file].  
<b>USAGE:</b>  
python data_generate/dataset.py protein_trimmap protein_dataset protein_id  
python data_generate/dataset.py protein_trimmap protein.stride protein_dataset protein_id  
  
## Emap2sec SS identification (Phase1 and Phase2)  
  Run Emap2sec program for identification of secondary structures.  
Use emap2sec/Emap2sec.py when your input is an experimental EM map.  
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
<b>USAGE:</b>  
First run : echo [location of protein_dataset file] > dataset_location_file to save the location of your 
protein dataset file in dataset_location_file. You can then run emap2sec/Emap2sec.py as shown below.  
 
python emap2sec/Emap2sec.py dataset_location_file

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
visual/Visual.pl protein_trimmap outputP1_0 -p > out_fin1.pdb  
visual/Visual.pl protein_trimmap outputP2_0 -p > out_fin2.pdb  

Upon pymol installation, from pymol download directory you can run the below code from command line,  
<b>pymol out_fin2.pdb</b>  
or
Open Pymol GUI and load visual.pdb.

Then run <b>run pymol_script.py</b> from the pymol command line. This gives you the final clean secondary structure visualization.
