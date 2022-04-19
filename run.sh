#!/bin/bash

# Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.
# This file is a part of Emap2sec package with -
# Reference:  Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, and Daisuke Kihara. Protein Secondary Structure Detection in Intermediate Resolution Cryo-Electron Microscopy Maps Using Deep Learning. Nature Methods (2019).
## License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License V3 for more details.
#
# You should have received a copy of the GNU v3.0 General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.


chmod +x *
chmod -R 777 data/
chmod -R 777 map2train_src/
chmod -R 777 Visual/

#Folders
data_folder="data/"
result_folder="results/"
models_folder="models/"

#Inputs
default_map="${data_folder}1733.mrc"
default_contour_level=2.5

# Getting mrc file and countour from arguments. If they are not provided, defaut values are taken
map="${1:-$default_map}"
contour_level="${2:-$default_contour_level}"

#Vars
input_filename="${data_folder}input_file.txt"
trimmap="${data_folder}trimmap"
dataset_basename="proteinDataset"
dataset="${data_folder}${dataset_basename}"
output_1="outputP1_${dataset_basename}"
output_2="outputP2_${dataset_basename}"
visual_output_1="${result_folder}visual_1.pdb"
visual_output_2="${result_folder}visual_2.pdb"
tmp_files_patern="${data_folder}TMP_*"

#Code
# Checking number of arguments
if [ "$#" -gt 2 ]; then
    echo "ERROR: Wrong number of parameters"
    echo "USAGE:"
    echo "	default (1733.mrc and contour_level 2.5):	./run.sh"
    echo "	custom .mrc:					./run.sh path_to_your_file.mrc"
    echo "	custom .mrc and contour level:			./run.sh path_to_your_file.mrc contour_level"
    exit
fi

# Exit execution if models are not present
if [ ! -d "$models_folder" ]; then
  echo "Folder \"${models_folder}\" not found in project."
  echo "Please, download trained models from https://kiharalab.org/Emap2sec_models/"
  exit
fi

mkdir -p $result_folder
cd map2train_src
make
cd -
map2train_src/bin/map2train $map -c $contour_level >  $trimmap 
python data_generate/dataset_wo_stride.py $trimmap $dataset
echo $dataset > $input_filename
echo "INFO : Running Emap2sec.py with arguments ${dataset}"
python emap2sec/Emap2sec.py $input_filename --prefix $result_folder
echo "INFO : Running Visual.pl"
Visual/Visual.pl $trimmap $result_folder$output_1 -p > $visual_output_1
Visual/Visual.pl $trimmap $result_folder$output_2 -p > $visual_output_2
echo "INFO : Cleaning up"
rm -rf $tmp_files_patern $input_filename $trimapp $dataset
echo "INFO : Done"
