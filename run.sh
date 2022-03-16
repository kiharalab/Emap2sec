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

# Create results folder if does not exist
mkdir -p results

#Inputs
map='data/1733.mrc'
contour_level=2.5


#Vars
filename='data/input_file.txt'
trimmap='data/trimmap'
dataset='data/dataset'
output='results/outputP2_0'
visual_output='results/visual.pdb'
tmp_files_pattern='data/TMP_*'

#Code
cd map2train_src
make
cd -
map2train_src/bin/map2train $map -c $contour_level >  $trimmap 
python data_generate/dataset_wo_stride.py $trimmap $dataset
echo $dataset > $filename
echo "INFO : Running Emap2sec.py with arguments $filename"
python emap2sec/Emap2sec.py "$filename"
echo "INFO : Running Visual.pl"
Visual/Visual.pl "$trimmap" "$output" -p > "$visual_output"
cp map2train_src/bin/map2train results/
rm -rf $tmp_files_pattern
echo "INFO : Done"

