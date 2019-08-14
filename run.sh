#!/bin/bash


chmod +x *
chmod -R 777 ../data
chmod -R 777 map2train_src/

#Inputs
map='../data/1733.mrc'
contour_level=2.5


#Vars
filename='../data/input_file.txt'
trimmap='../data/trimmap'
dataset='../data/dataset'
output='../results/outputP2_0'
visual_output='../results/visual.pdb'

#Code
cd ../code/map2train_src
make
cd -
../code/map2train_src/bin/map2train $map -c $contour_level >  $trimmap 
python dataset.py $trimmap $dataset
echo $dataset > $filename
echo "INFO : Running Emap2sec.py with arguments $filename"
python Emap2sec.py "$filename"
echo "INFO : Running Visual.pl"
./Visual.pl "$trimmap" "$output" -p > "$visual_output"
cp ../code/map2train_src/bin/map2train ../results
echo "INFO : Done"

