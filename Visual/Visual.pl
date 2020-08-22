#!/usr/bin/perl
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

if(@ARGV!=3){
 print "$0 [output of map2train] [output of Sai's script] [-n or -p]\n";
 print "-p : Show predicted data\n";
 print "-n : Show native data\n";
 exit;
}

($file,$file2,$opt)=@ARGV;


@B=firstfileComment("$file");
@C=firstfile("$file2");

@cd=();
$cnt=0;

@NB=();
$a=1;
foreach $l (@B){ #inside
 if($$l[0]=~/^#C:/){
  next if($$l[2]==-2);
  $NB[$$l[14]][$$l[15]][$$l[16]]=1;
  print "#IN $a [$$l[14]][$$l[15]][$$l[16]]\n";
  $a++;
 }
}

print "#Inside $a\n";

foreach $l (@B){
 if($$l[0]=~/^#C:/){
  #next if($$l[2]==-1); 
  $bx=$$l[10];
  $by=$$l[11];
  $bz=$$l[12];
  $step=$$l[8];
  $Nstep=$$l[6];

  $px=$$l[14];
  $py=$$l[15];
  $pz=$$l[16];
  $flag=0;
  for($x=-4;$x<5;$x+=4){
   next if($px+$x<0);
  for($y=-4;$y<5;$y+=4){
   next if($py+$y<0);
  for($z=-4;$z<5;$z+=4){
   next if($pz+$z<0);
   if($NB[$px+$x][$py+$y][$pz+$z]==1){
    $flag=1;
    next;
   }
  }}}
  next if($flag==0);
  #Center xyz
  $back=int(($Nstep-1)/2);
  $cx=$bx+$back*$step;
  $cy=$by+$back*$step;
  $cz=$bz+$back*$step;
  #print"$cx $cy $cz <- $bx $by $bz\n";
  $cd[$cnt][0]=$cx;
  $cd[$cnt][1]=$cy;
  $cd[$cnt][2]=$cz;
  $cnt++;
 }
}


$cnt2=0;

if($opt eq "-n"){
foreach $l (@{$C[0]}){
 #print"$l\n";
 $pre[$cnt2]=$l;
 $cnt2++;
}
}else{
$cnt2=0;
foreach $l (@{$C[1]}){
 #print"$cnt2 $l\n";
 $pre[$cnt2]=$l;
 $cnt2++;
}
}
#Show results in PDB format

print"#Cd cnt= $cnt\n"; #Coordinates from trimmap output.
print"#Data= $cnt2\n";

$natm=1;
for($i=0;$i<$cnt2;$i++){
 if($pre[$i]==0){
  printf("ATOM%7d  %3s %3s%2s%4d    ",$natm,"CA ","ALA"," A",$natm);
  printf("%8.3f%8.3f%8.3f%6.2f%6.2f\n",$cd[$i][0],$cd[$i][1],$cd[$i][2],1,1);
  $natm++;
 }
}
print "TER\n";
$natm=1;
for($i=0;$i<$cnt2;$i++){
 if($pre[$i]==1){
  printf("ATOM%7d  %3s %3s%2s%4d    ",$natm,"CA ","ALA"," B",$natm);
  printf("%8.3f%8.3f%8.3f%6.2f%6.2f\n",$cd[$i][0],$cd[$i][1],$cd[$i][2],1,1);
  $natm++;
 }
}
print "TER\n";
$natm=1;
for($i=0;$i<$cnt2;$i++){
 if($pre[$i]==2){
  printf("ATOM%7d  %3s %3s%2s%4d    ",$natm,"CA ","ALA"," C",$natm);
  printf("%8.3f%8.3f%8.3f%6.2f%6.2f\n",$cd[$i][0],$cd[$i][1],$cd[$i][2],1,1);
  $natm++;
 }
}

print "TER\n";
$natm=1;
for($i=0;$i<$cnt2;$i++){
 if($pre[$i]==3){
  printf("ATOM%7d  %3s %3s%2s%4d    ",$natm,"CA ","ALA"," D",$natm);
  printf("%8.3f%8.3f%8.3f%6.2f%6.2f\n",$cd[$i][0],$cd[$i][1],$cd[$i][2],1,1);
  $natm++;
 }
}





#@key = sort { $hash{$a} <=> $hash{$b} || $a <=> $b} keys %hash;


sub firstfile{
my $cnt=0;
my @A;
open(IN,$_[0]) or die;
while(<IN>){
  next if(/^[#\/\n]/);
 chomp;
 my $item;
 @{$item}=split(/[\s\t;]+/,$_);
 push @A, $item
}
close(IN);
return @A;
}

sub firstfileComment{
my $cnt=0;
my @A;
open(IN,$_[0]) or die;
while(<IN>){
  next unless(/^#/);
 chomp;
 my $item;
 @{$item}=split(/[\s\t]+/,$_);
 push @A, $item
}
close(IN);
return @A;
}


sub onetothree{
 %amin123 = ("W"=>"TRP","F"=>"PHE","Y"=>"TYR","L"=>"LEU","I"=>"ILE","V"=>"VAL","M"=>"MET","A"=>"ALA","G"=>"GLY","P"=>"PRO","C"=>"CYS","T"=>"THR","S"=>"SER","Q"=>"GLN","N"=>"ASN","E"=>"GLU","D"=>"ASP","H"=>"HIS","K"=>"LYS","R"=>"ARG");
 %amin321 = ("TRP"=>"W","PHE"=>"F","TYR"=>"Y","LEU"=>"L","ILE"=>"I","VAL"=>"V","MET"=>"M","ALA"=>"A","GLY"=>"G","PRO"=>"P","CYS"=>"C","THR"=>"T","SER"=>"S","GLN"=>"Q","ASN"=>"N","GLU"=>"E","ASP"=>"D","HIS"=>"H","LYS"=>"K","ARG"=>"R");
}
sub firstfile_line{
my $cnt=0;
my @A;
open(IN,$_[0]) or die;
while(<IN>){
  next if(/^#/);
 chomp;
 push @A, $_;
}
close(IN);
return @A;
}

sub readpdb{
 my $cnt=0;
 my @A;
 my ($x,$y,$z);
 my ($file)=@_;

 open(IN,$file) or die;
 while(<IN>){
  next unless(/^ATOM/);
  chomp;

  $x=substr($_,30,8);
  $y=substr($_,38,8);
  $z=substr($_,46,8);

  my $atm=substr($_,13,3);
  my $res=substr($_,17,3);
  my $rnum=substr($_,22,4);
  #my $m_tag=substr($_,17,9);
  my $m_tag=substr($_,13,13);

  my $item;
  @{$item}=($res,$atm,$x,$y,$z,$rnum,$m_tag);
  push @A, $item;
 }
 close(IN);
 return @A;
}

