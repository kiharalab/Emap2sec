#!/usr/bin/perl

# Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.
# This file is a part of Emap2sec package with -
# Reference: Maddhuri S,Terashi G, Kihara D. Protein Secondary Structure Detection in Intermediate Resolution Cryo-Electron Microscopy Maps Using Deep Learning. In submission (2018).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
if(@ARGV!=3){
 print "$0 [output of map2train] [output of Sai's script] [-n or -p] > tmp.pdb\n";
 print "-p : Show predicted data\n";
 print "-n : Show native data\n";
 print "Example\n";
 print "$0 train.out Sai.out -p > tmp.pdb \n";
 print "THEN; pymol -u ./tmp.pdb \n";
 exit;
}

($file,$file2,$opt)=@ARGV;


@B=firstfileComment("$file");
@C=firstfile("$file2");

@cd=();
$cnt=0;
foreach $l (@B){
 if($$l[0]=~/^#C:/){
  $bx=$$l[10];
  $by=$$l[11];
  $bz=$$l[12];
  $step=$$l[8];
  $Nstep=$$l[6];

  next if($$l[2]==-1);

  #Center xyz
  $back=int(($Nstep-1)/2);
  $cx=$bx+$back;
  $cy=$by+$back;
  $cz=$bz+$back;
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
 #print"$l\n";
 $pre[$cnt2]=$l;
 $cnt2++;
}
}

#Heat Map

print "load tmp.pdb,tmp\n";
print "show spheres\n";

$Nmap=0;
foreach $l (@C){
 if($$l[-1] =~/]$/){
  $$l[-3]=~s/\[//g;
  $$l[-1]=~s/\]//g;


  #print "$$l[-3] $$l[-2]  $$l[-1]\n";
  $map[$Nmap][0]=$$l[3];#H Red
  $map[$Nmap][1]=$$l[1];#C Green
  $map[$Nmap][2]=$$l[-2];#E Blue
  print "# $map[$Nmap][0],$map[$Nmap][1],$map[$Nmap][2]\n";
  printf("set_color p%d, [ %.2f, %.2f, %.2f, ]\n",$Nmap+1,$map[$Nmap][0],$map[$Nmap][1],$map[$Nmap][2]);
  printf("color p%d, (resi %d)\n",$Nmap+1,$Nmap+1);
  $Nmap++;
 }
}

#Show results in PDB format

print "MODEL 1\n";
$natm=1;
for($i=0;$i<$cnt2;$i++){
  printf("ATOM%7d  %3s %3s%2s%4d    ",$natm,"CA ","ALA"," A",$natm);
  printf("%8.3f%8.3f%8.3f%6.2f%6.2f\n",$cd[$i][0],$cd[$i][1],$cd[$i][2],1,$map[$i][0]);
  $natm++;
}
print "TER\n";
print "ENDMDL\n";

#@key = sort { $hash{$a} <=> $hash{$b} || $a <=> $b} keys %hash;


sub firstfile{
my $cnt=0;
my @A;
open(IN,$_[0]) or die;
while(<IN>){
  next if(/^#/);
 
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
  $_=~s/^[\s]+//g;
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

