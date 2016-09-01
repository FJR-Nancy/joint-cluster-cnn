#!/usr/local/bin/perl

@command1=('ls *ppm');
@fileListppm=`@command1`;
#print "@fileListpng";
for $flnm(@fileListppm){
    $flnmppm=$flnm;
    $flnmppm=~s/\n//;
    $flnmpng=$flnmppm;
    $flnmpng=~s/ppm/png/g;
    @command2=("convert", $flnmppm, $flnmpng);
    print "@command2\n"; 
    `@command2`;
}
