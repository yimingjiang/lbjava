#!/usr/bin/perl

die "usage: ./shuffle.pl <file 1>\n" unless @ARGV == 1;
$file = $ARGV[0];

open IN, $file or die "Can't open $file for input: $!";
@data = <IN>;
close IN;

for ($i = 0; $i < @data; ++$i) {
  $j = $i + int(rand(scalar(@data) - $i));
  $temp = $data[$i];
  $data[$i] = $data[$j];
  $data[$j] = $temp;
}

print @data;

