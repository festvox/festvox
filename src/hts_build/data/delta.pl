#!/usr/bin/perl
#  ----------------------------------------------------------------  #
#      The HMM-Based Speech Synthesis System (HTS): version 1.1      #
#                        HTS Working Group                           #
#                                                                    #
#                   Department of Computer Science                   #
#                   Nagoya Institute of Technology                   #
#                                and                                 #
#    Interdisciplinary Graduate School of Science and Engineering    #
#                   Tokyo Institute of Technology                    #
#                      Copyright (c) 2001-2003                       #
#                        All Rights Reserved.                        #
#                                                                    #
#  Permission is hereby granted, free of charge, to use and          #
#  distribute this software and its documentation without            #
#  restriction, including without limitation the rights to use,      #
#  copy, modify, merge, publish, distribute, sublicense, and/or      #
#  sell copies of this work, and to permit persons to whom this      #
#  work is furnished to do so, subject to the following conditions:  #
#                                                                    #
#    1. The code must retain the above copyright notice, this list   #
#       of conditions and the following disclaimer.                  #
#                                                                    #
#    2. Any modifications must be clearly marked as such.            #
#                                                                    #     
#  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,   #
#  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM     #
#  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL        #
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT    #
#  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF         #
#  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE     #
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY         #
#  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,   #
#  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS    #
#  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR           #
#  PERFORMANCE OF THIS SOFTWARE.                                     #
#                                                                    #
#  ----------------------------------------------------------------  # 
#      delta.pl : load regression window and add delta coefficients  #
#                                                                    #
#                                    2003/05/09 by Heiga Zen         #
#  ----------------------------------------------------------------  #

if (@ARGV<3) {
  print "delta.pl dimensionality infile winfile1 winfile2 ... \n";
  exit(0);
}

# dimensionality of input vector

$dim = $ARGV[0];


# open infile as a sequence of static coefficients 

open(INPUT,"$ARGV[1]");
@STAT=stat(INPUT);
read(INPUT,$data,$STAT[7]);
close(INPUT);

$nwin = @ARGV-1;

$n = $STAT[7]/4;  # number of data
$T = $n/$dim;     # length of input vector 


# static coefficients 
@static = unpack("f$n",$data);   # unpack static coefficients

for ($t=0; $t<$T; $t++) {
  for ($j=0; $j<$dim; $j++) {
     $delta[$t*$nwin*$dim+$j] = $static[$t*$dim+$j];  # store static coefficients
  }
}


# dynamic coefficients

$nwin = @ARGV-1;   # the number of delta window ( include static window )

for($i=2;$i<=$nwin;$i++) {
  # load $i-th dynamic coefficients window 
  open(INPUT,"$ARGV[$i]");
  @STAT=stat(INPUT);
  read(INPUT,$data,$STAT[7]);
  close(INPUT);

  $w = $STAT[7]/4;            
  @win = unpack("f$w",$data);
  $size = @win;   # size of this window

  if ($size % 2 != 1) {             
    die "Size of window must be 2*n + 1 and float"; 
  }

  $nlr = ($size-1)/2; 

  # calcurate $i-th dynamic coefficients

  for ($t=0; $t<$T; $t++) {
    for ($j=0; $j<$dim; $j++) {
      # check voiced/unvoiced boundary
      $voiced = 1;
      for ($k=-$nlr; $k<=$nlr; $k++) {      
        if ($static[($t+$k)*$dim+$j] == -1.0E10) {
          $voiced = 0;
        }
      }
      if ($voiced) {
        $delta[$t*$nwin*$dim+$dim*($i-1)+$j] = 0.0;
        for ($k=-$nlr; $k<=$nlr; $k++) {
          $delta[$t*$nwin*$dim+$dim*($i-1)+$j] += $win[$k+$nlr] * $static[($t+$k)*$dim+$j];
        }
      }
      else {
        $delta[$t*$nwin*$dim+$dim*($i-1)+$j] = -1.0E10;
      }
    }
  }
}

$n = $n*$nwin;

$data = pack("f$n",@delta);

print $data;
