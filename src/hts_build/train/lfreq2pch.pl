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
#   lfreq2pch.pl : convert log frequency sequence to pitch           #
#                                                                    #
#                                    2003/05/09 by Heiga Zen         #
#  ----------------------------------------------------------------  #

# if you use other sampling frequency, please change this variable
$SF = 16000;

open(INPUT,"$ARGV[0]");
@STAT=stat(INPUT);
read(INPUT,$data,$STAT[7]);
close(INPUT);

$n = $STAT[7]/4;
@frq = unpack("f$n",$data);
for ($i=0; $i<$n; $i++) {
  if ($frq[$i] == 0.0) {
    $out[$i] = 0.0;
  } else {
    $out[$i] = $SF/exp($frq[$i]);
  }
}

$data = pack("f$n",@out);

print $data;
