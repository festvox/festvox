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
#   convstats.pl : convert stats file to duration stats file         #
#                                                                    #
#                                    2003/05/09 by Heiga Zen         #
#  ----------------------------------------------------------------  #

$| = 1;

(@ARGV == 1) || (die "Usage : convstats StatsFile\n");

while(<>){
    @LINE = split(' ');
    printf("%4d %14s %4d %4d\n",$LINE[0],$LINE[1],$LINE[2],$LINE[2]);
}
