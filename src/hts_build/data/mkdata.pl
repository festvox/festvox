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
#    mkdata.pl : make training data for HMM-based speech synthesis   #
#                                                                    #
#                                    2003/05/09 by Heiga Zen         #
#  ----------------------------------------------------------------  #

$|=1;

use File::Basename;

# directory  ==============================
$basedir     = "CURRENTDIR";
$SPTKdir     = "SPTKDIR";
$rawdir      = "$basedir/raw";
$mcepdir     = "$basedir/mcep";
$f0dir       = "$basedir/f0";
$lf0dir      = "$basedir/log_f0";
$windir      = "$basedir/win";
$cmpdir      = "$basedir/cmp";
$scpdir      = "$basedir/scripts";

# speech analysis setting ===============
$sampfreq    = 16000;  # 16kHz sampling frequency 
$framelength = 0.025;  # 25ms window length
$frameshift  = 0.005;  # 5ms frame shift
$windowtype  = 0;      # window type ->   0: Blackman  1: Hamming  2: Hanning
$normtype    = 1;      # normalization of window -> 0: none  1: by power  2: by magnitude
$FFTLength   = 512;    # FFT length for Mel-cepstral analysis
$freqwarp    = 0.42;   # frequency warping for mel
$mceporder   = 24;     # order of Mel-cepstral analysis

# regression windows for calcurate dynamic and acceralation coefficients
$mcepwin[0]  = "$windir/mcep_dyn.win";
$mcepwin[1]  = "$windir/mcep_acc.win";
$lf0win[0]   = "$windir/lf0_dyn.win";
$lf0win[1]   = "$windir/lf0_acc.win";

# Main Program ==========================

# speech analysis  -------------------------

@RAW      = glob("$rawdir/*.raw");

$frame_length = $sampfreq * $framelength;
$frame_shift  = $sampfreq * $frameshift;
$mcepdim      = $mceporder + 1;
$lf0dim       = 1;
$nmcepwin     = @mcepwin + 1;
$nlf0win      = @lf0win  + 1;
$byte         = 4 * ( $mcepdim * $nmcepwin + $lf0dim * $nlf0win );
$period       = 1000 * $frameshift;

foreach $data (@RAW) {
    $base = basename($data,'.raw');
    print " make training data $base.cmp from $base.raw \n";

    # mel-cepstral analysis
    $line = "$SPTKdir/x2x +sf $data | "               
          . "$SPTKdir/frame +f -l $frame_length -p $frame_shift | "
          . "$SPTKdir/window -l $frame_length -L $FFTLength -w $windowtype -n $normtype | "
          . "$SPTKdir/mcep -a $freqwarp -m $mceporder -l $FFTLength > $mcepdir/$base.mcep ";
    system "$line \n"; 

    # convert f0 to log f0
    system "perl $scpdir/freq2lfreq.pl $f0dir/$base.f0 > $lf0dir/$base.lf0\n";
    
    # add delta and delta delta coefficients 
    system "perl $scpdir/delta.pl $mcepdim $mcepdir/${base}.mcep @mcepwin > $cmpdir/tmp.mcep";
    system "perl $scpdir/delta.pl $lf0dim   $lf0dir/${base}.lf0  @lf0win  > $cmpdir/tmp.lf0";

    # merge mel-cepstrum and log f0 
    system "$SPTKdir/merge +f -s 0 -l ".($nlf0win*$lf0dim)." -L ".($nmcepwin*$mcepdim)." $cmpdir/tmp.mcep < $cmpdir/tmp.lf0 | $SPTKdir/swab +f >  $cmpdir/tmp.cmp";
 
    # make HTK header
    @STAT = stat "$cmpdir/tmp.cmp";
    $size = $STAT[7]/$byte;
    system "echo $size ".($frameshift * 10000000)." | $SPTKdir/x2x +al | $SPTKdir/swab +f > $cmpdir/tmp.head";
    system "echo $byte 9 | $SPTKdir/x2x +as | $SPTKdir/swab +s >> $cmpdir/tmp.head";   # number 9 corresponds to user specified parameter in HTK

    # combine HTK header and sequence of feature vector 
    system "cat $cmpdir/tmp.head $cmpdir/tmp.cmp > $cmpdir/${base}.cmp";
} 

`rm -f $cmpdir/tmp.*`;
