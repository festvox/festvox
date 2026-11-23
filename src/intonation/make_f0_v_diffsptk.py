# MIT License
#####################################################-*-mode:shell-script-*-
##                                                                       ##
##                     Carnegie Mellon University                        ##
##                         Copyright (c) 2026                            ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  Author: Pallas13 (edsondasilvaguedes0@gmail) 2025                    ##
##                                                                       ##
###########################################################################
##                                                                       ##
## Uses DiffSPTK for extracting F0 and Voicing Information  		     ##
##                   (Basead in DiffSPTK Doc)                     		 ##
###########################################################################

import argparse

import diffsptk
import numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_period", required=True, type=int)
    parser.add_argument("--sample_rate", required=True, type=int)
    parser.add_argument("--f0_max", required=True, type=int)
    parser.add_argument("--f0_min", required=True, type=int)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()


fp = args.frame_period  # Frame period.
sample_rate = args.sample_rate  # Sample Rate
f_min = args.f0_min  # F0 Min
f_max = args.f0_max  # F0 Max
input = args.input  # Wave input
output = args.output  # f0 output

# Read waveform.
X, sr = diffsptk.read(input)

# Estimate f0 of x.
pitch = diffsptk.Pitch(
    frame_period=fp,
    sample_rate=sample_rate,
    f_min=f_min,
    f_max=f_max,
    voicing_threshold=0.4,
    out_format="f0",
)

f0 = pitch(X)

numpy.savetxt(output, f0, delimiter="\n")
