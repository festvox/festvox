###########################################################################
##                                                                       ##
##                  Language Technologies Institute                      ##
##                     Carnegie Mellon University                        ##
##                      Copyright (c) 2003-2007                          ##
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
##  Make a voice for Nagoya Institutte of Technologies HTS, HMM-based    ##
##  generation synthesis system  (http://hts.ics.nitech.ac.jp)           ##
##                                                                       ##
##  Requires a number of extras installed before this build process will ##
##  work                                                                 ##
##                                                                       ##
##  Specifically Cambridge Univeristy's HTK, Nagoya's SPTK and other HTS ##
##  extensions                                                           ##
##                                                                       ##
############################################################################
TOP=../../
DIRNAME=src/hts_build
BUILD_DIRS = 
ALL_DIRS= data train $(BUILD_DIRS)
SCHEMEFILES = VOICE_hts.scm VOICE_2.0_hts.scm
SCRIPTS = build_hts do_hts
OTHERS = f0range

FILES = Makefile $(SCRIPTS) $(SCHEMEFILES) $(OTHERS)

ALL = 

include $(TOP)/config/common_make_rules


