###########################################################################
##                                                                       ##
##             Carnegie Mellon University, Pittsburgh, PA                ##
##                     Copyright (c) 2007-2010                           ##
##                       All Rights Reserved.                            ##
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
##          Author :  S P Kishore (skishore@cs.cmu.edu)                  ##
##          Date   :  August 2010                                        ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  Interslice, segment long audio streams                               ##
##  Version 2 (2010)                                                     ##
##  As described in "Automatic Building of Synthetic Voices from         ##
##  Audio Books" Kishore Prahallad, PhD Thesis, LTI, CMU, August 2010    ##
##                                                                       ##
##  English is well supported                                            ##
##                                                                       ##
##  For Non-English (experimental)                                       ##
##                                                                       ##
##  This has been used for other languages, but you need an acoustic     ##
##  model for that language (trained with ehmm)                          ##
##                                                                       ##
##  Call setup, then edit script/phseq.scm to call the right voice       ##
##  (language) so it'll generate split text, and generate the right      ##
##  phones                                                               ##
##  copy in ehmm/mod/model101.txt and etc/ph_list.int into etc/ from a   ##
##  voice trained in the target language                                 ##
##                                                                       ##
###########################################################################
TOP=../../
DIRNAME=src/interslice
BUILD_DIRS = src scripts
ALL_DIRS= $(BUILD_DIRS) model ml_model
SCHEMEFILES = 

FILES = Makefile

ALL = 

include $(TOP)/config/common_make_rules


