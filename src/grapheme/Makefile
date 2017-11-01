###########################################################################
##                                                                       ##
##                   Language Technologies Institute                     ##
##                     Carnegie Mellon University                        ##
##                         Copyright (c) 2012                            ##
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
##  Makefile for various grapheme (and sampa) based scripts              ##
##                                                                       ##
##  Note the unicode_sampa_mapping.scm file is copyright UIUC under the  ##
##  apache license.  It provides a mapping for (many) unicode characters ##
##  to sampa phonemes.  The original code can be found here              ##
##  https://github.com/nltk/nltk_contrib/tree/master/nltk_contrib/scripttranscriber/Unitran
##  
##  The sampa phonemes and phonetic features have random origins and     ##
##  not as good as they should be                                        ##
##                                                                       ##
###########################################################################
TOP=../..
DIRNAME=src/grapheme
SCHEMEFILES = grapheme_unicode.scm unicode_sampa_mapping.scm
SCRIPTS = make_cg_grapheme build_cg_grapheme_voice latin_norm
OTHERS = sampa.top sampa.table sampa.bottom
MAINS = 

FILES = Makefile $(SCRIPTS) $(SCHEMEFILES) $(OTHERS) 

ALL = 

include $(TOP)/config/common_make_rules

CFLAGS += -I$(FLITEDIR)/include




