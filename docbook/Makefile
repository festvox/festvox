###########################################################################
##                                                                       ##
##                   Carnegie Mellon University and                      ##
##                   Alan W Black and Kevin A. Lenzo                     ##
##                      Copyright (c) 1998-2000                          ##
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
###                                                                      ##
###  The festvox document itself as a Docbook document                   ##
###                                                                      ##
###########################################################################
TOP=..
DIRNAME=docbook
BUILD_DIRS = 
ALL_DIRS = figures
TEXT = bsv.sgml \
   eval.sgml concept.sgml corpus.sgml deploy.sgml diphone.sgml \
   estman.sgml festival.sgml festman.sgml future.sgml \
   install.sgml intro.sgml \
   grapheme.sgml jpdiphone.sgml ldom.sgml ldomfull.sgml \
   lexicons.sgml machlearn.sgml markup.sgml phonetable.sgml \
   prosody.sgml require.sgml resources.sgml scheme.sgml speech.sgml \
   text.sgml unitsel.sgml usukdiphone.sgml waveform.sgml  \
   nonenldom.sgml label.sgml sps.sgml
IMAGESPS = $(IMAGESPNG:.png=.eps)
IMAGES = $(IMAGESPNG) $(IMAGESPS) 
OTHERS = texi2docbook
FILES = Makefile $(TEXT) $(OTHERS) $(IMAGES) stuff.ed
HTMLDIR = $(TOP)/html
DOCBOOK = /usr/lib/sgml/stylesheets/nwalsh-modular/html/docbook.dsl

ALL = doc

include $(TOP)/config/common_make_rules

doc: bsv/book1.html bsv.pdf
	@ cp -p bsv.pdf bsv
	@ rm -rf $(TOP)/html
	@ cp -pr bsv $(TOP)/html

bsv/book1.html: $(TEXT)
	db2html bsv.sgml
	mkdir bsv/figures
	cp -p figures/*.gif bsv/figures/
	(cd bsv && cat ../stuff.ed | ed index.html)
bsv.pdf: $(TEXT)
	db2pdf bsv.sgml
refs:
	texirefs $(TEXT)

