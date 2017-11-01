###########################################################################
##                                                                       ##
##                  Language Technologies Institute                      ##
##                    Carnegie Mellon University                         ##
##                         Copyright (c) 2002                            ##
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
##  Evaluation tools for speech synthesis                                ##
##                                                                       ##
##  This work was partially funded under a grant from the US National    ##
##  Science Foundation, "ITR/CIS Evaluation and Personalization of       ##
##  Synthetic Voices #0219687"                                           ##
##                                                                       ##
###########################################################################
TOP=../../
DIRNAME=src/eval
SCHEMEFILES = 
SCRIPTS = abtest mostest wave2html.scm evallog.cgi
OTHERS = 

FILES = Makefile $(SCRIPTS) $(SCHEMEFILES) $(OTHERS)

ALL = wave2html

include $(TOP)/config/common_make_rules

wave2html: wave2html.scm
	@echo making wave2html script
	@rm -f wave2html
	@echo "#!/bin/sh" >wave2html
	@echo "\"true\" ; exec "$(FESTDIR)/bin/festival --script '$$0 $$*' >>wave2html
	@cat wave2html.scm >> wave2html
	@chmod +x wave2html


