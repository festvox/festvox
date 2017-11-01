###########################################################################
##                                                                       ##
##                                                                       ##
##              Carnegie Mellon University, Pittsburgh, PA               ##
##                      Copyright (c) 2007-2010                          ##
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
##          Author :  S P Kishore (skishore@cs.cmu.edu)                  ##
##          Date   :  February 2007                                      ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  C++ part of the splitter                                             ##
##                                                                       ##
###########################################################################
TOP=../../../
DIRNAME=src/interslice/src
BUILD_DIRS = 
ALL_DIRS = $(BUILD_DIRS)
SRCS = edec_v3.cc wavchop.cc
H = header.h hmmstate.h hmmword.h my_header.h
FILES = Makefile $(SRCS) $(H)

ISLICEBINDIR=../bin

ALL = $(ISLICEBINDIR)/edec.exe $(ISLICEBINDIR)/wavchop.exe 

include $(TOP)/config/common_make_rules

# Try and see if EST's config is where its said to be
estconfig_dummy := $(shell test -f $(ESTDIR)/config/config || { echo '*** '; echo '*** Speech Tools config not found in '; echo '*** $(ESTDIR)/config/config.'; echo '*** Check EST installation and/or ESTDIR in festvox/config/config '; echo '*** '; }  >&2)

# GCC and systems settings got from EST's config
EST=$(ESTDIR)
include $(ESTDIR)/config/config
include $(ESTDIR)/config/rules/defaults.mak
include $(ESTDIR)/config/rules/compile_options.mak

ESTINC=
ESTLIBS=

CXXFLAGS += -O2 $(ESTINC)
CXX=g++

LOCAL_CLEAN = $(ALL)

$(ISLICEBINDIR)/edec.exe: edec_v3.o $(H)
	if [ ! -d $(ISLICEBINDIR) ] ; \
        then \
           mkdir $(ISLICEBINDIR) ;\
	fi
	$(strip $(LINK_COMMAND)) -o $@ edec_v3.o

$(ISLICEBINDIR)/wavchop.exe: wavchop.o $(H)
	if [ ! -d $(ISLICEBINDIR) ] ; \
        then \
           mkdir $(ISLICEBINDIR) ;\
	fi
	$(strip $(LINK_COMMAND)) -o $@ wavchop.o

