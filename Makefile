###########################################################################
##                                                                       ##
##                   Carnegie Mellon University and                      ##
##                   Alan W Black and Kevin A. Lenzo                     ##
##                      Copyright (c) 1998-2017                          ##
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
##  Authors: Alan W Black (awb@cs.cmu.edu)                               ##
##           Kevin A. Lenzo (lenzo@cs.cmu.edu)                           ##
##  Version: festvox-2.8 December 2017                                   ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  Documentation, tools and scripts to aid building of new synthetic    ##
##  voice for the Festival Speech Synthesis System.                      ##
##                                                                       ##
##  This project's home page is http://www.festvox.org                   ##
##                                                                       ##
##  This release corresponds to the Festival 2.5 release (which          ##
##  incorporates Edinburgh Speech Tools 2.5)                             ##
##                                                                       ##
###########################################################################
TOP=.
DIRNAME=.
BUILD_DIRS = src 
ALL_DIRS=config docbook $(BUILD_DIRS)
CONFIG=configure configure.in config.sub config.guess \
       missing install-sh mkinstalldirs
OTHERS = README.md ACKNOWLEDGEMENTS 
FILES = Makefile $(OTHERS) $(CONFIG)

ALL = $(BUILD_DIRS)

# Try and say if config hasn't been created
config_dummy := $(shell test -f config/config || ( echo '*** '; echo '*** Making default config file ***'; echo '*** '; ./configure; )  >&2)

include $(TOP)/config/common_make_rules

config/config: config/config.in config.status
	./config.status

configure: configure.in
	autoconf

release: 
	$(MAKE) dist
	cp -p $(PROJECT_PREFIX)-$(PROJECT_VERSION)-$(PROJECT_STATE).tar.gz html/

backup: time-stamp
	@ $(RM) -f $(TOP)/FileList
	@ $(MAKE) file-list
	@ echo .time-stamp >>FileList
	@ sed 's/^\.\///' <FileList | sed 's/^/festvox\//' >.file-list-all
	@ (cd ..; tar zcvf festvox/$(PROJECT_PREFIX)-$(PROJECT_VERSION)-$(PROJECT_STATE).tar.gz `cat festvox/.file-list-all`)
	@ $(RM) -f $(TOP)/.file-list-all
	@ ls -l $(PROJECT_PREFIX)-$(PROJECT_VERSION)-$(PROJECT_STATE).tar.gz

# dist doesn't include the festvox.org site html files or the course
dist: time-stamp
	@ $(RM) -f $(TOP)/FileList
	@ $(MAKE) file-list
	@ echo .time-stamp >>FileList
	@ sed 's/^\.\///' <FileList | grep -v "festvox.org" | grep -v "^course/" | sed 's/^/festvox\//' | grep -v ANNOUNCE >.file-list-all
	@ echo ANNOUNCE-$(PROJECT_VERSION) | sed 's/^/festvox\//' >>.file-list-all
	@ find html -type f -print | sed 's/^/festvox\//' >>.file-list-all
	@ (cd ..; tar zcvf festvox/$(PROJECT_PREFIX)-$(PROJECT_VERSION)-$(PROJECT_STATE).tar.gz `cat festvox/.file-list-all`)
	@ $(RM) -f $(TOP)/.file-list-all
	@ ls -l $(PROJECT_PREFIX)-$(PROJECT_VERSION)-$(PROJECT_STATE).tar.gz

time-stamp :
	@ echo $(PROJECT_NAME) >.time-stamp
	@ echo $(PROJECT_PREFIX) >>.time-stamp
	@ echo $(PROJECT_VERSION) >>.time-stamp
	@ echo $(PROJECT_DATE) >>.time-stamp
	@ echo $(PROJECT_STATE) >>.time-stamp
	@ echo $(LOGNAME) >>.time-stamp
	@ hostname >>.time-stamp
	@ date >>.time-stamp

