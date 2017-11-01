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
##                                                                       ##
##  Makefile for general scripts                                         ##
##                                                                       ##
###########################################################################
TOP=../../
DIRNAME=src/general
BUILD_DIRS = 
ALL_DIRS = $(BUILD_DIRS)
SRCS = phonealign_main.cc fix_pm_main.cc smooth_f0_main.cc \
       powcont_norm_main.cc cmn_main.cc
EMUTPL = emu_lab.tpl emu_sg.tpl emu_f0.tpl emu_pm.tpl emu_single_f0.tpl \
         emu_hier.tpl 
WSTPL = ws_festvox.conf
SCRIPTS = find_unknowns.sh make_dirs make_lpc make_pm \
          make_pm_epochs make_pm_wave make_f0 \
          make_pmlab_pm make_pm_pmlab \
          get_lars get_wavs \
          display_sg display_f0 find_powerfactors \
          prompt_them make_labs play_them phonealign_test \
          align_phones make_pm_fix simple_powernormalize traintest \
          find_powercontours contour_powernormalize \
          make_dist guess_voice_defs make_cmn do_build \
          make_samples reduce_prompts synthfile find_db_duration \
          prune_silence find_num_available_cpu ws prune_middle_silences \
	  add_noise
FILES = Makefile $(SCRIPTS) $(EMUTPL) $(WSTPL) $(SRCS) 

ALL = find_unknowns phonealign fix_pm smooth_f0 powcont_norm cmn

include $(TOP)/config/common_make_rules

# Try and see if EST's config is where its said to be
estconfig_dummy := $(shell test -f $(ESTDIR)/config/config || { echo '*** '; echo '*** Speech Tools config not found in '; echo '*** $(ESTDIR)/config/config.'; echo '*** Check EST installation and/or ESTDIR in festvox/config/config '; echo '*** '; }  >&2)

# GCC and systems settings got from EST's config
EST=$(ESTDIR)
include $(ESTDIR)/config/config
include $(ESTDIR)/config/rules/defaults.mak
include $(ESTDIR)/config/rules/compile_options.mak

ESTINC=-I$(ESTDIR)/include
ESTLIBS=-L$(ESTDIR)/lib -lestools -lestbase -leststring $(COMPILERLIBS) -lm $(OS_LIBS)

CXXFLAGS += -O2 $(ESTINC)

LOCAL_CLEAN = phonealign fix_pm smooth_f0 powcont_norm

phonealign: phonealign_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o phonealign phonealign_main.o $(ESTLIBS)
phonealign_main.o: phonealign_main.cc 
	$(strip $(CXX_COMMAND)) -c -o phonealign_main.o phonealign_main.cc
fix_pm: fix_pm_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o fix_pm fix_pm_main.o $(ESTLIBS)
fix_pm_main.o: fix_pm_main.cc 
	$(strip $(CXX_COMMAND)) -c -o fix_pm_main.o fix_pm_main.cc
fix_egg: fix_egg_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o fix_egg fix_egg_main.o $(ESTLIBS)
fix_egg_main.o: fix_egg_main.cc 
	$(strip $(CXX_COMMAND)) -c -o fix_egg_main.o fix_egg_main.cc
smooth_f0: smooth_f0_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o smooth_f0 smooth_f0_main.o $(ESTLIBS)
smooth_f0_main.o: smooth_f0_main.cc 
	$(strip $(CXX_COMMAND)) -c -o smooth_f0_main.o smooth_f0_main.cc
powcont_norm: powcont_norm_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o powcont_norm powcont_norm_main.o $(ESTLIBS)
powcont_norm_main.o: powcont_norm_main.cc 
	$(strip $(CXX_COMMAND)) -c -o powcont_norm_main.o powcont_norm_main.cc
cmn: cmn_main.o $(LIBDEPS)
	$(strip $(LINK_COMMAND)) -o cmn cmn_main.o $(ESTLIBS)
cmn_main.o: cmn_main.cc 
	$(strip $(CXX_COMMAND)) -c -o cmn_main.o cmn_main.cc

find_unknowns: find_unknowns.sh
	@echo making find_unknowns script
	@rm -f find_unknowns
	@echo "#!/bin/sh" >find_unknowns
	@echo "\"true\" ; exec "$(FESTDIR)/bin/festival --script '$$0 $$*' >>find_unknowns
	@cat find_unknowns.sh >> find_unknowns
	@chmod +x find_unknowns
