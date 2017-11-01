###########################################################################
##                                                                       ##
##                   Language Technologies Institute                     ##
##                     Carnegie Mellon University                        ##
##                       Copyright (c) 2005-2008                         ##
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
###  Makefile for clustergen specific scripts                            ##
###                                                                      ##
###  A statistical corpus based synthesizer.                             ##
###  See Black, A. (2006), CLUSTERGEN: A Statistical Parametric          ##
###  Synthesizer using Trajectory Modeling", Interspeech 2006 - ICSLP,   ##
###  Pittsburgh, PA.                                                     ##
###                                                                      ##
###  Was originally started as a better integration of HTS into FestVox  ##
###  but superceded the base part of HTS (called cgp here) and grew to   ##
###  to include trajectory modeling.  This is also language independent  ##
###                                                                      ##
###########################################################################
TOP=../..
DIRNAME=src/clustergen
SCHEMEFILES = CLUSTERGEN_cg.scm CLUSTERGEN_statenames.scm \
              clustergen.scm clustergen_build.scm xsent.scm
SCRIPTS = setup_cg setup_cg_indic \
          make_dur_model_mcep do_clustergen cg_resynth \
          d2a.pl a2d.pl a2l.pl \
          cg_get_feats cg_get_f0_feats cg_get_track \
          cg_get_feats_all cg_get_feats_all_traj \
          cg_test track_diff track_diff_nof0 track_diff_f0 track_diff_mcd \
	  track_diff_n cg_mco cg_mlpg make_voicing \
          track_diff_mm track_diff_mm_z track_diff_mm_each track_diff_only \
          track_diff_lpc \
          do_move_label do_move_label_me \
          cg_get_mcd get_cd_dtw get_cd_dtw_mgc get_str_sptk \
          build_cg_voice find_wstop find_nstates freq_response_vector_gen \
          do_xsent find_dg wagon_rf wagon_rf_f0 rf_post_build build_rfs \
          make_dur_model_mcep_rf build_cg_rfs_voice reduce_senones
OTHERS = mcep.desc mceptraj.desc statedur.feats statedur.feats_flite HOWTO \
         mcep.desc.graph statedur.feats.graph statedur.feats_flite.graph \
	 SPTK-3.6.patch xsent.desc

MAINS = cg_lpc_resynth_main.c

CGA = do_cga 

FILES = Makefile $(SCRIPTS) $(SCHEMEFILES) $(OTHERS) $(CGA) $(MAINS)

ALL = cg_lpc_resynth

include $(TOP)/config/common_make_rules

CFLAGS += -I$(FLITEDIR)/include

cg_lpc_resynth: cg_lpc_resynth_main.o
	$(CC) -o cg_lpc_resynth cg_lpc_resynth_main.o -L$(FLITEDIR)/build/i386-linux-gnu/lib -lflite -lm




