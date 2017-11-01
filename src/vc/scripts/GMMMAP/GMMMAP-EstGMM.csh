#! /bin/csh -f
#######################################################################
##                                                                   ##
##            Nagoya Institute of Technology, Aichi, Japan,          ##
##       Nara Institute of Science and Technology, Nara, Japan       ##
##                                and                                ##
##             Carnegie Mellon University, Pittsburgh, PA            ##
##                      Copyright (c) 2003-2004                      ##
##                        All Rights Reserved.                       ##
##                                                                   ##
##  Permission is hereby granted, free of charge, to use and         ##
##  distribute this software and its documentation without           ##
##  restriction, including without limitation the rights to use,     ##
##  copy, modify, merge, publish, distribute, sublicense, and/or     ##
##  sell copies of this work, and to permit persons to whom this     ##
##  work is furnished to do so, subject to the following conditions: ##
##                                                                   ##
##    1. The code must retain the above copyright notice, this list  ##
##       of conditions and the following disclaimer.                 ##
##    2. Any modifications must be clearly marked as such.           ##
##    3. Original authors' names are not deleted.                    ##
##                                                                   ##    
##  NAGOYA INSTITUTE OF TECHNOLOGY, NARA INSTITUTE OF SCIENCE AND    ##
##  TECHNOLOGY, CARNEGIE MELLON UNiVERSITY, AND THE CONTRIBUTORS TO  ##
##  THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,  ##
##  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, ##
##  IN NO EVENT SHALL NAGOYA INSTITUTE OF TECHNOLOGY, NARA           ##
##  INSTITUTE OF SCIENCE AND TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, ##
##  NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR      ##
##  CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM   ##
##  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,  ##
##  NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN        ##
##  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         ##
##                                                                   ##
#######################################################################
##                                                                   ##
##          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          ##
##          Date   :  June 2004                                      ##
##                                                                   ##
#######################################################################
##                                                                   ##
##  Iterative Training of Spectral Conversion Function               ##
##                                                                   ##
#######################################################################

##############################
set work_dir = $argv[1]	# work directory
set src_dir = $argv[2]	# src directory
set csh_dir = $argv[3]	# csh directory
set org = $argv[4]	# original label
set tar = $argv[5]	# target label
set olist = $argv[6]	# original list
set tlist = $argv[7]	# target list
set clsnum = $argv[8]	# the number of classes
set itnum = $argv[9]	# the number of iterations
##############################

set dtw_dir = $work_dir/dtw/$org"-"$tar
set cb_dir = $work_dir/cbook/$org"-"$tar
set init_dir = $work_dir/gmm_init/$org"-"$tar
set gmm_dir = $work_dir/gmm_jde/$org"-"$tar
mkdir -p $dtw_dir
mkdir -p $cb_dir
mkdir -p $init_dir
mkdir -p $gmm_dir

set res = $dtw_dir/$org"-"$tar"_frmcd.res"

foreach list ($olist $tlist)
	echo Extracting features from $list ...
	# Convert Static Mel-Cep to Static and Dynamic Mel-Cep
	csh $csh_dir/get_delta.csh \
		$work_dir \
		$src_dir \
		$list \
		mcep
	# Extract Speech Frames Using Power Information
	csh $csh_dir/get_extfrm.csh \
		$work_dir \
		$src_dir \
		$list \
		mcep
end
if (-r DUMMY-ERROR-LOG) then
	exit
endif

# Calculate Time Warping Function
echo Calculating TWF ...
csh $csh_dir/get_twf.csh \
	$work_dir/mcep \
	$dtw_dir \
	$src_dir \
	$olist \
	$tlist \
	mcep \
	$res
if (-r DUMMY-ERROR-LOG) then
	exit
endif

set itidx = 0
while ($itidx <= $itnum)
	if ($itidx == 0) then
		set flbl = $org"-"$tar
	else
		set flbl = it$itidx"_"$org"-"$tar
	endif
	set jnt = $dtw_dir/$flbl".mat"
	set cblbl = $cb_dir/$flbl"_sdmcep_cb"
	set initlbl = $init_dir/$flbl"_sdmcep_cb"
	set gmmlbl = $gmm_dir/$flbl"_sdmcep_cb"
	set otparam = $gmmlbl$clsnum.param

	# DTW and Constructing Joint Feature
	echo Constructing Joint Feature ...
	csh $csh_dir/get_dtw.csh \
		$work_dir/mcep \
		$dtw_dir \
		$src_dir \
		$olist \
		$tlist \
		$jnt
	if (-r DUMMY-ERROR-LOG) then
		exit
	endif

	if (!(-r $otparam)) then
		# VQ with LBG Algorithm
		csh $csh_dir/get_cbook.csh \
			$src_dir \
			$clsnum \
			$jnt \
			$cblbl
		if (-r DUMMY-ERROR-LOG) then
			exit
		endif

		# Calculate Initial Parameters of GMM
		echo VQ ...
		csh $csh_dir/get_gmm_init.csh \
			$src_dir \
			$clsnum \
			$jnt \
			$cblbl \
			$initlbl
		if (-r DUMMY-ERROR-LOG) then
			exit
		endif

		# Estimate GMM Parameters with EM Algorithm
		csh $csh_dir/get_gmm_jde.csh \
			$src_dir \
			$clsnum \
			$jnt \
			$cblbl \
			$initlbl \
			$gmmlbl
		if (-r DUMMY-ERROR-LOG) then
			exit
		endif
	endif

	# Remove Joint Feature
	rm -f $jnt

	# GMM-Based Conversion Using ML Parameter Generation
	echo Converting $org mel-cep to $tar mel-cep ...
	csh $csh_dir/get_gmmmap.csh \
		$work_dir \
		$src_dir \
		$clsnum \
		$otparam \
		$olist
	if (-r DUMMY-ERROR-LOG) then
		exit
	endif

	echo "### "$itidx"-th iteration has been finished. ###"
	@ itidx = $itidx + 1

	set res = $dtw_dir/it$itidx"_"$org"-"$tar"_frmcd.res"

	echo Extracting features from $olist ...
	foreach ext (mcep cvmcep)
		# Convert Static Mel-Cep to Static and Dynamic Mel-Cep
		csh $csh_dir/get_delta.csh \
			$work_dir \
			$src_dir \
			$olist \
			$ext
		# Extract Speech Frames Using Power Information
		csh $csh_dir/get_extfrm.csh \
			$work_dir \
			$src_dir \
			$olist \
			$ext
	end
	if (-r DUMMY-ERROR-LOG) then
		exit
	endif

	echo Extracting features from $tlist ...
	# Convert Static Mel-Cep to Static and Dynamic Mel-Cep
	csh $csh_dir/get_delta.csh \
		$work_dir \
		$src_dir \
		$tlist \
		mcep
	# Extract Speech Frames Using Power Information
	csh $csh_dir/get_extfrm.csh \
		$work_dir \
		$src_dir \
		$tlist \
		mcep
	if (-r DUMMY-ERROR-LOG) then
		exit
	endif

	# Calculate Time Warping Function
	echo Calculating TWF ...
	csh $csh_dir/get_twf.csh \
		$work_dir/mcep \
		$dtw_dir \
		$src_dir \
		$olist \
		$tlist \
		cvmcep \
		$res
	if (-r DUMMY-ERROR-LOG) then
		exit
	endif
end

# Remove Files
foreach list ($olist $tlist)
	foreach lbl (`cat $list`)
		if (-r $work_dir/mcep/$lbl.exmcep) then
			rm -f $work_dir/mcep/$lbl.exmcep
		endif
	end
end
foreach lbl (`cat $olist`)
	if (-r $dtw_dir/$lbl.twf) then
		rm -f $dtw_dir/$lbl.twf
	endif
end
