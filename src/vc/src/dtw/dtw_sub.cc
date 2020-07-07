/*********************************************************************/
/*                                                                   */
/*            Nagoya Institute of Technology, Aichi, Japan,          */
/*       Nara Institute of Science and Technology, Nara, Japan       */
/*                                and                                */
/*             Carnegie Mellon University, Pittsburgh, PA            */
/*                      Copyright (c) 2003-2004                      */
/*                        All Rights Reserved.                       */
/*                                                                   */
/*  Permission is hereby granted, free of charge, to use and         */
/*  distribute this software and its documentation without           */
/*  restriction, including without limitation the rights to use,     */
/*  copy, modify, merge, publish, distribute, sublicense, and/or     */
/*  sell copies of this work, and to permit persons to whom this     */
/*  work is furnished to do so, subject to the following conditions: */
/*                                                                   */
/*    1. The code must retain the above copyright notice, this list  */
/*       of conditions and the following disclaimer.                 */
/*    2. Any modifications must be clearly marked as such.           */
/*    3. Original authors' names are not deleted.                    */
/*                                                                   */    
/*  NAGOYA INSTITUTE OF TECHNOLOGY, NARA INSTITUTE OF SCIENCE AND    */
/*  TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, AND THE CONTRIBUTORS TO  */
/*  THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,  */
/*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, */
/*  IN NO EVENT SHALL NAGOYA INSTITUTE OF TECHNOLOGY, NARA           */
/*  INSTITUTE OF SCIENCE AND TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, */
/*  NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR      */
/*  CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM   */
/*  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,  */
/*  NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN        */
/*  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  Subroutine for DTW                                               */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "../sub/matope_sub.h"
#include "dtw_sub.h"

#define AMPSPG 0

DMATRIX xget_dist_mat(DMATRIX orgmat,
		      DMATRIX tarmat,
		      long startdim,
		      long lastdim,
		      long cep,
		      XBOOL noprint_flag)
{
    long k, sk = 1;
    long io, it;
    double cd;
    double cdk;
    double sumcdk;
    DVECTOR tarvec;
    DVECTOR orgvec;
    DMATRIX distmat = NODATA;

    // error check
    if (lastdim >= orgmat->col || lastdim >= tarmat->col) {
	fprintf(stderr, "lastdim error\n");
	return NODATA;
    }
    // memory allocation
    distmat = xdmzeros(tarmat->row, orgmat->row);
    // start dimension
    if (startdim > lastdim) {
	fprintf(stderr, "startdim error\n");
	return NODATA;
    } else {
	sk = startdim;
    }
    
    for (io = 0; io < tarmat->row; io++) {
	// get io-th target vec
	tarvec = xdmextractrow(tarmat, io);

	for (it = 0; it < orgmat->row; it++) {
	    // get it-th original vec
	    orgvec = xdmextractrow(orgmat, it);

	    if (cep != 0) {
		// calculate cepstrum distortion
		for (k = sk, sumcdk = 0.0; k <= lastdim; k++) {
		    cdk = pow((tarvec->data[k] - orgvec->data[k]), 2.0);
		    sumcdk += cdk;
		}
		if (AMPSPG) {	// amplitude spectrum
		    cd = 20.0 * sqrt(2.0 * sumcdk) / log(10.0);
		} else {	// power spectrum
		    cd = 10.0 * sqrt(2.0 * sumcdk) / log(10.0);
		}
	    } else {
		// errr check
		if (tarvec->length != 1 || orgvec->length != 1) {
		    fprintf(stderr, "error: xget_dist_mat");
		    return NODATA;
		}
		// calculate spectrum distortion
		if (tarvec->data[0] <= 0.0) tarvec->data[0] = 0.000001;
		if (orgvec->data[0] <= 0.0) orgvec->data[0] = 0.000001;
		if (AMPSPG) {	// amplitude spectrum
		    cd = pow((20.0 * log10(tarvec->data[0]) - 20.0 * log10(orgvec->data[0])), 2.0);
		} else {	// power spectrum
		    cd = pow((10.0 * log10(tarvec->data[0]) - 10.0 * log10(orgvec->data[0])), 2.0);
		}
	    }
	    
	    distmat->data[io][it] = cd;

	    // memory free
	    xdvfree(orgvec);
	}
	// memory free
	xdvfree(tarvec);
    }

    if (noprint_flag == XFALSE) fprintf(stderr, "get distance matrix\n");
    
    return  distmat;
}

DMATRIX xget_dtw_mat(
		    DMATRIX orgmat,
		    LMATRIX twfunc)
{
    long ri, ci;
    long dtwri = 0;
    DMATRIX dtwsgram;

    // memory allocation
    dtwsgram = xdmzeros(twfunc->row, orgmat->col);

    // Dynamic Time Warping
    for (ri = 0; ri < twfunc->row; ri++)
	if (twfunc->data[ri][0] >= 0) break;
    for (; ri < dtwsgram->row; ri++) {
	if (twfunc->data[ri][0] >= 0)
	    dtwri = twfunc->data[ri][0] + twfunc->data[ri][1] - 1;
	// error check
	if (dtwri >= orgmat->row) {
	    fprintf(stderr, "error: DTW on spectrogram\n");
	    return NODATA;
	}
	if (twfunc->data[ri][1] != 1) {
	    for (ci = 0; ci < dtwsgram->col; ci++)
		dtwsgram->data[ri][ci] = (orgmat->data[dtwri][ci] + orgmat->data[(twfunc->data[ri][0])][ci]) / 2.0;
	} else {
	    for (ci = 0; ci < dtwsgram->col; ci++)
		dtwsgram->data[ri][ci] = orgmat->data[dtwri][ci];
	}
    }

    return dtwsgram;
}

DMATRIX xget_dtw_orgmat_dbl(DMATRIX orgmat,
			    LMATRIX twfunc)
{
    long ri, ris, k;
    long dtwrow, dorgri, dtwri, pdtwri, itera;
    DMATRIX dtworgmat = NODATA;

    for (ris = 0; ris < twfunc->row; ris++)
	if (twfunc->data[ris][0] >= 0) break;
    pdtwri = twfunc->data[ris][0];
    for (ri = ris, dtwrow = 0; ri < twfunc->row; ri++) {
	if (twfunc->data[ri][0] < 0) break;
	if (pdtwri == twfunc->data[ri][0]) dtwrow++;
	dtwrow += twfunc->data[ri][1];
	pdtwri = twfunc->data[ri][0] + twfunc->data[ri][1];
    }

    // memory allocation
    dtworgmat = xdmalloc(dtwrow, orgmat->col);

    pdtwri = twfunc->data[ris][0];
    for (ri = ris, dorgri = 0; ri < twfunc->row; ri++) {
	if (twfunc->data[ri][0] < 0) break;
	dtwri = twfunc->data[ri][0];
	if (pdtwri == dtwri) {	// diagonal
	    for (k = 0; k < orgmat->col; k++)
		dtworgmat->data[dorgri][k] = orgmat->data[dtwri][k];
	    dorgri++;
	}
	for (itera = 0; itera < twfunc->data[ri][1]; itera++, dtwri++) {
	    for (k = 0; k < orgmat->col; k++)
		dtworgmat->data[dorgri][k] = orgmat->data[dtwri][k];
	    dorgri++;
	}
	pdtwri = dtwri;
    }
    if (dorgri != dtworgmat->row) {
	fprintf(stderr, "error: xget_dtw_orgmat\n");
	return NODATA;
    }
    
    return dtworgmat;
}

DMATRIX xget_dtw_tarmat_dbl(DMATRIX tarmat,
			    LMATRIX twfunc)
{
    long ri, ris, k;
    long dtwrow, dtarri, dtwri, pdtwri, itera;
    DMATRIX dtwtarmat = NODATA;

    for (ris = 0; ris < twfunc->row; ris++)
	if (twfunc->data[ris][0] >= 0) break;
    pdtwri = twfunc->data[ris][0];
    for (ri = ris, dtwrow = 0; ri < twfunc->row; ri++) {
	if (twfunc->data[ri][0] < 0) break;
	if (pdtwri == twfunc->data[ri][0]) dtwrow++;
	dtwrow += twfunc->data[ri][1];
	pdtwri = twfunc->data[ri][0] + twfunc->data[ri][1];
    }

    // memory allocation
    dtwtarmat = xdmalloc(dtwrow, tarmat->col);

    pdtwri = twfunc->data[ris][0];
    for (ri = ris, dtarri = 0; ri < twfunc->row; ri++) {
	dtwri = twfunc->data[ri][0];
	if (twfunc->data[ri][0] < 0) break;
	if (pdtwri == dtwri) {	// diagonal
	    for (k = 0; k < tarmat->col; k++)
		dtwtarmat->data[dtarri][k] = tarmat->data[ri][k];
	    dtarri++;
	}
	for (itera = 0; itera < twfunc->data[ri][1]; itera++, dtwri++) {
	    for (k = 0; k < tarmat->col; k++)
		dtwtarmat->data[dtarri][k] = tarmat->data[ri][k];
	    dtarri++;
	}
	pdtwri = dtwri;
    }
    if (dtarri != dtwtarmat->row) {
	fprintf(stderr, "error: xget_dtw_tarmat\n");
	return NODATA;
    }
    
    return dtwtarmat;
    
}

DMATRIX xget_sumdistmat_asym(DMATRIX distmat, long startl,
				 LMATRICES pathmats)
{
    long ci, ri, cie;
    double diadist, underdist, leftdist, dist;
    DMATRIX sumdistmat = NODATA;

    // memory allocation
    sumdistmat = xdmzeros(distmat->row, distmat->col);

    // calculate 0-th row sum distance
    for (ci = 0; ci < distmat->col; ci++) {
	if (ci < startl) {
	    sumdistmat->data[0][ci] = distmat->data[0][ci];
	    pathmats->matrix[0]->data[0][ci] = 0;
	    pathmats->matrix[1]->data[0][ci] = -ci;
	} else {
	    sumdistmat->data[0][ci] = -1.0;
	    pathmats->matrix[0]->data[0][ci] = -1;
	    pathmats->matrix[1]->data[0][ci] = -ci;
	} 
    }

    // calculate sum distance
    for (ri = 1; ri < distmat->row; ri++) {
	cie = MIN(startl + 2 * ri, distmat->col);
	for (ci = 0; ci < distmat->col; ci++) {
	    if (ci < cie) {
		// calculate distance at each blocks
		if (ci - 1 >= 0)
		    if (sumdistmat->data[ri - 1][ci - 1] != -1.0)
			diadist = sumdistmat->data[ri - 1][ci - 1]
			    + distmat->data[ri][ci];
		    else diadist = -1.0;
		else diadist = -1.0;
		if (sumdistmat->data[ri - 1][ci] != -1.0)
		    underdist = sumdistmat->data[ri - 1][ci]
			+ distmat->data[ri][ci];
		else underdist = -1.0;
		if (ci - 2 >= 0)
		    if (sumdistmat->data[ri - 1][ci - 2] != -1.0)
			leftdist = sumdistmat->data[ri - 1][ci - 2]
			    + distmat->data[ri][ci];
		    else leftdist = -1.0;
		else leftdist = -1.0;
		
		// select block
		if (diadist != -1.0) {
		    dist = diadist;
		    if (underdist != -1.0)
			if (underdist < dist) dist = underdist;
		    if (leftdist != -1.0)
			if (leftdist < dist) dist = leftdist;
		} else if (underdist != -1.0){
		    dist = underdist;
		    if (leftdist != -1.0)
			if (leftdist < dist) dist = leftdist;
		} else dist = leftdist;
		if (dist == -1.0) {
		    fprintf(stderr, "error: calculate sum distance matrix\n");
		    return NODATA;
		}
	
		sumdistmat->data[ri][ci] = dist;
		if (dist == diadist) {
		    pathmats->matrix[0]->data[ri][ci] = 1;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 1][ci - 1];
		} else if (dist == underdist) {
		    pathmats->matrix[0]->data[ri][ci] = 2;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 1][ci];
		} else if (dist == leftdist) {
		    pathmats->matrix[0]->data[ri][ci] = 3;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 1][ci - 2];
		} else {
		    fprintf(stderr, "Error: xget_sumdistmat_asym\n");
		    exit(1);
		}
	    } else {
		sumdistmat->data[ri][ci] = -1.0;
		pathmats->matrix[0]->data[ri][ci] = -1;
		pathmats->matrix[1]->data[ri][ci] = -1;
	    }
	}
    }

    return sumdistmat;
}

// DTW: start and end point is free
LMATRIX dtw_body_asym(DMATRIX distmat,
			  double shiftm,
			  double startm,
			  double endm,
			  XBOOL noprint_flag,
			  XBOOL sumdistprint_flag,
			  XBOOL sd_flag)
{
    long ci, cis;
    long startl, endl;
    long bestcis = 0;
    double mindist = 1000000.0, meandist;
    LMATRIX twfunc = NODATA;
    LMATRICES pathmats = NODATA;
    DMATRIX sumdistmat = NODATA;

    // error check
    if (shiftm < 0.0) {
	fprintf(stderr, "shift length > 0 [mm]\n");	return NODATA;
    }
    if (startm < 0.0) {
	fprintf(stderr, "start length > 0 [mm]\n");	return NODATA;
    }
    startl = (long)(startm / shiftm);
    if (endm < 0.0) {
	fprintf(stderr, "end length > 0 [mm]\n");	return NODATA;
    }
    endl = (long)(endm / shiftm);
    if ((startl + endl) > distmat->row || (startl + endl) > distmat->col) {
	fprintf(stderr, "start and end length are too long\n");
	return NODATA;
    }
    startl = MAX(startl, 1);

    // calculate sum distance
    pathmats = xlmsalloc(2);
    pathmats->matrix[0] = xlmzeros(distmat->row, distmat->col);
    pathmats->matrix[1] = xlmzeros(distmat->row, distmat->col);
    sumdistmat = xget_sumdistmat_asym(distmat, startl, pathmats);
    
    // select best path (end point col <= matrix col)
    for (cis = (sumdistmat->col - endl - 1); cis < sumdistmat->col; cis++) {
	meandist = get_meandist_free(sumdistmat->row - 1, cis, sumdistmat,
					 pathmats);
	if (cis == (sumdistmat->col - endl - 1)) {
	    mindist = meandist;	bestcis = cis;
	} else {
	    mindist = MIN(mindist, meandist);
	    if (mindist == meandist) bestcis = cis;
	}
    }
    ci = bestcis;

    if (mindist == 999999.9) {
	fprintf(stderr, "Error: Can't reach an end range\n");
	fprintf(stderr, "Should extend range by -start or -end\n");
	exit(1);
    }
    if (noprint_flag == XFALSE)
	printf("#sum distance [%ld][%ld]%f\n", sumdistmat->row - 1, ci,
	       sumdistmat->data[sumdistmat->row - 1][ci]);

    // get time warping function
    twfunc = xget_twfunc_asym(ci, pathmats, sumdistmat, noprint_flag,
				  sumdistprint_flag, sd_flag);

    // memory free
    xdmfree(sumdistmat);
    xlmsfree(pathmats);

    if (noprint_flag == XFALSE) fprintf(stderr, "DTW done\n");

    return twfunc;
}

LMATRIX xget_twfunc_asym(long ci,
			     LMATRICES pathmats,
			     DMATRIX sumdistmat,
			     XBOOL noprint_flag,
			     XBOOL sumdistprint_flag,
			     XBOOL sd_flag)
{
    long ri, cis;
    LMATRIX twfunc = NODATA;

    // memory allocation
    twfunc = xlmzeros(pathmats->matrix[0]->row, 2);

    ri = sumdistmat->row - 1;    cis = ci;
    while(ri != 0) {
	twfunc->data[ri][0] = ci;	twfunc->data[ri][1] = 1;
	if (pathmats->matrix[0]->data[ri][ci] == 2) {
	    // best path is under
	} else if (pathmats->matrix[0]->data[ri][ci] == 3) {
	    // best path is left
	    ci -= 2;
	} else if (pathmats->matrix[0]->data[ri][ci] == 1){
	    // best path is diagonal
	    ci--;
	} else {
	    if (pathmats->matrix[0]->data[ri][ci] < 1) {
		fprintf(stderr, "error: calculate best path\n");
		return NODATA;
	    }
	}
	ri--;
    }
    twfunc->data[ri][0] = ci;
    twfunc->data[ri][1] = 1;
    
    if (noprint_flag == XFALSE) {
	if (sumdistprint_flag == XFALSE) {
	    if (sd_flag == XFALSE)
		printf("normalized distance %f\n",
		       sumdistmat->data[sumdistmat->row - 1][cis] /
		       (double)sumdistmat->row);
	    else
		printf("normalized distance %f\n",
		       sqrt(sumdistmat->data[sumdistmat->row - 1][cis] /
			    (double)sumdistmat->row));
	} else {
	    if (sd_flag == XFALSE)
		printf("%f	%ld	%f\n",
		       sumdistmat->data[sumdistmat->row - 1][cis],
		       sumdistmat->row,
		       sumdistmat->data[sumdistmat->row - 1][cis] /
		       (double)sumdistmat->row);
	    else
		printf("%f	%ld	%f\n",
		       sumdistmat->data[sumdistmat->row - 1][cis],
		       sumdistmat->row,
		       sqrt(sumdistmat->data[sumdistmat->row - 1][cis] /
			    (double)sumdistmat->row));
	}
    }

    return twfunc;
}

// DTW: start and end point is fixed
LMATRIX dtw_body_fixed(DMATRIX distmat,
			   XBOOL noprint_flag,
			   XBOOL sumdistprint_flag,
			   XBOOL sd_flag)
{
    long ri, ci;
    LMATRIX twfunc = NODATA;
    LMATRIX pathmat = NODATA;
    DMATRIX sumdistmat = NODATA;

    // calculate sum distance matrix
    pathmat = xlmzeros(distmat->row, distmat->col);
    sumdistmat = xget_sumdistmat_fixed(distmat, pathmat);

    ri = distmat->row - 1;    ci = distmat->col - 1;
    if (noprint_flag == XFALSE) {
	if (sumdistprint_flag == XFALSE) {
	    printf("#sum distance [%ld][%ld]%f\n", ri, ci,
		   sumdistmat->data[ri][ci]);
	    if (sd_flag == XFALSE) {
		printf("#normalized distance %f\n",
		       sumdistmat->data[ri][ci] / (double)(ri + ci + 2));
	    } else {
		printf("#normalized distance %f\n",
		       sqrt(sumdistmat->data[ri][ci] / (double)(ri + ci + 2)));
	    }
	} else {
	    if (sd_flag == XFALSE) {
		printf("%f	%ld	%f\n", sumdistmat->data[ri][ci],
		       ri + ci + 2, sumdistmat->data[ri][ci] /
		       (double)(ri + ci + 2));
	    } else {
		printf("%f	%ld	%f\n", sumdistmat->data[ri][ci],
		       ri + ci + 2, sqrt(sumdistmat->data[ri][ci] /
					 (double)(ri + ci + 2)));
	    }
	}
    }

    // calculate time warping function
    twfunc = xget_twfunc_fixed(sumdistmat, pathmat);

    // memory free
    xdmfree(sumdistmat);
    xlmfree(pathmat);

    if (noprint_flag == XFALSE) fprintf(stderr, "DTW done\n");

    return twfunc;
}

DMATRIX xget_sumdistmat_fixed(DMATRIX distmat, LMATRIX pathmat)
{
    long ri, ci;
    double diadist, underdist, leftdist, dist;
    DMATRIX sumdistmat = NODATA;

    // memory allocation
    sumdistmat = xdmzeros(distmat->row, distmat->col);

    // calculate 0-th row sum distance
    sumdistmat->data[0][0] = 2.0 * distmat->data[0][0];
    pathmat->data[0][0] = 0;
    for (ci = 1; ci < distmat->col; ci++) {
	sumdistmat->data[0][ci] = sumdistmat->data[0][ci-1]
	    + distmat->data[0][ci];
	pathmat->data[0][ci] = 3;
    }
    // calculate 0-th col sum distance
    for (ri = 1; ri < distmat->row; ri++) {
	sumdistmat->data[ri][0] = sumdistmat->data[ri-1][0]
	    + distmat->data[ri][0];
	pathmat->data[ri][0] = 2;
    }

    // calculate sum distance
    for (ri = 1; ri < distmat->row; ri++) {
	for (ci = 1; ci < distmat->col; ci++) {
	    // calculate distance at each blocks
	    diadist = sumdistmat->data[ri-1][ci-1]
		+ 2.0 * distmat->data[ri][ci];
	    underdist = sumdistmat->data[ri-1][ci] + distmat->data[ri][ci];
	    leftdist = sumdistmat->data[ri][ci-1] + distmat->data[ri][ci];

	    // select block
	    if (underdist < leftdist) dist = underdist;
	    else dist = leftdist;
	    if (diadist < dist) dist = diadist;

	    sumdistmat->data[ri][ci] = dist;
	    if (dist == diadist) pathmat->data[ri][ci] = 1;
	    else if (dist == underdist) pathmat->data[ri][ci] = 2;
	    else if (dist == leftdist) pathmat->data[ri][ci] = 3;
	    else {
		fprintf(stderr, "Error: xget_sumdistmat_fixed\n");
		exit(1);
	    }
	}
    }

    return sumdistmat;
}

LMATRIX xget_twfunc_fixed(DMATRIX sumdistmat,
			      LMATRIX pathmat)
{
    long ri, ci;
    long leftnum = 1;
    LMATRIX twfunc = NODATA;

    ri = sumdistmat->row - 1;    ci = sumdistmat->col - 1;

    // memory allocation
    twfunc = xlmzeros(sumdistmat->row, 2);// first index, number of index
    while(ri != 0 && ci != 0) {
	if (pathmat->data[ri][ci] == 2) {	// best path is under
	    twfunc->data[ri][0] = ci;
	    twfunc->data[ri][1] = leftnum;
	    leftnum = 1;	ri--;
	} else if (pathmat->data[ri][ci] == 3) {// best path is left
	    leftnum++;		ci--;
	} else if (pathmat->data[ri][ci] == 1){	// best path is diagonal
	    twfunc->data[ri][0] = ci;
	    twfunc->data[ri][1] = leftnum;
	    leftnum = 1;	ri--;	ci--;
	}
    }

    if (ri != 0) {
	twfunc->data[ri][0] = 0;
	twfunc->data[ri][1] = leftnum;	ri--;
	for (; ri >= 0; ri--) {
	    twfunc->data[ri][0] = 0;
	    twfunc->data[ri][1] = 1;
	}
    } else {
	twfunc->data[ri][0] = 0;
	twfunc->data[ri][1] = ci + 1;
    }

    return twfunc;
}

// DTW: start and end point is free
LMATRIX dtw_body_free(DMATRIX distmat,
			  double shiftm,
			  double startm,
			  double endm,
			  XBOOL noprint_flag,
			  XBOOL sumdistprint_flag,
			  XBOOL sd_flag)
{
    long ri, ci, ris, cis;
    long startl, endl;
    long bestcis = 0, bestris = 0;
    double mindist = 1000000.0, mindistci = 1000000.0, mindistri = 1000000.0;
    double meandistci, meandistri;
    LMATRIX twfunc = NODATA;
    LMATRICES pathmats = NODATA;
    DMATRIX sumdistmat = NODATA;

    // error check
    if (shiftm < 0.0) {
	fprintf(stderr, "shift length > 0 [mm]\n");	return NODATA;
    }
    if (startm < 0.0) {
	fprintf(stderr, "start length > 0 [mm]\n");	return NODATA;
    }
    startl = (long)(startm / shiftm);
    if (endm < 0.0) {
	fprintf(stderr, "end length > 0 [mm]\n");	return NODATA;
    }
    endl = (long)(endm / shiftm);
    if ((startl + endl) > distmat->row || (startl + endl) > distmat->col) {
	fprintf(stderr, "start and end length are too long\n");
	return NODATA;
    }
    startl = MAX(startl, 1);

    // calculate sum distance
    pathmats = xlmsalloc(2);
    pathmats->matrix[0] = xlmzeros(distmat->row, distmat->col);
    pathmats->matrix[1] = xlmzeros(distmat->row, distmat->col);
    sumdistmat = xget_sumdistmat_free(distmat, startl, pathmats);
    
    // select best path (end point col <= matrix col)
    for (cis = (sumdistmat->col - endl - 1), ris = (sumdistmat->row - 1);
	 cis < sumdistmat->col; cis++) {
	meandistci = get_meandist_free(ris, cis, sumdistmat, pathmats);
	if (cis == (sumdistmat->col - endl - 1)) {
	    mindistci = meandistci;	bestcis = cis;
	} else {
	    mindistci = MIN(mindistci, meandistci);
	    if (mindistci == meandistci) bestcis = cis;
	}
    }
    // select best path (end point row <= matrix row)
    for (ris = (sumdistmat->row - endl - 1), cis = (sumdistmat->col - 1);
	 ris < sumdistmat->row; ris++) {
	meandistri = get_meandist_free(ris, cis, sumdistmat, pathmats);
	if (ris == (sumdistmat->row - endl - 1)) {
	    mindistri = meandistri;	bestris = ris;
	} else {
	    mindistri = MIN(mindistri, meandistri);
	    if (mindistri == meandistri) bestris = ris;
	}
    }

    mindist = MIN(mindistci, mindistri);
    if (mindist == 999999.9) {
	fprintf(stderr, "Error: Can't reach an end range\n");
	fprintf(stderr, "Should extend range by -start or -end\n");
	exit(1);
    }
    if (mindist == mindistci) {
	ri = sumdistmat->row - 1;	ci = bestcis;
    } else {
	ri = bestris;	ci = sumdistmat->col - 1;
    }

    if (noprint_flag == XFALSE)
	printf("#sum distance [%ld][%ld]%f\n", ri, ci,
	       sumdistmat->data[ri][ci]);

    // get time warping function
    twfunc = xget_twfunc_free(ri, ci, pathmats, sumdistmat, noprint_flag,
				  sumdistprint_flag, sd_flag);

    // memory free
    xdmfree(sumdistmat);
    xlmsfree(pathmats);

    if (noprint_flag == XFALSE) fprintf(stderr, "DTW done\n");

    return twfunc;
}

DMATRIX xget_sumdistmat_free(DMATRIX distmat, long startl,
				 LMATRICES pathmats)
{
    long ci, ri, cis, cie;
    double diadist, underdist, leftdist, dist;
    DMATRIX sumdistmat = NODATA;

    // memory allocation
    sumdistmat = xdmzeros(distmat->row, distmat->col);

    // calculate 0-th row sum distance
    for (ci = 0; ci < distmat->col; ci++) {
	if (ci < startl) {
	    sumdistmat->data[0][ci] = 2.0 * distmat->data[0][ci];
	    pathmats->matrix[0]->data[0][ci] = 0;
	    pathmats->matrix[1]->data[0][ci] = -ci;
	} else {
	    sumdistmat->data[0][ci] = -1.0;
	    pathmats->matrix[0]->data[0][ci] = -1;
	    pathmats->matrix[1]->data[0][ci] = -ci;
	} 
    }
    // calculate 0-th col sum distance
    for (ri = 1; ri < distmat->row; ri++) {
	if (ri < startl) {
	    sumdistmat->data[ri][0] = 2.0 * distmat->data[ri][0];
	    pathmats->matrix[0]->data[ri][0] = 0;
	    pathmats->matrix[1]->data[ri][0] = ri;
	} else {
	    sumdistmat->data[ri][0] = -1.0;
	    pathmats->matrix[0]->data[ri][0] = -1;
	    pathmats->matrix[1]->data[ri][0] = ri;
	}
    }

    // calculate sum distance
    for (ri = 1; ri < distmat->row; ri++) {
	cis = MAX(((ri - startl) / 2 + 1), 1);
	cie = MIN(startl + 2 * ri, distmat->col);
	for (ci = 1; ci < distmat->col; ci++) {
	    if (ci >= cis && ci < cie) {
		// calculate distance at each blocks
		dist = -1.0;
		diadist = -1.0;	underdist = -1.0;	leftdist = -1.0;
		if (sumdistmat->data[ri - 1][ci - 1] != -1.0)
		    diadist = sumdistmat->data[ri - 1][ci - 1]
			+ 2.0 * distmat->data[ri][ci];
		if (ri - 2 >= 0)
		    if (sumdistmat->data[ri - 2][ci - 1] != -1.0)
			underdist = sumdistmat->data[ri - 2][ci - 1]
			    + 2.0 * distmat->data[ri - 1][ci]
			    + distmat->data[ri][ci];
		if (ci - 2 >= 0)
		    if (sumdistmat->data[ri - 1][ci - 2] != -1.0)
			leftdist = sumdistmat->data[ri - 1][ci - 2]
			    + 2.0 * distmat->data[ri][ci - 1]
			    + distmat->data[ri][ci];

		// select block
		if (diadist != -1.0) {
		    dist = diadist;
		    if (underdist != -1.0)
			if (underdist < dist) dist = underdist;
		    if (leftdist != -1.0)
			if (leftdist < dist) dist = leftdist;
		} else if (underdist != -1.0){
		    dist = underdist;
		    if (leftdist != -1.0)
			if (leftdist < dist) dist = leftdist;
		} else dist = leftdist;
		if (dist == -1.0) {
		    fprintf(stderr, "error: calculate sum distance matrix\n");
		    return NODATA;
		}

		sumdistmat->data[ri][ci] = dist;
		if (dist == diadist) {
		    pathmats->matrix[0]->data[ri][ci] = 1;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 1][ci - 1];
		} else if (dist == underdist) {
		    pathmats->matrix[0]->data[ri][ci] = 2;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 2][ci - 1];
		} else if (dist == leftdist) {
		    pathmats->matrix[0]->data[ri][ci] = 3;
		    pathmats->matrix[1]->data[ri][ci] =
			pathmats->matrix[1]->data[ri - 1][ci - 2];
		} else {
		    fprintf(stderr, "Error 2\n");
		    exit(1);
		}
	    } else {
		sumdistmat->data[ri][ci] = -1.0;
		pathmats->matrix[0]->data[ri][ci] = -1;
		pathmats->matrix[1]->data[ri][ci] = -1;
	    }
	}
    }

    return sumdistmat;
}

double get_meandist_free(long ris,
			     long cis,
			     DMATRIX sumdistmat,
			     LMATRICES pathmats)
{
    long ri, ci;
    double meandist;

    if (pathmats->matrix[1]->data[ris][cis] < 0) {
	ri = 0;	ci = -1 * pathmats->matrix[1]->data[ris][cis];
    } else if (pathmats->matrix[1]->data[ris][cis] > 0) {
	ri = pathmats->matrix[1]->data[ris][cis];	ci = 0;
    } else {
	ri = 0;	ci = 0;
    }
    meandist =  sumdistmat->data[ris][cis] / (double)(ris - ri + cis - ci+ 2);
    if (sumdistmat->data[ris][cis] < 0.0) meandist = 999999.9;

    return meandist;
}

LMATRIX xget_twfunc_free(long ri,
			     long ci,
			     LMATRICES pathmats,
			     DMATRIX sumdistmat,
			     XBOOL noprint_flag,
			     XBOOL sumdistprint_flag,
			     XBOOL sd_flag)
{
    long ris, cis;
    LMATRIX twfunc = NODATA;

    // memory allocation
    twfunc = xlmzeros(pathmats->matrix[0]->row, 2);

    ris = ri;    cis = ci;

    while(ri != 0 && ci != 0) {
	if (pathmats->matrix[0]->data[ri][ci] == 2) {
	    // best path is under
	    twfunc->data[ri][0] = ci;		twfunc->data[ri][1] = 1;
	    twfunc->data[ri - 1][0] = ci;	twfunc->data[ri - 1][1] = 1;
	    ri -= 2;	    ci--;
	} else if (pathmats->matrix[0]->data[ri][ci] == 3) {
	    // best path is left
	    twfunc->data[ri][0] = ci - 1;	twfunc->data[ri][1] = 2;
	    ri--;	    ci -= 2;
	} else if (pathmats->matrix[0]->data[ri][ci] == 1){
	    // best path is diagonal
	    twfunc->data[ri][0] = ci;		twfunc->data[ri][1] = 1;
	    ri--;	    ci--;
	} else {
	    if (pathmats->matrix[0]->data[ri][ci] < 1) {
		fprintf(stderr, "error: calculate best path\n");
		fprintf(stderr, "%ld %ld  %ld\n", ri, ci, pathmats->matrix[0]->data[ri][ci]);
		return NODATA;
	    }
	}
    }
    
    if (noprint_flag == XFALSE) {
	if (sumdistprint_flag == XFALSE) {
	    if (sd_flag == XFALSE) {
		printf("normalized distance %f\n",
		       sumdistmat->data[ris][cis] /
		       (double)(ris - ri + cis - ci + 2));
	    } else {
		printf("normalized distance %f\n",
		       sqrt(sumdistmat->data[ris][cis] /
			    (double)(ris - ri + cis - ci + 2)));
	    }
	} else {
	    if (sd_flag == XFALSE) {
		printf("%f	%ld	%f\n", sumdistmat->data[ris][cis],
		       ris - ri + cis - ci + 2,
		       sumdistmat->data[ris][cis] /
		       (double)(ris - ri + cis - ci + 2));
	    } else {
		printf("%f	%ld	%f\n", sumdistmat->data[ris][cis],
		       ris - ri + cis - ci + 2,
		       sqrt(sumdistmat->data[ris][cis] /
			    (double)(ris - ri + cis - ci + 2)));
	    }
	}
    }
    
    if (ri != 0) {
	twfunc->data[ri][0] = 0;
	twfunc->data[ri][1] = 1;	ri--;
	for (; ri >= 0; ri--) {
	    //twfunc->data[ri][0] = 0;
	    twfunc->data[ri][0] = -1;
	    twfunc->data[ri][1] = 1;
	}
    } else {
	//twfunc->data[ri][0] = 0;
	//twfunc->data[ri][1] = ci + 1;
	twfunc->data[ri][0] = ci;
	twfunc->data[ri][1] = 1;
    }
    
    for (ri = ris + 1; ri < twfunc->row; ri++) {
	//twfunc->data[ri][0] = twfunc->data[ris][0];
	twfunc->data[ri][0] = -1;
	twfunc->data[ri][1] = 1;
    }

    return twfunc;
}

void getcd(DMATRIX orgmat,
	   DMATRIX tarmat,
	   long startdim,
	   long lastdim)
{
    long ri, rie, k, sumri = 0, sk;
    double cd, cdri, sumcdk, cdk;
    DVECTOR orgvec;
    DVECTOR tarvec;

    // error check
    if (lastdim >= orgmat->col || lastdim >= tarmat->col) {
	fprintf(stderr, "lastdim error\n");
	exit(1);
    }
    if (orgmat->row != tarmat->row) {
	fprintf(stderr, "error: getcd_new (DTWed matrix)\n");
	exit(1);
    }
    // start dimension
    if (startdim > lastdim) {
	fprintf(stderr, "startdim error\n");
	exit(1);
    } else {
	sk = startdim;
    }

    rie = tarmat->row;

    for (ri = 0, cdri = 0.0; ri < rie; ri++) {
	// get io-th original vec
	orgvec = xdmextractrow(orgmat, ri);
	// get it-th original vec
	tarvec = xdmextractrow(tarmat, ri);

	// calculate cepstrum distortion
	for (k = sk, sumcdk = 0.0; k <= lastdim; k++) {
	    cdk = pow((tarvec->data[k] - orgvec->data[k]), 2.0);
	    sumcdk += cdk;
	}
	if (AMPSPG) {	// amplitude spectrum
	    cdri += 20.0 * sqrt(2.0 * sumcdk) / log(10.0);
	} else {	// power spectrum
	    cdri += 10.0 * sqrt(2.0 * sumcdk) / log(10.0);
	}
	sumri += 1;

	xdvfree(orgvec);
	xdvfree(tarvec);
    }
    // calculate mean CD
    cd = cdri / (double)sumri;

    printf("#Cepstrum Distortion [dB]\n");
    printf("%f\n", cd);
}

void getcd(DMATRIX orgmat,
	   DMATRIX tarmat,
	   long startdim,
	   long lastdim,
	   char *outf)
{
    long ri, rie, k, sumri = 0, sk;
    double cd, cdri, sumcdk, cdk;
    DVECTOR orgvec;
    DVECTOR tarvec;
    FILE *fp;

    // error check
    if (lastdim >= orgmat->col || lastdim >= tarmat->col) {
	fprintf(stderr, "lastdim error\n");
	exit(1);
    }
    if (orgmat->row != tarmat->row) {
	fprintf(stderr, "error: getcd_new (DTWed matrix)\n");
	exit(1);
    }
    // start dimension
    if (startdim > lastdim) {
	fprintf(stderr, "startdim error\n");
	exit(1);
    } else {
	sk = startdim;
    }
    // file open
    if ((fp = fopen(outf, "wt")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", outf);
	exit(1);
    }

    rie = tarmat->row;

    for (ri = 0, cdri = 0.0; ri < rie; ri++) {
	// get io-th original vec
	orgvec = xdmextractrow(orgmat, ri);
	// get it-th original vec
	tarvec = xdmextractrow(tarmat, ri);

	// calculate cepstrum distortion
	for (k = sk, sumcdk = 0.0; k <= lastdim; k++) {
	    cdk = pow((tarvec->data[k] - orgvec->data[k]), 2.0);
	    sumcdk += cdk;
	}
	if (AMPSPG) {	// amplitude spectrum
	    sumcdk = 20.0 * sqrt(2.0 * sumcdk) / log(10.0);
	} else {	// power spectrum
	    sumcdk = 10.0 * sqrt(2.0 * sumcdk) / log(10.0);
	}
	cdri += sumcdk;
	sumri += 1;
	fprintf(fp, "%f\n", sumcdk);

	xdvfree(orgvec);
	xdvfree(tarvec);
    }
    // calculate mean CD
    cd = cdri / (double)sumri;
    // file close
    fclose(fp);

    printf("#Cepstrum Distortion [dB]\n");
    printf("%f\n", cd);
}
