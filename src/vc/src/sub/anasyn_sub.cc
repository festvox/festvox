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
/*  Analysis-synthesis subroutine                                    */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/basic.h"
#include "../include/fileio.h"
#include "../include/filter.h"
#include "../include/memory.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "sptk_sub.h"
#include "anasyn_sub.h"

// read header of wav file
char *xreadheader_wav(char *filename)
{
    char *header = NULL;
    FILE *fp;

    // memory allocate
    if ((header = (char *)malloc((int)44 * sizeof(char))) == NULL) {
	fprintf(stderr, "Read header: Memory allcation is failed\n");
	exit(0);
    }

    // open file
    if (NULL == (fp = fopen(filename, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", filename);
	return NULL;
    }

    // read data
    fread(header, sizeof(char), (int)44, fp);

    // close file
    fclose(fp);

    return header;
}

// write header of wav file
void writessignal_wav(char *filename, SVECTOR vector, double fs)
{
    char riff[5] = "RIFF";
    char riffsize[4] = "";
    char wave[5] = "WAVE";
    char fmt[5] = "fmt ";
    char fmtsize[4] = "";
    unsigned short format = 1;
    unsigned short channel = 1;
    char sampling[4] = "";
    char bps[4] = "";
    char block = 2;
    char dummy1 = 0;
    char bit = 16;
    char dummy2 = 0;
    char data[5] = "data";
    char datasize[4] = "";
    unsigned long tmp;
    FILE *fp;

    fmtsize[3] = 0x00;	fmtsize[2] = 0x00;
    fmtsize[1] = 0x00;	fmtsize[0] = 0x10;
    
    tmp = (unsigned long)(vector->length * 2);
    datasize[3] = (char)(tmp >> 24);	datasize[2] = (char)(tmp >> 16);
    datasize[1] = (char)(tmp >> 8);	datasize[0] = (char)tmp;

    tmp += (unsigned long)36;
    riffsize[3] = (char)(tmp >> 24);	riffsize[2] = (char)(tmp >> 16);
    riffsize[1] = (char)(tmp >> 8);	riffsize[0] = (char)tmp;

    tmp = (unsigned long)fs;
    sampling[3] = (char)(tmp >> 24);	sampling[2] = (char)(tmp >> 16);
    sampling[1] = (char)(tmp >> 8);	sampling[0] = (char)tmp;

    tmp += tmp;
    bps[3] = (char)(tmp >> 24);	bps[2] = (char)(tmp >> 16);
    bps[1] = (char)(tmp >> 8);	bps[0] = (char)tmp;
    
    // open file
    check_dir(filename);
    if (NULL == (fp = fopen(filename, "wb"))) {
	fprintf(stderr, "can't open file: %s\n", filename);
	return;
    }

    // write header
    fwrite(riff, sizeof(char), 4, fp);
    fwrite(riffsize, sizeof(char), 4, fp);
    fwrite(wave, sizeof(char), 4, fp);
    fwrite(fmt, sizeof(char), 4, fp);
    fwrite(fmtsize, sizeof(char), 4, fp);
    fwrite(&format, sizeof(unsigned short), 1, fp);
    fwrite(&channel, sizeof(unsigned short), 1, fp);
    fwrite(sampling, sizeof(char), 4, fp);
    fwrite(bps, sizeof(char), 4, fp);
    fwrite(&block, sizeof(char), 1, fp);
    fwrite(&dummy1, sizeof(char), 1, fp);
    fwrite(&bit, sizeof(char), 1, fp);
    fwrite(&dummy2, sizeof(char), 1, fp);
    fwrite(data, sizeof(char), 4, fp);
    fwrite(datasize, sizeof(char), 4, fp);
    
    // write data
    fwriteshort(vector->data, vector->length, 0, fp);

    // close file
    if (fp != stdout)
	fclose(fp);

    return;
}

void check_header(char *file, double *fs, XBOOL *float_flag, XBOOL msg_flag)
{
    char *header = NULL;
    
    // get header
    if ((header = xreadheader_wav(file)) == NULL) {
	fprintf(stderr, "Can't read header %s\n", file);
	exit(1);
    }
    // read header-information
    if ((strncmp(header, "RIFF", 4) != 0 ||
	 strncmp(header + 8, "WAVE", 4) != 0) ||
	(strncmp(header + 12, "fmt", 3) != 0 ||
	 strncmp(header + 36, "data", 4) != 0)) {
	fprintf(stderr, "no wav file: %s\n", file);
	exit(1);
    } else {
	if (fs != NULL) {
	    *fs = (double)(((((header[27] << 24) & 0xff000000) |
			     ((header[26] << 16) & 0xff0000))) |
			   ((((header[25] << 8) & 0xff00) |
			     ((header[24]) & 0xff))));
	    if (msg_flag == XTRUE)
		fprintf(stderr, "Sampling frequency %5.0f [Hz]\n", *fs);
	}
    }
    if (header[34] == 16) {
	if (msg_flag == XTRUE) fprintf(stderr, "16bit short wave\n");
	*float_flag = XFALSE;
    } else if (header[34] == 32) {
	if (msg_flag == XTRUE) fprintf(stderr, "32bit float wave\n");
	*float_flag = XTRUE;
    } else {
	fprintf(stderr, "no wav file: %s\n", file);
	fprintf(stderr, "Please use this type: signed 16 bit or float 32 bit\n");
	exit(1);
    }
    xfree(header);

    return;
}


DVECTOR xread_wavfile(char *file, double *fs, XBOOL msg_flag)
{
    long headlen = 44;
    SVECTOR xs = NODATA;
    FVECTOR xf = NODATA;
    DVECTOR xd = NODATA;
    XBOOL float_flag = XFALSE;

    // read header
    check_header(file, fs, &float_flag, msg_flag);
    // read waveform
    if (float_flag == XFALSE) {
	// read short wave data
	if ((xs = xreadssignal(file, headlen, 0)) == NODATA) {
	    exit(1);
	} else {
	    xd = xsvtod(xs);	xsvfree(xs);
	}
    } else {
	// read float wave data
	if ((xf = xreadfsignal(file, headlen, 0)) == NODATA) {
	    exit(1);
	} else {
	    xd = xfvtod(xf);	xfvfree(xf);
	    dvscoper(xd, "*", 32000.0);
	}
    }
    if (msg_flag == XTRUE) fprintf(stderr, "read wave: %s\n", file);

    return xd;
}

// cut low noise (f < f0floor)
void cleaninglownoise(DVECTOR x,
		      double fs,
		      double f0floor)
{
    long nn, flp, k;
    double flm;
    DVECTOR wlp = NODATA;
    DVECTOR tx = NODATA;
    DVECTOR ttx = NODATA;

    flm = 50.0;
    flp = (long)round(fs * flm / 1000.0);
    nn = x->length;
    wlp = fir1(flp * 2, f0floor / (fs / 2.0));

    wlp->data[flp] = wlp->data[flp] - 1.0;
    dvscoper(wlp, "*", -1.0);
    tx = xdvzeros(x->length + 2 * wlp->length);
    dvcopy(tx, x);
    ttx = xdvfftfiltm(wlp, tx, wlp->length * 2);
    for (k = 0; k < nn; k++) x->data[k] = ttx->data[k + flp];

    // memory free
    xdvfree(wlp);
    xdvfree(tx);
    xdvfree(ttx);

    return;
}

// lowpass FIR digital filter by using Hamming window
DVECTOR fir1(long len,		// return [length + 1]
	     double cutoff)	// cut-off frequency
{
    long k, length;
    double half;
    double value, a;
    DVECTOR filter;

    // 0 <= cufoff <= 1
    cutoff = MIN(cutoff, 1.0);
    cutoff = MAX(cutoff, 0.0);

    half = (double)len / 2.0;
    length = len + 1;

    // memory allocate
    filter = xdvalloc(length);

    // hamming window
    for (k = 0; k < length; k++) {
	filter->data[k] = 0.54 - 0.46 * cos(2.0 * PI * (double)k / (double)(length - 1));
    }

    // calculate lowpass filter
    for (k = 0; k < length; k++) {
	a = PI * ((double)k - half);
	if (a == 0.0) {
	    value = cutoff;
	} else {
	    value = sin(PI * cutoff * ((double)k - half)) / a;
	}
	filter->data[k] *= value;
    }

    return filter;
}
