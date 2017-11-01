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
/*          Author :  Hideki Banno                                   */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  Slightly modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)       */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef VARARGS
#include <varargs.h>
#else
#include <stdarg.h>
#endif

#include "../include/defs.h"
#include "../include/memory.h"
#include "../include/option.h"
#include "../include/vector.h"
#include "../include/matrix.h"
#include "../include/fileio.h"

/*
 *	get file size
 */
long getfilesize(const char *filename, int headlen)
{
    long size;
    char *basicname;
    struct stat status;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	/* stat file */
	if (fstat(0, &status)) {
	    fprintf(stderr, "can't stat stdin\n");
	    return 0;
	}
    } else {
	/* stat file */
	if (stat(filename, &status)) {
	    fprintf(stderr, "can't stat file: %s\n", filename);
	    return 0;
	}
    }

    /* get data size */
    size = (long)MAX(status.st_size - headlen, 0);

    /* memory free */
    xfree(basicname);

    return size;
}

/*
 *	swap data
 */
void swapshort(short *data, long length)
{
    long k;

    for (k = 0; k < length; k++) {
	data[k] = ((data[k] << 8) & 0xff00) | ((data[k] >> 8) & 0xff);
    }

    return;
}

void swaplong(long *data, long length)
{
    long k;
    int i;
    int num_unit;
    char *pi, *po;
    long vi, vo;

    num_unit = sizeof(long) - 1;

    for (k = 0; k < length; k++) {
	vi = data[k];
	pi = (char *)&vi;
	po = (char *)&vo;
	for (i = 0; i <= num_unit; i++) {
	    po[i] = pi[num_unit - i];
	}
	data[k] = vo;
    }
    
    return;
}

void swapfloat(float *data, long length)
{
    long k;
    int i;
    int num_unit;
    char *pi, *po;
    float vi, vo;

    num_unit = sizeof(float) - 1;

    for (k = 0; k < length; k++) {
	vi = data[k];
	pi = (char *)&vi;
	po = (char *)&vo;
	for (i = 0; i <= num_unit; i++) {
	    po[i] = pi[num_unit - i];
	}
	data[k] = vo;
    }
    
    return;
}

void swapdouble(double *data, long length)
{
    long k;
    int i;
    int num_unit;
    char *pi, *po;
    double vi, vo;

    num_unit = sizeof(double) - 1;

    for (k = 0; k < length; k++) {
	vi = data[k];
	pi = (char *)&vi;
	po = (char *)&vo;
	for (i = 0; i <= num_unit; i++) {
	    po[i] = pi[num_unit - i];
	}
	data[k] = vo;
    }
    
    return;
}

/*
 *	read data
 */
void freadshort(short *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    /* read file */
    fread((char *)data, sizeof(short), (int)length, fp);

    if (swap) {
	/* byte swap */
	swapshort(data, length);
    }

    return;
}

void freadlong(long *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    /* read file */
    fread((char *)data, sizeof(long), (int)length, fp);

    if (swap) {
	/* byte swap */
	swaplong(data, length);
    }

    return;
}

void freadfloat(float *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    /* read file */
    fread((char *)data, sizeof(float), (int)length, fp);

    if (swap) {
	/* byte swap */
	swapfloat(data, length);
    }

    return;
}

void freaddouble(double *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    /* read file */
    fread((char *)data, sizeof(double), (int)length, fp);

    if (swap) {
	/* byte swap */
	swapdouble(data, length);
    }

    return;
}

void freadshorttod(double *data, long length, int swap, FILE *fp)
{
    long k;
    short value;
    
    if (data == NULL) return;

    for (k = 0; k < length; k++) {
	/* read file */
	fread((char *)&value, sizeof(short), 1, fp);
	
	if (swap) {
	    /* byte swap */
	    swapshort(&value, 1);
	}
	
	data[k] = (double)value;
    }

    return;
}


/*
 *	write data
 */
void fwriteshort(short *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    if (swap) {
	long k;
	short value;

	for (k = 0; k < length; k++) {
	    value = data[k];
	    
	    /* byte swap */
	    swapshort(&value, 1);

	    /* write file */
	    fwrite((char *)&value, sizeof(short), 1, fp);
	}
    } else {
	/* write file */
	fwrite((char *)data, sizeof(short), (int)length, fp);
    }

    return;
}

void fwritelong(long *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    if (swap) {
	long k;
	long value;

	for (k = 0; k < length; k++) {
	    value = data[k];
	    
	    /* byte swap */
	    swaplong(&value, 1);

	    /* write file */
	    fwrite((char *)&value, sizeof(long), 1, fp);
	}
    } else {
	/* write file */
	fwrite((char *)data, sizeof(long), (int)length, fp);
    }

    return;
}

void fwritefloat(float *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    if (swap) {
	long k;
	float value;

	for (k = 0; k < length; k++) {
	    value = data[k];
	    
	    /* byte swap */
	    swapfloat(&value, 1);

	    /* write file */
	    fwrite((char *)&value, sizeof(float), 1, fp);
	}
    } else {
	/* write file */
	fwrite((char *)data, sizeof(float), (int)length, fp);
    }

    return;
}

void fwritedouble(double *data, long length, int swap, FILE *fp)
{
    if (data == NULL) return;
    
    if (swap) {
	long k;
	double value;

	for (k = 0; k < length; k++) {
	    value = data[k];
	    
	    /* byte swap */
	    swapdouble(&value, 1);

	    /* write file */
	    fwrite((char *)&value, sizeof(double), 1, fp);
	}
    } else {
	/* write file */
	fwrite((char *)data, sizeof(double), (int)length, fp);
    }

    return;
}

void fwritedoubletos(double *data, long length, int swap, FILE *fp)
{
    long k;
    short value;
    
    if (data == NULL) return;
    
    for (k = 0; k < length; k++) {
	value = (short)data[k];
	
	if (swap) {
	    /* byte swap */
	    swapshort(&value, 1);
	}

	/* write file */
	fwrite((char *)&value, sizeof(short), 1, fp);
    }

    return;
}

/*
 *	read short signal
 */
SVECTOR xreadssignal(const char *filename, int headlen, int swap)
{
    long length;
    char *basicname;
    SVECTOR vector;
    FILE *fp;

    /* get signal length */
    if ((length = getsiglen(filename, headlen, short)) <= 0) {
	return NODATA;
    }

    /* memory allocate */
    vector = xsvalloc(length);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* skip header */
    if (headlen > 0)
	fseek(fp, (long)headlen, 0);

    /* read data */
    freadshort(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return vector;
}

/*
 *	read float signal
 */
FVECTOR xreadfsignal(const char *filename, int headlen, int swap)
{
    long length;
    char *basicname;
    FVECTOR vector;
    FILE *fp;

    /* get signal length */
    if ((length = getsiglen(filename, headlen, float)) <= 0) {
	return NODATA;
    }

    /* memory allocate */
    vector = xfvalloc(length);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* skip header */
    if (headlen > 0)
	fseek(fp, (long)headlen, 0);

    /* read data */
    freadfloat(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return vector;
}

DVECTOR xdvreadssignal(const char *filename, int headlen, int swap)
{
    long length;
    char *basicname;
    DVECTOR vector;
    FILE *fp;

    /* get signal length */
    if ((length = getsiglen(filename, headlen, short)) <= 0) {
	return NODATA;
    }

    /* memory allocate */
    vector = xdvalloc(length);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* skip header */
    if (headlen > 0)
	fseek(fp, (long)headlen, 0);

    /* read data */
    freadshorttod(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return vector;
}

/*
 *	read double signal
 */
DVECTOR xreaddsignal(const char *filename, int headlen, int swap)
{
    long length;
    char *basicname;
    DVECTOR vector;
    FILE *fp;

    /* get signal length */
    if ((length = getsiglen(filename, headlen, double)) <= 0) {
	return NODATA;
    }

    /* memory allocate */
    vector = xdvalloc(length);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* skip header */
    if (headlen > 0)
	fseek(fp, (long)headlen, 0);

    /* read data */
    freaddouble(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return vector;
}

DVECTOR xreadf2dsignal(const char *filename, int headlen, int swap)
{
    long length, k;
    char *basicname;
    FVECTOR fvec = NODATA;
    DVECTOR vector;
    FILE *fp;

    /* get signal length */
    if ((length = getsiglen(filename, headlen, float)) <= 0) {
	return NODATA;
    }

    /* memory allocate */
    vector = xdvalloc(length);
    fvec = xfvalloc(length);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* skip header */
    if (headlen > 0)
	fseek(fp, (long)headlen, 0);

    /* read data */
    freadfloat(fvec->data, vector->length, swap, fp);
    for (k = 0; k < fvec->length; k++) vector->data[k] = (double)fvec->data[k];

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    xfvfree(fvec);

    return vector;
}
#if 0
#endif

/*
 *	write short signal
 */
void writessignal(const char *filename, SVECTOR vector, int swap)
{
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write data */
    fwriteshort(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    
    return;
}

void dvwritessignal(const char *filename, DVECTOR vector, int swap)
{
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write data */
    fwritedoubletos(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    
    return;
}

/*
 *	write double signal
 */
void writedsignal(const char *filename, DVECTOR vector, int swap)
{
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write data */
    fwritedouble(vector->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    
    return;
}

void writed2fsignal(const char *filename, DVECTOR vector, int swap)
{
    long k;
    char *basicname;
    FVECTOR fvec = NODATA;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    fvec = xfvalloc(vector->length);
    for (k = 0; k < vector->length; k++)
	fvec->data[k] = (float)vector->data[k];

    /* write data */
    fwritefloat(fvec->data, vector->length, swap, fp);

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    xfvfree(fvec);
    
    return;
}
#if 0
#endif

LMATRIX xreadlmatrix(const char *filename, long ncol, int swap)
{
    long k;
    long nrow;
    long length;
    char *basicname;
    LMATRIX mat;
    FILE *fp;

    /* get data length */
    if ((length = getsiglen(filename, 0, long)) <= 0) {
	return NODATA;
    }
    if (length % ncol != 0) {
	fprintf(stderr, "wrong data format: %s\n", filename);
	return NODATA;
    }
    nrow = length / ncol;

    /* memory allocate */
    mat = xlmalloc(nrow, ncol);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* read data */
    for (k = 0; k < mat->row; k++) {
	freadlong(mat->data[k], mat->col, swap, fp);
    }

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return mat;
}

DMATRIX xreaddmatrix(const char *filename, long ncol, int swap)
{
    long k;
    long nrow;
    long length;
    char *basicname;
    DMATRIX mat;
    FILE *fp;

    /* get data length */
    if ((length = getsiglen(filename, 0, double)) <= 0) {
	return NODATA;
    }
    if (length % ncol != 0) {
	fprintf(stderr, "wrong data format: %s\n", filename);
	return NODATA;
    }
    nrow = length / ncol;

    /* memory allocate */
    mat = xdmalloc(nrow, ncol);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* read data */
    for (k = 0; k < mat->row; k++) {
	freaddouble(mat->data[k], mat->col, swap, fp);
    }

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return mat;
}

DMATRIX xreadf2dmatrix(const char *filename, long ncol, int swap)
{
    long k, l;
    long nrow;
    long length;
    char *basicname;
    FVECTOR fvec = NODATA;
    DMATRIX mat;
    FILE *fp;

    /* get data length */
    if ((length = getsiglen(filename, 0, float)) <= 0) {
	return NODATA;
    }
    if (length % ncol != 0) {
	fprintf(stderr, "wrong data format: %s\n", filename);
	return NODATA;
    }
    nrow = length / ncol;

    /* memory allocate */
    mat = xdmalloc(nrow, ncol);
    fvec = xfvalloc(ncol);

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdin")) {
	fp = stdin;
    } else {
	/* open file */
	if (NULL == (fp = fopen(filename, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return NODATA;
	}
    }

    /* read data */
    for (k = 0; k < mat->row; k++) {
	freadfloat(fvec->data, mat->col, swap, fp);
	for (l = 0; l < mat->col; l++) mat->data[k][l] = (double)fvec->data[l];
    }

    /* close file */
    if (fp != stdin)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    xfvfree(fvec);

    return mat;
}

void writelmatrix(const char *filename, LMATRIX mat, int swap)
{
    long k;
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write data */
    for (k = 0; k < mat->row; k++) {
	fwritelong(mat->data[k], mat->col, swap, fp);
    }

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return;
}

void writedmatrix(const char *filename, DMATRIX mat, int swap)
{
    long k;
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    /* write data */
    for (k = 0; k < mat->row; k++) {
	fwritedouble(mat->data[k], mat->col, swap, fp);
    }

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return;
}

void writed2fmatrix(const char *filename, DMATRIX mat, int swap)
{
    long k, l;
    float *fdata = NULL;
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
	/* open file */
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wb"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return;
	}
    }

    // memory allocation
    fdata = xalloc(mat->col, float);
    /* write data */
    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) fdata[l] = (float)mat->data[k][l];
	fwritefloat(fdata, mat->col, swap, fp);
    }

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);
    xfree(fdata);

    return;
}

/*
 *	file i/o for text data
 */
long getfilesize_txt(const char *filename)
{
    long size;
    double value;
    char line[MAX_PATHNAME];
    char string[MAX_LINE];
    FILE *fp;

    if (NULL == (fp = fopen(filename, "r"))) {
        fprintf(stderr, "can't open file: %s\n", filename);
	return FAILURE;
    }

    size = 0;
    while (fgetline(line, fp) != EOF) {
        sscanf(line, "%s", string);
        if (sscanf(string, "%lf", &value) == 1) {
	    size++;
        }
    }

    fclose(fp);

    return size;
}

int readdvector_txt(const char *filename, DVECTOR vector)
{
    long lc;
    double value;
    char line[MAX_PATHNAME];
    char string[MAX_LINE];
    FILE *fp;

    if (NULL == (fp = fopen(filename, "r"))) {
        fprintf(stderr, "can't open file: %s\n", filename);
	return FAILURE;
    }

    lc = 0;
    while (fgetline(line, fp) != EOF) {
	if (lc >= vector->length) {
	    break;
	}

        sscanf(line, "%s", string);
        if (sscanf(string, "%lf", &value) == 1) {
	    vector->data[lc] = value;
	    lc++;
        }
    }

    for (; lc < vector->length; lc++) {
	vector->data[lc] = 0.0;
    }

    fclose(fp);

    return SUCCESS;
}

DVECTOR xreaddvector_txt(const char *filename)
{
    long length;
    DVECTOR vector;

    if ((length = getfilesize_txt(filename)) == FAILURE) {
	return NODATA;
    }

    vector = xdvalloc(length);

    if (readdvector_txt(filename, vector) == FAILURE) {
	xdvfree(vector);
	return NODATA;
    }

    return vector;
}

int getnumrow_txt(const char *filename)
{
    int nrow, ncol;
    FILE *fp;
    char buf[MAX_LINE];

    if (NULL == (fp = fopen(filename, "r"))) {
        fprintf(stderr, "can't open file: %s\n", filename);
	return -1;
    }

    nrow = 0;
    while (1) {
	ncol = fgetcol_txt(buf, 0, fp);
	if (ncol == EOF) {
	    break;
	}
	nrow++;
    }

    fclose(fp);

    return nrow;
}

int getnumcol_txt(const char *filename)
{
    int ncol;
    FILE *fp;
    char buf[MAX_LINE];

    if (NULL == (fp = fopen(filename, "r"))) {
        fprintf(stderr, "can't open file: %s\n", filename);
	return -1;
    }

    ncol = fgetcol_txt(buf, 0, fp);

    fclose(fp);

    return ncol;
}

/*
 *	get column from text file
 * 	(reason of using fscanf is SGI's bug(?).)
 */
int fgetcol_txt(char *buf, int col, FILE *fp)
{
    char c, prev_c;
    int n;
    int end;
    int ncol;
    int nchar;

    buf[0] = NUL;
    prev_c = NUL;
    end = 0;
    ncol = 0;
    nchar = 0;
    while (1) {
	n = fscanf(fp, "%c", &c);
	if (c == EOF || n <= 0) {
	    break;
	} else if (c == '#' && prev_c != '\\') {
	    while (1) {
		n = fscanf(fp, "%c", &c);
		if (c == EOF || n <= 0) {
		    end = 1;
		    break;
		} else if (c == '\n') {
		    if (ncol > 0) {
			end = 1;
		    }
		    break;
		}
	    }
	    
	    if (end == 1) {
		break;
	    }
	} else if (c == '\n') {
	    if (prev_c == '\\' || prev_c == NUL) {
	    } else if (ncol != 0 || nchar != 0) {
		break;
	    }
	} else if ((c == ' ' || c == '\t') && prev_c != '\\') {
	    if (nchar > 0) {
		if (ncol == col) {
		    buf[nchar] = NUL;
		}
		ncol++;
	    }
	    nchar = 0;
	} else if (c == '\\' && prev_c != '\\') {
	    if (nchar > 0) {
		nchar--;
	    }
	} else {
	    if (ncol == col) {
		buf[nchar] = c;
	    }
	    nchar++;

	    if (c == '\\') {
		c = NUL;
	    }
	}

	prev_c = c;
    }

    if (nchar > 0) {
	if (ncol == col) {
	    buf[nchar] = NUL;
	}
	ncol++;
    }

    if (ncol <= col && (c == EOF || n <= 0)) {
	ncol = EOF;
    }

    return ncol;
}

int sgetcol(char *buf, int col, char *line)
{
    int i;
    int n;
    int end;
    int ncol;
    int nchar;
    char c, prev_c;

    buf[0] = NUL;
    prev_c = NUL;
    end = 0;
    ncol = 0;
    nchar = 0;
    for (i = 0;; i++) {
	c = line[i];
	n = 1;
	if (c == NUL) {
	    n = -1;
	    break;
	} else if (c == '#' && prev_c != '\\') {
	    for (i = 0;; i++) {
		c = line[i];
		n = 1;
		if (c == NUL) {
		    end = 1;
		    break;
		} else if (c == '\n') {
		    if (ncol > 0) {
			end = 1;
		    }
		    break;
		}
	    }
	    
	    if (end == 1) {
		n = -1;
		break;
	    }
	} else if (c == '\n') {
	    if (prev_c == '\\' || prev_c == NUL) {
	    } else if (ncol != 0 || nchar != 0) {
		n = -1;
		break;
	    }
	} else if ((c == ' ' || c == '\t') && prev_c != '\\') {
	    if (nchar > 0) {
		if (ncol == col) {
		    buf[nchar] = NUL;
		}
		ncol++;
	    }
	    nchar = 0;
	} else if (c == '\\' && prev_c != '\\') {
	    if (nchar > 0) {
		nchar--;
	    }
	} else {
	    if (ncol == col) {
		buf[nchar] = c;
	    }
	    nchar++;

	    if (c == '\\') {
		c = NUL;
	    }
	}

	prev_c = c;
    }

    if (nchar > 0) {
	if (ncol == col) {
	    buf[nchar] = NUL;
	}
	ncol++;
    }

    if (ncol <= col && (c == EOF || n <= 0)) {
	ncol = EOF;
    }

    return ncol;
}

int fgetline(char *buf, FILE *fp)
{
    char c, prev_c;
    int end;
    int ncol;
    int nchar;
    int pos;
    int n;

    buf[0] = NUL;
    prev_c = NUL;
    end = 0;
    ncol = 0;
    nchar = 0;
    pos = 0;
    while (1) {
	n = fscanf(fp, "%c", &c);
	if (c == EOF || n <= 0) {
	    break;
	} else if (c == '#' && prev_c != '\\') {
	    while (1) {
		n = fscanf(fp, "%c", &c);
		if (c == EOF || n <= 0) {
		    end = 1;
		    break;
		} else if (c == '\n') {
		    if (ncol > 0) {
			end = 1;
		    }
		    break;
		}
	    }
	    
	    if (end == 1) {
		break;
	    }
	} else if (c == '\n') {
	    if (prev_c == '\\') {
		pos--;
		nchar--;
	    } else if (ncol != 0 || nchar != 0) {
		break;
	    }
	} else if (c == ' ' || c == '\t') {
	    if (nchar > 0) {
		ncol++;
	    }
	    buf[pos] = c;
	    pos++;
	    nchar = 0;
	} else {
	    buf[pos] = c;
	    pos++;
	    nchar++;
	}

	prev_c = c;
    }

    buf[pos] = NUL;

    if (pos <= 0 && (c == EOF || n <= 0)) {
	pos = EOF;
    }

    return pos;
}

int getline(char *buf)
{
    char c, prev_c;
    int n;
    int end;
    int ncol;
    int nchar;
    int pos;

    buf[0] = NUL;
    prev_c = NUL;
    end = 0;
    ncol = 0;
    nchar = 0;
    pos = 0;
    while (1) {
	n = fscanf(stdin, "%c", &c);
	if (c == EOF || n <= 0) {
	    break;
	} else if (c == '#' && prev_c != '\\') {
	    while (1) {
		n = fscanf(stdin, "%c", &c);
		if (c == EOF || n <= 0) {
		    end = 1;
		    break;
		} else if (c == '\n') {
		    if (ncol > 0) {
			end = 1;
		    }
		    break;
		}
	    }
	    
	    if (end == 1) {
		break;
	    }
	} else if (c == '\n') {
	    if (prev_c == '\\') {
		pos--;
		nchar--;
	    } else {/*if (ncol != 0 || nchar != 0) {*/
		break;
	    }
	} else if (c == ' ' || c == '\t') {
	    if (nchar > 0) {
		ncol++;
	    }
	    buf[pos] = c;
	    pos++;
	    nchar = 0;
	} else {
	    buf[pos] = c;
	    pos++;
	    nchar++;
	}

	prev_c = c;
    }

    buf[pos] = NUL;

    if (pos <= 0 && (c == EOF || n <= 0)) {
	pos = EOF;
    }

    return pos;
}

char *gets0(char *buf, int size)
{
    char *err;
    char *p;

    err = fgets(buf, size, stdin);
    if ((p = strchr(buf, '\n')) != NULL) {
	*p = NUL;
    }

    return err;
}

int sscanf_setup(char *line, char *name, char *value)
{
    int i;
    int n;
    int end;
    int ncol;
    int nchar;
    char c, prev_c;

    name[0] = NUL;
    value[0] = NUL;
    
    prev_c = NUL;
    end = 0;
    ncol = 0;
    nchar = 0;
    for (i = 0;; i++) {
	c = line[i];
	n = 1;
	if (c == NUL) {
	    n = -1;
	    break;
	} else if (c == '#' && prev_c != '\\') {
	    for (i = 0;; i++) {
		c = line[i];
		n = 1;
		if (c == NUL) {
		    end = 1;
		    break;
		} else if (c == '\n') {
		    if (ncol > 0) {
			end = 1;
		    }
		    break;
		}
	    }
	    
	    if (end == 1) {
		n = -1;
		break;
	    }
	} else if (c == '\n') {
	    if (prev_c == '\\' || prev_c == NUL) {
	    } else if (ncol != 0 || nchar != 0) {
		n = -1;
		break;
	    }
	} else if ((c == ' ' || c == '\t') && prev_c != '\\') {
	    if (nchar > 0) {
		if (ncol == 0) {
		    name[nchar] = NUL;
		    nchar = 0;
		    ncol++;
		} else {
		    value[nchar] = c;
		    nchar++;
		}
	    }
	} else if (c == '\\' && prev_c != '\\') {
	    if (nchar > 0) {
		nchar--;
	    }
	} else {
	    if (ncol == 0) {
		name[nchar] = c;
	    } else {
		value[nchar] = c;
	    }
	    nchar++;

	    if (c == '\\') {
		c = NUL;
	    }
	}

	prev_c = c;
    }

    if (nchar > 0) {
	if (ncol == 0) {
	    name[nchar] = NUL;
	} else {
	    value[nchar] = NUL;
	}
	ncol++;
    }

    if (ncol <= 0 && (c == EOF || n <= 0)) {
	ncol = EOF;
    }

    return ncol;
}

int dvreadcol_txt(const char *filename, int col, DVECTOR vector)
{
    long k;
    char buf[MAX_LINE];
    FILE *fp;

    if (NULL == (fp = fopen(filename, "r"))) {
        fprintf(stderr, "can't open file: %s\n", filename);
	return FAILURE;
    }

    for (k = 0; k < vector->length; k++) {
	if (fgetcol_txt(buf, col, fp) == EOF) {
	    break;
	}
	sscanf(buf, "%lf", &vector->data[k]);
    }

    return SUCCESS;
}

DVECTOR xdvreadcol_txt(const char *filename, int col)
{
    long length;
    DVECTOR vector;

    length = (long)getnumrow_txt(filename);

    if (length <= 0) {
	return NODATA;
    } else {
	vector = xdvzeros(length);
	dvreadcol_txt(filename, col, vector);
    }

    return vector;
}

int writedvector_txt(const char *filename, DVECTOR vector)
{
    long lc;
    char *basicname;
    FILE *fp;

    /* get basic name */
    basicname = xgetbasicname(filename);

    if (streq(basicname, "-") || streq(basicname, "stdout")) {
	fp = stdout;
    } else {
        check_dir(filename);
	if (NULL == (fp = fopen(filename, "wt"))) {
	    fprintf(stderr, "can't open file: %s\n", filename);
	    return FAILURE;
	}
    }

    for (lc = 0; lc < vector->length; lc++) {
	fprintf(fp, "%f\n", vector->data[lc]);
    }

    /* close file */
    if (fp != stdout)
	fclose(fp);

    /* memory free */
    xfree(basicname);

    return SUCCESS;
}

/*
 *	dump data
 */
void svdump(SVECTOR vec)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0) {
	    printf("%d\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0) {
		printf("%d - %di\n", vec->data[k], -vec->imag[k]);
	    } else {
		printf("%d + %di\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    printf("\n");

    return;
}

void lvdump(LVECTOR vec)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0) {
	    printf("%ld\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0) {
		printf("%ld - %ldi\n", vec->data[k], -vec->imag[k]);
	    } else {
		printf("%ld + %ldi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    printf("\n");

    return;
}

void fvdump(FVECTOR vec)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0.0) {
	    printf("%f\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0.0) {
		printf("%f - %fi\n", vec->data[k], -vec->imag[k]);
	    } else {
		printf("%f + %fi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    printf("\n");

    return;
}

void dvdump(DVECTOR vec)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0.0) {
	    printf("%f\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0.0) {
		printf("%f - %fi\n", vec->data[k], -vec->imag[k]);
	    } else {
		printf("%f + %fi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    printf("\n");

    return;
}

void svfdump(SVECTOR vec, FILE *fp)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0) {
	    fprintf(fp, "%d\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0) {
		fprintf(fp, "%d - %di\n", vec->data[k], -vec->imag[k]);
	    } else {
		fprintf(fp, "%d + %di\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    fprintf(fp, "\n");

    return;
}

void lvfdump(LVECTOR vec, FILE *fp)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0) {
	    fprintf(fp, "%ld\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0) {
		fprintf(fp, "%ld - %ldi\n", vec->data[k], -vec->imag[k]);
	    } else {
		fprintf(fp, "%ld + %ldi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    fprintf(fp, "\n");

    return;
}

void fvfdump(FVECTOR vec, FILE *fp)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0.0) {
	    fprintf(fp, "%f\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0.0) {
		fprintf(fp, "%f - %fi\n", vec->data[k], -vec->imag[k]);
	    } else {
		fprintf(fp, "%f + %fi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    fprintf(fp, "\n");

    return;
}

void dvfdump(DVECTOR vec, FILE *fp)
{
    long k;

    for (k = 0; k < vec->length; k++) {
	if (vec->imag == NULL || vec->imag[k] == 0.0) {
	    fprintf(fp, "%f\n", vec->data[k]);
	} else {
	    if (vec->imag[k] < 0.0) {
		fprintf(fp, "%f - %fi\n", vec->data[k], -vec->imag[k]);
	    } else {
		fprintf(fp, "%f + %fi\n", vec->data[k], vec->imag[k]);
	    }
	}
    }
    fprintf(fp, "\n");

    return;
}

void lmfdump(LMATRIX mat, FILE *fp)
{
    long k, l;

    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) {
	    if (mat->imag == NULL || mat->imag[k][l] == 0) {
		fprintf(fp, "%ld  ", mat->data[k][l]);
	    } else {
		if (mat->imag[k][l] < 0) {
		    fprintf(fp, "%ld - %ldi  ", mat->data[k][l], -mat->imag[k][l]);
		} else {
		    fprintf(fp, "%ld + %ldi  ", mat->data[k][l], mat->imag[k][l]);
		}
	    }
	}
	fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    return;
}

void dmfdump(DMATRIX mat, FILE *fp)
{
    long k, l;

    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) {
	    if (mat->imag == NULL || mat->imag[k][l] == 0.0) {
		fprintf(fp, "%f  ", mat->data[k][l]);
	    } else {
		if (mat->imag[k][l] < 0.0) {
		    fprintf(fp, "%f - %fi  ", mat->data[k][l], -mat->imag[k][l]);
		} else {
		    fprintf(fp, "%f + %fi  ", mat->data[k][l], mat->imag[k][l]);
		}
	    }
	}
	fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    return;
}

#ifdef VARARGS
void dvnfdump(va_alist)
va_dcl
{
    FILE *fp;
    int nrow, ncol;
    int end;
    char buf[MAX_LINE];
    char line[MAX_PATHNAME];
    DVECTOR p;
    va_list argp;

    end = 0;
    nrow = 0;
    while (1) {
	line[0] = NUL;

	va_start(argp);
	fp = va_arg(argp, FILE *);

	ncol = 0;
	while ((p = va_arg(argp, DVECTOR)) != NULL) {
	    if (nrow >= p->length) {
		end = 1;
		break;
	    } else {
		sprintf(buf, "%f ", p->data[nrow]);
		strcat(line, buf);
	    }
	    ncol++;
	}

	va_end(argp);

	if (end == 1) {
	    break;
	} else {
	    fprintf(fp, "%s\n", line);
	}
	nrow++;
    }

    return;
}
#else
void dvnfdump(FILE *fp, DVECTOR vec, ...)
{
    int nrow, ncol;
    int end;
    char buf[MAX_LINE];
    char line[MAX_PATHNAME];
    DVECTOR p;
    va_list argp;

    end = 0;
    nrow = 0;
    while (1) {
	line[0] = NUL;

	va_start(argp, vec);

	ncol = 0;
	while ((p = va_arg(argp, DVECTOR)) != NULL) {
	    if (nrow >= p->length) {
		end = 1;
		break;
	    } else {
		sprintf(buf, "%f ", p->data[nrow]);
		strcat(line, buf);
	    }
	    ncol++;
	}

	va_end(argp);

	if (end == 1) {
	    break;
	} else {
	    fprintf(fp, "%s\n", line);
	}
	nrow++;
    }

    return;
}
#endif


void check_dir(const char *file)
{
    int k, len;
    char dir[MAX_LINE] = "";
    char tmp[MAX_MESSAGE] = "";

    len = strlen(file);
    dir[0] = file[0];
    for (k = 1; k < len; k++) {
	dir[k] = file[k];
//	if ((dir[k] == '/' && dir[k - 1] != '/') ||
//	    (dir[k] == '\' && dir[k - 1] != '\')) {
	if (dir[k] == '/' && dir[k - 1] != '/') {
	    dir[k + 1] = '\0';
	    sprintf(tmp, "if [ ! -r %s ];then\n mkdir %s\n fi", dir, dir);
	    system(tmp);
	}
    }

    return;
}


long get_dnum_file(const char *file, long dim)
{
    long length, dnum;

    if ((length = getsiglen(file, 0, double)) <= 0) {
	fprintf(stderr, "Error: file format %s\n", file);
	exit(1);
    }
    if (length % dim != 0) {
	fprintf(stderr, "Error: file format: %s\n", file);
	exit(1);
    }
    dnum = length / dim;

    return dnum;
}

long get_fnum_file(const char *file, long dim)
{
    long length, fnum;

    if ((length = getsiglen(file, 0, float)) <= 0) {
	fprintf(stderr, "Error: file format %s\n", file);
	exit(1);
    }
    if (length % dim != 0) {
	fprintf(stderr, "Error: file format: %s\n", file);
	exit(1);
    }
    fnum = length / dim;

    return fnum;
}

long get_lnum_file(const char *file, long dim)
{
    long length, fnum;

    if ((length = getsiglen(file, 0, long)) <= 0) {
	fprintf(stderr, "Error: file format %s\n", file);
	exit(1);
    }
    if (length % dim != 0) {
	fprintf(stderr, "Error: file format: %s\n", file);
	exit(1);
    }
    fnum = length / dim;

    return fnum;
}

