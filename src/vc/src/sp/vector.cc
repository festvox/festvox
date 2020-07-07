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
#include <math.h>

#include "../include/defs.h"
#include "../include/memory.h"
#include "../include/vector.h"

/*
 *	allocate and free memory
 */
SVECTOR xsvalloc(long length)
{
    SVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct SVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), short);
    x->imag = NULL;
    x->length = length;

    return x;
}

LVECTOR xlvalloc(long length)
{
    LVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct LVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), long);
    x->imag = NULL;
    x->length = length;

    return x;
}

FVECTOR xfvalloc(long length)
{
    FVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct FVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), float);
    x->imag = NULL;
    x->length = length;

    return x;
}

DVECTOR xdvalloc(long length)
{
    DVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct DVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), double);
    x->imag = NULL;
    x->length = length;

    return x;
}

void xsvfree(SVECTOR x)
{
    if (x != NULL) {
	if (x->data != NULL) {
	    xfree(x->data);
	}
	if (x->imag != NULL) {
	    xfree(x->imag);
	}
	xfree(x);
    }

    return;
}

void xlvfree(LVECTOR x)
{
    if (x != NULL) {
	if (x->data != NULL) {
	    xfree(x->data);
	}
	if (x->imag != NULL) {
	    xfree(x->imag);
	}
	xfree(x);
    }

    return;
}

void xfvfree(FVECTOR x)
{
    if (x != NULL) {
	if (x->data != NULL) {
	    xfree(x->data);
	}
	if (x->imag != NULL) {
	    xfree(x->imag);
	}
	xfree(x);
    }

    return;
}

void xdvfree(DVECTOR x)
{
    if (x != NULL) {
	if (x->data != NULL) {
	    xfree(x->data);
	}
	if (x->imag != NULL) {
	    xfree(x->imag);
	}
	xfree(x);
    }

    return;
}

void svialloc(SVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    x->imag = xalloc(x->length, short);

    return;
}

void lvialloc(LVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    x->imag = xalloc(x->length, long);

    return;
}

void fvialloc(FVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    x->imag = xalloc(x->length, float);

    return;
}

void dvialloc(DVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    x->imag = xalloc(x->length, double);

    return;
}

void svifree(SVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    
    return;
}

void lvifree(LVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    
    return;
}

void fvifree(FVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    
    return;
}

void dvifree(DVECTOR x)
{
    if (x->imag != NULL) {
	xfree(x->imag);
    }
    
    return;
}

SVECTOR xsvrialloc(long length)
{
    SVECTOR x;

    x = xsvalloc(length);
    svialloc(x);

    return x;
}

LVECTOR xlvrialloc(long length)
{
    LVECTOR x;

    x = xlvalloc(length);
    lvialloc(x);

    return x;
}

FVECTOR xfvrialloc(long length)
{
    FVECTOR x;

    x = xfvalloc(length);
    fvialloc(x);

    return x;
}

DVECTOR xdvrialloc(long length)
{
    DVECTOR x;

    x = xdvalloc(length);
    dvialloc(x);

    return x;
}

SVECTOR xsvrealloc(SVECTOR x, long length)
{
    long k;
    
    if (x == NODATA) {
	x = xsvzeros(length);
    } else {
	if (length > x->length) {
	    x->data = xrealloc(x->data, length, short);
#ifdef EXIST_REALLOC_BUG
	    /* I don't know why +1 is necessary */
	    for (k = x->length + 1; k < length; k++) {
		x->data[k] = 0;
	    }
#else
	    for (k = x->length; k < length; k++) {
		x->data[k] = 0;
	    }
#endif
	}

	x->length = length;
    }

    return x;
}

LVECTOR xlvrealloc(LVECTOR x, long length)
{
    long k;
    
    if (x == NODATA) {
	x = xlvzeros(length);
    } else {
	if (length > x->length) {
	    x->data = xrealloc(x->data, length, long);
#ifdef EXIST_REALLOC_BUG
	    /* I don't know why +1 is necessary */
	    for (k = x->length + 1; k < length; k++) {
		x->data[k] = 0;
	    }
#else
	    for (k = x->length; k < length; k++) {
		x->data[k] = 0;
	    }
#endif
	}
	x->length = length;
    }

    return x;
}

FVECTOR xfvrealloc(FVECTOR x, long length)
{
    long k;
    
    if (x == NODATA) {
	x = xfvzeros(length);
    } else {
	if (length > x->length) {
	    x->data = xrealloc(x->data, length, float);
#ifdef EXIST_REALLOC_BUG
	    /* I don't know why +1 is necessary */
	    for (k = x->length + 1; k < length; k++) {
		x->data[k] = 0.0;
	    }
#else
	    for (k = x->length; k < length; k++) {
		x->data[k] = 0.0;
	    }
#endif
	}
	x->length = length;
    }

    return x;
}

DVECTOR xdvrealloc(DVECTOR x, long length)
{
    long k;
    
    if (x == NODATA) {
	x = xdvzeros(length);
    } else {
	if (length > x->length) {
	    x->data = xrealloc(x->data, length, double);
#ifdef EXIST_REALLOC_BUG
	    /* I don't know why +1 is necessary */
	    for (k = x->length + 1; k < length; k++) {
		x->data[k] = 0.0;
	    }
#else
	    for (k = x->length; k < length; k++) {
		x->data[k] = 0.0;
	    }
#endif
	}
	x->length = length;
    }

    return x;
}

SVECTORS xsvsalloc(long num)
{
    long k;
    SVECTORS xs;

    xs = xalloc(1, struct SVECTORS_STRUCT);
    xs->vector = xalloc(MAX(num, 1), SVECTOR);
    xs->num_vector = num;
    
    for (k = 0; k < xs->num_vector; k++) {
	xs->vector[k] = NODATA;
    }

    return xs;
}

LVECTORS xlvsalloc(long num)
{
    long k;
    LVECTORS xs;

    xs = xalloc(1, struct LVECTORS_STRUCT);
    xs->vector = xalloc(MAX(num, 1), LVECTOR);
    xs->num_vector = num;
    
    for (k = 0; k < xs->num_vector; k++) {
	xs->vector[k] = NODATA;
    }

    return xs;
}

FVECTORS xfvsalloc(long num)
{
    long k;
    FVECTORS xs;

    xs = xalloc(1, struct FVECTORS_STRUCT);
    xs->vector = xalloc(MAX(num, 1), FVECTOR);
    xs->num_vector = num;
    
    for (k = 0; k < xs->num_vector; k++) {
	xs->vector[k] = NODATA;
    }

    return xs;
}

DVECTORS xdvsalloc(long num)
{
    long k;
    DVECTORS xs;

    xs = xalloc(1, struct DVECTORS_STRUCT);
    xs->vector = xalloc(MAX(num, 1), DVECTOR);
    xs->num_vector = num;
    
    for (k = 0; k < xs->num_vector; k++) {
	xs->vector[k] = NODATA;
    }

    return xs;
}

void xsvsfree(SVECTORS xs)
{
    long k;

    if (xs != NULL) {
	if (xs->vector != NULL) {
	    for (k = 0; k < xs->num_vector; k++) {
		if (xs->vector[k] != NODATA) {
		    xsvfree(xs->vector[k]);
		}
	    }
	    xfree(xs->vector);
	}
	xfree(xs);
    }

    return;
}

void xlvsfree(LVECTORS xs)
{
    long k;

    if (xs != NULL) {
	if (xs->vector != NULL) {
	    for (k = 0; k < xs->num_vector; k++) {
		if (xs->vector[k] != NODATA) {
		    xlvfree(xs->vector[k]);
		}
	    }
	    xfree(xs->vector);
	}
	xfree(xs);
    }

    return;
}

void xfvsfree(FVECTORS xs)
{
    long k;

    if (xs != NULL) {
	if (xs->vector != NULL) {
	    for (k = 0; k < xs->num_vector; k++) {
		if (xs->vector[k] != NODATA) {
		    xfvfree(xs->vector[k]);
		}
	    }
	    xfree(xs->vector);
	}
	xfree(xs);
    }

    return;
}

void xdvsfree(DVECTORS xs)
{
    long k;

    if (xs != NULL) {
	if (xs->vector != NULL) {
	    for (k = 0; k < xs->num_vector; k++) {
		if (xs->vector[k] != NODATA) {
		    xdvfree(xs->vector[k]);
		}
	    }
	    xfree(xs->vector);
	}
	xfree(xs);
    }

    return;
}

SVECTOR xsvcplx(SVECTOR xr, SVECTOR xi)
{
    long k;
    SVECTOR z;

    if (xr != NODATA && xi != NODATA) {
	z = xsvrialloc(MIN(xr->length, xi->length));
    } else if (xr != NODATA) {
	z = xsvrialloc(xr->length);
    } else if (xi != NODATA) {
	z = xsvrialloc(xi->length);
    } else {
#if 0
	z = xsvnull();
	return z;
#else
	return NODATA;
#endif
    }

    for (k = 0; k < z->length; k++) {
	if (xr != NODATA) {
	    z->data[k] = xr->data[k];
	} else {
	    z->data[k] = 0;
	}
	if (xi != NODATA) {
	    z->imag[k] = xi->data[k];
	} else {
	    z->imag[k] = 0;
	}
    }

    return z;
}

LVECTOR xlvcplx(LVECTOR xr, LVECTOR xi)
{
    long k;
    LVECTOR z;

    if (xr != NODATA && xi != NODATA) {
	z = xlvrialloc(MIN(xr->length, xi->length));
    } else if (xr != NODATA) {
	z = xlvrialloc(xr->length);
    } else if (xi != NODATA) {
	z = xlvrialloc(xi->length);
    } else {
#if 0
	z = xlvnull();
	return z;
#else
	return NODATA;
#endif
    }

    for (k = 0; k < z->length; k++) {
	if (xr != NODATA) {
	    z->data[k] = xr->data[k];
	} else {
	    z->data[k] = 0;
	}
	if (xi != NODATA) {
	    z->imag[k] = xi->data[k];
	} else {
	    z->imag[k] = 0;
	}
    }

    return z;
}

FVECTOR xfvcplx(FVECTOR xr, FVECTOR xi)
{
    long k;
    FVECTOR z;

    if (xr != NODATA && xi != NODATA) {
	z = xfvrialloc(MIN(xr->length, xi->length));
    } else if (xr != NODATA) {
	z = xfvrialloc(xr->length);
    } else if (xi != NODATA) {
	z = xfvrialloc(xi->length);
    } else {
#if 0
	z = xfvnull();
	return z;
#else
	return NODATA;
#endif
    }

    for (k = 0; k < z->length; k++) {
	if (xr != NODATA) {
	    z->data[k] = xr->data[k];
	} else {
	    z->data[k] = 0.0;
	}
	if (xi != NODATA) {
	    z->imag[k] = xi->data[k];
	} else {
	    z->imag[k] = 0.0;
	}
    }

    return z;
}

DVECTOR xdvcplx(DVECTOR xr, DVECTOR xi)
{
    long k;
    DVECTOR z;

    if (xr != NODATA && xi != NODATA) {
	z = xdvrialloc(MIN(xr->length, xi->length));
    } else if (xr != NODATA) {
	z = xdvrialloc(xr->length);
    } else if (xi != NODATA) {
	z = xdvrialloc(xi->length);
    } else {
#if 0
	z = xdvnull();
	return z;
#else
	return NODATA;
#endif
    }

    for (k = 0; k < z->length; k++) {
	if (xr != NODATA) {
	    z->data[k] = xr->data[k];
	} else {
	    z->data[k] = 0.0;
	}
	if (xi != NODATA) {
	    z->imag[k] = xi->data[k];
	} else {
	    z->imag[k] = 0.0;
	}
    }

    return z;
}

void svreal(SVECTOR x)
{
    if (x->imag != NULL) {
	svifree(x);
    }

    return;
} 

void lvreal(LVECTOR x)
{
    if (x->imag != NULL) {
	lvifree(x);
    }

    return;
} 

void fvreal(FVECTOR x)
{
    if (x->imag != NULL) {
	fvifree(x);
    }

    return;
} 

void dvreal(DVECTOR x)
{
    if (x->imag != NULL) {
	dvifree(x);
    }

    return;
} 

void svimag(SVECTOR x)
{
    if (x->imag == NULL) {
	svzeros(x, x->length);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;

    return;
} 

void lvimag(LVECTOR x)
{
    if (x->imag == NULL) {
	lvzeros(x, x->length);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;

    return;
} 

void fvimag(FVECTOR x)
{
    if (x->imag == NULL) {
	fvzeros(x, x->length);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;

    return;
} 

void dvimag(DVECTOR x)
{
    if (x->imag == NULL) {
	dvzeros(x, x->length);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;

    return;
} 

SVECTOR xsvreal(SVECTOR x)
{
    long k;
    SVECTOR y;

    y = xsvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

LVECTOR xlvreal(LVECTOR x)
{
    long k;
    LVECTOR y;

    y = xlvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

FVECTOR xfvreal(FVECTOR x)
{
    long k;
    FVECTOR y;

    y = xfvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

DVECTOR xdvreal(DVECTOR x)
{
    long k;
    DVECTOR y;

    y = xdvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

SVECTOR xsvimag(SVECTOR x)
{
    long k;
    SVECTOR y;

    if (x->imag == NULL) {
	y = xsvzeros(x->length);

	return y;
    }
    y = xsvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

LVECTOR xlvimag(LVECTOR x)
{
    long k;
    LVECTOR y;

    if (x->imag == NULL) {
	y = xlvzeros(x->length);

	return y;
    }
    y = xlvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

FVECTOR xfvimag(FVECTOR x)
{
    long k;
    FVECTOR y;

    if (x->imag == NULL) {
	y = xfvzeros(x->length);

	return y;
    }
    y = xfvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

DVECTOR xdvimag(DVECTOR x)
{
    long k;
    DVECTOR y;

    if (x->imag == NULL) {
	y = xdvzeros(x->length);

	return y;
    }
    y = xdvalloc(x->length);

    /* copy data */
    for (k = 0; k < x->length; k++) {
	y->data[k] = x->data[k];
    }

    return y;
} 

void svconj(SVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	return;
    }

    for (k = 0; k < x->length; k++) {
	x->imag[k] = -x->imag[k];
    }

    return;
}

void lvconj(LVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	return;
    }

    for (k = 0; k < x->length; k++) {
	x->imag[k] = -x->imag[k];
    }

    return;
}

void fvconj(FVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	return;
    }

    for (k = 0; k < x->length; k++) {
	x->imag[k] = -x->imag[k];
    }

    return;
}

void dvconj(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	return;
    }

    for (k = 0; k < x->length; k++) {
	x->imag[k] = -x->imag[k];
    }

    return;
}

SVECTOR xsvconj(SVECTOR x)
{
    SVECTOR y;

    y = xsvclone(x);
    svconj(y);

    return y;
}

LVECTOR xlvconj(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvconj(y);

    return y;
}

FVECTOR xfvconj(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvconj(y);

    return y;
}

DVECTOR xdvconj(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvconj(y);

    return y;
}

void svriswap(SVECTOR x)
{
    short *p;

    if (x->imag == NULL) {
	svizeros(x, x->length);
    }

    p = x->data;
    x->data = x->imag;
    x->imag = p;

    return;
}

void lvriswap(LVECTOR x)
{
    long *p;

    if (x->imag == NULL) {
	lvizeros(x, x->length);
    }

    p = x->data;
    x->data = x->imag;
    x->imag = p;

    return;
}

void fvriswap(FVECTOR x)
{
    float *p;

    if (x->imag == NULL) {
	fvizeros(x, x->length);
    }

    p = x->data;
    x->data = x->imag;
    x->imag = p;

    return;
}

void dvriswap(DVECTOR x)
{
    double *p;

    if (x->imag == NULL) {
	dvizeros(x, x->length);
    }

    p = x->data;
    x->data = x->imag;
    x->imag = p;

    return;
}

SVECTOR xsvriswap(SVECTOR x)
{
    SVECTOR y;

    y = xsvclone(x);
    svriswap(y);

    return y;
}

LVECTOR xlvriswap(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvriswap(y);

    return y;
}

FVECTOR xfvriswap(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvriswap(y);

    return y;
}

DVECTOR xdvriswap(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvriswap(y);

    return y;
}

/*
 *	copy data of x into y
 */
void svcopy(SVECTOR y, SVECTOR x)
{
    long k;
    long length;

    length = MIN(y->length, x->length);

    /* copy data */
    for (k = 0; k < length; k++) {
	y->data[k] = x->data[k];
    }
    if (x->imag != NULL && y->imag != NULL) {
	for (k = 0; k < length; k++) {
	    y->imag[k] = x->imag[k];
	}
    }

    return;
}

void lvcopy(LVECTOR y, LVECTOR x)
{
    long k;
    long length;

    length = MIN(y->length, x->length);

    /* copy data */
    for (k = 0; k < length; k++) {
	y->data[k] = x->data[k];
    }
    if (x->imag != NULL && y->imag != NULL) {
	for (k = 0; k < length; k++) {
	    y->imag[k] = x->imag[k];
	}
    }

    return;
}

void fvcopy(FVECTOR y, FVECTOR x)
{
    long k;
    long length;

    length = MIN(y->length, x->length);

    /* copy data */
    for (k = 0; k < length; k++) {
	y->data[k] = x->data[k];
    }
    if (x->imag != NULL && y->imag != NULL) {
	for (k = 0; k < length; k++) {
	    y->imag[k] = x->imag[k];
	}
    }

    return;
}

void dvcopy(DVECTOR y, DVECTOR x)
{
    long k;
    long length;

    length = MIN(y->length, x->length);

    /* copy data */
    for (k = 0; k < length; k++) {
	y->data[k] = x->data[k];
    }
    if (x->imag != NULL && y->imag != NULL) {
	for (k = 0; k < length; k++) {
	    y->imag[k] = x->imag[k];
	}
    }

    return;
}

/*
 *	make clone of input vector
 */
SVECTOR xsvclone(SVECTOR x)
{
    SVECTOR y;

    /* memory allocate */
    y = xsvalloc(x->length);
    if (x->imag != NULL) {
	svialloc(y);
    }
    
    /* copy data */
    svcopy(y, x);

    return y;
}

LVECTOR xlvclone(LVECTOR x)
{
    LVECTOR y;

    /* memory allocate */
    y = xlvalloc(x->length);
    if (x->imag != NULL) {
	lvialloc(y);
    }
    
    /* copy data */
    lvcopy(y, x);

    return y;
}

FVECTOR xfvclone(FVECTOR x)
{
    FVECTOR y;

    /* memory allocate */
    y = xfvalloc(x->length);
    if (x->imag != NULL) {
	fvialloc(y);
    }
    
    /* copy data */
    fvcopy(y, x);

    return y;
}

DVECTOR xdvclone(DVECTOR x)
{
    DVECTOR y;

    /* memory allocate */
    y = xdvalloc(x->length);
    if (x->imag != NULL) {
	dvialloc(y);
    }
    
    /* copy data */
    dvcopy(y, x);

    return y;
}

/*
 *	concatenate vector
 */
SVECTOR xsvcat(SVECTOR x, SVECTOR y)
{
    long k;
    SVECTOR z;

    /* memory allocate */
    z = xsvalloc(x->length + y->length);
    if (x->imag != NULL || y->imag != NULL) {
	svialloc(z);
    }
    
    /* concatenate data */
    for (k = 0; k < z->length; k++) {
	if (k < x->length) {
	    z->data[k] = x->data[k];
	} else {
	    z->data[k] = y->data[k - x->length];
	}
    }
    if (z->imag != NULL) {
	for (k = 0; k < z->length; k++) {
	    if (k < x->length) {
		if (x->imag != NULL) {
		    z->imag[k] = x->imag[k];
		} else {
		    z->imag[k] = 0;
		}
	    } else {
		if (y->imag != NULL) {
		    z->imag[k] = y->imag[k - x->length];
		} else {
		    z->imag[k] = 0;
		}
	    }
	}
    }

    return z;
}

LVECTOR xlvcat(LVECTOR x, LVECTOR y)
{
    long k;
    LVECTOR z;

    /* memory allocate */
    z = xlvalloc(x->length + y->length);
    if (x->imag != NULL || y->imag != NULL) {
	lvialloc(z);
    }
    
    /* concatenate data */
    for (k = 0; k < z->length; k++) {
	if (k < x->length) {
	    z->data[k] = x->data[k];
	} else {
	    z->data[k] = y->data[k - x->length];
	}
    }
    if (z->imag != NULL) {
	for (k = 0; k < z->length; k++) {
	    if (k < x->length) {
		if (x->imag != NULL) {
		    z->imag[k] = x->imag[k];
		} else {
		    z->imag[k] = 0;
		}
	    } else {
		if (y->imag != NULL) {
		    z->imag[k] = y->imag[k - x->length];
		} else {
		    z->imag[k] = 0;
		}
	    }
	}
    }

    return z;
}

FVECTOR xfvcat(FVECTOR x, FVECTOR y)
{
    long k;
    FVECTOR z;

    /* memory allocate */
    z = xfvalloc(x->length + y->length);
    if (x->imag != NULL || y->imag != NULL) {
	fvialloc(z);
    }
    
    /* concatenate data */
    for (k = 0; k < z->length; k++) {
	if (k < x->length) {
	    z->data[k] = x->data[k];
	} else {
	    z->data[k] = y->data[k - x->length];
	}
    }
    if (z->imag != NULL) {
	for (k = 0; k < z->length; k++) {
	    if (k < x->length) {
		if (x->imag != NULL) {
		    z->imag[k] = x->imag[k];
		} else {
		    z->imag[k] = 0.0;
		}
	    } else {
		if (y->imag != NULL) {
		    z->imag[k] = y->imag[k - x->length];
		} else {
		    z->imag[k] = 0.0;
		}
	    }
	}
    }

    return z;
}

DVECTOR xdvcat(DVECTOR x, DVECTOR y)
{
    long k;
    DVECTOR z;

    /* memory allocate */
    z = xdvalloc(x->length + y->length);
    if (x->imag != NULL || y->imag != NULL) {
	dvialloc(z);
    }
    
    /* concatenate data */
    for (k = 0; k < z->length; k++) {
	if (k < x->length) {
	    z->data[k] = x->data[k];
	} else {
	    z->data[k] = y->data[k - x->length];
	}
    }
    if (z->imag != NULL) {
	for (k = 0; k < z->length; k++) {
	    if (k < x->length) {
		if (x->imag != NULL) {
		    z->imag[k] = x->imag[k];
		} else {
		    z->imag[k] = 0.0;
		}
	    } else {
		if (y->imag != NULL) {
		    z->imag[k] = y->imag[k - x->length];
		} else {
		    z->imag[k] = 0.0;
		}
	    }
	}
    }

    return z;
}

/*
 *	get initialized vector
 */
void svinit(SVECTOR x, long j, long incr, long n)
{
    long k;
    long num;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->data[k] = (short)(j + (k * incr));
    }

    return;
}

void lvinit(LVECTOR x, long j, long incr, long n)
{
    long k;
    long num;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->data[k] = j + (k * incr);
    }

    return;
}

void fvinit(FVECTOR x, float j, float incr, float n)
{
    long k;
    long num;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->data[k] = j + (k * incr);
    }

    return;
}

void dvinit(DVECTOR x, double j, double incr, double n)
{
    long k;
    long num;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->data[k] = j + (k * incr);
    }

    return;
}

SVECTOR xsvinit(long j, long incr, long n)
{
    long k;
    long num;
    SVECTOR x;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	x = xsvnull();
	return x;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
	    x = xsvnull();
	    return x;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    
    /* memory allocate */
    x = xsvalloc(num);

    /* initailize data */
    for (k = 0; k < x->length; k++) {
	x->data[k] = (short)(j + (k * incr));
    }

    return x;
}

LVECTOR xlvinit(long j, long incr, long n)
{
    long k;
    long num;
    LVECTOR x;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	x = xlvnull();
	return x;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
	    x = xlvnull();
	    return x;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    
    /* memory allocate */
    x = xlvalloc(num);

    /* initailize data */
    for (k = 0; k < x->length; k++) {
	x->data[k] = j + (k * incr);
    }

    return x;
}

FVECTOR xfvinit(float j, float incr, float n)
{
    long k;
    long num;
    FVECTOR x;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	x = xfvnull();
	return x;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
	    x = xfvnull();
	    return x;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* memory allocate */
    x = xfvalloc(num);

    /* initailize data */
    for (k = 0; k < x->length; k++) {
	x->data[k] = j + (k * incr);
    }

    return x;
}

DVECTOR xdvinit(double j, double incr, double n)
{
    long k;
    long num;
    DVECTOR x;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	x = xdvnull();
	return x;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
	    x = xdvnull();
	    return x;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* memory allocate */
    x = xdvalloc(num);

    /* initailize data */
    for (k = 0; k < x->length; k++) {
	x->data[k] = j + (k * incr);
    }

    return x;
}

void sviinit(SVECTOR x, long j, long incr, long n)
{
    long k;
    long num;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    if (x->imag == NULL) {
	svialloc(x);
	svizeros(x, x->length);
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->imag[k] = (short)(j + (k * incr));
    }

    return;
}

void lviinit(LVECTOR x, long j, long incr, long n)
{
    long k;
    long num;

    if ((incr > 0 && j > n) || (incr < 0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0) {
	num = n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((n - j) / incr) + 1;
    }
    if (x->imag == NULL) {
	lvialloc(x);
	lvizeros(x, x->length);
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->imag[k] = j + (k * incr);
    }

    return;
}

void fviinit(FVECTOR x, float j, float incr, float n)
{
    long k;
    long num;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    if (x->imag == NULL) {
	fvialloc(x);
	fvizeros(x, x->length);
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->imag[k] = j + (k * incr);
    }

    return;
}

void dviinit(DVECTOR x, double j, double incr, double n)
{
    long k;
    long num;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
	return;
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    num = x->length;
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    if (x->imag == NULL) {
	dvialloc(x);
	dvizeros(x, x->length);
    }
    
    /* initailize data */
    for (k = 0; k < num; k++) {
	if (k >= x->length) {
	    break;
	}
	x->imag[k] = j + (k * incr);
    }

    return;
}

SVECTOR xsvriinit(long j, long incr, long n)
{
    SVECTOR x;

    x = xsvinit(j, incr, n);
    svialloc(x);
    sviinit(x, j, incr, n);

    return x;
}

LVECTOR xlvriinit(long j, long incr, long n)
{
    LVECTOR x;

    x = xlvinit(j, incr, n);
    lvialloc(x);
    lviinit(x, j, incr, n);

    return x;
}

FVECTOR xfvriinit(float j, float incr, float n)
{
    FVECTOR x;

    x = xfvinit(j, incr, n);
    fvialloc(x);
    fviinit(x, j, incr, n);

    return x;
}

DVECTOR xdvriinit(double j, double incr, double n)
{
    DVECTOR x;

    x = xdvinit(j, incr, n);
    dvialloc(x);
    dviinit(x, j, incr, n);

    return x;
}

/*
 *	cut vector
 */
SVECTOR xsvcut(SVECTOR x, long offset, long length)
{
    long k;
    long pos;
    SVECTOR y;
    
    y = xsvalloc(length);
    if (x->imag != NULL) {
	svialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < x->length) {
	    y->data[k] = x->data[pos];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[pos];
	    }
	} else {
	    y->data[k] = 0;
	    if (y->imag != NULL) {
		y->imag[k] = 0;
	    }
	}
    }

    return y;
}

LVECTOR xlvcut(LVECTOR x, long offset, long length)
{
    long k;
    long pos;
    LVECTOR y;
    
    y = xlvalloc(length);
    if (x->imag != NULL) {
	lvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < x->length) {
	    y->data[k] = x->data[pos];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[pos];
	    }
	} else {
	    y->data[k] = 0;
	    if (y->imag != NULL) {
		y->imag[k] = 0;
	    }
	}
    }

    return y;
}

FVECTOR xfvcut(FVECTOR x, long offset, long length)
{
    long k;
    long pos;
    FVECTOR y;
    
    y = xfvalloc(length);
    if (x->imag != NULL) {
	fvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < x->length) {
	    y->data[k] = x->data[pos];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[pos];
	    }
	} else {
	    y->data[k] = 0.0;
	    if (y->imag != NULL) {
		y->imag[k] = 0.0;
	    }
	}
    }

    return y;
}

DVECTOR xdvcut(DVECTOR x, long offset, long length)
{
    long k;
    long pos;
    DVECTOR y;
    
    y = xdvalloc(length);
    if (x->imag != NULL) {
	dvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < x->length) {
	    y->data[k] = x->data[pos];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[pos];
	    }
	} else {
	    y->data[k] = 0.0;
	    if (y->imag != NULL) {
		y->imag[k] = 0.0;
	    }
	}
    }

    return y;
}

/*
 *	paste data of x in y
 *	if length equal to 0, length is set length of x
 */
void svpaste(SVECTOR y, SVECTOR x, long offset, long length, int overlap)
{
    long k;
    long pos;

    if (length <= 0 || length > x->length) {
	length = x->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] += x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] += x->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] = x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] = x->imag[k];
		}
	    }
	}
    }

    return;
}

void lvpaste(LVECTOR y, LVECTOR x, long offset, long length, int overlap)
{
    long k;
    long pos;
    
    if (length <= 0 || length > x->length) {
	length = x->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] += x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] += x->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] = x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] = x->imag[k];
		}
	    }
	}
    }

    return;
}

void fvpaste(FVECTOR y, FVECTOR x, long offset, long length, int overlap)
{
    long k;
    long pos;
    
    if (length <= 0 || length > x->length) {
	length = x->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] += x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] += x->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] = x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] = x->imag[k];
		}
	    }
	}
    }

    return;
}

void dvpaste(DVECTOR y, DVECTOR x, long offset, long length, int overlap)
{
    long k;
    long pos;
    
    if (length <= 0 || length > x->length) {
	length = x->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] += x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] += x->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= y->length) {
		break;
	    }
	    if (pos >= 0) {
		y->data[pos] = x->data[k];
		if (x->imag != NULL && y->imag != NULL) {
		    y->imag[pos] = x->imag[k];
		}
	    }
	}
    }

    return;
}

/*
 *	convert type of vector
 */
LVECTOR xsvtol(SVECTOR x)
{
    long k;
    LVECTOR y;

    y = xlvalloc(x->length);
    if (x->imag != NULL) {
	lvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (long)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (long)x->imag[k];
	}
    }

    return y;
}

FVECTOR xsvtof(SVECTOR x)
{
    long k;
    FVECTOR y;

    y = xfvalloc(x->length);
    if (x->imag != NULL) {
	fvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (float)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (float)x->imag[k];
	}
    }

    return y;
}

DVECTOR xsvtod(SVECTOR x)
{
    long k;
    DVECTOR y;

    y = xdvalloc(x->length);
    if (x->imag != NULL) {
	dvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (double)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (double)x->imag[k];
	}
    }

    return y;
}

DVECTOR xfvtod(FVECTOR x)
{
    long k;
    DVECTOR y;

    y = xdvalloc(x->length);
    if (x->imag != NULL) {
	dvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (double)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (double)x->imag[k];
	}
    }

    return y;
}

SVECTOR xdvtos(DVECTOR x)
{
    long k;
    SVECTOR y;

    y = xsvalloc(x->length);
    if (x->imag != NULL) {
	svialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (short)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (short)x->imag[k];
	}
    }

    return y;
}

LVECTOR xdvtol(DVECTOR x)
{
    long k;
    LVECTOR y;

    y = xlvalloc(x->length);
    if (x->imag != NULL) {
	lvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (long)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (long)x->imag[k];
	}
    }

    return y;
}

FVECTOR xdvtof(DVECTOR x)
{
    long k;
    FVECTOR y;

    y = xfvalloc(x->length);
    if (x->imag != NULL) {
	fvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = (float)x->data[k];
    }
    if (y->imag != NULL) {
	for (k = 0; k < y->length; k++) {
	    y->imag[k] = (float)x->imag[k];
	}
    }

    return y;
}

SVECTOR xsvset(short *data, long length)
{
    SVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct SVECTOR_STRUCT);
    x->data = data;
    x->imag = NULL;
    x->length = length;
    
    return x;
}

SVECTOR xsvsetnew(short *data, long length)
{
    long k;
    SVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct SVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), short);
    for (k = 0; k < length; k++) {
	x->data[k] = data[k];
    }
    x->imag = NULL;
    x->length = length;
    
    return x;
}

LVECTOR xlvset(long *data, long length)
{
    LVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct LVECTOR_STRUCT);
    x->data = data;
    x->imag = NULL;
    x->length = length;
    
    return x;
}

LVECTOR xlvsetnew(long *data, long length)
{
    long k;
    LVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct LVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), long);
    for (k = 0; k < length; k++) {
	x->data[k] = data[k];
    }
    x->imag = NULL;
    x->length = length;
    
    return x;
}

FVECTOR xfvset(float *data, long length)
{
    FVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct FVECTOR_STRUCT);
    x->data = data;
    x->imag = NULL;
    x->length = length;
    
    return x;
}

FVECTOR xfvsetnew(float *data, long length)
{
    long k;
    FVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct FVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), float);
    for (k = 0; k < length; k++) {
	x->data[k] = data[k];
    }
    x->imag = NULL;
    x->length = length;
    
    return x;
}

DVECTOR xdvset(double *data, long length)
{
    DVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct DVECTOR_STRUCT);
    x->data = data;
    x->imag = NULL;
    x->length = length;
    
    return x;
}

DVECTOR xdvsetnew(double *data, long length)
{
    long k;
    DVECTOR x;

    length = MAX(length, 0);
    x = xalloc(1, struct DVECTOR_STRUCT);
    x->data = xalloc(MAX(length, 1), double);
    for (k = 0; k < length; k++) {
	x->data[k] = data[k];
    }
    x->imag = NULL;
    x->length = length;
    
    return x;
}
