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
#include "../include/basic.h"
#include "../include/memory.h"
#include "../include/vector.h"
#include "../include/voperate.h"

extern int sp_warning;

void fvoper(FVECTOR x, const char *op, FVECTOR y)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (y->imag != NULL && x->imag == NULL) {
	fvizeros(x, x->length);
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < x->length; k++) {
	    if (k < y->length) {
		x->data[k] = x->data[k] + y->data[k];
		if (x->imag != NULL) {
		    if (y->imag != NULL) {
			x->imag[k] = x->imag[k] + y->imag[k];
		    }
		}
	    }
	}
    } else if (strveq(op2, "-")) {
	if (reverse) {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    x->data[k] = y->data[k] - x->data[k];
		    if (x->imag != NULL) {
			if (y->imag != NULL) {
			    x->imag[k] = y->imag[k] - x->imag[k];
			} else {
			    x->imag[k] = -x->imag[k];
			}
		    }
		} else {
		    x->data[k] = -x->data[k];
		    if (x->imag != NULL) {
			x->imag[k] = -x->imag[k];
		    }
		}
	    }
	} else {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    x->data[k] = x->data[k] - y->data[k];
		    if (x->imag != NULL) {
			if (y->imag != NULL) {
			    x->imag[k] = x->imag[k] - y->imag[k];
			}
		    }
		}
	    }
	}
    } else if (strveq(op2, "*")) {
	float xr, xi;
	for (k = 0; k < x->length; k++) {
	    if (k < y->length) {
		if (x->imag != NULL) {
		    if (y->imag != NULL) {
			xr = x->data[k] * y->data[k] - x->imag[k] * y->imag[k];
			xi = x->data[k] * y->imag[k] + x->imag[k] * y->data[k];
			x->data[k] = xr;
			x->imag[k] = xi;
		    } else {
			x->data[k] = x->data[k] * y->data[k];
			x->imag[k] = x->imag[k] * y->data[k];
		    }
		} else {
		    x->data[k] = x->data[k] * y->data[k];
		}
	    } else {
		x->data[k] = 0.0;
		if (x->imag != NULL) {
		    x->imag[k] = 0.0;
		}
	    }
	}
    } else if (strveq(op2, "/")) {
	float a;
	float xr, xi;
	if (reverse) {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    if (x->imag != NULL) {
			if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: fvoper: divide by zero\n");
			    
			    if (y->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = y->data[k] / (float)ALITTLE_NUMBER;
			    }
			    if (y->imag != NULL) {
				if (y->imag[k] == 0.0) {
				    x->imag[k] = 0.0;
				} else {
				    x->imag[k] = y->imag[k] / (float)ALITTLE_NUMBER;
				}
			    } else {
				x->imag[k] = 0.0;
			    }
			} else {
			    a = CSQUARE(x->data[k], x->imag[k]);
			    if (y->imag != NULL) {
				xr = x->data[k] * y->data[k] + x->imag[k] * y->imag[k];
				xi = x->data[k] * y->imag[k] - x->imag[k] * y->data[k];
				x->data[k] = xr / a;
				x->imag[k] = xi / a;
			    } else {
				x->data[k] =  x->data[k] * y->data[k] / a;
				x->imag[k] = -x->imag[k] * y->data[k] / a;
			    }
			}
		    } else {
			if (x->data[k] != 0.0) {
			    x->data[k] = y->data[k] / x->data[k];
			} else {
			    if (sp_warning)
				fprintf(stderr, "warning: fvoper: divide by zero\n");

			    if (y->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = y->data[k] / (float)ALITTLE_NUMBER;
			    }
			}
		    }
		} else {
		    x->data[k] = 0.0;
		    if (x->imag != NULL) {
			x->imag[k] = 0.0;
		    }
		}
	    }
	} else {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    if (x->imag != NULL && y->imag != NULL) {
			if (y->data[k] == 0.0 && y->imag[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: fvoper: divide by zero\n");

			    if (x->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = x->data[k] / (float)ALITTLE_NUMBER;
			    }
			    if (x->imag[k] == 0.0) {
				x->imag[k] = 0.0;
			    } else {
				x->imag[k] = x->imag[k] / (float)ALITTLE_NUMBER;
			    }
			} else {
			    a = CSQUARE(y->data[k], y->imag[k]);
			    xr = x->data[k] * y->data[k] + x->imag[k] * y->imag[k];
			    xi = -x->data[k] * y->imag[k] + x->imag[k] * y->data[k];
			    x->data[k] = xr / a;
			    x->imag[k] = xi / a;
			}
		    } else {
			if (y->data[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: fvoper: divide by zero\n");

			    if (x->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = x->data[k] / (float)ALITTLE_NUMBER;
			    }
			    if (x->imag != NULL) {
				if (x->imag[k] == 0.0) {
				    x->imag[k] = 0.0;
				} else {
				    x->imag[k] = x->imag[k] / (float)ALITTLE_NUMBER;
				}
			    }
			} else {
			    x->data[k] = x->data[k] / y->data[k];
			    if (x->imag != NULL) {
				x->imag[k] = x->imag[k] / y->data[k];
			    }
			}
		    }
		} else {
		    x->data[k] = 0.0;
		    if (x->imag != NULL) {
			x->imag[k] = 0.0;
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	float xr, xi;
	float yr, yi;
	if (reverse) {
	    if (x->imag != NULL) {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			if (y->imag == NULL) {
			    yr = y->data[k];
			    yi = 0.0;
			} else {
			    yr = y->data[k];
			    yi = y->imag[k];
			}
			if (yr == 0.0 && yi == 0.0) {
			    x->data[k] = 0.0;
			    x->imag[k] = 0.0;
			} else if (x->imag[k] == 0.0 && yi == 0.0) {
			    x->data[k] = (float)pow((double)y->data[k], 
						    (double)x->data[k]);
			} else {
			    clogf(&yr, &yi);
			    xr = x->data[k] * yr - x->imag[k] * yi;
			    xi = x->data[k] * yi + x->imag[k] * yr;
			    cexpf(&xr, &xi);
			    x->data[k] = xr;
			    x->imag[k] = xi;
			}
		    } else {
			x->data[k] = 0.0;
			x->imag[k] = 0.0;
		    }
		}
	    } else {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			x->data[k] = (float)pow((double)y->data[k], 
						(double)x->data[k]);
		    } else {
			x->data[k] = 0.0;
		    }
		}
	    }
	} else {
	    if (x->imag != NULL) {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			    x->data[k] = 0.0;
			    x->imag[k] = 0.0;
			} else {
			    if (y->imag == NULL) {
				yr = y->data[k];
				yi = 0.0;
			    } else {
				yr = y->data[k];
				yi = y->imag[k];
			    }
			    if (x->imag[k] == 0.0 && yi == 0.0) {
				x->data[k] = (float)pow((double)x->data[k],
							(double)y->data[k]);
			    } else {
				clogf(&x->data[k], &x->imag[k]);
				xr = x->data[k] * yr - x->imag[k] * yi;
				xi = x->data[k] * yi + x->imag[k] * yr;
				cexpf(&xr, &xi);
				x->data[k] = xr;
				x->imag[k] = xi;
			    }
			}
		    } else {
			x->data[k] = 1.0;
			x->imag[k] = 1.0;
		    }
		}
	    } else {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			x->data[k] = (float)pow((double)x->data[k], 
						(double)y->data[k]);
		    } else {
			x->data[k] = 1.0;
		    }
		}
	    }
	}
    } else {
	fprintf(stderr, "fvoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

void dvoper(DVECTOR x, const char *op, DVECTOR y)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (y->imag != NULL && x->imag == NULL) {
	dvizeros(x, x->length);
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < x->length; k++) {
	    if (k < y->length) {
		x->data[k] = x->data[k] + y->data[k];
		if (x->imag != NULL) {
		    if (y->imag != NULL) {
			x->imag[k] = x->imag[k] + y->imag[k];
		    }
		}
	    }
	}
    } else if (strveq(op2, "-")) {
	if (reverse) {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    x->data[k] = y->data[k] - x->data[k];
		    if (x->imag != NULL) {
			if (y->imag != NULL) {
			    x->imag[k] = y->imag[k] - x->imag[k];
			} else {
			    x->imag[k] = -x->imag[k];
			}
		    }
		} else {
		    x->data[k] = -x->data[k];
		    if (x->imag != NULL) {
			x->imag[k] = -x->imag[k];
		    }
		}
	    }
	} else {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    x->data[k] = x->data[k] - y->data[k];
		    if (x->imag != NULL) {
			if (y->imag != NULL) {
			    x->imag[k] = x->imag[k] - y->imag[k];
			}
		    }
		}
	    }
	}
    } else if (strveq(op2, "*")) {
	double xr, xi;
	for (k = 0; k < x->length; k++) {
	    if (k < y->length) {
		if (x->imag != NULL) {
		    if (y->imag != NULL) {
			xr = x->data[k] * y->data[k] - x->imag[k] * y->imag[k];
			xi = x->data[k] * y->imag[k] + x->imag[k] * y->data[k];
			x->data[k] = xr;
			x->imag[k] = xi;
		    } else {
			x->data[k] = x->data[k] * y->data[k];
			x->imag[k] = x->imag[k] * y->data[k];
		    }
		} else {
		    x->data[k] = x->data[k] * y->data[k];
		}
	    } else {
		x->data[k] = 0.0;
		if (x->imag != NULL) {
		    x->imag[k] = 0.0;
		}
	    }
	}
    } else if (strveq(op2, "/")) {
	double a;
	double xr, xi;
	if (reverse) {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    if (x->imag != NULL) {
			if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: dvoper: divide by zero\n");

			    if (y->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = y->data[k] / ALITTLE_NUMBER;
			    }
			    if (y->imag != NULL) {
				if (y->imag[k] == 0.0) {
				    x->imag[k] = 0.0;
				} else {
				    x->imag[k] = y->imag[k] / ALITTLE_NUMBER;
				}
			    } else {
				x->imag[k] = 0.0;
			    }
			} else {
			    a = CSQUARE(x->data[k], x->imag[k]);
			    if (y->imag != NULL) {
				xr = x->data[k] * y->data[k] + x->imag[k] * y->imag[k];
				xi = x->data[k] * y->imag[k] - x->imag[k] * y->data[k];
				x->data[k] = xr / a;
				x->imag[k] = xi / a;
			    } else {
				x->data[k] =  x->data[k] * y->data[k] / a;
				x->imag[k] = -x->imag[k] * y->data[k] / a;
			    }
			}
		    } else {
			if (x->data[k] != 0.0) {
			    x->data[k] = y->data[k] / x->data[k];
			} else {
			    if (sp_warning)
				fprintf(stderr, "warning: dvoper: divide by zero\n");

			    if (y->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = y->data[k] / ALITTLE_NUMBER;
			    }
			}
		    }
		} else {
		    x->data[k] = 0.0;
		    if (x->imag != NULL) {
			x->imag[k] = 0.0;
		    }
		}
	    }
	} else {
	    for (k = 0; k < x->length; k++) {
		if (k < y->length) {
		    if (x->imag != NULL && y->imag != NULL) {
			if (y->data[k] == 0.0 && y->imag[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: dvoper: divide by zero\n");

			    if (x->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = x->data[k] / ALITTLE_NUMBER;
			    }
			    if (x->imag[k] == 0.0) {
				x->imag[k] = 0.0;
			    } else {
				x->imag[k] = x->imag[k] / ALITTLE_NUMBER;
			    }
			} else {
			    a = CSQUARE(y->data[k], y->imag[k]);
			    xr = x->data[k] * y->data[k] + x->imag[k] * y->imag[k];
			    xi = -x->data[k] * y->imag[k] + x->imag[k] * y->data[k];
			    x->data[k] = xr / a;
			    x->imag[k] = xi / a;
			}
		    } else {
			if (y->data[k] == 0.0) {
			    if (sp_warning)
				fprintf(stderr, "warning: dvoper: divide by zero\n");

			    if (x->data[k] == 0.0) {
				x->data[k] = 0.0;
			    } else {
				x->data[k] = x->data[k] / ALITTLE_NUMBER;
			    }
			    if (x->imag != NULL) {
				if (x->imag[k] == 0.0) {
				    x->imag[k] = 0.0;
				} else {
				    x->imag[k] = x->imag[k] / ALITTLE_NUMBER;
				}
			    }
			} else {
			    x->data[k] = x->data[k] / y->data[k];
			    if (x->imag != NULL) {
				x->imag[k] = x->imag[k] / y->data[k];
			    }
			}
		    }
		} else {
		    x->data[k] = 0.0;
		    if (x->imag != NULL) {
			x->imag[k] = 0.0;
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	double xr, xi;
	double yr, yi;
	if (reverse) {
	    if (x->imag != NULL) {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			if (y->imag == NULL) {
			    yr = y->data[k];
			    yi = 0.0;
			} else {
			    yr = y->data[k];
			    yi = y->imag[k];
			}
			if (yr == 0.0 && yi == 0.0) {
			    x->data[k] = 0.0;
			    x->imag[k] = 0.0;
			} else if (x->imag[k] == 0.0 && yi == 0.0) {
			    x->data[k] = pow(y->data[k], x->data[k]);
			} else {
			    clog(&yr, &yi);
			    xr = x->data[k] * yr - x->imag[k] * yi;
			    xi = x->data[k] * yi + x->imag[k] * yr;
			    cexp(&xr, &xi);
			    x->data[k] = xr;
			    x->imag[k] = xi;
			}
		    } else {
			x->data[k] = 0.0;
			x->imag[k] = 0.0;
		    }
		}
	    } else {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			x->data[k] = pow(y->data[k], x->data[k]);
		    } else {
			x->data[k] = 0.0;
		    }
		}
	    }
	} else {
	    if (x->imag != NULL) {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			    x->data[k] = 0.0;
			    x->imag[k] = 0.0;
			} else {
			    if (y->imag == NULL) {
				yr = y->data[k];
				yi = 0.0;
			    } else {
				yr = y->data[k];
				yi = y->imag[k];
			    }
			    if (x->imag[k] == 0.0 && yi == 0.0) {
				x->data[k] = pow(x->data[k], y->data[k]);
			    } else {
				clog(&x->data[k], &x->imag[k]);
				xr = x->data[k] * yr - x->imag[k] * yi;
				xi = x->data[k] * yi + x->imag[k] * yr;
				cexp(&xr, &xi);
				x->data[k] = xr;
				x->imag[k] = xi;
			    }
			}
		    } else {
			x->data[k] = 1.0;
			x->imag[k] = 1.0;
		    }
		}
	    } else {
		for (k = 0; k < x->length; k++) {
		    if (k < y->length) {
			x->data[k] = pow(x->data[k], y->data[k]);
		    } else {
			x->data[k] = 1.0;
		    }
		}
	    }
	}
    } else {
	fprintf(stderr, "dvoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

FVECTOR xfvoper(FVECTOR a, const char *op, FVECTOR b)
{
    FVECTOR c;

    c = xfvclone(a);
    fvoper(c, op, b);

    return c;
}

DVECTOR xdvoper(DVECTOR a, const char *op, DVECTOR b)
{
    DVECTOR c;

    c = xdvclone(a);
    dvoper(c, op, b);

    return c;
}

void fvscoper(FVECTOR x, const char *op, float t)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = x->data[k] + t;
	}
    } else if (strveq(op2, "-")) {
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		x->data[k] = t - x->data[k];
		if (x->imag != NULL) {
		    x->imag[k] = -x->imag[k];
		}
	    } else {
		x->data[k] = x->data[k] - t;
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = x->data[k] * t;
	    if (x->imag != NULL) {
		x->imag[k] = x->imag[k] * t;
	    }
	}
    } else if (strveq(op2, "/")) {
	float a;
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		if (x->imag != NULL) {
		    if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			if (sp_warning)
			    fprintf(stderr, "warning: fvscoper: divide by zero\n");

			if (t == 0.0) {
			    x->data[k] = 0.0;
			} else {
			    x->data[k] = t / (float)ALITTLE_NUMBER;
			}
			x->imag[k] = 0.0;
		    } else {
			a = CSQUARE(x->data[k], x->imag[k]);
			x->data[k] = x->data[k] * t / a;
			x->imag[k] = -x->imag[k] * t / a;
		    }
		} else {
		    if (x->data[k] != 0.0) {
			x->data[k] = t / x->data[k];
		    } else {
			if (sp_warning)
			    fprintf(stderr, "warning: fvscoper: divide by zero\n");

			if (t == 0.0) {
			    x->data[k] = 0.0;
			} else {
			    x->data[k] = t / (float)ALITTLE_NUMBER;
			}
		    }
		}
	    } else {
		if (t != 0.0) {
		    x->data[k] = x->data[k] / t;
		    if (x->imag != NULL) {
			x->imag[k] = x->imag[k] / t;
		    }
		} else {
		    if (x->data[k] == 0.0) {
			x->data[k] = 0.0;
		    } else {
			x->data[k] = x->data[k] / (float)ALITTLE_NUMBER;
		    }
		    if (x->imag != NULL) {
			if (x->imag[k] == 0.0) {
			    x->imag[k] = 0.0;
			} else {
			    x->imag[k] = x->imag[k] / (float)ALITTLE_NUMBER;
			}
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	float a; 
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		if (x->imag != NULL && x->imag[k] != 0.0) {
		    a = (float)log(t);
		    x->data[k] *= a;
		    x->imag[k] *= a;
		    cexpf(&x->data[k], &x->imag[k]);
		} else {
		    x->data[k] = (float)pow((double)t, (double)x->data[k]);
		}
	    } else {
		if (x->imag != NULL && x->imag[k] != 0.0) {
		    clogf(&x->data[k], &x->imag[k]);
		    x->data[k] *= t;
		    x->imag[k] *= t;
		    cexpf(&x->data[k], &x->imag[k]);
		} else {
		    x->data[k] = (float)pow((double)x->data[k], (double)t);
		}
	    }
	}
    } else {
	fprintf(stderr, "fvscoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

void dvscoper(DVECTOR x, const char *op, double t)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = x->data[k] + t;
	}
    } else if (strveq(op2, "-")) {
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		x->data[k] = t - x->data[k];
		if (x->imag != NULL) {
		    x->imag[k] = -x->imag[k];
		}
	    } else {
		x->data[k] = x->data[k] - t;
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = x->data[k] * t;
	    if (x->imag != NULL) {
		x->imag[k] = x->imag[k] * t;
	    }
	}
    } else if (strveq(op2, "/")) {
	double a;
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		if (x->imag != NULL) {
		    if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
			if (sp_warning)
			    fprintf(stderr, "warning: dvscoper: divide by zero\n");

			if (t == 0.0) {
			    x->data[k] = 0.0;
			} else {
			    x->data[k] = t / ALITTLE_NUMBER;
			}
			x->imag[k] = 0.0;
		    } else {
			a = CSQUARE(x->data[k], x->imag[k]);
			x->data[k] = x->data[k] * t / a;
			x->imag[k] = -x->imag[k] * t / a;
		    }
		} else {
		    if (x->data[k] != 0.0) {
			x->data[k] = t / x->data[k];
		    } else {
			if (sp_warning)
			    fprintf(stderr, "warning: dvscoper: divide by zero\n");

			if (t == 0.0) {
			    x->data[k] = 0.0;
			} else {
			    x->data[k] = t / ALITTLE_NUMBER;
			}
		    }
		}
	    } else {
		if (t != 0.0) {
		    x->data[k] = x->data[k] / t;
		    if (x->imag != NULL) {
			x->imag[k] = x->imag[k] / t;
		    }
		} else {
		    if (x->data[k] == 0.0) {
			x->data[k] = 0.0;
		    } else {
			x->data[k] = x->data[k] / ALITTLE_NUMBER;
		    }
		    if (x->imag != NULL) {
			if (x->imag[k] == 0.0) {
			    x->imag[k] = 0.0;
			} else {
			    x->imag[k] = x->imag[k] / ALITTLE_NUMBER;
			}
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	double a; 
	for (k = 0; k < x->length; k++) {
	    if (reverse) {
		if (x->imag != NULL && x->imag[k] != 0.0) {
		    a = log(t);
		    x->data[k] *= a;
		    x->imag[k] *= a;
		    cexp(&x->data[k], &x->imag[k]);
		} else {
		    x->data[k] = pow(t, x->data[k]);
		}
	    } else {
		if (x->imag != NULL && x->imag[k] != 0.0) {
		    clog(&x->data[k], &x->imag[k]);
		    x->data[k] *= t;
		    x->imag[k] *= t;
		    cexp(&x->data[k], &x->imag[k]);
		} else {
		    x->data[k] = pow(x->data[k], t);
		}
	    }
	}
    } else {
	fprintf(stderr, "dvscoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

FVECTOR xfvscoper(FVECTOR a, const char *op, float t)
{
    FVECTOR c;

    c = xfvclone(a);
    fvscoper(c, op, t);

    return c;
}

DVECTOR xdvscoper(DVECTOR a, const char *op, double t)
{
    DVECTOR c;

    c = xdvclone(a);
    dvscoper(c, op, t);

    return c;
}

void svoper(SVECTOR a, const char *op, SVECTOR b)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < a->length; k++) {
	    if (k < b->length) {
		a->data[k] = a->data[k] + b->data[k];
	    }
	}
    } else if (strveq(op2, "-")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = b->data[k] - a->data[k];
		} else {
		    a->data[k] = -(a->data[k]);
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = a->data[k] - b->data[k];
		}
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < a->length; k++) {
	    if (k < b->length) {
		a->data[k] = a->data[k] * b->data[k];
	    } else {
		a->data[k] = 0;
	    }
	}
    } else if (strveq(op2, "/")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    if (a->data[k] != 0) {
			a->data[k] = b->data[k] / a->data[k];
		    } else {
			if (sp_warning)
			    fprintf(stderr, "warning: svoper: divide by zero\n");

			if (b->data[k] == 0) {
			    a->data[k] = 0;
			} else {
			    a->data[k] = (short)((double)b->data[k] / ALITTLE_NUMBER);
			}
		    }
		} else {
		    a->data[k] = 0;
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = a->data[k] / b->data[k];
		} else {
		    a->data[k] = 0;
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = (short)pow(b->data[k], (double)a->data[k]);
		} else {
		    a->data[k] = 0;
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = (short)pow((double)a->data[k], b->data[k]);
		} else {
		    a->data[k] = 1;
		}
	    }
	}
    } else {
	fprintf(stderr, "svoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

void lvoper(LVECTOR a, const char *op, LVECTOR b)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < a->length; k++) {
	    if (k < b->length) {
		a->data[k] = a->data[k] + b->data[k];
	    }
	}
    } else if (strveq(op2, "-")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = b->data[k] - a->data[k];
		} else {
		    a->data[k] = -(a->data[k]);
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = a->data[k] - b->data[k];
		}
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < a->length; k++) {
	    if (k < b->length) {
		a->data[k] = a->data[k] * b->data[k];
	    } else {
		a->data[k] = 0;
	    }
	}
    } else if (strveq(op2, "/")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    if (a->data[k] != 0) {
			a->data[k] = b->data[k] / a->data[k];
		    } else {
			if (sp_warning)
			    fprintf(stderr, "warning: lvoper: divide by zero\n");

			if (b->data[k] == 0) {
			    a->data[k] = 0;
			} else {
			    a->data[k] = (long)((double)b->data[k] / ALITTLE_NUMBER);
			}
		    }
		} else {
		    a->data[k] = 0;
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = a->data[k] / b->data[k];
		} else {
		    a->data[k] = 0;
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	if (reverse) {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = (long)pow(b->data[k], (double)a->data[k]);
		} else {
		    a->data[k] = 0;
		}
	    }
	} else {
	    for (k = 0; k < a->length; k++) {
		if (k < b->length) {
		    a->data[k] = (long)pow((double)a->data[k], b->data[k]);
		} else {
		    a->data[k] = 1;
		}
	    }
	}
    } else {
	fprintf(stderr, "lvoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

SVECTOR xsvoper(SVECTOR a, const char *op, SVECTOR b)
{
    SVECTOR c;

    c = xsvclone(a);
    svoper(c, op, b);

    return c;
}

LVECTOR xlvoper(LVECTOR a, const char *op, LVECTOR b)
{
    LVECTOR c;

    c = xlvclone(a);
    lvoper(c, op, b);

    return c;
}

void svscoper(SVECTOR a, const char *op, double t)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < a->length; k++) {
	    a->data[k] = a->data[k] + (short)t;
	}
    } else if (strveq(op2, "-")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		a->data[k] = (short)t - a->data[k];
	    } else {
		a->data[k] = a->data[k] - (short)t;
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < a->length; k++) {
	    a->data[k] = (short)((double)a->data[k] * t);
	}
    } else if (strveq(op2, "/")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		if (a->data[k] != 0.0) {
		    a->data[k] = (short)(t / (double)a->data[k]);
		} else {
		    if (sp_warning)
			fprintf(stderr, "warning: svscoper: divide by zero\n");

		    if (t == 0) {
			a->data[k] = 0;
		    } else {
			a->data[k] = (short)(t / ALITTLE_NUMBER);
		    }
		}
	    } else {
		if (t != 0.0) {
		    a->data[k] = (short)((double)a->data[k] / t);
		} else {
		    if (sp_warning)
			fprintf(stderr, "warning: svscoper: divide by zero\n");

		    if (a->data[k] != 0) {
			a->data[k] = (short)((double)a->data[k] / ALITTLE_NUMBER);
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		a->data[k] = (short)pow(t, (double)a->data[k]);
	    } else {
		a->data[k] = (short)pow((double)a->data[k], t);
	    }
	}
    } else {
	fprintf(stderr, "svscoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}

void lvscoper(LVECTOR a, const char *op, double t)
{
    long k;
    int reverse = 0;
    const char *op2 = op;

    if (strveq(op2, "!")) {
	reverse = 1;
	op2++;
    }

    if (strveq(op2, "+")) {
	for (k = 0; k < a->length; k++) {
	    a->data[k] = a->data[k] + (long)t;
	}
    } else if (strveq(op2, "-")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		a->data[k] = (long)t - a->data[k];
	    } else {
		a->data[k] = a->data[k] - (long)t;
	    }
	}
    } else if (strveq(op2, "*")) {
	for (k = 0; k < a->length; k++) {
	    a->data[k] = (long)((double)a->data[k] * t);
	}
    } else if (strveq(op2, "/")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		if (a->data[k] != 0.0) {
		    a->data[k] = (long)(t / (double)a->data[k]);
		} else {
		    if (sp_warning)
			fprintf(stderr, "warning: lvscoper: divide by zero\n");

		    if (t == 0) {
			a->data[k] = 0;
		    } else {
			a->data[k] = (long)(t / ALITTLE_NUMBER);
		    }
		}
	    } else {
		if (t != 0.0) {
		    a->data[k] = (long)((double)a->data[k] / t);
		} else {
		    if (sp_warning)
			fprintf(stderr, "warning: lvscoper: divide by zero\n");

		    if (a->data[k] != 0) {
			a->data[k] = (long)((double)a->data[k] / ALITTLE_NUMBER);
		    }
		}
	    }
	}
    } else if (strveq(op2, "^")) {
	for (k = 0; k < a->length; k++) {
	    if (reverse) {
		a->data[k] = (long)pow(t, (double)a->data[k]);
	    } else {
		a->data[k] = (long)pow((double)a->data[k], t);
	    }
	}
    } else {
	fprintf(stderr, "lvscoper: unknouwn operation: %s\n", op2);
	exit(1);
    }

    return;
}


SVECTOR xsvscoper(SVECTOR a, const char *op, double t)
{
    SVECTOR c;

    c = xsvclone(a);
    svscoper(c, op, t);

    return c;
}

LVECTOR xlvscoper(LVECTOR a, const char *op, double t)
{
    LVECTOR c;

    c = xlvclone(a);
    lvscoper(c, op, t);

    return c;
}

void svabs(SVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = ABS(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = (short)CABS(x->data[k], x->imag[k]);
	}
	svifree(x);
    }

    return;
}

void lvabs(LVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = ABS(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = (long)CABS(x->data[k], x->imag[k]);
	}
	lvifree(x);
    }

    return;
}

void fvabs(FVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = FABS(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = (float)CABS(x->data[k], x->imag[k]);
	}
	fvifree(x);
    }

    return;
}

void dvabs(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = FABS(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = CABS(x->data[k], x->imag[k]);
	}
	dvifree(x);
    }

    return;
}

SVECTOR xsvabs(SVECTOR x)
{
    SVECTOR y;

    y = xsvclone(x);
    svabs(y);

    return y;
}

LVECTOR xlvabs(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvabs(y);

    return y;
}

FVECTOR xfvabs(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvabs(y);

    return y;
}

DVECTOR xdvabs(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvabs(y);

    return y;
}

void svsquare(SVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = SQUARE(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = CSQUARE(x->data[k], x->imag[k]);
	}
	svifree(x);
    }

    return;
}

void lvsquare(LVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = SQUARE(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = CSQUARE(x->data[k], x->imag[k]);
	}
	lvifree(x);
    }

    return;
}

void fvsquare(FVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = SQUARE(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = CSQUARE(x->data[k], x->imag[k]);
	}
	fvifree(x);
    }

    return;
}

void dvsquare(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = SQUARE(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = CSQUARE(x->data[k], x->imag[k]);
	}
	dvifree(x);
    }

    return;
}

SVECTOR xsvsquare(SVECTOR x)
{
    SVECTOR y;

    y = xsvclone(x);
    svsquare(y);

    return y;
}

LVECTOR xlvsquare(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvsquare(y);

    return y;
}

FVECTOR xfvsquare(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvsquare(y);

    return y;
}

DVECTOR xdvsquare(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvsquare(y);

    return y;
}

void svsign(SVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] > 0) {
		x->data[k] = 1;
	    } else if (x->data[k] == 0) {
		x->data[k] = 0;
	    } else {
		x->data[k] = -1;
	    }
	}
    } else {
	double value;
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] != 0 || x->imag[k] != 0) {
		value = CABS(x->data[k], x->imag[k]);
		x->data[k] = (short)((double)x->data[k] / value);
		x->imag[k] = (short)((double)x->imag[k] / value);
	    }
	}
    }

    return;
}

void lvsign(LVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] > 0) {
		x->data[k] = 1;
	    } else if (x->data[k] == 0) {
		x->data[k] = 0;
	    } else {
		x->data[k] = -1;
	    }
	}
    } else {
	double value;
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] != 0 || x->imag[k] != 0) {
		value = CABS(x->data[k], x->imag[k]);
		x->data[k] = (long)((double)x->data[k] / value);
		x->imag[k] = (long)((double)x->imag[k] / value);
	    }
	}
    }

    return;
}

void fvsign(FVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] > 0.0) {
		x->data[k] = 1.0;
	    } else if (x->data[k] == 0.0) {
		x->data[k] = 0.0;
	    } else {
		x->data[k] = -1.0;
	    }
	}
    } else {
	double value;
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] != 0.0 || x->imag[k] != 0.0) {
		value = CABS(x->data[k], x->imag[k]);
		x->data[k] = (float)((double)x->data[k] / value);
		x->imag[k] = (float)((double)x->imag[k] / value);
	    }
	}
    }

    return;
}

void dvsign(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] > 0.0) {
		x->data[k] = 1.0;
	    } else if (x->data[k] == 0.0) {
		x->data[k] = 0.0;
	    } else {
		x->data[k] = -1.0;
	    }
	}
    } else {
	double value;
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] != 0.0 || x->imag[k] != 0.0) {
		value = CABS(x->data[k], x->imag[k]);
		x->data[k] = (double)x->data[k] / value;
		x->imag[k] = (double)x->imag[k] / value;
	    }
	}
    }

    return;
}

SVECTOR xsvsign(SVECTOR x)
{
    SVECTOR y;

    y = xsvclone(x);
    svsign(y);

    return y;
}

LVECTOR xlvsign(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvsign(y);

    return y;
}

FVECTOR xfvsign(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvsign(y);

    return y;
}

DVECTOR xdvsign(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvsign(y);

    return y;
}

SVECTOR xsvremap(SVECTOR x, LVECTOR map)
{
    long k;
    SVECTOR y;

    y = xsvalloc(map->length);
    if (x->imag != NULL) {
	svialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	if (map->data[k] >= 0 && map->data[k] < x->length) {
	    y->data[k] = x->data[map->data[k]];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[map->data[k]];
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

LVECTOR xlvremap(LVECTOR x, LVECTOR map)
{
    long k;
    LVECTOR y;

    y = xlvalloc(map->length);
    if (x->imag != NULL) {
	lvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	if (map->data[k] >= 0 && map->data[k] < x->length) {
	    y->data[k] = x->data[map->data[k]];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[map->data[k]];
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

FVECTOR xfvremap(FVECTOR x, LVECTOR map)
{
    long k;
    FVECTOR y;

    y = xfvalloc(map->length);
    if (x->imag != NULL) {
	fvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	if (map->data[k] >= 0 && map->data[k] < x->length) {
	    y->data[k] = x->data[map->data[k]];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[map->data[k]];
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

DVECTOR xdvremap(DVECTOR x, LVECTOR map)
{
    long k;
    DVECTOR y;

    y = xdvalloc(map->length);
    if (x->imag != NULL) {
	dvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	if (map->data[k] >= 0 && map->data[k] < x->length) {
	    y->data[k] = x->data[map->data[k]];
	    if (y->imag != NULL) {
		y->imag[k] = x->imag[map->data[k]];
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

SVECTOR xsvcodiff(SVECTOR x, double coef)
{
    long k;
    SVECTOR y;

    if (x->length <= 1) {
	y = xsvnull();
	return y;
    }
    y = xsvalloc(x->length - 1);
    if (x->imag != NULL) {
	svialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = x->data[k + 1] - (short)(coef * (double)x->data[k]);
	if (y->imag != NULL) {
	    y->imag[k] = x->imag[k + 1] - (short)(coef * (double)x->imag[k]);
	}
    }

    return y;
}

LVECTOR xlvcodiff(LVECTOR x, double coef)
{
    long k;
    LVECTOR y;

    if (x->length <= 1) {
	y = xlvnull();
	return y;
    }
    y = xlvalloc(x->length - 1);
    if (x->imag != NULL) {
	lvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = x->data[k + 1] - (long)(coef * (double)x->data[k]);
	if (y->imag != NULL) {
	    y->imag[k] = x->imag[k + 1] - (long)(coef * (double)x->imag[k]);
	}
    }

    return y;
}

FVECTOR xfvcodiff(FVECTOR x, double coef)
{
    long k;
    FVECTOR y;

    if (x->length <= 1) {
	y = xfvnull();
	return y;
    }
    y = xfvalloc(x->length - 1);
    if (x->imag != NULL) {
	fvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = x->data[k + 1] - (float)(coef * (double)x->data[k]);
	if (y->imag != NULL) {
	    y->imag[k] = x->imag[k + 1] - (float)(coef * (double)x->imag[k]);
	}
    }

    return y;
}

DVECTOR xdvcodiff(DVECTOR x, double coef)
{
    long k;
    DVECTOR y;

    if (x->length <= 1) {
	y = xdvnull();
	return y;
    }
    y = xdvalloc(x->length - 1);
    if (x->imag != NULL) {
	dvialloc(y);
    }

    for (k = 0; k < y->length; k++) {
	y->data[k] = x->data[k + 1] - coef * x->data[k];
	if (y->imag != NULL) {
	    y->imag[k] = x->imag[k + 1] - coef * x->imag[k];
	}
    }

    return y;
}

LVECTOR xsvfind(SVECTOR x)
{
    long k, l;
    LVECTOR y;

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0) {
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0) {
	    l++;
	}
    }

    y = xlvalloc(l);

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0) {
	    y->data[l] = k;
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0) {
	    y->data[l] = k;
	    l++;
	}
    }

    return y;
}

LVECTOR xlvfind(LVECTOR x)
{
    long k, l;
    LVECTOR y;

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0) {
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0) {
	    l++;
	}
    }

    y = xlvalloc(l);

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0) {
	    y->data[l] = k;
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0) {
	    y->data[l] = k;
	    l++;
	}
    }

    return y;
}

LVECTOR xfvfind(FVECTOR x)
{
    long k, l;
    LVECTOR y;

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    l++;
	}
    }

    y = xlvalloc(l);

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    y->data[l] = k;
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    y->data[l] = k;
	    l++;
	}
    }

    return y;
}

LVECTOR xdvfind(DVECTOR x)
{
    long k, l;
    LVECTOR y;

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    l++;
	}
    }

    y = xlvalloc(l);

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    y->data[l] = k;
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    y->data[l] = k;
	    l++;
	}
    }

    return y;
}

DVECTOR xdvfindv(DVECTOR x)
{
    long k, l;
    DVECTOR y;

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    l++;
	}
    }

    if (x->imag != NULL) {
	y = xdvrialloc(l);
    } else {
	y = xdvalloc(l);
    }

    for (k = 0, l = 0; k < x->length; k++) {
	if (x->data[k] != 0.0) {
	    y->data[l] = x->data[k];
	    if (x->imag != NULL) {
		y->imag[l] = x->imag[k];
	    }
	    l++;
	} else if (x->imag != NULL && x->imag[k] != 0.0) {
	    y->data[l] = x->data[k];
	    y->imag[l] = x->imag[k];
	    l++;
	}
    }

    return y;
}

DVECTOR xdvsceval(DVECTOR x, const char *op, double t)
{
    long k;
    DVECTOR y;

    y = xdvzeros(x->length);

    if (strveq(op, "<=")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] <= t) {
		y->data[k] = 1.0;
	    }
	}
    } else if (strveq(op, "<")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] < t) {
		y->data[k] = 1.0;
	    }
	}
    } else if (strveq(op, ">=")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] >= t) {
		y->data[k] = 1.0;
	    }
	}
    } else if (strveq(op, ">")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] > t) {
		y->data[k] = 1.0;
	    }
	}
    } else if (strveq(op, "==")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] == t) {
		y->data[k] = 1.0;
	    }
	}
    } else if (strveq(op, "!=")) {
	for (k = 0; k < x->length; k++) {
	    if (x->data[k] != t) {
		y->data[k] = 1.0;
	    }
	}
    } else {
	fprintf(stderr, "xdvsceval: unknouwn operation: %s\n", op);
	exit(1);
    }

    return y;
}

LVECTOR xdvscfind(DVECTOR x, const char *op, double t)
{
    DVECTOR y;
    LVECTOR z;

    y = xdvsceval(x, op, t);
    z = xdvfind(y);

    xdvfree(y);

    return z;
}

DVECTOR xdvscfindv(DVECTOR x, const char *op, double t)
{
    DVECTOR y;
    LVECTOR idx;
    DVECTOR xd;

    y = xdvsceval(x, op, t);
    idx = xdvfind(y);
    xd = xdvremap(x, idx);

    xdvfree(y);
    xlvfree(idx);

    return xd;
}
#if 0
#endif

void lvcumsum(LVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += x->data[k];
	x->data[k] = sum;
    }
    if (x->imag != NULL) {
	for (k = 0, sum = 0; k < x->length; k++) {
	    sum += x->imag[k];
	    x->imag[k] = sum;
	}
    }

    return;
}

void fvcumsum(FVECTOR x)
{
    long k;
    float sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += x->data[k];
	x->data[k] = sum;
    }
    if (x->imag != NULL) {
	for (k = 0, sum = 0.0; k < x->length; k++) {
	    sum += x->imag[k];
	    x->imag[k] = sum;
	}
    }

    return;
}

void dvcumsum(DVECTOR x)
{
    long k;
    double sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += x->data[k];
	x->data[k] = sum;
    }
    if (x->imag != NULL) {
	for (k = 0, sum = 0.0; k < x->length; k++) {
	    sum += x->imag[k];
	    x->imag[k] = sum;
	}
    }

    return;
}

LVECTOR xlvcumsum(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvcumsum(y);

    return y;
}

FVECTOR xfvcumsum(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvcumsum(y);

    return y;
}

DVECTOR xdvcumsum(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvcumsum(y);

    return y;
}

void lvcumprod(LVECTOR x)
{
    long k;
    long prod;

    for (k = 0, prod = 1; k < x->length; k++) {
	prod *= x->data[k];
	x->data[k] = prod;
    }
    if (x->imag != NULL) {
	for (k = 0, prod = 1; k < x->length; k++) {
	    prod *= x->imag[k];
	    x->imag[k] = prod;
	}
    }

    return;
}

void fvcumprod(FVECTOR x)
{
    long k;
    float prod;

    for (k = 0, prod = 1.0; k < x->length; k++) {
	prod *= x->data[k];
	x->data[k] = prod;
    }
    if (x->imag != NULL) {
	for (k = 0, prod = 1.0; k < x->length; k++) {
	    prod *= x->imag[k];
	    x->imag[k] = prod;
	}
    }

    return;
}

void dvcumprod(DVECTOR x)
{
    long k;
    double prod;

    for (k = 0, prod = 1.0; k < x->length; k++) {
	prod *= x->data[k];
	x->data[k] = prod;
    }
    if (x->imag != NULL) {
	for (k = 0, prod = 1.0; k < x->length; k++) {
	    prod *= x->imag[k];
	    x->imag[k] = prod;
	}
    }

    return;
}

LVECTOR xlvcumprod(LVECTOR x)
{
    LVECTOR y;

    y = xlvclone(x);
    lvcumprod(y);

    return y;
}

FVECTOR xfvcumprod(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvcumprod(y);

    return y;
}

DVECTOR xdvcumprod(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvcumprod(y);

    return y;
}

void fvexp(FVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = (float)exp((double)x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    cexpf(&x->data[k], &x->imag[k]);
	}
    }

    return;
}

void dvexp(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	for (k = 0; k < x->length; k++) {
	    x->data[k] = exp(x->data[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    cexp(&x->data[k], &x->imag[k]);
	}
    }

    return;
}

FVECTOR xfvexp(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvexp(y);

    return y;
}

DVECTOR xdvexp(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvexp(y);

    return y;
}

void fvlog(FVECTOR x)
{
    int flag = 0;
    long k;

    for (k = 0; k < x->length; k++) {
	if (x->imag != NULL || x->data[k] < 0.0) {
	    flag = 1;
	    break;
	}
    }

    if (flag) {
	if (x->imag == NULL) {
	    fvizeros(x, x->length);
	}
	for (k = 0; k < x->length; k++) {
	    clogf(&x->data[k], &x->imag[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    clogf(&x->data[k], NULL);
	}
    }

    return;
}

void dvlog(DVECTOR x)
{
    int flag = 0;
    long k;

    for (k = 0; k < x->length; k++) {
	if (x->imag != NULL || x->data[k] < 0.0) {
	    flag = 1;
	    break;
	}
    }

    if (flag) {
	if (x->imag == NULL) {
	    dvizeros(x, x->length);
	}
	for (k = 0; k < x->length; k++) {
	    clog(&x->data[k], &x->imag[k]);
	}
    } else {
	for (k = 0; k < x->length; k++) {
	    clog(&x->data[k], NULL);
	}
    }

    return;
}

FVECTOR xfvlog(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvlog(y);

    return y;
}

DVECTOR xdvlog(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvlog(y);

    return y;
}

void fvdecibel(FVECTOR x)
{
    long k;

    fvsquare(x);
    
    for (k = 0; k < x->length; k++) {
	if (x->data[k] <= 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: fvdecibel: log of zero\n");

	    x->data[k] = (float)10.0 * (float)log10(ALITTLE_NUMBER);
	} else {
	    x->data[k] = (float)10.0 * (float)log10((double)x->data[k]);
	}
    }

    return;
}

void dvdecibel(DVECTOR x)
{
    long k;

    dvsquare(x);
    
    for (k = 0; k < x->length; k++) {
	if (x->data[k] <= 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: dvdecibel: log of zero\n");

	    x->data[k] = 10.0 * log10(ALITTLE_NUMBER);
	} else {
	    x->data[k] = 10.0 * log10(x->data[k]);
	}
    }

    return;
}

FVECTOR xfvdecibel(FVECTOR x)
{
    FVECTOR y;

    y = xfvclone(x);
    fvdecibel(y);

    return y;
}

DVECTOR xdvdecibel(DVECTOR x)
{
    DVECTOR y;

    y = xdvclone(x);
    dvdecibel(y);

    return y;
}

FVECTOR xfvrandn(long length)
{
    long k;
    FVECTOR x;

    x = xfvalloc(length);
    
    for (k = 0; k < x->length; k++) {
	x->data[k] = (float)randn();
    }

    return x;
}

DVECTOR xdvrandn(long length)
{
    long k;
    DVECTOR x;

    x = xdvalloc(length);
    
    for (k = 0; k < x->length; k++) {
	x->data[k] = randn();
    }

    return x;
}

static int qsort_numcmp(const void *x, const void *y)
{
    double tx, ty;

    tx = *(double *)x;
    ty = *(double *)y;

    if (tx < ty) return (-1);
    if (tx > ty) return (1);
    return (0);
}

void dvsort(DVECTOR x)
{
    if (x == NODATA || x->length <= 1)
	return;

    qsort(x->data, (unsigned)x->length, sizeof(double), qsort_numcmp);

    return;
}

long svsum(SVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += (long)x->data[k];
    }

    return sum;
}

long lvsum(LVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += x->data[k];
    }

    return sum;
}

float fvsum(FVECTOR x)
{
    long k;
    float sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += x->data[k];
    }

    return sum;
}

double dvsum(DVECTOR x)
{
    long k;
    double sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += x->data[k];
    }

    return sum;
}

long svsqsum(SVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += SQUARE((long)x->data[k]);
    }

    return sum;
}

long lvsqsum(LVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += SQUARE(x->data[k]);
    }

    return sum;
}

float fvsqsum(FVECTOR x)
{
    long k;
    float sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += SQUARE(x->data[k]);
    }

    return sum;
}

double dvsqsum(DVECTOR x)
{
    long k;
    double sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += SQUARE(x->data[k]);
    }

    return sum;
}

long svabssum(SVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += FABS((long)x->data[k]);
    }

    return sum;
}

long lvabssum(LVECTOR x)
{
    long k;
    long sum;

    for (k = 0, sum = 0; k < x->length; k++) {
	sum += FABS(x->data[k]);
    }

    return sum;
}

float fvabssum(FVECTOR x)
{
    long k;
    float sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += FABS(x->data[k]);
    }

    return sum;
}

double dvabssum(DVECTOR x)
{
    long k;
    double sum;

    for (k = 0, sum = 0.0; k < x->length; k++) {
	sum += FABS(x->data[k]);
    }

    return sum;
}

short svmax(SVECTOR x, long *index)
{
    long k;
    long ind;
    short max;

    ind = 0;
    max = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (max < x->data[k]) {
	    ind = k;
	    max = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return max;
}

long lvmax(LVECTOR x, long *index)
{
    long k;
    long ind;
    long max;

    ind = 0;
    max = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (max < x->data[k]) {
	    ind = k;
	    max = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return max;
}

float fvmax(FVECTOR x, long *index)
{
    long k;
    long ind;
    float max;

    ind = 0;
    max = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (max < x->data[k]) {
	    ind = k;
	    max = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return max;
}

double dvmax(DVECTOR x, long *index)
{
    long k;
    long ind;
    double max;

    ind = 0;
    max = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (max < x->data[k]) {
	    ind = k;
	    max = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return max;
}

short svmin(SVECTOR x, long *index)
{
    long k;
    long ind;
    short min;

    ind = 0;
    min = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (min > x->data[k]) {
	    ind = k;
	    min = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return min;
}

long lvmin(LVECTOR x, long *index)
{
    long k;
    long ind;
    long min;

    ind = 0;
    min = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (min > x->data[k]) {
	    ind = k;
	    min = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return min;
}

float fvmin(FVECTOR x, long *index)
{
    long k;
    long ind;
    float min;

    ind = 0;
    min = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (min > x->data[k]) {
	    ind = k;
	    min = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return min;
}

double dvmin(DVECTOR x, long *index)
{
    long k;
    long ind;
    double min;

    ind = 0;
    min = x->data[ind];
    for (k = 1; k < x->length; k++) {
	if (min > x->data[k]) {
	    ind = k;
	    min = x->data[k];
	}
    }

    if (index != NULL) {
	*index = ind;
    }

    return min;
}

void svscmax(SVECTOR x, short a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MAX(x->data[k], a);
    }

    return;
}

void lvscmax(LVECTOR x, long a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MAX(x->data[k], a);
    }

    return;
}

void fvscmax(FVECTOR x, float a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MAX(x->data[k], a);
    }

    return;
}

void dvscmax(DVECTOR x, double a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MAX(x->data[k], a);
    }

    return;
}

void svscmin(SVECTOR x, short a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MIN(x->data[k], a);
    }

    return;
}

void lvscmin(LVECTOR x, long a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MIN(x->data[k], a);
    }

    return;
}

void fvscmin(FVECTOR x, float a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MIN(x->data[k], a);
    }

    return;
}

void dvscmin(DVECTOR x, double a)
{
    long k;

    for (k = 0; k < x->length; k++) {
	x->data[k] = MIN(x->data[k], a);
    }

    return;
}

float fvdot(FVECTOR x, FVECTOR y)
{
    long k;
    float a = 0.0;

    if (x == NODATA || y == NODATA) {
	a = 0.0;
    } else if (x->length != y->length) {
	fprintf(stderr, "fvdot: vector length must agree\n");
	exit(1);
    } else {
	for (k = 0; k < x->length; k++) {
	    a += x->data[k] * y->data[k];
	}
    }
    
    return a;
}

double dvdot(DVECTOR x, DVECTOR y)
{
    long k;
    double a = 0.0;

    if (x == NODATA || y == NODATA) {
	a = 0.0;
    } else if (x->length != y->length) {
	fprintf(stderr, "dvdot: vector length must agree\n");
	exit(1);
    } else {
	for (k = 0; k < x->length; k++) {
	    a += x->data[k] * y->data[k];
	}
    }
    
    return a;
}

void dvmorph(DVECTOR x, DVECTOR y, double rate)
{
    long k;
    double a, b;

    if (x == NODATA)
	return;
    
    b = rate;
    a = 1.0 - b;
    
    for (k = 0; k < x->length; k++) {
	if (y != NODATA && k < y->length) {
	    x->data[k] = a * x->data[k] + b * y->data[k];
	} else {
	    x->data[k] = a * x->data[k];
	}
    }
    
    if (x->imag != NULL) {
	for (k = 0; k < x->length; k++) {
	    if (y != NODATA && (y->imag != NULL && k < y->length)) {
		x->imag[k] = a * x->imag[k] + b * y->imag[k];
	    } else {
		x->imag[k] = a * x->imag[k];
	    }
	}
    }
    
    return;
}

DVECTOR xdvmorph(DVECTOR x, DVECTOR y, double rate)
{
    DVECTOR z;

    if (x == NODATA && y == NODATA) {
	z = NODATA;
    } else if (x == NODATA) {
	z = xdvzeros(y->length);
    } else {
	z = xdvclone(x);
    }
    dvmorph(z, y, rate);
    
    return z;
}
#if 0
#endif
