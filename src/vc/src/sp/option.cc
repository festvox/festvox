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
/*#include <unistd.h>*/
#include <sys/types.h>
#include <sys/stat.h>
/*#include <pwd.h>*/
#include <math.h>
#ifdef VARARGS
#include <varargs.h>
#else
#include <stdarg.h>
#endif

#include "../include/defs.h"
#include "../include/memory.h"
#include "../include/option.h"
#include "../include/fileio.h"

/*
 *	get basic name
 */
const char *getbasicname(const char *name)
{
    int i;
    int len;
    const char *string;

    if (name == NULL || *name == NUL)
	return NULL;

    len = strlen(name) - 1;
    for (i = len;; i--) {
	if (*(name + i) == '/') {
	    if (i == len)
	    	return NULL;
	    string = name + i + 1;
	    break;
	}
	if (i <= 0) {
	    string = name;
	    break;
	}
    }

    return string;
}

/*
 *	get allocated basic name
 */
char *xgetbasicname(const char *name)
{
    const char *string;
    char *basicname;

    if (name == NULL || *name == NUL)
	return NULL;

    string = getbasicname(name);
    basicname = strclone(string);

    return basicname;
}

/*
 *	get directory name
 */
char *xgetdirname(const char *filename)
{
    char *p;
    char *dirname;

    /* get directory name */
    if ((dirname = xgetexactname(filename)) == NULL) {
	dirname = strclone("/");
    } else {
	if ((p = strrchr(dirname, '/')) == NULL) {
	    xfree(dirname);
	    dirname = strclone("/");
	} else {
	    *p = NUL;
	}
    } 

    return dirname;
}

/*
 *	get exact name
 */
/*char *xgetexactname(char *name)
{
    int len;
    char buf[MAX_PATHNAME];
    char *home, *username;
    char *exactname;
    struct passwd *entry;

    if (name == NULL || *name == NUL) {
	getcwd(buf, MAX_PATHNAME);
	strcat(buf, "/");
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    } else if (*name == '~') {
	name++;
	if (*name == '/') {
	    name++;
	    if ((home = getenv("HOME")) == NULL) {
		if ((entry = getpwuid(getuid())) == NULL) {
		    return NULL;
		}
		home = entry->pw_dir;
	    }
	    len = strlen(home) + strlen(name) + 2;
	    exactname = xalloc(len, char);
	    sprintf(exactname, "%s/%s", home, name);
	} else {
	    strcpy(buf, name);
            if ((username = strchr(buf, '/')) != NULL)
		*username = NUL;
	    if ((entry = getpwnam(buf)) != NULL) {
                home = entry->pw_dir;
            } else {
		return NULL;
	    }    
            while (*name != '/' && *name != NUL) {
	    	name++;
	    }
	    name++;
	    len = strlen(home) + strlen(name) + 2;
	    exactname = xalloc(len, char);
	    sprintf(exactname, "%s/%s", home, name);
	}
    } else if (streq(name, "..")) {
	getcwd(buf, MAX_PATHNAME);
	if ((username = strrchr(buf, '/')) != NULL)
	    *username = NUL;
	strcat(buf, "/");
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    } else if (strveq(name, "../")) {
	name += 2;
	getcwd(buf, MAX_PATHNAME);
	if ((username = strrchr(buf, '/')) != NULL)
	    *username = NUL;
	strcat(buf, name);
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    } else if (streq(name, ".")) {
	getcwd(buf, MAX_PATHNAME);
	strcat(buf, "/");
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    } else if (strveq(name, "./")) {
	name++;
	getcwd(buf, MAX_PATHNAME);
	strcat(buf, name);
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    } else if (strveq(name, "/")) {
	len = strlen(name) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", name);
    } else {
	getcwd(buf, MAX_PATHNAME);
	strcat(buf, "/");
	strcat(buf, name);
	len = strlen(buf) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", buf);
    }

    return exactname;
}
*/
char *xgetexactname(const char *name)
{
    int len;
    char *exactname;

    if (name == NULL || *name == NUL) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (*name == '~') {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (streq(name, "..")) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (streq(name, "../")) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (streq(name, ".")) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (streq(name, "./")) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else if (streq(name, "/")) {
	exactname = xalloc(5, char);
	sprintf(exactname, "gomi");
    } else {
	len = strlen(name) + 1;
	exactname = xalloc(len, char);
	sprintf(exactname, "%s", name);
    }

    return exactname;
}

/*
 *      get option number of being equal to flag
 */
int flageq(char *flag, OPTIONS options)
{
    int i;

    for (i = 0;; i++) {
        if (i >= options.num_option) {
	    return UNKNOWN;
        }
        if (streq(options.option[i].flag, flag) ||
	    streq(options.option[i].subflag, flag)) {
            break;
        }
    }

    return i;
}

/*
 *	convert option value
 */
int convoptvalue(char *value, OPTION *option)
{
    int incr;

    if (value == NULL || *value == NUL) {
	incr = UNKNOWN;
    } else if (eqtype(option->type, TYPE_INT)) {
	*(int *)option->value = atoi(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_SHORT)) {
	*(short *)option->value = (short)atol(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_LONG)) {
	*(long *)option->value = atol(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_FLOAT)) {
	*(float *)option->value = (float)atof(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_DOUBLE)) {
	*(double *)option->value = (double)atof(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_STRING)) {
	*(char **)option->value = strclone(value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_STRING_S)) {
	strcpy(*(char **)option->value, value);
	incr = 1;
    } else if (eqtype(option->type, TYPE_BOOLEAN)) {
	*(XBOOL *)option->value = str2bool(value);
	incr = 0;
    } else {
	fprintf(stderr, "unknown option data type\n");
	incr = UNKNOWN;
    }

    /* set changed flag true */
    if (incr != UNKNOWN) {
	option->changed = XTRUE;
    }

    return incr;
}

/*
 *	set option value
 */
int setoptvalue(char *value, OPTION *option)
{
    int incr;

    if (eqtype(option->type, TYPE_BOOLEAN)) {
	if (option->value == NULL || *(XBOOL *)option->value != XTRUE) {
	    *(XBOOL *)option->value = XTRUE;
	} else {
	    *(XBOOL *)option->value = XFALSE;
	}
	incr = 0;
    } else {
	incr = convoptvalue(value, option);
    }

    return incr;
}

/*
 *	get option
 */
int getoption(int argc, char *argv[], int *ac, OPTIONS *options)
{
    int i;
    int oc;
    int incr;

    if (*ac >= argc)
	usage(*options);

    i = *ac;
    if ((oc = flageq(argv[i], *options)) != UNKNOWN) {
	if (i + 1 >= argc) {
	    incr = setoptvalue((char *)NULL, &(options->option[oc]));
	} else {
	    incr = setoptvalue(argv[i + 1], &(options->option[oc]));
	}
	if (incr == UNKNOWN)
	    usage(*options);
    } else {
	return UNKNOWN;
    }

    if (incr != UNKNOWN)
	*ac += incr;

    return incr;
}

/*
 *	set changed flag
 */
void setchanged(int argc, char *argv[], OPTIONS *options)
{
    int i;
    int oc;

    for (i = 1; i < argc; i++) {
	if ((oc = flageq(argv[i], *options)) != UNKNOWN) {
	    options->option[oc].changed = XTRUE;
	}
    }

    return;
}

/*
 *	get arg file
 */
int getargfile(char *filename, int *fc, OPTIONS *options)
{
    int i;
    int incr = 1;

    if (!streq(filename, "-") && strveq(filename, "-")) {
	printerr(*options, "unknown option %s", filename);
    }
    if (fc == NULL) {
	i = 0;
    } else {
	i = *fc;
	*fc += incr;
    }

    if (i >= options->num_file) {
	printerr(*options, "too many files");
    }

    options->file[i].name = xgetexactname(filename);

    return incr;
}

#ifdef VARARGS
/*
 *	print help
 */
void printhelp(va_alist)
va_dcl
{
    va_list args;
    char *format;
    OPTIONS options;
    char buf[MAX_LINE];
    char message[MAX_LINE];

    va_start(args);
    options = va_arg(args, OPTIONS);
    format = va_arg(args, char *);
    vsprintf(message, format, args);
    va_end(args);

    sprintf(buf, "%s (%d)", options.progname, options.section);
    fprintf(stderr, "%-24s- %s\n", buf, message);

    usage(options);
}

/*
 *	print error
 */
void printerr(va_alist)
va_dcl
{
    va_list args;
    char *format;
    OPTIONS options;
    char message[MAX_LINE];

    va_start(args);
    options = va_arg(args, OPTIONS);
    format = va_arg(args, char *);
    vsprintf(message, format, args);
    va_end(args);

    fprintf(stderr, "%s: %s\n", options.progname, message);

    usage(options);
}
#else
/*
 *	print help
 */
void printhelp(OPTIONS options, const char *format, ...)
{
    va_list args;
    char buf[MAX_LINE];
    char message[MAX_LINE];

    va_start(args, format);
    vsprintf(message, format, args);
    va_end(args);

    sprintf(buf, "%s (%d)", options.progname, options.section);
    fprintf(stderr, "%-24s- %s\n", buf, message);

    usage(options);
}

/*
 *	print error
 */
void printerr(OPTIONS options, const char *format, ...)
{
    va_list args;
    char message[MAX_LINE];

    va_start(args, format);
    vsprintf(message, format, args);
    va_end(args);

    fprintf(stderr, "%s: %s\n", options.progname, message);

    usage(options);
}
#endif

/*
 *	get option number of being equal to label
 */
int labeleq(char *label, OPTIONS *options)
{
    int i;

    for (i = 0;; i++) {
	if (i >= options->num_option) {
	    i = UNKNOWN;
	    break;
	}
	if (!strnone(options->option[i].label) &&
	    streq(options->option[i].label, label)) {
	    break;
	}
    }

    return i;
}

/*
 *	read setup file
 */
void readsetup(char *filename, OPTIONS *options)
{
    int j;
    char *exactname;
    char name[MAX_LINE] = "";
    char value[MAX_LINE] = "";
    char line[MAX_MESSAGE] = "";
    FILE *fp;

    if (strnone(filename))
	return;

    exactname = xgetexactname(filename);

    if (NULL == (fp = fopen(exactname, "r"))) {
	return;
    }

    while (fgetline(line, fp) != EOF) {
	/*sscanf(line, "%s %s", name, value);*/
	sscanf_setup(line, name, value);
	if (!strnone(value) && (j = labeleq(name, options)) >= 0) {
	    if (options->option[j].changed != XTRUE) {
		if (eqtype(options->option[j].type, TYPE_BOOLEAN)) {
		    *(XBOOL *)options->option[j].value = str2bool(value);
		} else if (eqtype(options->option[j].type, TYPE_INT)) {
		    *(int *)options->option[j].value = atoi(value);
		} else if (eqtype(options->option[j].type, TYPE_SHORT)) {
		    *(short *)options->option[j].value = atoi(value);
		} else if (eqtype(options->option[j].type, TYPE_LONG)) {
		    *(long *)options->option[j].value = atol(value);
		} else if (eqtype(options->option[j].type, TYPE_FLOAT)) {
		    *(float *)options->option[j].value = (float)atof(value);
		} else if (eqtype(options->option[j].type, TYPE_DOUBLE)) {
		    *(double *)options->option[j].value = atof(value);
		} else if (eqtype(options->option[j].type,  TYPE_STRING)) {
		    *(char **)options->option[j].value = strclone(value);
		} else if (eqtype(options->option[j].type,  TYPE_STRING_S)) {
		    strcpy(*(char **)options->option[j].value, value);
		}
	    }
	}
	strcpy(name, "");
	strcpy(value, "");
    }
    fclose(fp);

    xfree(exactname);

    return;
}

/*
 *	write setup file
 */
void writesetup(char *filename, OPTIONS options)
{
    int i;
    char *exactname;
    FILE *fp;

    if (strnone(filename))
	return;

    exactname = xgetexactname(filename);

    if (NULL == (fp = fopen(exactname, "w"))) {
	fprintf(stderr, "can't open file: %s\n", exactname);
	return;
    }
    
    for (i = 0; i < options.num_option; i++) {
	if (strnone(options.option[i].label))
	    continue;
	
	fprintf(fp, "%s ", options.option[i].label);
        if (eqtype(options.option[i].type, TYPE_BOOLEAN)) {
	    fprintf(fp, "%s", bool2str(options.option[i].value));
	} else if (eqtype(options.option[i].type, TYPE_INT)) {
	    fprintf(fp, "%d", *(int *)options.option[i].value);
	} else if (eqtype(options.option[i].type, TYPE_SHORT)) {
	    fprintf(fp, "%d", *(short *)options.option[i].value);
	} else if (eqtype(options.option[i].type, TYPE_LONG)) {
	    fprintf(fp, "%ld", *(long *)options.option[i].value);
	} else if (eqtype(options.option[i].type, TYPE_FLOAT)) {
	    fprintf(fp, "%f", *(float *)options.option[i].value);
	} else if (eqtype(options.option[i].type, TYPE_DOUBLE)) {
	    fprintf(fp, "%f", *(double *)options.option[i].value);
	} else if (eqtype(options.option[i].type,  TYPE_STRING) ||
		   eqtype(options.option[i].type,  TYPE_STRING_S)) {
	    fprintf(fp, "%s", *(char **)options.option[i].value);
	}
	fprintf(fp, "\n");
    }
    fclose(fp);

    xfree(exactname);

    return;
}

/* 
 *      print usage
 */
void usage(OPTIONS options)
{
    int i;
    char buf[MAX_LINE] = "";
    char label[MAX_LINE];
    char filename[MAX_LINE];

    for (i = 0; i < options.num_file; i++) {
	sprintf(filename, " %s", options.file[i].label);
	strcat(buf, filename);
    }

    if (options.num_option <= 0) {
	fprintf(stderr, "usage: %s%s\n", options.progname, buf);
    } else {
	fprintf(stderr, "usage: %s [options...]%s\n", options.progname, buf);
	fprintf(stderr, "options:\n");
    }
    for (i = 0; i < options.num_option; i++) {
	if (strnone(options.option[i].flag) || strnone(options.option[i].desc))
	    continue;
	
	if (!strnone(options.option[i].label)) {
	    strcpy(label, options.option[i].label);
	} else {
	    strcpy(label, USAGE_LABEL_STRING);
	}
	    
        if (eqtype(options.option[i].type, TYPE_BOOLEAN)) {
            fprintf(stderr, "\t%-32s: %s\n",
                    options.option[i].flag, options.option[i].desc);
        } else if (options.option[i].value != NULL) {
	    if (eqtype(options.option[i].type, TYPE_INT)) {
		sprintf(buf, "%s %s[%d]", options.option[i].flag, label,
			*(int *)options.option[i].value);
	    } else if (eqtype(options.option[i].type, TYPE_SHORT)) {
		sprintf(buf, "%s %s[%d]", options.option[i].flag, label,
			*(short *)options.option[i].value);
	    } else if (eqtype(options.option[i].type, TYPE_LONG)) {
		sprintf(buf, "%s %s[%ld]", options.option[i].flag, label,
			*(long *)options.option[i].value);
	    } else if (eqtype(options.option[i].type, TYPE_FLOAT) ||
		       eqtype(options.option[i].type, TYPE_DOUBLE)) {
		int j;
		char value[MAX_LINE];

		if (eqtype(options.option[i].type, TYPE_FLOAT)) {
		    sprintf(value, "%f", *(float *)options.option[i].value);
		} else {
		    sprintf(value, "%f", *(double *)options.option[i].value);
		}
		for (j = strlen(value) - 1; j >= 0; j--) {
		    if (value[j] == '.') {
			value[j + 2] = NUL;
			break;
		    } else if (value[j] != '0') {
			value[j + 1] = NUL;
			break;
		    }
		}
		sprintf(buf, "%s %s[%s]", options.option[i].flag, label,
			value);
	    } else if (eqtype(options.option[i].type, TYPE_STRING) ||
		       eqtype(options.option[i].type, TYPE_STRING_S)) {
		if (strnone(*(char **)options.option[i].value)) {
		    sprintf(buf, "%s %s", options.option[i].flag, label);
		} else {
		    sprintf(buf, "%s %s[%s]", options.option[i].flag, label,
			    *(char **)options.option[i].value);
		}
	    } else {
		fprintf(stderr, "unknown option data type\n");
		exit(1);
	    }

	    fprintf(stderr, "\t%-32s: %s\n", buf, options.option[i].desc);
        } else {
            sprintf(buf, "%s %s", options.option[i].flag, label);
            fprintf(stderr, "\t%-32s: %s\n", buf, options.option[i].desc);
        }
    }
    fprintf(stderr, "\n");

    exit(1);
}
