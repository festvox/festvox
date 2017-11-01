dnl########################################################################
dnl                                                                      ##
dnl                 Language Technologies Institute                      ##
dnl                    Carnegie Mellon University                        ##
dnl                     Copyright (c) 1999-2002                          ##
dnl                       All Rights Reserved.                           ##
dnl                                                                      ##
dnl Permission is hereby granted, free of charge, to use and distribute  ##
dnl this software and its documentation without restriction, including   ##
dnl without limitation the rights to use, copy, modify, merge, publish,  ##
dnl distribute, sublicense, and/or sell copies of this work, and to      ##
dnl permit persons to whom this work is furnished to do so, subject to   ##
dnl the following conditions:                                            ##
dnl  1. The code must retain the above copyright notice, this list of    ##
dnl     conditions and the following disclaimer.                         ##
dnl  2. Any modifications must be clearly marked as such.                ##
dnl  3. Original authors' names are not deleted.                         ##
dnl  4. The authors' names are not used to endorse or promote products   ##
dnl     derived from this software without specific prior written        ##
dnl     permission.                                                      ##
dnl                                                                      ##
dnl CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
dnl DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
dnl ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
dnl SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
dnl FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
dnl WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
dnl AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
dnl ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
dnl THIS SOFTWARE.                                                       ##
dnl                                                                      ##
dnl########################################################################
AC_INIT(src/unitsel/setup_clunits)

AC_CANONICAL_SYSTEM
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_INSTALL
AC_CHECK_TOOL(AR, ar)
AC_C_BIGENDIAN

if test "x$GCC" = "xyes"; then
	CFLAGS="$CFLAGS -Wall"
fi

SHFLAGS=
OTHERLIBS=
if test "$shared" = true; then
   case "$host_os" in
	linux*)
		SHFLAGS="-fPIC"
        ;;
	solaris*)
		SHFLAGS="-fPIC"
		OTHERLIBS="-lsocket -ldl"
        ;;
	*)
	;;
   esac
fi
AC_SUBST(SHFLAGS)

dnl Apple doesn't support the elif construct, so else if them ...
if test "x$ESTDIR" = x; then
  if test -f /usr/src/speech_tools-1.4.2/include/EST.h; then
    ESTDIR=/usr/src/speech_tools-1.4.2
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/local/src/speech_tools-1.4.2/include/EST.h; then
    ESTDIR=/usr/local/src/speech_tools-1.4.2
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/local/speech_tools-1.4.2/include/EST.h; then
    ESTDIR=/usr/local/speech_tools-1.4.2
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /opt/speech_tools-1.4.2/include/EST.h; then
    ESTDIR=/opt/speech_tools-1.4.2
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/src/speech_tools/include/EST.h; then
    ESTDIR=/usr/src/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/local/src/speech_tools/include/EST.h; then
    ESTDIR=/usr/local/src/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/local/speech_tools/include/EST.h; then
    ESTDIR=/usr/local/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /opt/speech_tools/include/EST.h; then
    ESTDIR=/opt/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /usr/include/EST.h; then
    ESTDIR=/usr
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /home/awb/projects/1.4.3/speech_tools/include/EST.h; then
    ESTDIR=/home/awb/projects/1.4.3/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f /home/awb/projects/1.4.2/speech_tools/include/EST.h; then
    ESTDIR=/home/awb/projects/1.4.2/speech_tools
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f "$PWD/../speech_tools/include/EST.h"; then

    ESTDIR="$PWD/../speech_tools"
  fi
fi
if test "x$ESTDIR" = x; then
  if test -f "$cwd/../speech_tools/include/EST.h"; then
    ESTDIR="$cwd/../speech_tools"
  fi
fi
if test "x$ESTDIR" = x; then
    AC_MSG_ERROR("Cannot locate Edinburgh Speech Tools. please specify ESTDIR explicitly.")
fi
echo Edinburgh Speech Tools found in $ESTDIR
AC_SUBST(ESTDIR)

dnl Add support to check for thread support
NOPTHREADFLAGS=""
multithreading=true
AC_ARG_ENABLE( multithreading,
               [  --disable-multithreading	build without multithreading support],
               [ multithreading=false ])

if test "x$multithreading" = xtrue; then
  have_pthreads=no
  AC_SEARCH_LIBS([pthread_create], [pthread],
    [have_pthreads=yes])

  if test "x${have_pthreads}" = xyes; then
    AC_CHECK_HEADER([pthread.h], [],
      [have_pthreads=no])
  fi

  if test "x${have_pthreads}" = xno; then
    echo "------------------------------------------"
    echo " Unable to find pthread on this system.  "
    echo " Building a single-threaded version.      "
    echo "------------------------------------------"

    NOPTHREADFLAGS="${NOPTHREADFLAGS} -DFESTVOX_NO_THREADS"

  fi
else
  NOPTHREADFLAGS="${NOPTHREADFLAGS} -DFESTVOX_NO_THREADS"
  echo "------------------------------------------"
  echo " Multithreading support disabled.         "
  echo "                                          " 
  echo " Run without --disable-multithreading     "
  echo " if you want thread support               "
  echo "------------------------------------------"
fi dnl test for multithreading
AC_SUBST(NOPTHREADFLAGS)

AC_OUTPUT(config/config)
