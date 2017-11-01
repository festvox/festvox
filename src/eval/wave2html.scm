;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;-*-mode:scheme-*-
;;;                                                                     ;;;
;;;                  Language Technologies Institute                    ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2003                          ;;;
;;;                        All Rights Reserved.                         ;;;
;;;                                                                     ;;;
;;; Permission is hereby granted, free of charge, to use and distribute ;;;
;;; this software and its documentation without restriction, including  ;;;
;;; without limitation the rights to use, copy, modify, merge, publish, ;;;
;;; distribute, sublicense, and/or sell copies of this work, and to     ;;;
;;; permit persons to whom this work is furnished to do so, subject to  ;;;
;;; the following conditions:                                           ;;;
;;;  1. The code must retain the above copyright notice, this list of   ;;;
;;;     conditions and the following disclaimer.                        ;;;
;;;  2. Any modifications must be clearly marked as such.               ;;;
;;;  3. Original authors' names are not deleted.                        ;;;
;;;  4. The authors' names are not used to endorse or promote products  ;;;
;;;     derived from this software without specific prior written       ;;;
;;;     permission.                                                     ;;;
;;;                                                                     ;;;
;;; CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK        ;;;
;;; DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     ;;;
;;; ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  ;;;
;;; SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE     ;;;
;;; FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   ;;;
;;; WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  ;;;
;;; AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         ;;;
;;; ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      ;;;
;;; THIS SOFTWARE.                                                      ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;             Author: Alan W Black (awb@cs.cmu.edu)                   ;;;
;;;               Date: June 2003                                       ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;; Generate a webpage to present waveform files for evaluation         ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;; Because this is a --script type file it has to explicitly
;;; load the initfiles: init.scm and user's .festivalrc
(load (path-append libdir "init.scm"))

(set_backtrace t)


;;; Process command line arguments
(define (wave2html_help)
  (format t "%s\n"
  "wave2html [options] textfile(s)
  Generate html listening experiment from wavefiles
  Options
  -comment         experiment description
  -name            experiment name (one token)
  -o ofile         output file for html
  -testtype        mos (default), ab, abx, xabc
  -itype <string>  input type, data, abx or wav (default)
  -waveprefixdir <string>  Prefix for wavefiles (default \".\")
  -cgicript <string>   url to cgi script for loging results
  -eval <string>   File or Lisp expression to be loaded/evaluated
")
  (quit))

;;; No gc messages
(gc-status nil)

;;; Default argument values
(defvar outfile "-")
(defvar waveprefixdir ".")
(defvar comment "Wavefile test")
(defvar testtype "mos")
(defvar name "yyz001")
(defvar wave_files '())
(defvar itype 'wav)
(defvar cgiscript "http://www.festvox.org/cgi-bin/evallog.cgi")

;;; Get options
(define (get_options)
  (let ((files nil)
	(o argv))
    (if (or (member_string "-h" argv)
	    (member_string "-help" argv)
	    (member_string "--help" argv)
	    (member_string "-?" argv))
	(wave2html_help))
    (while o
      (begin
	(cond
	 ((string-equal "-o" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no output file specified"))
	  (set! outfile (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-comment" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no comment specified"))
	  (set! comment (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-name" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no name specified"))
	  (set! name (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-cgiscript" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no cgiscript specified"))
	  (set! cgiscript (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-testtype" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no testtype specified"))
	  (set! testtype (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-waveprefixdir" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no waveprefixdir specified"))
	  (set! waveprefixdir (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-itype" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no itype specified"))
	  (if (not (member_string (car (cdr o)) '("wav" "data" "abx")))
	      (wave2html_error "unknown itype specified"))
	  (set! itype (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-eval" (car o))
	  (if (not (cdr o))
	      (wave2html_error "no file specified to load"))
	  (if (string-matches (car (cdr o)) "^(.*")
	      (eval (read-from-string (car (cdr o))))
	      (load (car (cdr o))))
	  (set! o (cdr o)))
	 (t
	  (set! files (cons (car o) files))))
	(set! o (cdr o))))
    (if files
	(set! wave_files (reverse files)))))

(define (wave2html_error message)
  (format stderr "%s: %s\n" "wave2html" message)
  (wave2html_help))

(define (output_header)
  (format ofd "<HTML>\n")
  (format ofd "<HEAD>\n")
  (format ofd "<TITLE>%s\n" name)
  (format ofd "</TITLE>\n")
  (format ofd "<BODY bgcolor=\"#ffffff\">\n")
)  

(define (output_footer)
  (format ofd "</BODY>\n")
  (format ofd "</HTML>\n")
)  



(define (output_mos_table fd files)

  (format fd "<h1>%s: %s</h1>\n" name comment)
  (format fd "<hr>\n")
  (format fd "<form name=\"evalform\" action=\"%s\" method=\"POST\">\n"
	  cgiscript)
  (format fd "Listener ID: <input type=text size=8 maxlength=8 name=\"listener_id\" value=awb\n<br>")
  (format fd "Test name: <input type=text size=8 maxlength=8 name=\"test_name\" value=\"%s\">\n<br>" name)
  (format fd "Listen to each example and choose 1 (bad) thru 5 (good).\n<br>")
  (format fd "<TABLE>\n")
  (format fd "<th> <th> \n")

(mapcar
   (lambda (w)
     (if (consp w)
	 (begin
	   (set! fname (car w))
	   (set! ftext (cadr w)))
	 (begin
	   (set! fname (basename w ".wav"))
	   (set! ftext fname)))
     (format fd "<tr>\n")
     (format fd "<td>\n")
     (format fd "<select size=1 name=\"%s\"><option>0<option>1<option>2<option>3<option>4<option>5</select>" fname)
;     (format fd "<input type=\"radio\" group=\"%s\" value=\"1\">1" fname)
;     (format fd "<input type=\"radio\" group=\"%s\" value=\"2\">2" fname)
;     (format fd "<input type=\"radio\" group=\"%s\" value=\"3\">3" fname)
;     (format fd "<input type=\"radio\" group=\"%s\" value=\"4\">4" fname)
;     (format fd "<input type=\"radio\" group=\"%s\" value=\"5\">5" fname)
     (format fd "<td>\n")
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">%s</A>" fname ftext)
	 (format fd "<A HREF=\"%s/%s.wav\">%s</A>" waveprefixdir fname ftext))
     (format fd "<td>\n")
     (format fd "</tr>\n")
     )
   files)

  (format fd "</TABLE>\n")
  (format fd "<input type=submit value=\"DONE\">\n")
  (format fd "</form>\n")
  (format fd "<hr>\n")
)

(define (output_abx_table fd waves)

  (format fd "<h1>%s: %s</h1>\n" name comment)
  (format fd "<hr>\n")
  (format fd "<form name=\"evalform\" action=\"%s\" method=\"POST\">\n"
	  cgiscript)
  (format fd "Listener ID: <input type=text size=8 maxlength=8 name=\"listener_id\" value=awb\n<br>")
  (format fd "Test name: <input type=text size=8 maxlength=8 name=\"test_name\" value=\"%s\">\n<br>" name)
  (format fd "Identify which of A or B is closest to X.\n<br>")
  (format fd "<TABLE>\n")
  (format fd "<th> <th> \n")
  
  (while waves
     (set! wav_a (car waves))
     (set! wav_x (car (cdr waves)))
     (set! wav_b (car (cddr waves)))
     (set! waves (cdr (cdr (cdr waves))))

     (if (string-equal "1" (nint (% (rand) 2))) ; random 1 or 0 
	 (begin
	   (set! a wav_a)
	   (set! b wav_b))
	 (begin
	   (set! b wav_a)
	   (set! a wav_b)))
     (set! fname_a (basename a ".wav"))
     (set! fname_x (basename wav_x ".wav"))
     (set! fname_b (basename b ".wav"))
     (set! fname (format nil "%s_%s_%s" fname_a fname_x fname_b))
     (format fd "<tr>\n")
     (format fd "<td>\n")
     (format fd "<select size=1 name=\"%s\"><option>A<option>B</select>" fname)
     (format fd "<td>\n")
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">A&nbsp;</A>" fname_a)
	 (format fd "<A HREF=\"%s/%s.wav\">A&nbsp;</A>" waveprefixdir fname_a))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">X&nbsp;</A>" fname_x)
	 (format fd "<A HREF=\"%s/%s.wav\">X&nbsp;</A>" waveprefixdir fname_x))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">B&nbsp;</A>" fname_b)
	 (format fd "<A HREF=\"%s/%s.wav\">B&nbsp;</A>" waveprefixdir fname_b))
     (format fd "<td>\n")
     (format fd "</tr>\n")
   )

  (format fd "</TABLE>\n")
  (format fd "<input type=submit value=\"DONE\">\n")
  (format fd "</form>\n")
  (format fd "<hr>\n")

)

(define (output_xabc_table fd waves)

  (format fd "<h1>%s: %s</h1>\n" name comment)
  (format fd "<hr>\n")
  (format fd "<form name=\"evalform\" action=\"%s\" method=\"POST\">\n"
	  cgiscript)
  (format fd "Listener ID: <input type=text size=8 maxlength=8 name=\"listener_id\" value=awb\n<br>")
  (format fd "Test name: <input type=text size=8 maxlength=8 name=\"test_name\" value=\"%s\">\n<br>" name)
  (format fd "Identify which of speaker A, B, C or don't know is closest to X.\n<br>")
  (format fd "<TABLE>\n")
  (format fd "<th> <th> \n")
  
  (while waves
     (set! wav_x (car (car waves)))
     (set! wav_a (cadr (car waves)))
     (set! wav_b (car (cddr (car waves))))
     (set! wav_c (cadr (cddr (car waves))))
     (set! waves (cdr waves))

     (set! fname (format nil "%s_%s_%s_%s" wav_x wav_a wav_b wav_c))
     (set! spaces "&nbsp;&nbsp;&nbsp;")
     (format fd "<tr>\n")
     (format fd "<td>\n")
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">A</A>%s" wav_a)
	 (format fd "<A HREF=\"%s/%s.wav\">A</A>%s" waveprefixdir wav_a spaces ))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">B</A>%s" wav_b)
	 (format fd "<A HREF=\"%s/%s.wav\">B</A>%s" waveprefixdir wav_b spaces))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">C</A>%s" wav_c)
	 (format fd "<A HREF=\"%s/%s.wav\">C</A>%s%s" waveprefixdir wav_c spaces spaces ))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">X</A>%s" wav_x spaces)
	 (format fd "<A HREF=\"%s/%s.wav\">X</A>%s%s" waveprefixdir wav_x spaces spaces))

     (format fd "<select size=1 name=\"%s\"><option>A<option>B<option>C<option>Don't know</select>" fname)
     (format fd "<td>\n")
     (format fd "<td>\n")
     (format fd "</tr>\n")
   )

  (format fd "</TABLE>\n")
  (format fd "<input type=submit value=\"DONE\">\n")
  (format fd "</form>\n")
  (format fd "<hr>\n")

)

(define (output_ab_table fd waves)

  (format fd "<h1>%s: %s</h1>\n" name comment)
  (format fd "<hr>\n")
  (format fd "<form name=\"evalform\" action=\"%s\" method=\"POST\">\n"
	  cgiscript)
  (format fd "Listener ID: <input type=text size=8 maxlength=8 name=\"listener_id\" value=awb\n<br>")
  (format fd "Test name: <input type=text size=8 maxlength=8 name=\"test_name\" value=\"%s\">\n<br>" name)
  (format fd "Identify which of A or B is better.\n<br>")
  (format fd "<TABLE>\n")
  (format fd "<th> <th> \n")
  
  (while waves
     (set! wav_a (car waves))
     (set! wav_b (car (cdr waves)))
     (set! waves (cdr (cdr waves)))

     (if (string-equal "1" (nint (% (rand) 2))) ; random 1 or 0 
	 (begin
	   (set! a wav_a)
	   (set! b wav_b))
	 (begin
	   (set! b wav_a)
	   (set! a wav_b)))
     (set! fname_a (basename a ".wav"))
     (set! fname_b (basename b ".wav"))
     (set! fname (format nil "%s_%s" fname_a fname_b))
;     (format stderr "a is %s\n" a)
;     (format stderr "b is %s\n" b)
;     (format stderr "fname A is %s\n" fname_a)
;     (format stderr "fname B is %s\n" fname_b)
     (format fd "<tr>\n")
     (format fd "<td>\n")
     (format fd "<select size=1 name=\"%s\"><option>A<option>B</select>" fname)
     (format fd "<td>\n")
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">A&nbsp;</A>" fname_a)
	 (format fd "<A HREF=\"%s/%s.wav\">A&nbsp;</A>" waveprefixdir fname_a))
     (if (string-equal "." waveprefixdir)
	 (format fd "<A HREF=\"%s.wav\">B&nbsp;</A>" fname_b)
	 (format fd "<A HREF=\"%s/%s.wav\">B&nbsp;</A>" waveprefixdir fname_b))
     (format fd "<td>\n")
     (format fd "</tr>\n")
     )
     
  (format fd "</TABLE>\n")
  (format fd "<input type=submit value=\"DONE\">\n")
  (format fd "</form>\n")
  (format fd "<hr>\n")

)

(define (main)
  (get_options)

  (if (string-equal "-" outfile)
      (set! ofd t)
      (set! ofd (fopen outfile "w")))

  (output_header)
;  (format stderr "testtype is %s\n " testtype)
;  (mapcar (lambda (c) (format stderr "wave_files is %s\n" c)) wave_files)
  (cond 
   ((string-equal testtype "abx")
;    (output_abx_table ofd (load (car wave_files) t)))
    (output_abx_table ofd wave_files))
   ((string-equal testtype "ab")
;    (output_ab_table ofd (load (car wave_files) t)))
    (output_ab_table ofd wave_files))
   ((string-equal testtype "xabc")
    (output_xabc_table ofd (load (car wave_files) t)))
   (t
    (if (string-equal itype "data")
	(set! wave_files (load (car wave_files) t)))
    (output_mos_table ofd wave_files)
    ))

  (output_footer)

  (if (not (string-equal "-" outfile))
      (fclose ofd))

)

;;;  Do the work
(main)
