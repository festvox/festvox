;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;-*-mode:scheme-*-
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                  and Alan W Black and Kevin Lenzo                   ;;;
;;;                      Copyright (c) 1998-2000                        ;;;
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
;;;                                                                     ;;;
;;;  Builds a diphone index from a diphone list and a set label files   ;;;
;;;                                                                     ;;;
;;;  Essentially copied from a awk script that did the same thing       ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Because this is a --script type file it has to explicitly
;;; load the initfiles: init.scm and user's .festivalrc
(if (not (symbol-bound? 'caar))
    (begin
      (load (path-append libdir "init.scm"))
      (if (probe_file (format nil "%s/.festivalrc" (getenv "HOME")))
	  (load (format nil "%s/.festivalrc" (getenv "HOME"))))))


(define (make_diph_index_help)
  (format t "%s\n"
  "make_diph_index [options] diphlist dic/diphdic.est
  Build a diphone index from a diphone list and set of label files
  Options
  -eval <file>
             Load in scheme file with run specific code, if file name
             starts with a left parent the string itself is interpreted.
             This can be used to select the appropriate phoneset.
  -lab_dir <directory>
             Directory containing the label files (default \"lab/\")
  -lab_ext <string>
             File extention for label files (default \".lab\")
")
  (quit))

(defvar diphlist_file "diphlist")
(defvar diphindex_file "dic/diphdic.est")
(defvar lab_dir "lab/")
(defvar lab_ext ".lab")
(defvar diphindex nil)

;;; Get options
(define (get_options)
  (let ((files nil) (o argv))
    (if (or (member_string "-h" argv)
	    (member_string "-help" argv)
	    (member_string "--help" argv)
	    (member_string "-?" argv))
	(make_diph_index_help))
    (while o
      (begin
	(cond
	 ((string-equal "-eval" (car o))
	  (if (not (cdr o))
	      (make_diph_index_error "no file specified to load"))
	  (if (string-matches (car (cdr o)) "^(.*")
	      (eval (read-from-string (car (cdr o))))
	      (load (car (cdr o))))
	  (set! o (cdr o)))
	 ((string-equal "-lab_dir" (car o))
	  (if (not (cdr o))
	      (make_diph_index_error "no lab_dir file specified"))
	  (set! lab_dir (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-lab_ext" (car o))
	  (if (not (cdr o))
	      (make_diph_index_error "no lab_ext file specified"))
	  (set! lab_ext (car (cdr o)))
	  (set! o (cdr o)))
	 (t
	  (set! files (cons (car o) files))))
	(set! o (cdr o))))
    (if (not (equal? 2 (length files)))
	(make_diph_index_help))
    (set! diphlist_file (car (cdr files)))
    (set! diphindex_file (car files))
))

(define (make_diph_index_error message)
  (format stderr "%s: %s\n" "make_diph_index" message)
  (make_diph_index_help))

(define (find_diphone_boundaries fname diph)
  "(find_diphone_boundaries fname diph)
Find diph in labelfile fname and return index entry."
  (let ((utt (Utterance Text ""))
	(left (string-before diph "-"))
	(right (string-after diph "-"))
	(startp 0)
	(midp 0)
	(lp 0))
    (set! oleft left)
    (set! oright right)
    (if (string-matches left "_.*")
	(set! left (string-after left "_")))
    (if (string-matches left ".*_")
	(set! left (string-before left "_")))
    (if (string-matches left ".*\\$")
	(set! left (string-before left "$")))
    (if (string-matches right "_.*")
	(set! right (string-after right "_")))
    (if (string-matches right "\\$.*")
	(set! right (string-after right "$")))
    (utt.relation.load utt 'Segment
		       (string-append lab_dir "/" fname lab_ext))
    (set! segs (utt.relation.items utt 'Segment))
    (set! diphinfo nil)
    (while (and segs (not diphinfo))
      ;; iterate through the segments to find the match 
      ;; for the desired diphone
;      (format t "%s %s %s %f\n" fname diph
;	      (item.name (car segs))
;	      (item.feat (car segs) "end"))
      (cond
       ((string-equal (item.name (car segs)) "DB")
	;; explciit diphone boundary marker
	(set! endp (item.feat (car segs) "end"))
	(set! segs (cdr segs)))
       ((string-matches (item.name (car segs)) ".cl")
	;; closure so break is in the closed bit
	(set! endp (/ (+ midp (item.feat (car segs) "end")) 2.0))
	(set! segs (cdr segs)))
       ((string-matches (item.name (car segs)) "[tdpdkg]")
	;; rather specific check for stop (without closure)
	(set! endp (+ midp (/ (- (item.feat (car segs) "end")
				 midp) 3.0))))
       (t
	(set! endp (/ (+ midp (item.feat (car segs) "end")) 2.0))))

      (if (car segs)
	  (begin
	    (if (and (string-equal lp left)
		     (string-equal (item.name (car segs)) right))
		(set! diphinfo
		      (list diph fname startp midp endp)))
	    (set! startp endp)
	    (set! midp (item.feat (car segs) "end"))
	    (set! lp (item.name (car segs)))
	    (set! segs (cdr segs)))))
    (if (not diphinfo)
	(set! diphinfo (list diph fname 0 0 0)))
    (format t "%l\n" diphinfo)
    diphinfo))

(define (make_diph_index_main)
  (get_options)
  (let ((dlist (fopen diphlist_file "r"))
	(dout (fopen diphindex_file "w")))
    (if (not dlist)
	(make_dip_index_error 
	 (format nil "can't open diphone list input file \"%s\"" 
		 diphlist_file)))
    (if (not dout)
	(make_dip_index_error 
	 (format nil "can't open diphone index output file \"%s\""
		 diphindex_file)))
    (while (not (equal? (set! fname (readfp dlist)) (eof-val)))
      ;; may be a diphone or list of diphones
      (if (consp fname)
        (let ((t1 fname))
          (set! fname (car t1))
          (set! phones (cadr t1))
          (set! diphs (cadr (cdr t1))))
        (begin ;; hmm this is probably just very old compatability 
          (set! diphs (readfp dlist))
          (set! phones (readfp dlist))  ;; phones
         ))
      (if (not (consp diphs)) (set! diphs (list diphs)))
      (set! diphindex 
	    (append
	     (mapcar
	      (lambda (d)
		(find_diphone_boundaries 
		 fname                ;; the label fname
		 d                    ;; the diphone name itself
		 ))
	      diphs)
	     diphindex)))
    (format dout "EST_File index\n")
    (format dout "DataType ascii\n")
    (format dout "NumEntries %d\n" (length diphindex))
    (format dout "EST_Header_End\n")
    (mapcar
     (lambda (a)
       (format dout "%s %s %s %s %s\n"
	       (nth 0 a)  ; diphone
	       (nth 1 a)  ; file
	       (nth 2 a)  ; start
	       (nth 3 a)  ; mid
	       (nth 4 a)  ; end
	       ))
     diphindex)
    (fclose dout)
    (fclose dlist)
    ))

(make_diph_index_main)
