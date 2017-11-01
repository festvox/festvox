;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
;;;
;;;  log and/or zscore duration
;;;

(set! seg_dur_info (load "festival/dur/etc/durs.meanstd" t))

(define (zscore_log_dur seg)
  (let ((di (assoc_string (item.name seg) seg_dur_info)))
    (cond
     ((not di)
      (format stderr "zscore_log_dur: %s no info found\n" 
	      (item.name seg))
      (/ (- (log 0.100)
	    (car (cdr di)))
	 (car (cdr (cdr di))))      )
     (t
      (/ (- (log (item.feat seg "duration"))
	    (car (cdr di)))
	 (car (cdr (cdr di))))))))

(define (zscore_dur seg)
  (let ((di (assoc_string (item.name seg) seg_dur_info)))
    (cond
     ((not di)
      (format stderr "zscore_dur: %s no info found\n" 
	      (item.name seg))
      (/ (- 0.100
	    (car (cdr di)))
	 (car (cdr (cdr di))))      )
     (t
      (/ (- (item.feat seg "duration")
	    (car (cdr di)))
	 (car (cdr (cdr di))))))))

(define (onset_has_ctype seg type)
  ;; "1" if onset contains ctype
  (let ((syl (item.relation.parent seg 'SylStructure)))
    (if (not syl)
	"0" ;; a silence 
	(let ((segs (item.relation.daughters syl 'SylStructure))
	      (v "0"))
	  (while (and segs 
		      (not (string-equal 
			    "+" 
			    (item.feat (car segs) "ph_vc"))))
		 (if (string-equal 
		      type
		      (item.feat (car segs) "ph_ctype"))
		     (set! v "1"))
		 (set! segs (cdr segs)))
	  v))))

(define (coda_has_ctype seg type)
  ;; "1" if coda contains ctype
  (let ((syl (item.relation.parent seg 'SylStructure)))
    (if (not syl)
	"0" ;; a silence 
	(let ((segs (reverse (item.relation.daughters
			      syl 'SylStructure)))
	      (v "0"))
	  (while (and segs 
		      (not (string-equal 
			    "+" 
			    (item.feat (car segs) "ph_vc"))))
		 (if (string-equal 
		      type
		      (item.feat (car segs) "ph_ctype"))
		     (set! v "1"))
		 (set! segs (cdr segs)))
	  v))))

(define (onset_stop seg)
  (onset_has_ctype seg "s"))
(define (onset_fric seg)
  (onset_has_ctype seg "f"))
(define (onset_nasal seg)
  (onset_has_ctype seg "n"))
(define (onset_glide seg)
  (let ((l (onset_has_ctype seg "l")))
    (if (string-equal l "0")
	(onset_has_ctype seg "r")
	"1")))

(define (coda_stop seg)
  (coda_has_ctype seg "s"))
(define (coda_fric seg)
  (coda_has_ctype seg "f"))
(define (coda_nasal seg)
  (coda_has_ctype seg "n"))
(define (coda_glide seg)
  (let ((l (coda_has_ctype seg "l")))
    (if (string-equal l "0")
	(coda_has_ctype seg "r")
	"1")))
  
