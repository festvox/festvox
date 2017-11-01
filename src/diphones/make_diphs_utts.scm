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
;;;  Searches the given set of utterances for a diphone set             ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (load_all_utts promptfile)
  (mapcar
   (lambda (f)
     (format t "loading %s\n" (car f))
     (utt.load nil (format nil "festival/utts/%s.utt" (car f))))
   (load promptfile t)))

(defvar diphone_table nil)

(define (build_all_diphone_list utts)
  "(build_all_diphone_list utts)
Build a table of all instances of diphones so they can be ordered."
  (set! diphone_table nil)
  (mapcar
   (lambda (u)
     (format t "adding %s\n" (utt.feat u "fileid"))
     (mapcar 
      (lambda (s)
	(if (item.next s)
	    (add_diphone_to_table 
	     (intern (item.name s))
	     (intern (item.feat s "n.name"))
	     s)))
      (utt.relation.items u 'Segment)))
   utts)
  t)

(define (add_diphone_to_table p1 p2 s)
  "(add_diphone_to_table p1 p2 s)
Add diphone to table"
  (let ((a (assoc p1 diphone_table)))
    (if a
	(let ((b (assoc p2 (cdr a))))
	  (if b 
	      (set-cdr! b (cons s (cdr b)))
	      (set-cdr! a (cons (list p2 s) (cdr a)))))
	(set! diphone_table
	      (cons
	       (list p1 (list p2 s))
	       diphone_table)))
    t))

(define (select-diphones-1 table)
  "(select-diphones-1 table)
Naively find the first diphone in the list."
  (apply
   append
   (mapcar 
    (lambda (p1s)
      (mapcar 
       (lambda (p2s) 
	 (set! selected (car (cdr p2s)))
	 (let ((s1 selected)
	       (s2 (item.next selected))
	       (p1 (car p1s))
	       (p2 (car p2s)))
	   (list
	    (format nil "%s-%s" p1 p2)
	    (utt.feat (item.get_utt s1) "fileid")
	    (if (string-equal "s" (item.feat s1 "ph_ctype"))
		(+ (item.feat s1 "segment_start")
		   (/ (item.feat s1 "segment_duration") 3))
		(+ (item.feat s1 "segment_start")
		   (/ (item.feat s1 "segment_duration") 2)))
	    (item.feat s1 "end")
	    (if (or (string-equal "s" (item.feat s2 "ph_ctype"))
		    (phone_is_silence (item.name s2)))
		(+ (item.feat s2 "segment_start")
		   (/ (item.feat s2 "segment_duration") 3))
		(+ (item.feat s2 "segment_start")
		   (/ (item.feat s2 "segment_duration") 2))))))
       (cdr p1s)))
    table)))

(define (find_best_diphone_2 cands)
  "(find_best_diphone_2 cands)
Find the best out of this set."
  (cond
   ((null (cdr cands)) (car cands))
   ((and (string-equal "-" (item.feat (car cands) "ph_vc"))
	 (string-equal "-" (item.feat (car cands) "n.ph_vc")))
    (if (string-equal "0" (item.feat (car cands) "syl_final"))
	(find_best_diphone_2 (cdr cands))
	(car cands)))
   ((and (string-equal "+" (item.feat (car cands) "ph_vc"))
	 (string-equal "-" (item.feat (car cands) "n.ph_vc")))
    (if (string-equal "0" (item.feat (car cands) "R:SylStructure.parent.stress"))
	(find_best_diphone_2 (cdr cands))
	(car cands)))
   ((and (string-equal "-" (item.feat (car cands) "ph_vc"))
	 (string-equal "+" (item.feat (car cands) "n.ph_vc")))
    (if (or (string-equal "0" (item.feat (car cands) 
					 "R:SylStructure.parent.stress"))
	    (not (string-equal "mid" "R:SylStructure.parent.position_type")))
	(find_best_diphone_2 (cdr cands))
	(car cands)))
   (t
    (car cands))))

(define (select-diphones-2 table)
  "(select-diphones-2 table)
Exclude examples from non-stressed syllable if possible."
  (apply
   append
   (mapcar 
    (lambda (p1s)
      (mapcar 
       (lambda (p2s) 
	 (set! selected (find_best_diphone_2 (cdr p2s)))
	 (let ((s1 selected)
	       (s2 (item.next selected))
	       (p1 (car p1s))
	       (p2 (car p2s)))
	   (list
	    (format nil "%s-%s" p1 p2)
	    (utt.feat (item.get_utt s1) "fileid")
	    (if (string-equal "s" (item.feat s1 "ph_ctype"))
		(+ (item.feat s1 "segment_start")
		   (/ (item.feat s1 "segment_duration") 3))
		(+ (item.feat s1 "segment_start")
		   (/ (item.feat s1 "segment_duration") 2)))
	    (item.feat s1 "end")
	    (if (string-equal "s" (item.feat s2 "ph_ctype"))
		(+ (item.feat s2 "segment_start")
		   (/ (item.feat s2 "segment_duration") 3))
		(+ (item.feat s2 "segment_start")
		   (/ (item.feat s2 "segment_duration") 2))))))
       (cdr p1s)))
    table)))

(define (save_diphone_index diphs ofile)
  "(save_diphone_index diphs ofile)
Save a diphone index as best you can."
  (let ((dout (fopen ofile "w")))
    (format dout "EST_File index\n")
    (format dout "DataType ascii\n")
    (format dout "NumEntries %d\n" (length diphs))
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
     diphs)
    (fclose dout)))

(define (make_diphone_index promptfile indexfile)
  "(make_diphone_index utts indexfile)
Find diphones in utts and build a diphone index in indexfile."
  (set! utts (load_all_utts promptfile))
  (build_all_diphone_list utts)
  (set! diphlist (select-diphones-2 diphone_table))
  (save_diphone_index diphlist indexfile))
  

    





