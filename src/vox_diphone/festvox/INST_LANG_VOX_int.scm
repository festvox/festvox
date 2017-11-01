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
;;;  Accent and F0 prediction
;;;

;;; Phrase breaks
;;;    use punctuation 
(set! INST_LANG_phrase_cart_tree
'
((lisp_token_end_punc in ("'" "\"" "?" "." "," ":" ";"))
  ((B))
  ((n.name is 0)
   ((B))
   ((NB)))))

;;; Intonation
(set! INST_LANG_accent_cart_tree
  '
  (
   (R:SylStructure.parent.gpos is content)
    ( (stress is 1)
       ((Accented))
       ((NONE))
    )
  )
)

(define (INST_LANG_VOX_targ_func1 utt syl)
 "(INST_LANG_VOX_targ_func1 utt syl)
Simple hat accents."
  (let ((start (item.feat syl 'syllable_start))
        (end (item.feat syl 'syllable_end))
        (ulen (item.feat (utt.relation.last utt 'Segment ) 'segment_end))
        nstart nend fustart fuend fuend fstart fend)
        (set! nstart (/ start ulen))
        (set! nend (/ end ulen))
        (set! fustart '130)
        (set! fuend   '110)
        (set! fstart  (+ (* (- fuend fustart) nstart) fustart))
        (set! fend    (+ (* (- fuend fustart) nend) fustart))
      
    (cond
      ((equal? (item.feat syl "R:Intonation.daughter1.name") "Accented")
       (list
        (list start fstart)
        (list (+ start 0.010) (+ fstart 10 ))
        (list (- end   0.010) (+ fstart 8 ))   
        (list end   fend) 
	))
      ((not (item.next syl))
       (list
	(list end fuend)))
      ((not (item.prev syl))
       (list
	(list start fustart)))
      (t
       nil))))


(provide 'INST_LANG_VOX_int)
