;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                 and Alan W Black and Kevin A. Lenzo                 ;;;
;;;                         Copyright (c) 2000                          ;;;
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
;;; Simple pronouncing mode for telephone numbers (probably US centric) ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (saytel numberstring)
  (let ((utt))
    (set! tel_old_token_to_words token_to_words)
    (unwind-protect
     (begin
       (set! token_to_words tel_token_to_words)
       (set! utt (SayText numberstring))))
    (set! token_to_words tel_old_token_to_words)
    utt))

(set! tel_tiny_break (list '(name "<break>") '(pbreak mB)))
(set! tel_small_break (list '(name "<break>") '(pbreak B)))
(set! tel_large_break (list '(name "<break>") '(pbreak BB)))

(define (tel_token_to_words token name)
  "(tel_token_to_words token name)
Telephone recogniser."
  (cond
   ((and (equal? 3 (length name))
	 (string-matches name "[0-9][0-9][0-9]"))
    (item.set_feat token "token_pos" "digits")
    (cond
     ((or (string-matches (item.feat token "n.name")
			  "[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]")
	  (and (string-matches (item.feat token "n.name")
			       "[0-9][0-9][0-9]")
	       (string-matches (item.feat token "nn.name")
			       "[0-9][0-9][0-9][0-9]")))
      (mapcar
       (lambda (a)
	 (if (string-equal a "zero")
	     "oh"
	     a))
       (builtin_english_token_to_words token name)))
     (t
       (builtin_english_token_to_words token name))))
   ((string-matches name "[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]")
    (item.set_feat token "token_pos" "digits")
    (append
     (builtin_english_token_to_words token (string-before name "-"))
     (list tel_small_break)
     (builtin_english_token_to_words token (string-after name "-"))))
   ((string-matches name "[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]")
    (item.set_feat token "token_pos" "digits")
    (append
      (mapcar
       (lambda (a)
	 (if (string-equal a "zero")
	     "oh"
	     a))
       (builtin_english_token_to_words token (string-before name "-")))
      (list tel_small_break)
      (tel_token_to_words token (string-after name "-"))))
   ((string-matches name "[0-9][0-9]*")
    (item.set_feat token "token_pos" "digits")
    (builtin_english_token_to_words token name))
   (t
    ;; hopefully we don't get here
    (tel_old_token_to_words token name))))

(provide 'tel)