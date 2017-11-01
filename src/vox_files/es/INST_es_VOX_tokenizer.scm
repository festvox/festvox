;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                  and Alan W Black and Kevin Lenzo                   ;;;
;;;                  and Ariadna Font Llitjos                           ;;;
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
;;; Tokenizer for Spanish
;;;
;;; Particularly numbers and symbols.
;;; 
;;;  To share this among voices you need to promote this file to
;;;  to say festival/lib/cmu_es/ so others can use it.
;;;  Particularly numbers and symbols.
;;; 
;;; As the database has no dipthongs, numbers sound much
;;; better removing the weak vowel ("ventiuno" instead of "veintiuno")
;;; Many speakers do this, so no problem with it.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Load any other required files

(set! spanish_guess_pos 
'((fn
    el la lo los las 
    un una unos unas
;;
    mi tu su mis tus sus 
    nuestra vuestra nuestras vuestras nuestro vuestro nuestros vuestros
    me te le nos os les se
    al del
;;
    a ante bajo cabe con contra de desde en entre
    hacia hasta para por sin sobre tras mediante
;;
    y e ni mas o "\'o" u pero aunque si 
    porque que quien cuando como donde cual cuan
    aun pues tan mientras sino )
  (partnums
   dieci venti trentai cuarentai cincuentai sesentai ochentai noventai)
  )
)

;;; Voice/es token_to_word rules 
(define (INST_es_VOX::token_to_words token name)
  "(INST_es_VOX::token_to_words token name)
Specific token to word rules for the voice INST_es_VOX.  Returns a list
of words that expand given token with name."
  (cond
   ((string-matches name "[1-9][0-9]+")
    (cmu_es::number token name))
   (t ;; when no specific rules apply do the general ones
    (list name))))

; from old version
;   (cond
;    ((string-matches name "[1-9][0-9]+")
;     (spanish_number name))
;    ((not (lts.in.alphabet name 'spanish_downcase))
;     ;; It contains some other than the lts can deal with
;     (let ((subwords))
;       (item.set_feat token "pos" "nn")
;       (mapcar
;        (lambda (letter)
; 	 ;; might be symbols or digits
; 	 (set! subwords
; 	       (append
; 		subwords
; 		(cond
; 		 ((string-matches letter "[0-9]")
; 		  (spanish_number letter))
; 		 ((string-matches letter "[A-Z¡…Õ”⁄‹—]")
; 		    (spanish_downcase letter))
; 		 (t
; 		  (list letter))))))
;        (symbolexplode name))
;       subwords))
;    (t
;     (list name))))

(define (cmu_es::number token name)
  "(cmu_es::number token name)
Return list of words that pronounce this number in Spanish."

  (if (string-matches name "0")
      (list "cero")
      (cmu_es::number_from_digits (symbolexplode name)))) ;; spanish_number_from_digits

(define (just_zeros digits)
"(just_zeros digits)
If this only contains 0s then we just do something different."
 (cond
  ((not digits) t)
  ((string-equal "0" (car digits))
   (just_zeros (cdr digits)))
  (t nil)))

(define (cmu_es::number_from_digits digits)
  "(cmu_es::number_from_digits digits)
Takes a list of digits and converts it to a list of words
saying the number."
  (let ((l (length digits)))
    (cond
     ((equal? l 0)
      nil)
     ((string-equal (car digits) "0")
      (cmu_es::number_from_digits (cdr digits)))
     ((equal? l 1);; single digit
      (cond 
       ((string-equal (car digits) "0") (list "cero"))
       ((string-equal (car digits) "1") (list "un"))
       ((string-equal (car digits) "2") (list "dos"))
       ((string-equal (car digits) "3") (list "tres"))
       ((string-equal (car digits) "4") (list "cuatro"))
       ((string-equal (car digits) "5") (list "cinco"))
       ((string-equal (car digits) "6") (list "seis"))
       ((string-equal (car digits) "7") (list "siete"))
       ((string-equal (car digits) "8") (list "ocho"))
       ((string-equal (car digits) "9") (list "nueve"))
       ;; fill in the rest
       (t (list "equis"))));; $$$ what should say?
     ((equal? l 2);; less than 100
      (cond
       ((string-equal (car digits) "0");; 0x
	(cmu_es::number_from_digits (cdr digits)))
     
       ((string-equal (car digits) "1");; 1x
	(cond
	 ((string-equal (car (cdr digits)) "0") (list "diez"))
	 ((string-equal (car (cdr digits)) "1") (list "once"))
	 ((string-equal (car (cdr digits)) "2") (list "doce"))
	 ((string-equal (car (cdr digits)) "3") (list "trece"))
	 ((string-equal (car (cdr digits)) "4") (list "catorce"))
	 ((string-equal (car (cdr digits)) "5") (list "quince"))
	 (t 
	  (cons "dieci" (cmu_es::number_from_digits (cdr digits))))))
     
       ((string-equal (car digits) "2");; 2x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "veinte")
	    (cons "venti" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "3");; 3x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "treinta")
	    (cons "trentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "4");; 4x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "cuarenta")
	    (cons "cuarentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "5");; 5x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "cincuenta")
	    (cons "cincuentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "6");; 6x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "sesenta")
	    (cons "sesentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "7");; 7x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "setenta")
	    (cons "setentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "8");; 8x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "ochenta")
	    (cons "ochentai" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "9");; 9x
	(if (string-equal (car (cdr digits)) "0") 
	    (list "noventa")
	    (cons "noventai" (cmu_es::number_from_digits (cdr digits)))))

       ))

     ((equal? l 3);; in the hundreds
      (cond 
     
       ((string-equal (car digits) "1");; 1xx
	(if (just_zeros (cdr digits)) (list "cien")
	    (cons "ciento" (cmu_es::number_from_digits (cdr digits)))))

       ((string-equal (car digits) "5");; 5xx
	(cons "quinientos" (cmu_es::number_from_digits (cdr digits))))

       ((string-equal (car digits) "7");; 7xx
	(cons "setecientos" (cmu_es::number_from_digits (cdr digits))))

       ((string-equal (car digits) "9");; 9xx
	(cons "novecientos" (cmu_es::number_from_digits (cdr digits))))

       (t;; ?xx
	(append (cmu_es::number_from_digits (list (car digits))) 
		(list "cientos") 
		(cmu_es::number_from_digits (cdr digits))))
       ))

     ((< l 7)
      (let ((sub_thousands 
	     (list 
	      (car (cdr (cdr (reverse digits))))
	      (car (cdr (reverse digits)))
	      (car (reverse digits))))
	    (thousands (reverse (cdr (cdr (cdr (reverse digits)))))))
	(set! x (cmu_es::number_from_digits thousands))
	(append
	 (if (string-equal (car x) "un") nil x)
	 (list "mil")
	 (cmu_es::number_from_digits sub_thousands))))

     ((< l 13)
      (let ((sub_million 
	     (list 
	      (car (cdr (cdr (cdr (cdr (cdr(reverse digits)))))))
	      (car (cdr (cdr (cdr (cdr (reverse digits))))))
	      (car (cdr (cdr (cdr (reverse digits)))))
	      (car (cdr (cdr (reverse digits))))
	      (car (cdr (reverse digits)))
	      (car (reverse digits))
	      ))
	    (millions (reverse (cdr (cdr (cdr (cdr (cdr (cdr (reverse digits))))))))))
	(set! x (cmu_es::number_from_digits millions))
	(append
	 (if (string-equal (car x) "un") 
	     (list "un" "millon") 
	     (append x (list "millones")))
	 (cmu_es::number_from_digits sub_million))))

     (t
      (list "un" "numero" "muy" "gr'aaaaaandee")))))

(define (INST_es_VOX::select_tokenizer)
  "(INST_es_VOX::select_tokenizer)
Set up tokenizer for Spanish."
  (Parameter.set 'Language 'cmu_es)

  (set! token_to_words INST_es_VOX::token_to_words)
)

(define (INST_es_VOX::reset_tokenizer)
  "(INST_es_VOX::reset_tokenizer)
Reset any globals modified for this voice.  Called by 
(INST_es_VOX::voice_reset)."
  ;; None

  t
)

(provide 'INST_es_VOX_tokenizer)
