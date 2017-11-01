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
;;;                                                                       ;;
;;;  A Scheme for a Japanese set of diphones                               ;;
;;;  Inspired by Steve Isard's diphone schemas from CSTR, University of   ;;
;;;  Edinburgh                                                            ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; A diphone list for NHG_phones

(set! svowels '(a i u e o))
(set! lvowels '(a: i: u: e: o:))
(set! uvowels '(I U))  ;; so-called unvoiced vowels
(set! vowels svowels)  ;; Lets only do short vowels (long vowels are double)
(set! palatable_vowels '(a u o))

;; Ones which can go with all vowels (I know I'm gonna get this wrong)
(set! all_consonants '(k g h p b m n r t d ch ts j s sh z f v w y ))
(set! geminates '(Qk Qs Qsh Qt Qch Qts Qh Qf Qg Qz Qj Qd Qb Qp 
		     Qky Qshy Qchy Qpy))
(set! palatals '(ky shy chy ny hy my ty gy jy py by))
(set! silence 'pau)

;; Index by vowels of which consonants cannot preceed them
(set! invalid_cv
      '((a ch j sh ts Qch Qj Qsh Qts)
	(i t d ts s z w y Qt Qd Qts Qs Qz 
	   Qky Qshy Qchy Qpy ky shy chy ny hy my ty gy jy py by)
	(u t d sh ch Qt Qd Qsh Qch)
	(e ch j sh ts w y Qch Qj Qsh Qts
	   Qky Qshy Qchy Qpy ky shy chy ny hy my ty gy jy py by)
	(o ch ts j sh Qch Qts Qj Qsh)
	(U t d sh ch tt dd ssh cch)
	(I t d ts s z w y tt dd tts ss zz )
	(a: ch j sh ts Qch Qj Qsh Qts)
	(i: t d ts s z w y Qt Qd Qts Qs Qz 
	   Qky Qshy Qchy Qpy ky shy chy ny hy my ty gy jy py by)
	(u: t d sh ch Qt Qd Qsh Qch)
	(e: ch j sh ts w y Qch Qj Qsh Qts
	   Qky Qshy Qchy Qpy ky shy chy ny hy my ty gy jy py by)
	(o: ch ts j sh Qch Qts Qj Qsh)
	))
	
(set! cvc-carrier '((pau t a ) (pau)))
(set! vv-carrier '((pau t a k) (t a pau)))
(set! vNv-carrier '((pau t a k) (t a pau)))
(set! Nc-carrier '((pau t a N) (t a pau)))
(set! silv-carrier '(() (t a pau)))
(set! silc-carrier '(() (t a pau)))
(set! vsil-carrier '((pau t a k ) ()))
(set! Nsil-carrier '((pau t a k a ) ()))
(set! pc-carrier '((pau t a k ) ( t a pau)))
(set! gk-carrier '((pau t a k ) ( t a pau)))

;;; These functions simply fill out the nonsense words
;;; from the carriers and vowel consonant definitions

(define (list-cvc)
  (let ((longlist nil))
    (mapcar
     (lambda (v)
       (mapcar
	(lambda (c)
	  (let ((dlist (list (string-append v "-" c)))
		(rc c) (rv 'a))
	    (if (member c (cdr (assoc v invalid_cv)))
		(set! rc 'k)  ;; k always works
		(set! dlist
		      (cons (string-append c "-" v) dlist)))
	    (if (member c (cdr (assoc rv invalid_cv)))
		(if (member c (cdr (assoc (set! rv 'i) invalid_cv)))
		    (set! rv 'u)))
	    (set! longlist
		  (cons
		   (list dlist
			 (append (car cvc-carrier) (list rc v c rv) 
				 (car (cdr cvc-carrier))))
		   longlist))))
	all_consonants))
       vowels)
    (reverse longlist)))

(define (list-vv)
  (let ((longlist nil))
    (mapcar
    (lambda (v1)
      (mapcar
       (lambda (v2)
	 (set! longlist
	       (cons
		(list
		 (list (string-append v1 "-" v2))
		 (append (car vv-carrier) (list v1 v2) 
			 (car (cdr vv-carrier))))
		longlist)))
       vowels))
    vowels)
    (reverse longlist)))

(define (list-N)
  (let ((longlist nil))
   (mapcar
    (lambda (v) 
      (set! longlist
	    (cons
	     (list
	      (list (string-append v "-" 'N)
		    (string-append 'N "-" v))
	      (append (car vNv-carrier) (list v 'N v) (car (cdr vNv-carrier))))
	     longlist)))
    vowels)
   (mapcar
    (lambda (c) 
      (let ((rv 'a))
	(if (member c (cdr (assoc rv invalid_cv)))
	    (if (member c (cdr (assoc (set! rv 'i) invalid_cv)))
		(set! rv 'u)))      
	(set! longlist
	    (cons
	     (list
	      (list (string-append 'N "-" c))
	      (append (car Nc-carrier) (list c rv) (car (cdr Nc-carrier))))
	     longlist))))
    all_consonants)
   (reverse longlist)))


(define (list-sil)
  (let ((longlist nil))
   (mapcar
    (lambda (v) 
      (set! longlist
	    (cons
	     (list
	      (list (string-append silence "-" v))
	      (append (car silv-carrier) (list silence v) 
		      (car (cdr silv-carrier))))
	     longlist)))
    vowels)
   (mapcar
    (lambda (c) 
      (let ((rv 'a))
	(if (member c (cdr (assoc rv invalid_cv)))
	    (if (member c (cdr (assoc (set! rv 'i) invalid_cv)))
		(set! rv 'u)))      
	(set! longlist
	    (cons
	     (list
	      (list (string-append silence "-" c))
	      (append (car silc-carrier) (list silence c rv) 
		      (car (cdr silc-carrier))))
	     longlist))))
    all_consonants)
   (mapcar
    (lambda (v) 
      (set! longlist
	    (cons
	     (list
	      (list (string-append v "-" silence))
	      (append (car vsil-carrier) (list v silence) 
		      (car (cdr vsil-carrier))))
	     longlist)))
    vowels)
   (set! longlist
	 (cons
	  (list
	   (list (string-append 'N "-" silence))
	      (append (car Nsil-carrier) (list 'N silence) 
		      (car (cdr Nsil-carrier))))
	  longlist))
   (reverse longlist)))

(define (list-pal)
  (let ((longlist nil))
   (mapcar
    (lambda (v1) 
      (mapcar 
       (lambda (pc)
	 (let ((rv v1)
	       (dlist (list (string-append v1 "-" pc))))
	   (if (member rv palatable_vowels)
	       (set! dlist 
		     (cons
		      (string-append pc "-" rv)
		      dlist))
	       (set! rv 'a))
	   (set! longlist
		 (cons
		  (list
		   dlist
		   (append (car pc-carrier) (list v1 pc rv)
			   (car (cdr pc-carrier))))
		  longlist))))
       palatals))
    vowels)
   (reverse longlist)))

(define (list-gem)
  (let ((longlist nil))
   (mapcar
    (lambda (v1) 
      (mapcar 
       (lambda (gk)
	 (let ((rv v1)
	       (dlist (list (string-append v1 "-" gk))))
	    (if (member gk (cdr (assoc rv invalid_cv)))
		(if (member gk (cdr (assoc (set! rv 'i) invalid_cv)))
		    (set! rv 'u))
		(set! dlist 
		      (cons
		       (string-append gk "-" rv) dlist)))
	    (set! longlist
		  (cons
		   (list
		    dlist
		    (append (car gk-carrier) (list v1 gk rv)
			    (car (cdr gk-carrier))))
		   longlist))))
       geminates))
    vowels)
   (reverse longlist)))

;;; End of individual generation functions

(define (diphone-gen-list)
  "(diphone-gen-list)
Returns a list of nonsense words as phone strings."
  (append
   (list-cvc)   ;; consonant-vowel-consonant
   (list-vv)    ;; vowel-vowel
   (list-N)     ;; syllable final N
   (list-sil)   ;; into and out of silence
   (list-pal)   ;; palatals
   (list-gem)   ;; geminates
   (list
    '(("pau-pau") (pau k a k a pau pau)))
   ))

(define (Diphone_Prompt_Setup)
 "(Diphone_Prompt_Setup)
Called before synthesizing the prompt waveforms.  Cross language prompts
from US male (for gaijin male)."
 (voice_kal_diphone)  ;; US male voice
 (set! FP_F0 90)
 (set! diph_do_db_boundaries nil) ;; cross-lang confuses this
 )

;; Because I'm actually generating the prompts using US English
;; (apologies to all) I map the Japanese phone to one or more English
;; phone sthat'll be roughly correct.  Note this is *not* as hacky as it
;; appears the English phones are distinguiable enough that the auto-labelled
;; will (mostly) find the right types of phone when these are spoken in
;; actual Japanese.
(set! nhg2radio_map
      '((a aa)
	(i iy)
	(o ow)
	(u uw)
	(e eh)
	(ts t s)
	(N n)
	(h hh)
	(Qk k)
	(Qg g)
	(Qd d)
	(Qt t)
	(Qts t s)
	(Qch t ch)
	(Qh  hh hh)
	(Qf f f)
	(Qj jh)
	(j jh)
	(Qs s)
	(Qsh sh)
	(Qz z)
	(Qp p)
	(Qb b)
	(Qky k y)
	(Qshy sh y)
	(Qchy ch y)
	(Qpy p y )
	(ky k y)
	(gy g y)
	(jy jh y)
	(chy ch y)
	(ty t y)
	(shy sh y)
	(hy hh y)
	(py p y)
	(by b y)
	(my m y)
	(ny n y)
	(ry r y)))

(define (Diphone_Prompt_Word utt)
  "(Diphone_Prompt_Word utt)
Specify specific modifications of the utterance before synthesis
specific to this particular phone set."
  (mapcar
   (lambda (s)
     (let ((n (item.name s))
	   (newn (cdr (assoc_string (item.name s) nhg2radio_map))))
       (cond
	((cdr newn)  ;; its a dual one
	 (let ((newi (item.insert s (list (car (cdr newn))) 'after)))
	   (item.set_feat newi "end" (item.feat s "end"))
	   (item.set_feat s "end"
			  (/ (+ (item.feat s "segment_start")
				(item.feat s "end"))
			     2))
	   (item.set_name s (car newn))))
	(newn
	 (item.set_name s (car newn)))
	(t
	 ;; as is
	 ))))
   (utt.relation.items utt 'Segment))
  utt)


(provide 'ja_schema)
