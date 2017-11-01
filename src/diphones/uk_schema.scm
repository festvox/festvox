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
;;;  Scheme for a MRPA (RP) set of diphones                               ;;
;;;  Inspired by Steve Isard's diphone schemas from CSTR, University of   ;;
;;;  Edinburgh                                                            ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; A diphone list for DARPAbet

(set! vowels '(uh e a o i u ii uu oo aa @@ ai ei oi au ou e@ i@ u@ @ ))
(set! consonants '(p t k b d g s z sh zh f v th dh ch jh h m n ng l y r w ))
(set! onset-only '(h y w))
(set! nocvcs '(y w r l m n ng))
(set! coda-only '(ng))
;(set! syllabics '(el em en))
;(set! special '(dx))  ;; glottal stops ?
(set! silence '#)
;; for consonant clusters
(set! stops '(p b t d k g))
(set! nasals '(n m))
(set! liquids '(l r w))
(set! clusters1
      (append
       (apply
	append
	(mapcar (lambda (b) (mapcar (lambda (a) (list a b)) stops)) liquids))
       (mapcar (lambda (b) (list 's b)) '(l w n m p t k))
       (mapcar (lambda (b) (list 'f b)) '(l r ))))
(set! clusters2
      (mapcar (lambda (b) (list b 'y))  (append stops nasals '(s f v h))))

(set! cvc-carrier '((# t aa ) (aa #)))
(set! vc-carrier '((# t aa t) (aa #)))
(set! cv-carrier '((# t aa ) (t aa #)))
(set! cc-carrier '((# t aa ) (aa t aa #)))
(set! vv-carrier '((# t aa t) (t aa #)))
(set! silv-carrier '(() (t aa #)))
(set! silc-carrier '(() (aa t aa #)))
(set! vsil-carrier '((# t aa t ) ()))
(set! csil-carrier '((# t aa t aa) ()))

;;; These are only the minimum ones, you should (if possible)
;;; consider consonant clusters i.e to distringuish
;;; t aa t - r aa t and t aa - t r aa t
(set! cc1-carrier '((# t aa -) (aa t aa #)))
;;; for t aa - t _y_ uw t aa 
(set! cc2-carrier '((# t aa -) (uu t aa #)))
;;; Open and vowels to (syllable end after vowel)
(set! vcopen-carrier '((# t aa t) (aa #)))
;;; Syllabics
(set! syllabics-carrier1 '((pau t aa ) ( aa pau))) ;; c-syl
(set! syllabics-carrier2 '((pau t aa t) (aa pau))) ;; syl-c
(set! syllabics-carrier3 '((pau t aa t) (t aa pau))) ;; syl-v
(set! syllabics-carrier4 '((pau t aa t) (t aa pau))) ;; syl-syl

;;; These functions simply fill out the nonsense words
;;; from the carriers and vowel consonant definitions

(define (list-cvcs)
  (apply
   append
   (mapcar
    (lambda (v)
      (mapcar
       (lambda (c)
	 (list
	  (list (string-append c "-" v) (string-append v "-" c))
	  (append (car cvc-carrier) (list c v c) (car (cdr cvc-carrier)))))
       (remove-list consonants nocvcs)))
    vowels)))

(define (list-vcs)
  (apply
   append
   (mapcar
    (lambda (v)
      (mapcar
       (lambda (c)
	 (list
	  (list (string-append v "-" c))
	  (append (car vc-carrier) (list v c) (car (cdr vc-carrier)))))
       nocvcs))
    vowels)))

(define (list-cvs)
  (apply
   append
   (mapcar
    (lambda (c)
      (mapcar 
       (lambda (v) 
	 (list
	  (list (string-append c "-" v))
	  (append (car cv-carrier) (list c v) (car (cdr cv-carrier)))))
       vowels))
    nocvcs)))

(define (list-vvs)
  (apply
   append
   (mapcar
    (lambda (v1)
      (mapcar 
       (lambda (v2) 
	 (list
	  (list (string-append v1 "-" v2))
	  (append (car vv-carrier) (list v1 v2) (car (cdr vv-carrier)))))
       vowels))
    vowels)))

(define (list-ccs)
  (apply
   append
   (mapcar
    (lambda (c1)
      (mapcar 
       (lambda (c2) 
       (list
	(list (string-append c1 "-" c2))
	(append (car cc-carrier) (list c1 '- c2) (car (cdr cc-carrier)))))
       (remove-list consonants coda-only)))
    (remove-list consonants onset-only))))

(define (list-silv)
  (mapcar 
   (lambda (v) 
     (list
      (list (string-append silence "-" v))
      (append (car silv-carrier) (list silence v) (car (cdr silv-carrier)))))
   vowels))

(define (list-silc)
  (mapcar 
   (lambda (c) 
     (list
      (list (string-append silence "-" c))
      (append (car silc-carrier) (list silence c) (car (cdr silc-carrier)))))
   (remove-list consonants coda-only)))

(define (list-vsil)
  (mapcar 
   (lambda (v) 
     (list
      (list (string-append v "-" silence))
      (append (car vsil-carrier) (list v silence) (car (cdr vsil-carrier)))))
   vowels))

(define (list-csil)
  (mapcar 
   (lambda (c) 
     (list
      (list (string-append c "-" silence))
      (append (car csil-carrier) (list c silence) (car (cdr csil-carrier)))))
   (remove-list consonants onset-only)))

(define (list-ccclust1)
  (mapcar
   (lambda (c1c2)
     (list
      (list (string-append (car c1c2) "_-_" (car (cdr c1c2))))
      (append (car cc1-carrier) c1c2 (car (cdr cc1-carrier)))))
   clusters1))

(define (list-ccclust2)
  (mapcar
   (lambda (c1c2)
     (list
      (list (string-append (car c1c2) "_-_" (car (cdr c1c2))))
      (append (car cc2-carrier) c1c2 (car (cdr cc2-carrier)))))
   clusters2))

(define (list-vcopen)
  (apply
   append
   (mapcar
    (lambda (v)
      (mapcar 
       (lambda (c) 
	 (list
	  (list (string-append v "$-" c))
	  (append (car vc-carrier) (list v '- c) (car (cdr vc-carrier)))))
       consonants))
    vowels)))

(define (list-syllabics)
  (append
   (apply
    append
    (mapcar
     (lambda (s)
       (mapcar 
	(lambda (c) 
	  (list
	   (list (string-append c "-" s))
	   (append (car syllabics-carrier1) (list c s) (car (cdr syllabics-carrier1)))))
	(remove-list consonants onset-only)))
     syllabics))
   (apply
    append
    (mapcar
     (lambda (s)
       (mapcar 
	(lambda (c) 
	  (list
	   (list (string-append s "-" c))
	   (append (car syllabics-carrier2) (list s '- c) (car (cdr syllabics-carrier2)))))
	(remove-list consonants coda-only)))
     syllabics))
   (apply
    append
    (mapcar
     (lambda (s)
       (mapcar 
	(lambda (v) 
	  (list
	   (list (string-append s "-" v))
	   (append (car syllabics-carrier3) (list s '- v) (car (cdr syllabics-carrier3)))))
	vowels))
       syllabics))))

;;; End of individual generation functions

(define (diphone-gen-list)
  "(diphone-gen-list)
Returns a list of nonsense words as phone strings."
  (append
   (list-cvcs)  ;; consonant-vowel and vowel-consonant
   (list-vcs)  ;; one which don't go in cvc
   (list-cvs)  ;; 
   (list-vvs)  ;; vowel-vowel
   (list-ccs)  ;; consonant-consonant
   (list-ccclust1)   ;; consonant clusters
   (list-ccclust2)   ;; consonant clusters
;   (list-syllabics)
   (list-silv)
   (list-silc)
   (list-csil)
   (list-vsil)
   (list
    '(("#-#") (# t aa t aa # #)))
;   (list-vcopen)    ;; open vowels
   ))

(define (Diphone_Prompt_Word utt)
  "(Diphone_Prompt_Word utt)
Specify specific modifications of the utterance before synthesis
specific to this particulat phone set."
'  (mapcar
   (lambda (s)
     (let ((n (item.name s)))
       (cond
	((string-equal n "el")
	 (item.set_name s "l"))
	((string-equal n "em")
	 (item.set_name s "m"))
	((string-equal n "en")
	 (item.set_name s "n"))
	((string-equal n "er")
	 ;; ked doesn't deal with er properly so we need to insert 
	 ;; an r after the er to get this to work reasonably
	 (let ((newr (item.insert s (list 'r) 'after)))
	   (item.set_feat newr "end" (item.feat s "end"))
	   (item.set_feat s "end" 
			  (/ (+ (item.feat s "segment_start")
				(item.feat s "end"))
			     2))))
	)))
   (utt.relation.items utt 'Segment))
;  (set! phoneme_durations kd_durs)
;  (Parameter.set 'Duration_Stretch '1.2)
;  (Duration_Averages utt)
  )

(define (Diphone_Prompt_Setup)
 "(Diphone_Prompt_Setup)
Called before synthesizing the prompt waveforms.  Defined for AWB
speaker using rab diphone set (UK English) and setting F0."
 (voice_rab_diphone)  ;; UK male voice
 (set! FP_F0 90)
 )


(provide 'uk_schema)
