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
;;; Phonset for Spain Spanish (INST_es_VOX) -> should be called cmu_es_sp
;;;
;;; ari: 30 phones + silence (12 vowels)
;;;      tuned to adjust Spain's phoneset [trying to reproduce Castillian 
;;;      as it would be spoken in Valladolid or Salamanca]
;;;      /v/ and /ll/ are not differenciated from /b/ and /y/ since most
;;;      speakers don't make these distinctions.


(defPhoneSet
  cmu_es
  ;;;  Phone Features
  (;; 1 vowel or consonant
   (vc + -)  
   ;; 2 vowel length: short long dipthong schwa
   (vlng s l d a 0)
   ;; 3 vowel height: high mid low
   (vheight 1 2 3 - 0)
   ;; 4 vowel frontness: front mid back
   (vfront 1 2 3 - 0)
   ;; 5 lip rounding
   (vrnd + - 0)
   ;; 6 consonant type: stop fricative affricative nasal liquid
   (ctype s f a n l 0)
   ;; 7 place of articulation: labial alveolar palatal labio-dental
   ;;                         dental velar
   (cplace l a p b d v 0)
   ;; 8 consonant voicing
   (cvox + - 0)
   )

  ;; borja: all the features are almost ok, only some problems:
  ;; r is a tap, rr is a trill.  We would need "vibrant". Now, coded as liquid.
  ;; l and ll are lateral. Now, coded as liquid (probably it's the samething)
  ;; The bdg/BDG distinction (stop/aproximant) is not done.
  ;; The i and u aproximants (sampa j and w, labio, agua) are not considered,
  ;; normal i and u used instead.
  ;; The fricative 'y' (sampa jj, ayer) is not considered [Argentinian dialect].
  (
   ;;  1 2 3 4 5 6 7 8 
   (pau  - 0 0 0 0 0 0 -) ;; silence
   (a  + l 3 2 - 0 0 -) ;; ari: rosa 
   (e  + l 2 1 - 0 0 -) ;; ari: gest'o
   (i  + l 1 1 - 0 0 -) ;; ari: ni~nera
   (o  + l 2 3 + 0 0 -) ;; ari: zapato
   (u  + l 1 3 + 0 0 -) ;; ari: muchacho

   (iW  + s 1 1 - 0 0 -) ;; ari: iogur (weak vowels in dipthongs)
   (uW  + s 1 3 + 0 0 -) ;; ari: agua, washington (weak vowels in dipthongs)

   (aS + l 3 2 - 0 0 -) ;; ari: mano
   (eS + l 2 1 - 0 0 -) ;; ari: mesa
   (iS + l 1 1 - 0 0 -) ;; ari: pino
   (oS + l 2 3 + 0 0 -) ;; ari: 'opera
   (uS + l 1 3 + 0 0 -) ;; ari: tortura

   (p  - 0 - - - s l -) ;; ari: pollo
   (t  - 0 - - - s d -) ;; ari: torre
   (k  - 0 - - - s v -) ;; ari: casa, kilo, queso
   (b  - 0 - - - s l +) ;; ari: bola, vivir
   (d  - 0 - - - s d +) ;; ari: cada
   (g  - 0 - - - s v +) ;; ari: gato, guerra, wisky (huevo?)

   (f  - 0 - - - f b -) ;; ari: feo
   (s  - 0 - - - f a -) ;; ari: caso, excavar?
   (j  - 0 - - - f v -) ;; ari: mexico, juan, gesto, caja 
   (ch - 0 - - - a p -) ;; ari: muchacho
   (th - 0 - - - f d -) ;; ari: cebolla, zapato 

   (m  - 0 - - - n l +) ;; ari: mano
   (n  - 0 - - - n a +) ;; ari: nada
   (ny - 0 - - - n p +) ;; ari: ni~na

   (l  - 0 - - - l a +) ;; ari: lado
   (y  - 0 - - - l p +) ;; ari: coyote, ya (callar, llorar)
   (r  - 0 - - - l a +) ;; ari: trozo
   (rr - 0 - - - l a +) ;; ari: rosa, carro

;; we don't need the phoneme /x/ since 'x' maps to /k/+/s/...

;; ari: we could add this 'foreign' phoneme since it appears quite often in 
;; news texts, and Spanish people do pronunce it as /sh/ and not /s epsilon/ 
;;   (sh - 0 - - - f p -) ;; ari: Schubert, Sasha, Shirley
  )
)

(PhoneSet.silences '(pau))

(define (INST_es_VOX::select_phoneset)
  "(INST_es_VOX::select_phoneset)
Set up phone set for cmu_es."
  (Parameter.set 'PhoneSet 'cmu_es)
  (PhoneSet.select 'cmu_es)
)

(define (INST_es_VOX::reset_phoneset)
  "(INST_es_VOX::reset_phoneset)
Reset phone set for cmu_es."
  t
)

(provide 'INST_es_VOX_phoneset)








