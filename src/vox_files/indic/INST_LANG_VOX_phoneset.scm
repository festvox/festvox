;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                      Copyright (c) 1998-2017                        ;;;
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
;;; Phoneset for Indic Languages
;;;

;;; To give a "sampa" phoneset definition (though there are so many
;;; different sampas these may not be exactly the sampas you're looking for

(defPhoneSet
  cmu_indic
  ;;;  Phone Features
  (;; vowel or consonant
   (clst + - 0)
   (vc + - 0)
   ;; vowel length: short long dipthong schwa
   (vlng s l d a 0)
   ;; vowel height: high mid low
   (vheight 1 2 3 0 -)
   ;; vowel frontness: front mid back
   (vfront 1 2 3 0 -)
   ;; lip rounding
   (vrnd + - 0)
   ;; consonant type: stop fricative affricative nasal liquid approximant
   (ctype s f a n l r 0)
   ;; place of articulation: labial alveolar palatal
   ;; labio-dental dental velar glottal
   (cplace l a p b d v g 0)
   ;; consonant voicing
   (cvox + - 0)
   (asp  + - 0)
   (nuk + - 0)
   )
  (
   (    pau   -   -   0   0   0   0   0   0   -   -   -   ) 

   ;; Vowels
   (      A   -   +   s   2   2   -   0   0   0   -   0   ) ; अ
   (     A:   -   +   l   3   2   -   0   0   0   -   0   ) ; आ
   (      i   -   +   s   1   1   -   0   0   0   -   0   ) ; इ
   (     i:   -   +   l   1   1   -   0   0   0   -   0   ) ; ई
   (      u   -   +   s   1   3   +   0   0   0   -   0   ) ; उ
   (     uy   -   +   s   1   3   -   0   0   0   -   0   ) ; Tamil word(morpheme?)-final unrounded u
   (     u:   -   +   l   1   3   +   0   0   0   -   0   ) ; ऊ
   (    9r=   -   +   s   1   3   -   0   0   0   -   0   ) ; ऋ
   (    lr=   -   +   d   2   3   -   0   0   0   -   0   ) ; ऌ

   (      E   -   +   a   2   1   -   0   0   0   -   0   ) ; the schwa in hindi worde rEhna
   (      e   -   +   s   2   1   -   0   0   0   -   0   ) ; ए
   (     e:   -   +   l   2   1   -   0   0   0   -   0   ) ; ए long
   (     aI   -   +   d   2   1   -   0   0   0   -   0   ) ; ऐ
   (      o   -   +   s   2   3   +   0   0   0   -   0   ) ; ओ
   (     o:   -   +   l   2   3   +   0   0   0   -   0   ) ; ओ long
   (     aU   -   +   d   1   3   +   0   0   0   -   0   ) ; औ
   (      M   -   -   0   0   0   0   0   0   0   -   0   ) ; अं
   (      h   -   -   0   0   0   0   f   v   -   +   -   ) ; अः
   (     ay   -   +   s   2   2   -   0   0   0   -   0   ) ; अॅ
   (     ow   -   +   l   3   3   +   0   0   0   -   0   ) ; ऑ

   ;; Consonants
   ;; Velar
   (      k   -   -   0   0   0   0   s   v   -   -   -   ) ; क
   (     kh   -   -   0   0   0   0   s   v   -   +   -   ) ; ख
   (      g   -   -   0   0   0   0   s   v   +   -   -   ) ; ग
   (     gh   -   -   0   0   0   0   s   v   +   +   -   ) ; घ
   (      N   -   -   0   0   0   0   n   v   +   -   -   ) ; ङ

   ;; Palatal
   (      c   -   -   0   0   0   0   a   p   -   -   -   ) ; च
   (     ch   -   -   0   0   0   0   a   p   -   +   -   ) ; छ
   (      J   -   -   0   0   0   0   a   p   +   -   -   ) ; ज
   (     Jh   -   -   0   0   0   0   a   p   +   +   -   ) ; झ
   (     n~   -   -   0   0   0   0   n   p   +   -   -   ) ; ञ

   ;; Retroflex
   (     tr   -   -   0   0   0   0   s   a   -   -   -   ) ; ट
   (     tR   -   -   0   0   0   0   s   a   -   +   -   ) ; ठ
   (     dr   -   -   0   0   0   0   s   a   +   -   -   ) ; ड
   (     dR   -   -   0   0   0   0   s   a   +   +   -   ) ; ढ
   (     nr   -   -   0   0   0   0   n   a   +   -   -   ) ; ण
   
   ;; Dental
   (     tB   -   -   0   0   0   0   s   d   -   -   -   ) ; त
   (    tBh   -   -   0   0   0   0   s   d   -   +   -   ) ; थ
   (     dB   -   -   0   0   0   0   s   d   +   -   -   ) ; द
   (    dBh   -   -   0   0   0   0   s   d   +   +   -   ) ; ध
   (     nB   -   -   0   0   0   0   n   d   +   -   -   ) ; न
   
   ;; Alveolar
   
   (      n   -   -   0   0   0   0   n   a   +   -   -   ) ; ऩ

   ;; Labial
   (      p   -   -   0   0   0   0   s   l   -   -   -   ) ; प
   (     ph   -   -   0   0   0   0   s   l   -   +   -   ) ; फ
   (      b   -   -   0   0   0   0   s   l   +   -   -   ) ; ब
   (     bh   -   -   0   0   0   0   s   l   +   +   -   ) ; भ
   (      m   -   -   0   0   0   0   n   l   +   -   -   ) ; म

   ;; Approximants
   (      j   -   -   0   0   0   0   r   p   +   -   -   ) ; य
   (     9r   -   -   0   0   0   0   r   a   +   -   -   ) ; र
   (      l   -   -   0   0   0   0   r   d   +   -   -   ) ; ल
   (     lr   -   -   0   0   0   0   r   a   +   -   -   ) ; ळ
   (     zr   -   -   0   0   0   0   r   a   +   -   -   ) ; ऴ
   (      v   -   -   0   0   0   0   r   l   +   -   -   ) ; व

   ;; Fricatives
   (     c}   -   -   0   0   0   0   f   p   -   +   -   ) ; श
   (     sr   -   -   0   0   0   0   f   a   -   +   -   ) ; ष
   (      s   -   -   0   0   0   0   f   d   -   +   -   ) ; स
   (     hv   -   -   0   0   0   0   f   v   +   +   -   ) ; ह
	  
   ;; More consonants
   (      q   -   -   0   0   0   0   s   g   -   -   +   ) ; क़
   (      x   -   -   0   0   0   0   f   g   -   -   +   ) ; ख़
   (      G   -   -   0   0   0   0   f   g   +   -   +   ) ; ग़
   (      z   -   -   0   0   0   0   f   d   +   -   +   ) ; ज़
   (     rr   -   -   0   0   0   0   s   a   +   -   +   ) ; ड़ 
   (    rrh   -   -   0   0   0   0   s   a   +   +   +   ) ; ढ़ 
   (      f   -   -   0   0   0   0   f   b   -   -   +   ) ; फ़ 
   (     dh   -   -   0   0   0   0   f   d   +   -   +   ) ; Tamil tB allophone 
   (      B   -   -   0   0   0   0   f   l   +   -   +   ) ; Tamil p allophone 
   (     nX   -   -   0   0   0   0   n   v   +   -   -   )


   ;; Nasalized vowels
   (  Anas    -   +   s   2   2   -   0   0   0   -   0   ) ; अ
   ( A:nas    -   +   l   3   2   -   0   0   0   -   0   ) ; आ
   (  inas    -   +   s   1   1   -   0   0   0   -   0   ) ; इ
   ( i:nas    -   +   l   1   1   -   0   0   0   -   0   ) ; ई
   (  unas    -   +   s   1   3   +   0   0   0   -   0   ) ; उ
   ( u:nas    -   +   l   1   3   +   0   0   0   -   0   ) ; ऊ
   (  enas    -   +   s   2   1   -   0   0   0   -   0   ) ; ए
   ( e:nas    -   +   l   2   1   -   0   0   0   -   0   ) ; ए long
   ( aInas    -   +   d   2   1   -   0   0   0   -   0   ) ; ऐ
   (  onas    -   +   s   2   3   +   0   0   0   -   0   ) ; ओ
   ( o:nas    -   +   l   2   3   +   0   0   0   -   0   ) ; ओ long
   ( aUnas    -   +   d   1   3   +   0   0   0   -   0   ) ; औ
   ( aynas    -   +   s   2   2   -   0   0   0   -   0   ) ; अॅ
   ( ownas    -   +   s   3   3   +   0   0   0   -   0   ) ; ऑ

   ;; English phones, which may be used in the bilingual indic databases
   ;; Only those not already listed above
   ( aa       -   +   l   3   3   -   0   0   0   -   0   ) ;; father
   ( ae       -   +   s   3   1   -   0   0   0   -   0   ) ;; fat
   ( ah       -   +   s   2   2   -   0   0   0   -   0   ) ;; but
   ( ao       -   +   l   3   3   +   0   0   0   -   0   ) ;; lawn
   ( aw       -   +   d   3   2   -   0   0   0   -   0   ) ;; how
   ( ax       -   +   a   2   2   -   0   0   0   -   0   ) ;; about
   ( axr      -   +   a   2   2   -   r   a   +   -   0   )
   ( d        -   -   0   0   0   0   s   a   +   -   0   )
   ( eh       -   +   s   2   1   -   0   0   0   -   0   ) ;; get
   ( er       -   +   a   2   2   -   r   0   0   -   0   )
   ( ey  -      +     d   2   1   -   0   0   0   -   0   ) ;; gate
   ( hh  -      -     0   0   0   0   f   g   -   -   0   )
   ( ih  -      +     s   1   1   -   0   0   0   -   0   ) ;; bit
   ( iy  -      +     l   1   1   -   0   0   0   -   0   ) ;; beet
   ( jh  -      -     0   0   0   0   a   p   +   -   0   )
   ( ng  -      -     0   0   0   0   n   v   +   -   0   )
   ( oy  -      +     d   2   3   +   0   0   0   -   0   ) ;; toy
   ( r   -      -     0   0   0   0   r   a   +   -   0   )
   ( sh  -      -     0   0   0   0   f   p   -   -   0   )
   ( t   -      -     0   0   0   0   s   a   -   -   0   )
   ( th  -      -     0   0   0   0   f   d   -   -   0   )
   ( uh  -      +     s   1   3   +   0   0   0   -   0   ) ;; full
   ( uw  -      +     l   1   3   +   0   0   0   -   0   ) ;; fool
   ( w   -      -     0   0   0   0   r   l   +   -   0   )
   ( y   -      -     0   0   0   0   r   p   +   -   0   )
   ( zh  -      -     0   0   0   0   f   p   +   -   0   )

   )
)

(PhoneSet.silences '(pau))

(define (INST_LANG_VOX::select_phoneset)
  "(INST_LANG_VOX::select_phoneset)
Set up phone set for INST_LANG."
  (Parameter.set 'PhoneSet 'cmu_indic)
  (PhoneSet.select 'cmu_indic)
)

(define (INST_LANG_VOX::reset_phoneset)
  "(INST_LANG_VOX::reset_phoneset)
Reset phone set for INST_LANG."
  t
)

(provide 'INST_LANG_VOX_phoneset)
