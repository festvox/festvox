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
;;; Phoneset for lvl_is
;;;
 ( ph_vc + - 0)
 ( ph_ctype 0 a f l n r s -)
 ( ph_vheight 0 1 2 3 -)
 ( ph_vlng 0 a d l s -)
 ( ph_vfront 0 1 2 3 -)
 ( p.ph_vrnd + - 0)
 ( ph_cplace 0 a b d l o p v g -)
 ( p.ph_cvox + - 0)

(defPhoneSet
  lvl_is
  (
   (vc d + - 0)
   (cplace 0 a b d l o p v g -)
   (cplace 0 b l d o p v g -) ; bilabial, labiodental, dental, alveolar, palatal, velar, glottal
   (ctype s f n l r 0)
   (cvox + - 0)
   (casp + - 0 )
   (vfront 3 2 1 0)
   (vheight 3 2 1 0 -)
   (vrnd - + 0)
   (dfront2 3 2 1 0 -)
   (dheight2 3 2 1 0 -)
   (drnd2 - + 0)
   (vlength + - 0)
   )
  (
   (p    - b s - - 0 - 0 - - 0 0)
   (ph   - b s - + 0 - 0 - - 0 0)
   (m    - b n + 0 0 - 0 - - 0 0)
   (mz   - b n - 0 0 - 0 - - 0 0)
   (f    - l f - 0 0 - 0 - - 0 0)
   (v    - l f + 0 0 - 0 - - 0 0)
   (t    - o s - - 0 - 0 - - 0 0)
   (th   - o s - + 0 - 0 - - 0 0)
   (D    - d f + 0 0 - 0 - - 0 0)
   (T    - d f - 0 0 - 0 - - 0 0)
   (n    - o n + 0 0 - 0 - - 0 0)
   (nz   - o n - 0 0 - 0 - - 0 0)
   (l    - o l + 0 0 - 0 - - 0 0)
   (lz   - o l - 0 0 - 0 - - 0 0)
   (r    - o r + 0 0 - 0 - - 0 0)
   (rz   - o r - 0 0 - 0 - - 0 0)
   (s    - o f - 0 0 - 0 - - 0 0)
   (c    - p s - - 0 - 0 - - 0 0)
   (ch   - p s - + 0 - 0 - - 0 0)
   (j    - p f + 0 0 - 0 - - 0 0)
   (C    - p f - 0 0 - 0 - - 0 0)
   (J    - p n + 0 0 - 0 - - 0 0)
   (Jz   - p n - 0 0 - 0 - - 0 0)
   (k    - v s - - 0 - 0 - - 0 0)
   (kh   - v s - + 0 - 0 - - 0 0)
   (x    - v f - 0 0 - 0 - - 0 0)
   (G    - v f + 0 0 - 0 - - 0 0)
   (N    - v n + 0 0 - 0 - - 0 0)
   (Nz   - v n - 0 0 - 0 - - 0 0)
   (h    - g f - 0 0 - 0 - - 0 0)
   (i    + 0 0 0 0 3 3 - - - 0 -)
   (ii   + 0 0 0 0 3 3 - - - 0 +)
   (I    + 0 0 0 0 3 2 - - - 0 -)
   (II   + 0 0 0 0 3 2 - - - 0 +)
   (Y    + 0 0 0 0 3 2 + - - 0 -)
   (YY   + 0 0 0 0 3 2 + - - 0 +)
   (oe   + 0 0 0 0 3 1 + - - 0 -)
   (oee  + 0 0 0 0 3 1 + - - 0 +)
   (E    + 0 0 0 0 2 1 - - - 0 -)
   (EE   + 0 0 0 0 2 1 - - - 0 +)
   (u    + 0 0 0 0 1 3 + - - 0 -)
   (uu   + 0 0 0 0 1 3 + - - 0 +)
   (O    + 0 0 0 0 1 1 + - - 0 -)
   (OO   + 0 0 0 0 1 1 + - - 0 +)
   (aa   + 0 0 0 0 1 0 - - - 0 +)
   (a    + 0 0 0 0 1 0 - - - 0 -)
   (Yi   d 0 0 0 0 3 2 + 3 3 - -)
   (Oi   d 0 0 0 0 1 1 + 3 3 - -)
   (ou   d 0 0 0 0 1 1 + 1 3 + -)
   (ouu  d 0 0 0 0 1 1 + 1 3 + +)
   (ai   d 0 0 0 0 1 0 - 3 3 - -)
   (aii  d 0 0 0 0 1 0 - 3 3 - +)
   (au   d 0 0 0 0 1 0 - 1 3 + -)
   (auu  d 0 0 0 0 1 0 - 1 3 + +)
   (oei  d 0 0 0 0 1 0 - 3 2 + -)
   (oeii d 0 0 0 0 1 0 - 3 2 + +)
   (ei   d 0 0 0 0 2 1 - 3 3 - -)
   (eii  d 0 0 0 0 2 1 - 3 3 - +)
   (pau  0 0 0 0 0 0 - 0 - 0 0 0)
  )
)

(PhoneSet.silences '(pau))

(define (INST_is_VOX::select_phoneset)
  "(INST_is_VOX::select_phoneset)
Set up phone set for lvl_is."
  (Parameter.set 'PhoneSet 'lvl_is)
  (PhoneSet.select 'lvl_is)
)

(define (INST_is_VOX::reset_phoneset)
  "(INST_is_VOX::reset_phoneset)
Reset phone set for lvl_is."
  t
)

(provide 'INST_is_VOX_phoneset)
