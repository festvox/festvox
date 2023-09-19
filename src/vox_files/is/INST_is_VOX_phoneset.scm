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

(defPhoneSet
  lvl_is
  (
   (vc consonant vowel diph silence 0)
   (cplace bilabial labiodental dental alveolar palatal velar glottal 0)
   (ctype plosive fricative nasal lateral trill 0)
   (cvox voiced voiceless 0)
   (casp aspirated unaspirated 0 )
   (vfrontness front central back 0)
   (vcloseness close nearclose closemid mid openmid nearopen open 0)
   (vroundness unrounded rounded 0)
   (dfrontness2 front central back 0)
   (dcloseness2 close nearclose closemid mid openmid nearopen open 0)
   (droundness2 unrounded rounded 0)
   (vlength long short 0)
   )
  (
   (p   consonant bilabial  plosive       voiceless unaspirated 0 0 0 0 0 0 0)
   (ph  consonant bilabial  plosive       voiceless aspirated 0 0 0 0 0 0 0)
   (m   consonant bilabial  nasal         voiced 0 0 0 0 0 0 0 0)
   (mz  consonant bilabial  nasal         voiceless 0 0 0 0 0 0 0 0)
   (f   consonant labiodental    fricative     voiceless 0 0 0 0 0 0 0 0)
   (v   consonant labiodental    fricative     voiced 0 0 0 0 0 0 0 0)
   (t   consonant alveolar plosive       voiceless unaspirated 0 0 0 0 0 0 0)
   (th  consonant alveolar plosive       voiceless aspirated 0 0 0 0 0 0 0)
   (D   consonant dental fricative     voiced 0 0 0 0 0 0 0 0)
   (T   consonant dental fricative     voiceless 0 0 0 0 0 0 0 0)
   (n   consonant alveolar nasal         voiced 0 0 0 0 0 0 0 0)
   (nz  consonant alveolar nasal         voiceless 0 0 0 0 0 0 0 0)
   (l   consonant alveolar lateral       voiced 0 0 0 0 0 0 0 0)
   (lz  consonant alveolar lateral       voiceless 0 0 0 0 0 0 0 0)
   (r   consonant alveolar trill         voiced 0 0 0 0 0 0 0 0)
   (rz  consonant alveolar trill         voiceless 0 0 0 0 0 0 0 0)
   (s   consonant alveolar fricative     voiceless 0 0 0 0 0 0 0 0)
   (c   consonant palatal   plosive       voiceless unaspirated 0 0 0 0 0 0 0)
   (ch  consonant palatal   plosive       voiceless aspirated 0 0 0 0 0 0 0)
   (j   consonant palatal   fricative     voiced 0 0 0 0 0 0 0 0)
   (C  consonant palatal   fricative     voiceless 0 0 0 0 0 0 0 0)
   (J  consonant palatal   nasal         voiced 0 0 0 0 0 0 0 0)
   (Jz consonant palatal   nasal         voiceless 0 0 0 0 0 0 0 0)
   (k   consonant velar     plosive       voiceless unaspirated 0 0 0 0 0 0 0)
   (kh  consonant velar     plosive       voiceless aspirated 0 0 0 0 0 0 0)
   (x   consonant velar     fricative     voiceless 0 0 0 0 0 0 0 0)
   (G  consonant velar     fricative     voiced 0 0 0 0 0 0 0 0)
   (N  consonant velar     nasal         voiced 0 0 0 0 0 0 0 0)
   (Nz consonant velar     nasal         voiceless 0 0 0 0 0 0 0 0)
   (h   consonant glottal   fricative     voiceless 0 0 0 0 0 0 0 0)
   (i   vowel 0 0 0 0 front     close    unrounded 0         0    0       short)
   (ii  vowel 0 0 0 0 front     close    unrounded 0         0    0       long)
   (I   vowel 0 0 0 0 front     nearclose unrounded 0         0    0       short)
   (II  vowel 0 0 0 0 front     nearclose unrounded 0         0    0       long)
   (Y   vowel 0 0 0 0 front     nearclose rounded 0         0    0       short)
   (YY  vowel 0 0 0 0 front     nearclose rounded 0         0    0       long)
   (oe  vowel 0 0 0 0 front     openmid  rounded 0         0    0       short)
   (oee vowel 0 0 0 0 front     openmid  rounded 0         0    0       long)
   (E   vowel 0 0 0 0 central   openmid  unrounded 0         0    0       short)
   (EE  vowel 0 0 0 0 central   openmid  unrounded 0         0    0       long)
   (u   vowel 0 0 0 0 back      close    rounded 0         0    0       short)
   (uu  vowel 0 0 0 0 back      close    rounded 0         0    0       long)
   (O   vowel 0 0 0 0 back      openmid  rounded 0         0    0       short)
   (OO  vowel 0 0 0 0 back      openmid  rounded 0         0    0       long)
   (a   vowel 0 0 0 0 back      open     unrounded 0         0    0       short)
   (aa  vowel 0 0 0 0 back      open     unrounded 0         0    0       long)
   (Yi  diph 0 0 0 0 front     nearclose rounded front  close unrounded short)
   (Oi  diph  0 0 0 0 back      openmid  rounded front     close unrounded short)
   (ou  diph  0 0 0 0 back      openmid  rounded back      close rounded short)
   (ouu diph  0 0 0 0 back      openmid  rounded back      close rounded long)
   (ai  diph  0 0 0 0 back      open     unrounded front     close unrounded short)
   (aii diph  0 0 0 0 back      open     unrounded front     close unrounded long)
   (au  diph  0 0 0 0 back      open     unrounded back      close rounded short)
   (auu diph  0 0 0 0 back      open     unrounded back      close rounded long)
   (oei  diph  0 0 0 0 back      open     unrounded front     nearclose rounded short)
   (oeii diph  0 0 0 0 back      open     unrounded front     nearclose rounded long)
   (ei  diph  0 0 0 0 central   openmid  unrounded front     close unrounded short)
   (eii diph  0 0 0 0 central   openmid  unrounded front     close unrounded long)
   (pau silence 0 0 0 0 0 0 0 0 0 0 0)
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
