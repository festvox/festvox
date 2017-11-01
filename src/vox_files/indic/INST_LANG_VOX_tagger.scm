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
;;; POS tagger for LANG
;;;

;;; Load any necessary files here

;;; Load voice/language specific gpos list if it exists
(if (probe_file (path-append INST_LANG_VOX::dir "festvox/INST_LANG_VOX_gpos.scm"))
    (begin
      (load (path-append INST_LANG_VOX::dir "festvox/INST_LANG_VOX_gpos.scm"))
      (set! guess_pos INST_LANG_VOX_guess_pos)))

(set! INST_LANG_guess_pos 
'((fn
    ;; function words 
  )
  ;; Or split them into sub classes (but give them meaningful names)
  ; (pos_0 .. .. .. ..)
  ; (pos_1 .. .. .. ..)
  ; (pos_2 .. .. .. ..)
))

(define (INST_LANG_VOX::select_tagger)
  "(INST_LANG_VOX::select_tagger)
Set up the POS tagger for LANG."
  (set! pos_lex_name nil)
  (if (boundp 'INST_LANG_VOX_guess_pos)
      (set! guess_pos INST_LANG_VOX_guess_pos)   ;; voice specific gpos
      (set! guess_pos english_guess_pos))      ;; default English gpos
)

(define (INST_LANG_VOX::reset_tagger)
  "(INST_LANG_VOX::reset_tagger)
Reset tagging information."
  t
)

(provide 'INST_LANG_VOX_tagger)
