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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                                                                       ;;
;;  A mode for LDOM limited domain.  Allows specific tokenization,       ;;
;;  lexical entries etc for the LDOM domain                              ;;
;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (INST_LDOM_VOX_init_func)
  (let ()
    (voice_INST_LDOM_VOX_ldom)

    ;; Save the standard text2word rules
    (set! INST_LDOM_VOX::previous_t2w_func english_token_to_words)

    (set! english_token_to_words INST_LDOM_VOX::token_to_words)
    (set! token_to_words INST_LDOM_VOX::token_to_words)

    ;; If you want to make your limited domain synthesizer have a fall
    ;; back diphone synthesizer when out of vocabulary words/phrases
    ;; are found then uncomment the following
    (set! INST_LDOM_VOX::old_tts_hooks tts_hooks)
    (set! INST_LDOM_VOX::real_utt.synth utt.synth)
    (set! tts_hooks 
	  (cons
	   INST_LDOM_VOX::utt.synth 
	   (delq utt.synth tts_hooks)))
  )
)

(define (INST_LDOM_VOX_token_to_words token name)
  "(INST_LDOM_VOX_token_to_words token name)
Some LDOM specific token processing."
  (cond
   ;; Insert doamin specific rules here
   (t
    (INST_LDOM_VOX::previous_t2w_func token name))))


(define (INST_LDOM_VOX_exit_func)
  "(INST_LDOM_VOX_exit_func)
Called on exiting the LDOM mode."
  (set! tts_hooks INST_LDOM_VOX::old_tts_hooks)

  (set! english_token_to_words INST_LDOM_VOX::previous_t2w_func)
  (set! token_to_words INST_LDOM_VOX::previous_t2w_func)

)

(define (INST_LDOM_VOX::utt.synth utt)
  "(INST_LDOM_VOX::utt.synth utt)
Uses the LDOM voice to synthesize utt, but if that fails 
back off to the define diphone synthesizer."
  (unwind-protect 
   (begin
     (INST_LDOM_VOX::real_utt.synth utt)
     ;; Successful synthesis
     )
   (begin
     ;; The above failed, so resynthesize with the backup voice
;     (format stderr "ldom/clunits failed\n")
     (eval (list INST_LDOM_VOX_closest_voice))  ;; call backup voice
     (set! utt2 (INST_LDOM_VOX_copy_tokens utt))
     (set! utt (INST_LDOM_VOX::real_utt.synth utt2))
     (voice_INST_LDOM_VOX_ldom)                 ;; return to ldom voice
     ))
  utt
)

(define (INST_LDOM_VOX_copy_tokens utt)
  "(INST_LDOM_VOX_copy_tokens utt)
This should be standard library, of in fact not even needed.  It constructs
a new utterance from the tokens in utt so it may be safely resynthesized."
  (let ((utt2 (Utterance Tokens nil))
	(oldtok (utt.relation.first utt 'Token)))
    (utt.relation.create utt2 'Token)
    (while oldtok
      (let ((ntok (utt.relation.append utt2 'Token)))
	(mapcar
	 (lambda (fp)
	   (item.set_feat ntok (car fp) (car (cdr fp))))
	 (cdr (item.features oldtok)))
	(set! oldtok (item.next oldtok))))
    utt2))

(set! tts_text_modes
   (cons
    (list
      'INST_LDOM_VOX         ;; mode name
      (list         
       (list 'init_func INST_LDOM_VOX_init_func)
       (list 'exit_func INST_LDOM_VOX_exit_func)))
    tts_text_modes))

(provide 'INST_LDOM_VOX-mode)

