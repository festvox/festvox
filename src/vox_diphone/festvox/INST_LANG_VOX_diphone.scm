;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;                            [SOMEBODY]                                 ;;
;;;                         Copyright (c) 2000                            ;;
;;;                        All Rights Reserved.                           ;;
;;;                                                                       ;;
;;;  Distribution policy                                                  ;;
;;;     [CHOOSE ONE OF]                                                   ;;
;;;     Free for any use                                                  ;;
;;;     Free for non commercial use                                       ;;
;;;     something else                                                    ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  An example diphone voice
;;;
;;;  Authors: [The people who did the work]
;;;

;;; Try to find out where we are
(if (assoc 'INST_LANG_VOX_diphone voice-locations)
    (defvar INST_LANG_VOX_dir 
      (cdr (assoc 'INST_LANG_VOX_diphone voice-locations)))
    ;;; Not installed in Festival yet so assume running in place
    (defvar INST_LANG_VOX_dir (pwd)))

(if (not (probe_file (path-append INST_LANG_VOX_dir "festvox/")))
    (begin
     (format stderr "INST_LANG_VOX: Can't find voice scm files they are not in\n")
     (format stderr "   %s\n" (path-append INST_LANG_VOX_dir "festvox/"))
     (format stderr "   Either the voice isn't linked into Festival\n")
     (format stderr "   or you are starting festival in the wrong directory\n")
     (error)))

;;;  Add the directory contains general voice stuff to load-path
(set! load-path (cons (path-append INST_LANG_VOX_dir "festvox/") load-path))

;;; other files we need
(require 'INST_LANG_phones)
(require 'INST_LANG_lex)
(require 'INST_LANG_token)
(require 'INST_LANG_VOX_int)
(require 'INST_LANG_VOX_dur)

;;;  Ensure we have a festival with the right diphone support compiled in
(require_module 'UniSyn)

(set! INST_LANG_VOX_lpc_sep 
      (list
       '(name "INST_LANG_VOX_lpc_sep")
       (list 'index_file (path-append INST_LANG_VOX_dir "dic/VOXdiph.est"))
       '(grouped "false")
       (list 'coef_dir (path-append INST_LANG_VOX_dir "lpc"))
       (list 'sig_dir  (path-append INST_LANG_VOX_dir "lpc"))
       '(coef_ext ".lpc")
       '(sig_ext ".res")
       (list 'default_diphone 
	     (string-append
	      (car (cadr (car (PhoneSet.description '(silences)))))
	      "-"
	      (car (cadr (car (PhoneSet.description '(silences)))))))))

(set! INST_LANG_VOX_lpc_group 
      (list
       '(name "VOX_lpc_group")
       (list 'index_file 
	     (path-append INST_LANG_VOX_dir "group/VOXlpc.group"))
       '(grouped "true")
       (list 'default_diphone 
	     (string-append
	      (car (cadr (car (PhoneSet.description '(silences)))))
	      "-"
	      (car (cadr (car (PhoneSet.description '(silences)))))))))

;; Go ahead and set up the diphone db
(set! INST_LANG_VOX_db_name (us_diphone_init INST_LANG_VOX_lpc_sep))
;; Once you've built the group file you can comment out the above and
;; uncomment the following.
;(set! INST_LANG_VOX_db_name (us_diphone_init INST_LANG_VOX_lpc_group))

(define (INST_LANG_VOX_diphone_fix utt)
"(INST_LANG_VOX_diphone_fix UTT)
Map phones to phonological variants if the diphone database supports
them."
  (mapcar
   (lambda (s)
     (let ((name (item.name s)))
       ;; Check and do something maybe 
       ))
   (utt.relation.items utt 'Segment))
  utt)

(define (INST_LANG_VOX_voice_reset)
  "(INST_LANG_VOX_voice_reset)
Reset global variables back to previous voice."
  ;; whatever
)

;;;  Full voice definition 
(define (voice_INST_LANG_VOX_diphone)
"(voice_INST_LANG_VOX_diphone)
Set speaker to VOX in LANG from INST."
  (voice_reset)
  (Parameter.set 'Language 'INST_LANG)
  ;; Phone set
  (Parameter.set 'PhoneSet 'INST_LANG)
  (PhoneSet.select 'INST_LANG)

  ;; token expansion (numbers, symbols, compounds etc)
  (set! token_to_words INST_LANG_token_to_words)

  ;; No pos prediction (get it from lexicon)
  (set! pos_lex_name nil)
  (set! guess_pos INST_LANG_guess_pos) 
  ;; Phrase break prediction by punctuation
  (set! pos_supported nil) ;; well not real pos anyhow
  ;; Phrasing
  (set! phrase_cart_tree INST_LANG_phrase_cart_tree)
  (Parameter.set 'Phrase_Method 'cart_tree)
  ;; Lexicon selection
  (lex.select "INST_LANG")

  ;; No postlexical rules
  (set! postlex_rules_hooks nil)

  ;; Accent and tone prediction
  (set! int_accent_cart_tree INST_LANG_accent_cart_tree)

  (Parameter.set 'Int_Target_Method 'Simple)

  (Parameter.set 'Int_Method 'General)
  (set! int_general_params (list (list 'targ_func INST_LANG_VOX_targ_func1)))

  ;; Duration prediction
  (set! duration_cart_tree INST_LANG_VOX::zdur_tree)
  (set! duration_ph_info INST_LANG_VOX::phone_data)
  (Parameter.set 'Duration_Method 'Tree_ZScores)

  ;; Waveform synthesizer: diphones
  (set! UniSyn_module_hooks (list INST_LANG_VOX_diphone_fix))
  (set! us_abs_offset 0.0)
  (set! window_factor 1.0)
  (set! us_rel_offset 0.0)
  (set! us_gain 0.9)

  (Parameter.set 'Synth_Method 'UniSyn)
  (Parameter.set 'us_sigpr 'lpc)
  (us_db_select INST_LANG_VOX_db_name)

  ;; set callback to restore some original values changed by this voice
  (set! current_voice_reset INST_LANG_VOX_voice_reset)

  (set! current-voice 'INST_LANG_VOX_diphone)
)

(proclaim_voice
 'INST_LANG_VOX_diphone
 '((language LANG)
   (gender COMMENT)
   (dialect COMMENT)
   (description
    "COMMENT"
    )
   (builtwith festvox-1.2)))

(provide 'INST_LANG_VOX_diphone)
