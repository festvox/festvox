;;  ----------------------------------------------------------------  ;;
;;                 Nagoya Institute of Technology and                 ;;
;;                     Carnegie Mellon University                     ;;
;;                      Copyright (c) 2002-2007                       ;;
;;                        All Rights Reserved.                        ;;
;;                                                                    ;;
;;  Permission is hereby granted, free of charge, to use and          ;;
;;  distribute this software and its documentation without            ;;
;;  restriction, including without limitation the rights to use,      ;;
;;  copy, modify, merge, publish, distribute, sublicense, and/or      ;;
;;  sell copies of this work, and to permit persons to whom this      ;;
;;  work is furnished to do so, subject to the following conditions:  ;;
;;                                                                    ;;
;;    1. The code must retain the above copyright notice, this list   ;;
;;       of conditions and the following disclaimer.                  ;;
;;                                                                    ;;
;;    2. Any modifications must be clearly marked as such.            ;;
;;                                                                    ;;
;;    3. Original authors' names are not deleted.                     ;;
;;                                                                    ;;
;;    4. The authors' names are not used to endorse or promote        ;;
;;       products derived from this software without specific prior   ;;
;;       written permission.                                          ;;
;;                                                                    ;;
;;  NAGOYA INSTITUTE OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY AND    ;;
;;  THE CONTRIBUTORS TO THIS WORK DISCLAIM ALL WARRANTIES WITH        ;;
;;  REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF      ;;
;;  MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL NAGOYA INSTITUTE   ;;
;;  OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS    ;;
;;  BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR   ;;
;;  ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        ;;
;;  PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER    ;;
;;  TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR  ;;
;;  PERFORMANCE OF THIS SOFTWARE.                                     ;;
;;                                                                    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;     A voice based on "HTS" HMM-Based Speech Synthesis System.      ;;
;;     2.0 style training
;;          Author :  Alan W Black                                    ;;
;;          Date   :  August 2002/December 2007                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Try to find the directory where the voice is, this may be from
;;; .../festival/lib/voices/ or from the current directory
(if (assoc 'VOICENAME_hts voice-locations)
    (defvar VOICENAME_hts::dir 
      (cdr (assoc 'VOICENAME_hts voice-locations)))
    (defvar VOICENAME_hts::dir (string-append (pwd) "/")))

(defvar VOICENAME::dir VOICENAME_hts::dir)
(defvar VOICENAME::clunits_loaded nil)

;;; Did we succeed in finding it
(if (not (probe_file (path-append VOICENAME_hts::dir "festvox/")))
    (begin
     (format stderr "VOICENAME_hts: Can't find voice scm files they are not in\n")
     (format stderr "   %s\n" (path-append  VOICENAME_hts::dir "festvox/"))
     (format stderr "   Either the voice isn't linked in Festival library\n")
     (format stderr "   or you are starting festival in the wrong directory\n")
     (error)))

;;;  Add the directory contains general voice stuff to load-path
(set! load-path (cons (path-append VOICENAME_hts::dir "festvox/") 
		      load-path))

(require 'hts)
(require_module 'hts_engine)

;;; Voice specific parameter are defined in each of the following
;;; files
(require 'VOICENAME_phoneset)
(require 'VOICENAME_tokenizer)
(require 'VOICENAME_tagger)
(require 'VOICENAME_lexicon)
(require 'VOICENAME_phrasing)
(require 'VOICENAME_intonation)
(require 'VOICENAME_duration)
(require 'VOICENAME_f0model)
(require 'VOICENAME_other)
;; ... and others as required


(define (VOICENAME_hts::voice_reset)
  "(VOICENAME_hts::voice_reset)
Reset global variables back to previous voice."
  (VOICENAME::reset_phoneset)
  (VOICENAME::reset_tokenizer)
  (VOICENAME::reset_tagger)
  (VOICENAME::reset_lexicon)
  (VOICENAME::reset_phrasing)
  (VOICENAME::reset_intonation)
  (VOICENAME::reset_duration)
  (VOICENAME::reset_f0model)
  (VOICENAME::reset_other)

  t
)

(set! VOICENAME_hts::hts_feats_list
      (load (path-append VOICENAME_hts::dir "hts/label.feats") t))

(set! VOICENAME_hts::hts_engine_params
      (list
       (list "-dm1" (path-append VOICENAME_hts::dir "hts/mcp.win2"))
       (list "-dm2" (path-append VOICENAME_hts::dir "hts/mcp.win3"))
       (list "-df1" (path-append VOICENAME_hts::dir "hts/lf0.win2"))
       (list "-df2" (path-append VOICENAME_hts::dir "hts/lf0.win3"))
       (list "-td" (path-append VOICENAME_hts::dir "hts/tree-dur.inf"))
       (list "-tm" (path-append VOICENAME_hts::dir "hts/tree-mcp.inf"))
       (list "-tf" (path-append VOICENAME_hts::dir "hts/tree-lf0.inf"))
       (list "-md" (path-append VOICENAME_hts::dir "hts/dur.pdf"))
       (list "-mm" (path-append VOICENAME_hts::dir "hts/mcp.pdf"))
       (list "-mf" (path-append VOICENAME_hts::dir "hts/lf0.pdf"))
       '("-a" 0.420000)
       '("-r" 0.000000)   ; duration stretch; neg faster, pos slower
       '("-fs" 1.000000)
       '("-fm" 0.000000)
       '("-u" 0.500000)
       '("-l" 0.000000)
       ))

;; This function is called to setup a voice.  It will typically
;; simply call functions that are defined in other files in this directory
;; Sometime these simply set up standard Festival modules othertimes
;; these will be specific to this voice.
;; Feel free to add to this list if your language requires it

(define (voice_VOICENAME_hts)
  "(voice_VOICENAME_hts)
Define voice for limited domain: us."
  ;; *always* required
  (voice_reset)

  ;; Select appropriate phone set
  (VOICENAME::select_phoneset)

  ;; Select appropriate tokenization
  (VOICENAME::select_tokenizer)

  ;; For part of speech tagging
  (VOICENAME::select_tagger)

  (VOICENAME::select_lexicon)
  ;; For hts selection you probably don't want vowel reduction
  ;; the unit selection will do that
  (if (string-equal "americanenglish" (Param.get 'Language))
      (set! postlex_vowel_reduce_cart_tree nil))

  (VOICENAME::select_phrasing)

  (VOICENAME::select_intonation)

  (VOICENAME::select_duration)

  (VOICENAME::select_f0model)

  ;; Waveform synthesis model: hts
  (set! hts_engine_params VOICENAME_hts::hts_engine_params)
  (set! hts_feats_list VOICENAME_hts::hts_feats_list)
  (Parameter.set 'Synth_Method 'HTS)

  ;; This is where you can modify power (and sampling rate) if desired
  (set! after_synth_hooks nil)
;  (set! after_synth_hooks
;      (list
;        (lambda (utt)
;          (utt.wave.rescale utt 2.1))))

  (set! current_voice_reset VOICENAME_hts::voice_reset)

  (set! current-voice 'VOICENAME_hts)
)

(provide 'VOICENAME_hts)

