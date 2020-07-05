;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                 Nagoya Institute of Technology and                  ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2005                          ;;;
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
;;;
;;; Voice Transformation Voice definition filter from @SOURCE@ to @TARGET@
;;;

;;; Try to find the directory where the voice is, this may be from
;;; .../festival/lib/voices/ or from the current directory
(if (assoc 'VOICENAME_transform voice-locations)
    (defvar VOICENAME_transform::dir 
      (cdr (assoc 'VOICENAME_transform voice-locations)))
    (defvar VOICENAME_transform::dir (string-append (pwd) "/")))

(defvar VOICENAME::dir VOICENAME_transform::dir)
(defvar VOICENAME::clunits_loaded nil)

;;; Did we succeed in finding it
(if (not (probe_file (path-append VOICENAME_transform::dir "festvox/")))
    (begin
     (format stderr "VOICENAME_transform: Can't find voice scm files they are not in\n")
     (format stderr "   %s\n" (path-append  VOICENAME_transform::dir "festvox/"))
     (format stderr "   Either the voice isn't linked in Festival library\n")
     (format stderr "   or you are starting festival in the wrong directory\n")
     (error)))

;;;  Add the directory contains general voice stuff to load-path
(set! load-path (cons (path-append VOICENAME_transform::dir "festvox/") 
		      load-path))

(define (VOICENAME_transform::voice_reset)
  (set! after_synth_hooks VOICENAME_transform::old_after_synth_hooks)
  t
)

(define (VOICENAME_transform::convfilter utt)
  "(VOICENAME_transform::convfilter utt)
Filter synthesized voice with transformation filter and reload waveform."
   (let ((wfile1 (make_tmp_filename))
	 (wfile2 (make_tmp_filename))
         (wfile3 (make_tmp_filename)))

     (utt.save.wave utt wfile1)
     (utt.save utt wfile3)

     (system
      (format 
       nil
       "(cd %s && csh scripts/VConvFestival.csh src param/source-target_param.list %s %s %s)"
       vc_dir
       wfile1  ;; input file
       wfile3  ;; utterance  
       wfile2))

     (utt.import.wave utt wfile2)
     (delete-file wfile1)
     (delete-file wfile2)
     (delete-file wfile3)

     utt
     ))


;; This function is called to setup a voice.  It will typically
;; simply call functions that are defined in other files in this directory
;; Sometime these simply set up standard Festival modules othertimes
;; these will be specific to this voice.
;; Feel free to add to this list if your language requires it

(define (voice_VOICENAME_transform)
  "(voice_VOICENAME_transform)
Define voice for VOICENAME with transformation."
  ;; *always* required
  (voice_reset)

  ;; Whatever the default voice is 
  (voice_kal_diphone)

  ;; Waveform synthesis model: hts
  (set! vc_dir (path-append VOICENAME_transform::dir "vc/"))

  (set! VOICENAME_transform::old_after_synth_hooks after_synth_hooks)
  (set! after_synth_hooks
      (append
       (if (or (null after_synth_hooks) (consp after_synth_hooks))
	   after_synth_hooks
	   (list after_synth_hooks))
       (list VOICENAME_transform::convfilter)))

  (set! current_voice_reset VOICENAME_transform::voice_reset)

  (set! current-voice 'VOICENAME_transform)
)

(provide 'VOICENAME_transform)



