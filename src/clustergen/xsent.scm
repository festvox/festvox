;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;                     Carnegie Mellon University                        ;;
;;;                         Copyright (c) 2013                            ;;
;;;                        All Rights Reserved.                           ;;
;;;                                                                       ;;
;;;  Permission is hereby granted, free of charge, to use and distribute  ;;
;;;  this software and its documentation without restriction, including   ;;
;;;  without limitation the rights to use, copy, modify, merge, publish,  ;;
;;;  distribute, sublicense, and/or sell copies of this work, and to      ;;
;;;  permit persons to whom this work is furnished to do so, subject to   ;;
;;;  the following conditions:                                            ;;
;;;   1. The code must retain the above copyright notice, this list of    ;;
;;;      conditions and the following disclaimer.                         ;;
;;;   2. Any modifications must be clearly marked as such.                ;;
;;;   3. Original authors' names are not deleted.                         ;;
;;;   4. The authors' names are not used to endorse or promote products   ;;
;;;      derived from this software without specific prior written        ;;
;;;      permission.                                                      ;;
;;;                                                                       ;;
;;;  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ;;
;;;  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ;;
;;;  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO         ;;
;;;  EVENT SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE       ;;
;;;  LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY     ;;
;;;  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,      ;;
;;;  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS       ;;
;;;  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR              ;;
;;;  PERFORMANCE OF THIS SOFTWARE.                                        ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;  Author: Alan W Black (awb@cs.cmu.edu) Apr 2013                       ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;  Corss sentence contextual synthesis                                  ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(require 'build_clunits) ;; for suffstats

;; global that is set at tts time too
(defvar xsent_utts nil) 
(defvar xsent_wavefiles nil)  ;; only used at synthesis time

;; TRAINING

(define (xsent_setup ttd)
    (set_backtrace t)
    (set! datafile ttd)
    (build_clunits_init ttd)

    (set! clustergen_params 
          (append
           (list
            '(clunit_relation mcep)
;            '(clunit_name_feat lisp_cg_name)
;           '(wagon_cluster_size_mcep 50) ;; 50 70 normally
            (list 'wagon_cluster_size_mcep cg:mcep_clustersize) ;; 50 70 normally
            '(wagon_cluster_size_f0 200)   ;; 200
            '(cg_ccoefs_template "ccoefs/%s.mcep")
;            '(cg_ccoefs_template "hnm/%s.hnm")  ;; HNM
            )
           clunits_params))
)

(define (xsent_dumpfeats ttd outfile)

  (xsent_setup ttd)

  (defvar prev_utt_seqid 0)
  (set! outfd (fopen outfile "w"))

  (mapcar 
   (lambda (fs)
     (format t "%s xsent dump" (car fs))
     (set! this_utt_seqid 
           (parse-number 
            (substring
             (format nil "%s" (car fs))
             (- (length (format nil "%s" (car fs))) 2)
             2)))
     (if (not (equal? (+ 1 prev_utt_seqid) this_utt_seqid))
         (begin
           (format t "reset %l\n" testset_utts)
           (set! testset_utts nil)) ;; new "para"
         (format t "\n"))
     
     (set! this_utt (utt.load nil (format nil "festival/utts/%s.utt" (car fs))))
     (set! fileid (car fs))
     (clustergen::load_hmmstates_utt this_utt clustergen_params)
     (clustergen::load_ccoefs_utt this_utt clustergen_params)
     (utt_f0_params this_utt)
     (set! first_word (utt.relation.first this_utt 'Word))
     (mapcar
      (lambda (feat)
        (format outfd "%s " (item.feat first_word feat)))
      xsent_feats)
     (format outfd "\n")
     (set! prev_utt_seqid this_utt_seqid)
     (set! testset_utts (cons this_utt testset_utts))
     t)
   (load ttd t))

  (fclose outfd))

(set! xsent_feats (mapcar car (car (load "festival/xsent/xsent.desc" t))))

;;;  Used at synthesis time to modify F0

(define (cg:xsent_f0_conversion utt)
  (let ((param_track (utt.feat utt "param_track"))
        (predicted_f0_mean 
         (wagon_predict (utt.relation.first utt 'Word) xsent_f0_model_f0_mean))
        (predicted_f0_stddev
         (wagon_predict (utt.relation.first utt 'Word) xsent_f0_model_f0_stddev))
        (num_frames 0) (c 0))
    (set_backtrace t)

    (utt_f0_params utt)  ;; calculate existing f0_mean/stddev
    (set! cur_f0_mean (utt.feat utt "f0_mean"))
    (set! cur_f0_stddev (utt.feat utt "f0_stddev"))
    (set! num_frames (track.num_frames param_track))

    ;; Some tuning
;;    (set! predicted_f0_mean 
;;          (+ cur_f0_mean 
;;             (* 2.3 (- predicted_f0_mean cur_f0_mean))))
;;    (set! predicted_f0_stddev 
;;          (+ cur_f0_stddev
;;             (* 0.3 (- predicted_f0_stddev cur_f0_stddev))))
    
    (format t "From %f (%f) to %f (%f)\n"
            cur_f0_mean cur_f0_stddev
            predicted_f0_mean predicted_f0_stddev)
    (while (< c num_frames)
       (if (> (track.get param_track c 0) 5.0)
           (begin
             (set! f0_z (/ (- (track.get param_track c 0) cur_f0_mean)
                           cur_f0_stddev))
             (track.set param_track c 0 
                        (+ predicted_f0_mean 
                           (* predicted_f0_stddev f0_z)))))
       (set! c (+ 1 c)))

    (utt.set_feat utt "f0_mean" predicted_f0_mean)
    (utt.set_feat utt "f0_stddev" predicted_f0_stddev)
    utt
    )
)

(define (utt_f0_params utt)
  (let ((f0ss (suffstats.new))
        (highest 0) (lowest 500) (f0 0)
        (c 0) (num_frames 0))
    (set! param_track (utt.feat utt "param_track"))
    (set! num_frames (track.num_frames param_track))
    (while (< c num_frames)
       (if (> (track.get param_track c 0) 5.0)
           (begin
             (set! f0 (track.get param_track c 0))
             (if (< f0 lowest) (set! lowest f0))
             (if (> f0 highest) (set! highest f0))
             (suffstats.add f0ss f0)))
       (set! c (+ 1 c)))

    (utt.set_feat utt "f0_mean" (suffstats.mean f0ss))
    (utt.set_feat utt "f0_stddev" (suffstats.stddev f0ss))
    (utt.set_feat utt "f0_high" highest)
    (utt.set_feat utt "f0_low" lowest)
    utt
    ))

(define (utt_f0_mean w)
  (utt.feat (item.get_utt w) "f0_mean"))
(define (utt_f0_stddev w)
  (utt.feat (item.get_utt w) "f0_stddev"))
(define (utt_f0_high w)
  (utt.feat (item.get_utt w) "f0_high"))
(define (utt_f0_low w)
  (utt.feat (item.get_utt w) "f0_low"))
(define (utt_num_words w)
  (item.feat w "words_out"))

(define (prev_utt_f0_mean w)
  (if testset_utts
      (utt_f0_mean (utt.relation.first (car testset_utts) 'Word))
      0))
(define (prev_utt_f0_stddev w)
  (if testset_utts
      (utt_f0_stddev (utt.relation.first (car testset_utts) 'Word))
      0))
(define (prev_utt_f0_high w)
  (if testset_utts
      (utt_f0_high (utt.relation.first (car testset_utts) 'Word))
      0))
(define (prev_utt_f0_low w)
  (if testset_utts
      (utt_f0_low (utt.relation.first (car testset_utts) 'Word))
      0))
(define (prev_utt_num_words w)
  (if testset_utts
      (item.feat (utt.relation.first (car testset_utts) 'Word) "words_out")
      0))
(define (prev_utts_num w)
  (length testset_utts))

(define (numsyls w)
  (length (utt.relation.items (item.get_utt w) 'Syllable)))
(define (prev_numsyls w)
  (if testset_utts
      (numsyls (utt.relation.first (car testset_utts) 'Word))
      0))

(define (prev_numcontentwords w)
  (if testset_utts
      (numcontentwords (utt.relation.first (car testset_utts) 'Word))
      0))
(define (numcontentwords w)
  (set! nc 0)
  (mapcar (lambda (x) 
            (if (item.feat x 'contentp) 
                (set! nc (+ 1 nc)))) 
          (utt.relation.items (item.get_utt w) 'Word))
  nc)

(define (firstcontentwordnumsyls w)
  (set! w1 w)
  (while (not (item.feat w1 'contentp))(set! w1 (item.next w1)))
  (item.feat w1 'word_numsyls))

(define (lastcontentwordnumsyls w)
  (if testset_utts
      (begin
        (set! w1 (utt.relation.last (car testset_utts) 'Word))
        (while (not (item.feat w1 'contentp))(set! w1 (item.prev w1)))
        (item.feat w1 'word_numsyls))
      0))

;;; Run time synthesis that preserves previous utterance context
(define (xsent_synth_para_ttd ttd odir)
  ;; One paragraph per line, no para info 
  (mapcar
   (lambda (fs)
     (format t "TTS %s\n" (car fs))
     (set! xsent_wavefiles nil)
     (set! xset_utts nil)
     (tts_text (cadr fs) nil)
     ;; Now join the waveform files together 
     (set! wholeutt (xsent_combine_waves xsent_wavefiles))
     (utt.save.wave wholeutt (format nil "%s/%s.wav" odir (car fs)))
     t
     )
   (load ttd t))
  t)

(define (xsent_synth_ttd ttd odir)
  ;; One sentence per line, with para info 
  (set! xsent_wavefiles nil)
  (set! xset_utts nil)
  (mapcar
   (lambda (fs)
     (format t "TTS %s\n" (car fs))
     (tts_text (cadr fs) nil)
     ;; Now join the waveform files together 
     (set! wholeutt (xsent_combine_waves xsent_wavefiles))
     (utt.save.wave wholeutt (format nil "%s/%s.wav" odir (car fs)))
     t
     )
   (load ttd t))
  t)

(define (xsent_synth_file textfile ofile)
  ;; One sentence per line, with para info 
  (set! xsent_wavefiles nil)
  (set! xset_utts nil)
  (tts textfile nil)
  (set! wholeutt (xsent_combine_waves xsent_wavefiles))
  (utt.save.wave wholeutt ofile)
  t)

(define (xsent_combine_waves wavefiles )
  "Join all the waves together into the desired output file
and delete the intermediate ones."
  (let ((wholeutt (utt.synth (Utterance Text ""))))
    (mapcar
     (lambda (d) 
       (utt.import.wave wholeutt d t)
       (delete-file d))
     (reverse wavefiles))
    wholeutt
    ))

(define (xsent_save_context utt)
"Saves the waveform and records its so it can be joined into a 
a single waveform at the end."
  (let ((fn (make_tmp_filename)))
    (utt.save.wave utt fn)
    (set! xsent_wavefiles (cons fn xsent_wavefiles))
    (set! xsent_utts (cons utt xsent_utts))
    utt))

(set! tts_hooks (list utt.synth xsent_save_context))

(provide 'xsent)
