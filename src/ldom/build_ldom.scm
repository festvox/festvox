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
;;;                                                                     ;;;
;;; Code for building data for prompts, aligning and unit selection     ;;;
;;; synthesizer                                                         ;;;
;;;                                                                     ;;;
;;; This file is only used at database build time                       ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(require 'clunits_build)

(defvar INST_LDOM_VOX::ldom_dir ".")

;;; Basic voice definition file with voice defines and clunit
;;; parameter definitoin for run time.
(load "festvox/INST_LDOM_VOX_ldom.scm")

;;; Add Build time parameters
(set! INST_LDOM_VOX::dt_params
      (cons
       ;; in case INST_LDOM_VOX_ldom defines this too, put this at start
       (list 'db_dir (string-append INST_LDOM_VOX::ldom_dir "/"))
       (append
	INST_LDOM_VOX::dt_params
	(list
	;;; In INST_LDOM_VOX_ldom.scm
	 ;;'(coeffs_dir "lpc/")
	 ;;'(coeffs_ext ".lpc")
	 '(disttabs_dir "festival/disttabs/")
	 '(utts_dir "festival/utts/")
	 '(utts_ext ".utt")
	 '(dur_pen_weight 0.0)
	 '(f0_pen_weight 0.1)
	 '(get_stds_per_unit t)
	 '(ac_left_context 0.8)
	 '(ac_weights
	   (0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 ))
	 ;; Join weights in INST_LDOM_VOX_ldom.scm
	 ;; Features for extraction
	 '(feats_dir "festival/feats/")
	 '(feats 
	   (occurid
	    p.name p.ph_vc p.ph_ctype 
	    p.ph_vheight p.ph_vlng 
	    p.ph_vfront  p.ph_vrnd 
	    p.ph_cplace  p.ph_cvox    
	    n.name n.ph_vc n.ph_ctype 
	    n.ph_vheight n.ph_vlng 
	    n.ph_vfront  n.ph_vrnd 
	    n.ph_cplace  n.ph_cvox
	    segment_duration 
	    seg_pitch p.seg_pitch n.seg_pitch
	    R:SylStructure.parent.stress 
	    seg_onsetcoda n.seg_onsetcoda p.seg_onsetcoda
	    R:SylStructure.parent.accented 
	    pos_in_syl 
	    syl_initial
	    syl_final
	    R:SylStructure.parent.syl_break 
	    R:SylStructure.parent.R:Syllable.p.syl_break
	    pp.name pp.ph_vc pp.ph_ctype 
	    pp.ph_vheight pp.ph_vlng 
	    pp.ph_vfront  pp.ph_vrnd 
	    pp.ph_cplace pp.ph_cvox
	    R:SylStructure.parent.parent.R:Word.word_out
	    R:SylStructure.parent.syl_in
	    R:SylStructure.parent.syl_out
	    ))
	 ;; Wagon tree building params
;	 (trees_dir "festvox/")  ;; in INST_LDOM_VOX_ldom.scm
	 '(wagon_field_desc "festival/clunits/all.desc")
	 '(wagon_progname "$ESTDIR/bin/wagon")
	 '(wagon_cluster_size 10)
	 '(prune_reduce 0)
	 '(cluster_prune_limit 20)
	 ;; The dictionary of units used at run time
;	 (catalogue_dir "festvox/")   ;; in INST_LDOM_VOX_ldom.scm
	 ;;  Run time parameters 
	 ;; all in INST_LDOM_VOX_ldom.scm
	 ;; Files in db, filled in at build_clunits time
	 ;; (files ("time0001" "time0002" ....))
))))

(define (build_clunits file)
  "(build_clunits file)
Build cluster synthesizer for the given recorded data and domain."
  (build_clunits_init file)
  (do_all)  ;; someday I'll change the name of this function
)

(define (build_clunits_init file)
  "(build_clunits_init file)
Get setup ready for (do_all) (or (do_init))."
  (eval (list INST_LDOM_VOX::closest_voice))

  ;; Add specific fileids to the list for this run
  (set! INST_LDOM_VOX::dt_params
	(append
	 INST_LDOM_VOX::dt_params
	 (list
	  (list
	   'files
	   (mapcar car (load file t))))))
  
  (set! dt_params INST_LDOM_VOX::dt_params)
  (set! clunits_params INST_LDOM_VOX::dt_params)
)

(define (do_prompt name text) 
  "(do_prompt name text) 
Synthesize given text and save waveform and labels for prompts."
  (let ((utt1 (utt.synth (eval (list 'Utterance 'Text text)))))
    (utt.save utt1 (format nil "prompt-utt/%s.utt" name))
    (utt.save.segs utt1 (format nil "prompt-lab/%s.lab" name))
    (utt.save.wave utt1 (format nil "prompt-wav/%s.wav" name))))

(define (build_prompts file)
  "(build_prompt file) 
For each utterances in prompt file, synth and save waveform and
labels for prompts and aligning."
  (set! INST_LDOM_VOX::ldom_prompting_stage t)
  (voice_INST_LDOM_VOX_ldom)
 (let ((p (load file t)))
    (mapcar
     (lambda (l)
       (format t "%s\n" (car l))
       (do_prompt (car l) (cadr l))
       t)
     p)
    t))

(define (rebuild_prompts file)
  "(rebuild_prompt file) 
For each utterances in prompt file, synth from existing lab/ file
and save waveform and labels for prompts and aligning."
  (set! INST_LDOM_VOX::ldom_prompting_stage t)
  (voice_INST_LDOM_VOX_ldom)
 (let ((p (load file t)))
    (mapcar
     (lambda (l)
       (format t "%s\n" (car l))
       (do_reprompt (car l) (cadr l))
       t)
     p)
    t))

(define (do_reprompt name text) 
  "(do_reprompt name text) 
Synthesize from lab files."
  (let ((utt1 (eval (list 'Utterance 'Text text))))
    (utt.relation.load utt1 'Segment 
		       (format nil "lab/%s.lab" name))
    (Int_Targets_Default utt1)
    (Wave_Synth utt1)
    (utt.save.segs utt1 (format nil "prompt-lab/%s.lab" name))
    (utt.save.wave utt1 (format nil "prompt-wav/%s.wav" name))))


(define (build_utts file)
  "(build_utts file) 
For each utterances in prompt file, synthesize and merge aligned labels
to predicted labels building a new utetrances and saving it."
  (set! INST_LDOM_VOX::ldom_prompting_stage t)
  (voice_INST_LDOM_VOX_ldom)
  (let ((p (load file t)))
    (mapcar
     (lambda (l)
       (format t "%s\n" (car l))
       (align_utt (car l) (cadr l))
       t)
     p)
    t))

(define (align_utt name text)
  "(align_utts file) 
Synth an utterance and load in the actualed aligned segments and merge
them into the synthesizer utterance."
  (let ((utt1 (utt.load nil (format nil "prompt-utt/%s.utt" name)))
;	(utt1 (utt.synth (eval (list 'Utterance 'Text text))))
	(silence (car (cadr (car (PhoneSet.description '(silences))))))
	segments actual-segments)
	
    (utt.relation.load utt1 'actual-segment 
		       (format nil "lab/%s.lab" name))
    (set! segments (utt.relation.items utt1 'Segment))
    (set! actual-segments (utt.relation.items utt1 'actual-segment))

    ;; These should align, but if the labels had to be hand edited
    ;; then they may not, we cater here for insertions and deletions
    ;; of silences int he corrected hand labelled files (actual-segments)
    ;; If you need to something more elaborate you'll have to change the
    ;; code below.
    (while (and segments actual-segments)
      (cond
       ((and (not (string-equal (item.name (car segments))
				(item.name (car actual-segments))))
	     (or (string-equal (item.name (car actual-segments)) silence)
		 (string-equal (item.name (car actual-segments)) "H#")
		 (string-equal (item.name (car actual-segments)) "h#")))
	(item.insert
	 (car segments)
	 (list silence (list (list "end" (item.feat 
					(car actual-segments) "end"))))
	 'before)
	(set! actual-segments (cdr actual-segments)))
       ((and (not (string-equal (item.name (car segments))
				(item.name (car actual-segments))))
	     (string-equal (item.name (car segments)) silence))
	(item.delete (car segments))
	(set! segments (cdr segments)))
       ((string-equal (item.name (car segments))
		      (item.name (car actual-segments)))
	(item.set_feat (car segments) "end" 
		       (item.feat (car actual-segments) "end"))
	(set! segments (cdr segments))
	(set! actual-segments (cdr actual-segments)))
       (t
	(format stderr
		"align missmatch at %s (%f) %s (%f)\n"
		(item.name (car segments))
		(item.feat (car segments) "end")
		(item.name (car actual-segments))
		(item.feat (car actual-segments) "end"))
	(error)))
      )

    (mapcar
     (lambda (a)
      ;; shorten and split sliences
      (while (and (string-equal (item.name a) silence)
		  (> (item.feat a "segment_duration") 0.300))
;              (format t "splitting %s silence of %f at %f\n"
;		      (item.name a)
;                      (item.feat a "segment_duration")
;                      (item.feat a "end"))
              (cond
               ((string-equal "h#" (item.feat a "p.name"))
                (item.set_feat (item.prev a) "end"
                               (+ 0.150 (item.feat a "p.end"))))
               ((and (string-equal silence (item.feat a "p.name"))
                     (string-equal silence (item.feat a "p.p.name")))
                (item.set_feat (item.prev a) "end"
                               (+ 0.150 (item.feat a "p.end")))
                (item.set_feat (item.prev a) "name" silence))
               (t
                (item.insert a
                             (list silence
                                   (list 
                                    (list "end" 
				      (+ 0.150 
					(item.feat a "p.end")))))
                             'before)))))
     (utt.relation.items utt1 'Segment))

    (utt.relation.delete utt1 'actual-segment)
    (utt.set_feat utt1 "fileid" name)
    ;; If we have an F0 add in targets too
    (if (probe_file (format nil "f0/%s.f0" name))
	(build::add_targets utt1))
    (utt.save utt1 (format nil "festival/utts/%s.utt" name))
    t))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Some prosody modelling code
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (build::add_targets utt)
  "(build::add_targets utt)
Adds targets based on the F0 in f0/*.f0.  Adds a point to each vowel."
  (let ((fileid (utt.feat utt "fileid"))
	(f0_points))
    (set! f0_points (build::load_f0_points fileid))
    (set! awb_f0_points f0_points)
    ;; Get rid of the old one
    (utt.relation.delete utt 'Target)
    ;; Create a new one
    (utt.relation.create utt 'Target)
    (build::add_target 
     utt
     f0_points)
    utt))

(define (build::add_target utt f0_points)
  "(build::add_target utt f0_points)
Add F0 points at start or syllable, mid point of each vowel, and
last segment before silence.  The F0 continued over non-voiced
periods is such a naive and hopless way its embarrassing."
  (let ((s (utt.relation.first utt 'Segment))
	(f0s f0_points)
	targ)
    (while s
     (if (and (not (member_string
		    (item.name s)
		    (cadr (car (PhoneSet.description '(silences))))))
	      (or (string-equal "1" (item.feat s "syl_initial"))
		  (string-equal "+" (item.feat s "ph_vc"))
		  (member_string 
		   (item.feat s "n.name")
		   (cadr (car (PhoneSet.description '(silences)))))))
	 (begin
	   (set! targ (utt.relation.append utt 'Target s))
	   (if (string-equal "1" (item.feat s "syl_initial"))
	       (item.relation.append_daughter
		targ
		'Target
		(list
		 "0"
		 (list
		  (list 'f0 (build::get_f0_at f0s (item.feat s "segment_start")))
		  (list 'pos (item.feat s "segment_start"))))))
	   (if (string-equal "+" (item.feat s "ph_vc"))
	       (item.relation.append_daughter
		targ
		'Target
		(list
		 "0"
		 (list
		  (list 'f0 (build::get_f0_at f0s (item.feat s "segment_mid")))
		  (list 'pos (item.feat s "segment_mid"))))))
	   (if (member_string 
		(item.feat s "n.name")
		(cadr (car (PhoneSet.description '(silences)))))
	       (item.relation.append_daughter
		targ
		'Target
		(list
		 "0"
		 (list
		  (list 'f0 (build::get_f0_at f0s (item.feat s "segment_end")))
		  (list 'pos (item.feat s "segment_end"))))))))
     (set! s (item.next s))
     ))
)

(define (build::get_f0_at f0s position)
  "(build::get_f0_at f0s position)
Returns the non-zero F0 nearest to position."
  (build::get_f0_at_2
   -1
   f0s
   position))

(define (build::get_f0_at_2 f0 f0s position)
  "(build::get_f0_at f0 f0s position)
Returns the non-zero F0 nearest to position."
  (cond
   ((null f0s)
    (if (> f0 0)
	f0
	110 ;; aint nothing there at all at all
	))
   (t
    (if (> 0 (cadr (car f0s)))
	(set! f0 (cadr (car f0s))))
    (cond
     ((and (>= position (car (car f0s)))
	   (<= position (car (cadr f0s))))
      (if (< f0 1)
	  (build::find_first_f0 f0s)
	  f0))
     (t
      (build::get_f0_at_2 f0 (cdr f0s) position))))))

(define (build::find_first_f0 f0s)
  (cond
   ((null f0s) 
    110  ;; last resort
    )
   ((> (cadr (car f0s)) 0)
    (cadr (car f0s)))
   (t
    (build::find_first_f0 (cdr f0s)))))

(define (build::load_f0_points fileid)
  "(build::load_f0_points fileid)
Extract F0 as ascii times and values from the F0 file and load
it as a simple assoc list."
  (let ((f0asciifile (make_tmp_filename))
	f0fd point points
	(time 0))
    (system
     (format nil "$EST%s/bin/ch_track -otype ascii -o %s f0/%s.f0"
	     "DIR"  ;; to stop that var name being mapped.
	     f0asciifile 
	     fileid))
    (set! f0fd (fopen f0asciifile "r"))
    (while (not (equal? (set! point (readfp f0fd)) (eof-val)))
      (set! points 
	    (cons
	     (list time point) points))
      (set! time (+ 0.005 time))
      ;; skip the second field.
      (readfp f0fd))
    (fclose f0fd)
    (delete-file f0asciifile)
    (reverse points)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Code to try to find bad labelling by looking at duration distribution
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;  A simple sufficient statistics class
(define (suffstats.new)
  (list
   0    ;; n
   0    ;; sum
   0    ;; sumx
   ))

(define (suffstats.set_n x n)
  (set-car! x n))
(define (suffstats.set_sum x sum)
  (set-car! (cdr x) sum))
(define (suffstats.set_sumx x sumx)
  (set-car! (cdr (cdr x)) sumx))
(define (suffstats.n x)
  (car x))
(define (suffstats.sum x)
  (car (cdr x)))
(define (suffstats.sumx x)
  (car (cdr (cdr x))))
(define (suffstats.reset x)
  (suffstats.set_n x 0)
  (suffstats.set_sum x 0)
  (suffstats.set_sumx x 0))
(define (suffstats.add x d)
  (suffstats.set_n x (+ (suffstats.n x) 1))
  (suffstats.set_sum x (+ (suffstats.sum x) d))
  (suffstats.set_sumx x (+ (suffstats.sumx x) (* d d)))
)

(define (suffstats.mean x)
  (/ (suffstats.sum x) (suffstats.n x)))
(define (suffstats.variance x)
  (/ (- (* (suffstats.n x) (suffstats.sumx x))
        (* (suffstats.sum x) (suffstats.sum x)))
     (* (suffstats.n x) (- (suffstats.n x) 1))))
(define (suffstats.stddev x)
  (sqrt (suffstats.variance x)))

(define (cummulate_stats stats phone duration)
  (let ((pstat (car (cdr (assoc_string phone stats))))
	(newstats stats))
    (if (null pstat)
	(begin
	  (set! pstat (suffstats.new))
	  (set! newstats (cons (list phone pstat) stats))))
    (suffstats.add pstat duration)
    newstats))

(define (collect_dur_stats utts)
  (let ((stats nil))
    (mapcar
     (lambda (u)
       (mapcar 
	(lambda (s)
	  (set! stats (cummulate_stats
		       stats
		       (item.name s)
		       (item.feat s "segment_duration"))))
	(utt.relation.items u 'Segment)))
     utts)
    stats))

(define (score_utts utts durstats ofile)
  (let ((ofd (fopen ofile "w")))
    (mapcar
     (lambda (u)
       (let ((score 0) (tot 0))
	 (format ofd "%s " (utt.feat u "fileid"))
	 (mapcar 
	  (lambda (s)
	    (let ((stats (car (cdr (assoc_string (item.name s) durstats))))
		  (dur (item.feat s "segment_duration"))
		  (zscore))
	      (set! tot (+ 1 tot))
	      (set! zscore (/ (- dur (suffstats.mean stats))
			      (suffstats.stddev stats)))
	      (if (< zscore 0)
		  (set! zscore (* -1 zscore)))
	      (if (or (< dur 0.011)
		      (> zscore 3))
		  (set! score (+ 1 score)))))
	  (utt.relation.items u 'Segment))
	 (format ofd "%0.4f %d %d\n"
		 (/ score tot)
		 score
		 tot)))
     utts)))

(define (make_simple_utt fileid)
  (let ((utt (Utterance Text "")))
    (utt.relation.load utt 'Segment
		       (format nil "lab/%s.lab" fileid))
    (utt.set_feat utt "fileid" fileid)
    utt))

(define (find_outlier_utts file ofile)
  (voice_kal_diphone)
  (let ((p (load file t))
	utts dur_states)
    (set! utts (mapcar (lambda (l) (make_simple_utt (car l))) p))
    (set! dur_stats (collect_dur_stats utts))
    (score_utts utts dur_stats ofile)
    t))

(provide 'build_ldom)

