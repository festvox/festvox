;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;                     Carnegie Mellon University                        ;;
;;;                      Copyright (c) 2010-2016                          ;;
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
;;;         Authors: Gopala Krishna Anumanchipalli                        ;;
;;;         Email:   gopalakr@cs.cmu.edu                                  ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defvar cg:smooth_window 1)

(define (cg_syl_ratio x)
        (/ (+ 1 (item.feat x 'syl_in)) (+ 1 (+ (item.feat x 'syl_in) (item.feat x 'syl_out))))
)

(define (cg_phrase_ratio x)
        (/ (+ 1 (cg_find_phrase_number x))(+ 1 (item.feat x 'last.lisp_cg_find_phrase_number)))
)

(define (cg_syls_in_phrase x)
        (+ 1 (item.feat x 'daughter1.R:Word.R:SylStructure.daughter1.syl_out)))


(define (cg_content_words_in_phrase x)
(set! xyz '0)
(set! xyz (+ (item.feat x 'R:SylStructure.parent.parent.R:Word.content_words_in) (item.feat x 'R:SylStructure.parent.parent.R:Word.content_words_out)))
(if (string-equal "content" (item.feat x 'R:SylStructure.parent.parent.R:Word.gpos))(set! xyz (+ 1 xyz)))
xyz)

(define (cg_dumpstats ttdfile phrdesc )
   (mapcar
    (lambda (f)
;;; Load Utterance
      (format t "%s finding tilt params\n" (car f))
      (set! opd (fopen (format nil "logf0/lf0/phrase0.%s.feats" (car f)) "w"))
      (set! utt1 (utt.load nil (format nil "festival/utts/%s.utt" (car f))))
      (set! ltrack (track.resize nil (length (utt.relation.items utt1 'Syllable)) 6))
      (set! lftrack (track.load (format nil "logf0/lf0/%s.lf0" (utt.feat utt1 'fileid))))
      (set! basetrack (track.resize nil (track.num_frames lftrack) 1))
      (set! difftrack (track.resize nil (track.num_frames lftrack) 1))
      (set! basetrack (init_baseline utt1 lftrack basetrack))
      (set! difftrack (cg_difftrack difftrack lftrack basetrack))
      (set! tuttsyl 0)
      (set! tsyl 0)
      (set! phr (utt.relation.first utt1 'Phrase))
;;; Go Phrase -> Word -> Syllable
      (while phr
       (set! tphrsyl 0)
       (mapcar 
        (lambda (wrd)
          (set! wrd (item.relation wrd 'Word))
          (set! tphrsyl (+ tphrsyl (length (item.relation.daughters wrd 'SylStructure))))
          (set! tobisyl 0)
          (mapcar 
           (lambda (syl)
             (set! syl (item.relation syl 'Syllable))
             (set! bt (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'R:SylStructure.daughter1.R:Segment.p.end))))
             (set! et (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'R:SylStructure.daughtern.R:Segment.end))))
             (set! olist (cons (* cg:frame_shift bt) (cons (track.get basetrack bt 0) (cg_tiltanalyze difftrack bt et))))
                                        ;(format "%l\n" olist)
             (set! p 0)
             (mapcar (lambda (x)(track.set ltrack tsyl p x) (set! p (+ 1 p))) olist)
             (set! tsyl (+ 1 tsyl))
             )(item.relation.daughters wrd 'SylStructure))
          )(item.daughters phr))
             (set! phr (item.next phr)))
      (track.save ltrack (format nil "logf0/lf0/%s.ltilt" (car f)))
      (track.save basetrack (format nil "logf0/lf0/%s.base" (car f)))
;;; Dump segment level features of the initial basetrack 
      (set! seg (utt.relation.first utt1 'Segment))
      (while seg
             (if (not (string-equal 'pau (item.name seg)))
                 (begin
                                        ;(format opd "%s " (cg_mean basetrack (nint (* (item.feat seg 'segment_start) (/ 1 cg:frame_shift))) (nint (* (item.feat seg 'segment_end) (/ 1 cg:frame_shift)))))
                   (format opd "%s " (track.get basetrack (nint (* (item.feat seg 'segment_start) (/ 1 cg:frame_shift))) 0))
                   (mapcar (lambda (fff)
                             (format opd "%s " (item.feat seg (car fff)))
                             t) (cdr (car (load phrdesc t))))
                   (format opd "\n"))
                 )
             (set! seg (item.next seg)))
      (fclose opd)
      )
    (load ttdfile t))
(fclose opd)
t)

(define (cg_tiltanalyze l1track bt et)
(set! tbt bt)
(set! maxlf -10)
(set! maxlfind bt)
(while (< tbt et)
   (if (< maxlf (track.get l1track tbt 0))
   (begin (set! maxlf (track.get l1track tbt 0))
	  (set! maxlfind tbt)))
(set! tbt (+ 1 tbt)))
(set! arise (- maxlf (track.get l1track bt 0)))
(set! afall (- (track.get l1track (- et 1) 0) maxlf))
(set! drise (* (- maxlfind bt) cg:frame_shift))
(set! dfall (* (- et maxlfind) cg:frame_shift))
(set! tt (/ (+ (/ (- ( absval arise) (absval afall)) (+ 0.00001 (+ (absval arise) (absval afall))))
	       (/ (- (absval drise) (absval dfall)) (+ (absval drise) (absval dfall))))
         2))
(list maxlf (+ (absval arise) (absval afall)) (+ drise dfall) tt))

(define (init_baseline utt1 lftrack basetrack)
	(set! syl (utt.relation.first utt1 'Syllable))
	(while syl
		(set! bt (nint (* (item.feat syl 'syllable_start) (/ 1 cg:frame_shift))))
		(set! et (nint (* (item.feat syl 'syllable_end) (/ 1 cg:frame_shift))))
		(set! tbt bt)
		(set! minlf 10)
		(while (< tbt et)
			(if (< (track.get lftrack tbt 0) minlf)
			 (set! minlf (track.get lftrack tbt 0)))
		(set! tbt (+ tbt 1)))
		(set! tbt bt)
		;(format t "%s\n" minlf)
		(while (< tbt et)
			(track.set basetrack tbt 0 minlf)
		(set! tbt (+ tbt 1)))
	(set! syl (item.next syl)))
basetrack)

(define (cg_addtrack tracka track1 track2)
(set! itr1 0)
(while (< itr1 (track.num_frames track1))
     (track.set tracka itr1 0 (+ (track.get track1 itr1 0) (track.get track2 itr1 0)))
(set! itr1 (+ 1 itr1)))
tracka)

(define (cg_difftrack trackd track1 track2)
(set! itr1 0)
(while (< itr1 (track.num_frames track1))
     (track.set trackd itr1 0 (- (track.get track1 itr1 0) (track.get track2 itr1 0)))
(set! itr1 (+ 1 itr1)))
trackd)

(define (cg_synthtilt ntrack st base peak tamp tdur ttilt)
;(format t "%s %s %s %s\n" peak tamp tdur ttilt)
(set! ttilt (+ ttilt  0.000000001))
(set! arise (/ (* tamp (+ 1 ttilt)) 2))
(set! afall (/ (* tamp (- 1 ttilt)) 2))

(set! drise (/ (* tdur (+ 1 ttilt)) 2))
(set! dfall (/ (* tdur (- 1 ttilt)) 2))

;Synthesizing the rise region
(set! nt (nint (* drise (/ 1 cg:frame_shift))))
;(set! itr1 1)
(set! itr1 0)

(while (<= itr1 nt)
   (if (> (* itr1 cg:frame_shift) (/ drise 2))
	(set! f (* (* arise 2) (* (- 1 (/ (* itr1 cg:frame_shift) drise)) (- 1 (/ (* itr1 cg:frame_shift) drise)))))
	(set! f (- arise (* (* 2 arise) (* (/ (* itr1 cg:frame_shift) drise) (/ (* itr1 cg:frame_shift) drise)))))
	)
;	(track.set ntrack (- (+ (/ st cg:frame_shift) itr1) 1) 0 (- (+ base peak) f))
	(track.set ntrack (nint (+ (/ st cg:frame_shift) itr1))  0 (- (+ base peak) f))
(set! itr1 (+ 1 itr1)))
(set! st1  (+ (- itr1 1) (/ st cg:frame_shift)))
;(set! st1  (+ itr1 (/ st cg:frame_shift)))

;Synthesizing the fall region
;(set! nt (+ (nint (* dfall (/ 1 cg:frame_shift))) 1) )
(set! nt (- (nint (/ tdur cg:frame_shift)) nt))
(set! itr1 1)

(while (<= itr1 nt)
   (if (> (* itr1 cg:frame_shift) (/ dfall 2))
	(set! f (* (* afall 2) (* (- 1 (/ (* itr1 cg:frame_shift) dfall)) (- 1 (/ (* itr1 cg:frame_shift) dfall)))))
	(set! f (- afall (* (* 2 afall) (* (/ (* itr1 cg:frame_shift) dfall) (/ (* itr1 cg:frame_shift) dfall)))))
	)
	;(track.set ntrack (- (+ st1 itr1) 1)  0 (+ peak (- (+ base f) afall)))
	(track.set ntrack (+ st1 itr1)   0 (+ peak (- (+ base f) afall)))
(set! itr1 (+ 1 itr1)))
ntrack)


(define (absval val)
(if (< val 0) (set! val (* -1 val))) 
val)

(define (cg_max trackb bt et)
(set! itr1 bt)
(set! tmax 0)
(while (< itr1 et)
	(if (> (track.get trackb itr1 0) tmax)
	(set! tmax (track.get trackb itr1 0)))
(set! itr1 (+ 1 itr1)))
tmax)

(define (cg_mean trackb bt et)
(set! tsum 0)
(set! itr1 bt)
(while (< itr1 et)
(set! tsum (+ tsum (track.get trackb itr1 0)))
(set! itr1 (+ 1 itr1)))
(/ tsum (- et bt)))

(define (cg_splitf0 ttdfile itnum phrdesc accdesc)
    (set! aftree (load (format nil "logf0/cb%s.tree" (- itnum 1)) t))
    (set! aftrack (track.load (format nil "logf0/cb%s.params" (- itnum 1))))
    (set! basetree (load (format nil "logf0/phrase%s.tree" (- itnum 1)) t))

    (mapcar
      (lambda (f)
        (set! opd (fopen (format nil "logf0/lf%s/phrase%s.feats" itnum (car f)) "w"))
        (set! oad (fopen (format nil "logf0/lf%s/acc%s.feats" itnum (car f)) "w"))
        (set! oald (fopen (format nil "logf0/lf%s/acode%s.data" itnum (car f)) "w"))
	(set! utt1 (utt.load nil (format nil "festival/utts/%s.utt" (car f))))
	(set! lftrack (track.load (format nil "logf0/lf0/%s.lf0" (car f))))
	(set! pttrack (track.load (format nil "logf0/lf%s/%s.ltilt" (- itnum 1) (car f))))
    	(set! nttrack (track.resize nil (length (utt.relation.items utt1 'Syllable)) 6))
    	(set! difftrack (track.resize nil (track.num_frames lftrack) 1))
    	(set! newtrack (track.resize nil (track.num_frames lftrack) 1))
    	(set! basetrack (track.resize nil (track.num_frames lftrack) 1))
	;(set! basetrack (track.load (format nil "logf0/lf%s/%s.base" (- itnum 1) (car f))))
    	(set! newbase (track.resize nil (track.num_frames lftrack) 1))
    	(set! newacc (track.resize nil (track.num_frames lftrack) 1))
	(set! seg (utt.relation.first utt1 'Segment))
	(while seg
		(if (not (string-equal 'pau (item.name seg)))
		(begin
		    (set! val (cadr (wagon seg (car basetree))))
		    (set! tbt (nint (* (item.feat seg 'segment_start) (/ 1 cg:frame_shift))))
		    (set! et (nint (* (item.feat seg 'end) (/ 1 cg:frame_shift))))
		    (while  (< tbt et)
			(track.set basetrack tbt 0 val)
			(set! tbt (+ 1 tbt)))
		))
	(set! seg (item.next seg)))
	
	(track.save basetrack (format nil "logf0/lf%s/%s.pbase" itnum (car f)))
	(cg_difftrack difftrack lftrack basetrack)
	(set! syl (utt.relation.first utt1 'Syllable))
	(set! sylcnt 0)
	(format oald "# ")
	(while syl
	    (set! bt (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_start))))
	    (set! tbt bt)
	    (set! et (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_end))))
	    (set! ctlist (cg_tiltanalyze difftrack bt et))
	    (item.set_feat syl 'phraseval (track.get basetrack bt 0))
	    (item.set_feat syl 'tpeak (car ctlist))
	    (item.set_feat syl 'tamp (cadr ctlist))
	    (item.set_feat syl 'tdur (cadr (cdr ctlist)))
	    (item.set_feat syl 'ttilt (cadr (cddr ctlist)))
	    ;(set! tlist (car (wagon syl (car aftree))))
; Finding closest leaf instead of wagon tree traversal
	    (set! tlist (findclosest aftrack (list (item.feat syl 'tpeak) (item.feat syl 'tamp) (item.feat syl 'tdur) (item.feat syl 'ttilt))))

	    (format oad "%s " tlist)
	    (format oald "%s " tlist)
            (mapcar
            (lambda (fff)
           	(format oad "%s " (item.feat syl (car fff)))
            t) (cdr (car (load accdesc t))))
	    (format oad "\n")

	    (set! newtrack (cg_synthtilt newtrack (* bt cg:frame_shift) 0 (track.get aftrack tlist 0) (track.get aftrack tlist 2) (track.get pttrack sylcnt 4) (track.get aftrack tlist 6)))
	    (set! sylcnt (+ 1 sylcnt))
	    (set! syl (item.next syl)))
	    (format oald "#\n")
	    (set! newtrack (cg_addtrack newtrack newtrack basetrack))

;	  (set! phr (utt.relation.first utt1 'Phrase))
;	  (while phr
;	    (set! bt (nint (* (/ 1 cg:frame_shift) (item.feat phr 'R:Phrase.daughter1.R:Word.word_start))))
;	    (set! et (nint (* (/ 1 cg:frame_shift) (item.feat phr 'R:Phrase.daughtern.R:Word.word_end))))
;	    (set! newtrack (cg_smooth newtrack bt et))
;	  (set! phr (item.next phr)))

	    (track.save newtrack (format nil "logf0/lf%s/%s.lf0" itnum (car f)))
	    (set! difftrack (cg_difftrack difftrack lftrack newtrack))
	    (track.save (cg_addtrack newbase difftrack basetrack) (format nil "logf0/lf%s/%s.base" itnum (car f)))
	    (set! seg (utt.relation.first utt1 'Segment))
	    (while seg
		(if (not (string-equal 'pau (item.name seg)))
		   (begin (format opd "%s " (cg_mean newbase (nint (* (item.feat seg 'segment_start) (/ 1 cg:frame_shift))) (nint (* (item.feat seg 'segment_end) (/ 1 cg:frame_shift)))))	
	           (mapcar
        	   (lambda (fff)
          		(format opd "%s " (item.feat seg (car fff)))
	           t) (cdr (car (load phrdesc t))))
		   (format opd "\n")
		))
	(set! seg (item.next seg)))
	(set! newacc (cg_difftrack newacc lftrack newbase))
	(set! syl (utt.relation.first utt1 'Syllable))
	(set! sylcnt 0)
	(while syl
	    (set! bt (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_start))))
	    (set! tbt bt)
	    (set! et (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_end))))
	    (set! newtlist (cg_tiltanalyze newacc bt et))
	    (track.set nttrack sylcnt 0 (item.feat syl 'syllable_start))
	    (track.set nttrack sylcnt 1 (cg_mean newbase bt et))
	    (set! p 2)
	    (mapcar (lambda (x) (track.set nttrack sylcnt p x) (set! p (+ 1 p ))) newtlist)
	    (set! sylcnt (+ 1 sylcnt))
	(set! syl (item.next syl)))
	(track.save nttrack (format nil "logf0/lf%s/%s.ltilt" itnum (car f)))
        (fclose opd)
	(fclose oad)
	(fclose oald)
      )(load ttdfile t))
)

(define (cg_smooth strack bt et)
(set! tbt (+ cg:smooth_window bt))
(while (< tbt (- et cg:smooth_window))
	(set! itbt (- tbt cg:smooth_window))
	(set! td 0)
	(set! tcnt 0)
	(while (< itbt (+ tbt cg:smooth_window))
	(set! td (+ td (track.get strack itbt 0)))
	(set! tcnt (+ 1 tcnt))
	(set! itbt (+ 1 itbt)))
	(if (> tcnt 0)(track.set strack tbt 0 (/ td tcnt)))
(set! tbt (+ 1 tbt)))
strack)

(define (cg_mean trackb bt et)
(set! tsum 0)
(set! itr1 bt)
(while (< itr1 et)
(set! tsum (+ tsum (track.get trackb itr1 0)))
(set! itr1 (+ 1 itr1)))
(/ tsum (- et bt)))

(define (findclosest cbtrack flist)
(set! tbt 0)
(set! mind 1000)
(while (< tbt (track.num_frames cbtrack))
   (set! tbc 0)
   (set! d 0)
   (set! tlist flist)
   (set! d (+ 
	     (* (- (track.get cbtrack tbt 0) (nth 0 flist)) (- (track.get cbtrack tbt 0) (nth 0 flist))) ; tiltpeak
	     (+ 0 ; (* (- (track.get cbtrack tbt 2) (nth 1 flist)) (- (track.get cbtrack tbt 2) (nth 1 flist))) ; tiltamp
	     (* (- (track.get cbtrack tbt 6) (nth 3 flist)) (- (track.get cbtrack tbt 6) (nth 3 flist)))))) ; tilttilt
;   (while (< tbc (track.num_channels cbtrack))
;        (set! td (- (track.get cbtrack tbt tbc) (car tlist)))
;        (set! d (+ d (* td td)))
;        (set! tlist (cdr tlist))
;   (set! tbc (+ 2 tbc)))
   (if (< d mind) (begin (set! mind d)(set! mindx tbt)))
(set! tbt (+ 1 tbt)))
mindx)

(define (spamf0_utt utt1)

  (set! ptreefd (load "festival/trees/phrase.tree" t))
  (set! atreefd (load "festival/trees/acc.tree" t))
  (set! atrack (track.load "festival/trees/cb.params"))

;  (ngram.load 'acode_ngram "festival/trees/acode.2.ngram")
;  (wfst.load 'acode_wfst "wfst/codebook.wfst")

  (set! tmp_track (track.copy (utt.feat utt1 "param_track")))

  (set! acctrack (track.resize nil (nint (* (/ 1 cg:frame_shift) (item.feat (utt.relation.last utt1 'Segment) 'end))) 1))
  (set! newtrack (track.resize nil (nint (* (/ 1 cg:frame_shift) (item.feat (utt.relation.last utt1 'Segment) 'end))) 1))
    
  ;; Predict accents (by viterbi (ngram or wfst) or best)
  (if cg:spamf0_viterbi
      (begin
        (set! gen_vit_params
              (list
               (list 'Relation "Syllable")
               (list 'return_feat "acode")
               (list 'p_word "#")
               (list 'pp_word "0")
;               (list 'ngramname 'acode_ngram)
               (list 'wfstname 'acode_wfst)
               (list 'cand_function 'acode_cand_function)
               ))
        (Gen_Viterbi utt1)
        ;; Make the predicted codebook a number, not c99
        (mapcar
         (lambda (s)
           (item.set_feat
            s "acode" (parse-number (string-after (item.feat s "acode") "c")))
;           (format t "predicted acode %s\n" (item.feat s "acode"))
           )
         (utt.relation.items utt1 'Syllable)
        ))
      (begin ;; non viterbi method
        (mapcar
         (lambda (s)
           (item.set_feat 
            s "acode" 
            (car (wagon s (car atreefd))))
           (format t "predicted acode %s\n" (item.feat s "acode"))
           )
         (utt.relation.items utt1 'Syllable))))

  (set! syl (utt.relation.first utt1 'Syllable))
  (while syl
     (set! bt (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_start))))
     (set! et (nint (* (/ 1 cg:frame_shift ) (item.feat syl 'syllable_end))))
     (set! cbnum (item.feat syl "acode"))
; 1.380790 0.281137 1.922710 0.538871 0.408032 0.151673 -0.412051 0.138119
     (set! acctrack 
           (cg_synthtilt 
            acctrack 
            (* bt cg:frame_shift) 0 
            (track.get atrack cbnum 0) 
            (track.get atrack cbnum 2) 
            (item.feat syl 'syllable_duration) 
            (track.get atrack cbnum 6)))
;	(format t "%s %s %s %s\n" (track.get atrack cbnum 0) (track.get atrack cbnum 2) (item.feat syl 'syllable_duration) (track.get atrack cbnum 6))

     (mapcar 
      (lambda (seg)
        (set! bval (cadr (wagon seg (car ptreefd))))
;	(format t "%s\n" bval)
        (set! tbt (nint (* (/ 1 cg:frame_shift) (item.feat seg 'segment_start))))
        (set! tet (nint (* (/ 1 cg:frame_shift) (item.feat seg 'end))))
        (while (< tbt tet)
           (track.set newtrack tbt 0 (+ bval (track.get acctrack tbt 0)))
           (set! tbt (+ 1 tbt))))
      (item.relation.daughters syl 'SylStructure))

     (set! syl (item.next syl)))
  (if cg:F0_smooth (cg_F0_smooth newtrack 0))

  (set! tnr 0)
  (while (< tnr (track.num_frames tmp_track))
     (if (eq? 0 (track.get newtrack tnr 0))
         (track.set tmp_track tnr 0 0)
         (track.set tmp_track tnr 0 (exp (track.get newtrack tnr 0))))
     (set! tnr (+ tnr 1)))
        

  (utt.set_feat utt1 "param_track" tmp_track)
  utt1)


(define (acode_cand_function s)
;;  (set! tree "festival/trees/acc.tree")
  (set! probs
        (acode_reverse_probs
         (cdr (reverse (wagon s (car atreefd))))))
  ;; For WFST the classes need to be c99
  (set! probs
        (mapcar
         (lambda (p)
           (if (string-equal cg:spamf0_viterbi 'wfst)
               (list (format nil "c%02d" (car p)) (cadr p))
               p))
         probs))
  probs)

(define (acode_reverse_probs pdf)
  (cond    
   ((null pdf) nil)
   ((not (> (car (cdr (car pdf))) 0))  ;; ignore zero prob codebooks
    (acode_reverse_probs (cdr pdf)))
   (t      
    (cons
     (list (car (car pdf)) 
           (/ (car (cdr (car pdf)))
              (acode_prob (car (car pdf) )))) 
     (acode_reverse_probs (cdr pdf))))))

(define (acode_prob acode)
  (let ((xxx (assoc_string acode acode_probs)))
    (if xxx
        (car (cdr xxx))
        (begin
          (format t "unknown prob %s\n" acode)
          (errrr)
          0.000012))))
