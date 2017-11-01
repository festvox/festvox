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
;;;                                                                     ;;;
;;; For imposing (natural) intonation on synthesized utterances         ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (External_Int_Targets utt)
  "(External_Int_Targets utt)
Get the F0 from an external source.  There are assumptions about
durations here, that are *your* responsibility."
  (let (f0item f0track)
    (utt.relation.create utt 'f0)
    (set! f0item (utt.relation.append utt 'f0))
    (item.set_feat f0item "name" "f0") ;; to be tidy
    (set! f0track (track.load 
		   (format nil "f0/%s.f0" (utt.feat utt "fileid"))))
    (item.set_feat f0item "f0" f0track)
    (build::add_targets utt)
    utt))

(define (External_Duration utt)
  "(External_Duration utt)
Get the durations from an external source.  Do a little fudging in case
extra pauses have been inserted by the labelling."
  (let (actual-segs synth-segs)
    (utt.relation.load utt 'actual-segment 
		       (format nil "lab/%s.lab" (utt.feat utt "fileid")))
    (set! actual-segs (utt.relation.items utt 'actual-segment))
    (set! synth-segs (utt.relation.items utt 'Segment))
    (while (and actual-segs synth-segs)
     (if (string-equal (item.name (car actual-segs)) "H#")
	 (item.set_feat (car actual-segs) "name" "pau"))
     (cond
      ((and (not (string-equal (item.name (car synth-segs))
			       (item.name (car actual-segs))))
	    (string-equal "pau" (item.name (car actual-segs))))
       (item.insert
	(car synth-segs)
	(list "pau" (list (list "end" (item.feat 
				       (car actual-segs) "end"))))
	'before)
       (set! actual-segs (cdr actual-segs)))
      ((and (not (string-equal (item.name (car synth-segs))
			       (item.name (car actual-segs))))
	    (string-equal "pau" (item.name (car synth-segs))))
       (item.delete (car synth-segs))
       (set! synth-segs (cdr synth-segs)))
      ((string-equal (item.name (car synth-segs))
		     (item.name (car actual-segs)))
       (item.set_feat (car synth-segs) "end" 
		      (item.feat (car actual-segs) "end"))
       (set! synth-segs (cdr synth-segs))
       (set! actual-segs (cdr actual-segs)))
      (t
       (format stderr
	       "align missmatch at %s (%f) %s (%f)\n"
	       (item.name (car synth-segs))
	       (item.feat (car synth-segs) "end")
	       (item.name (car actual-segs))
	       (item.feat (car actual-segs) "end"))
       (error))))
    (utt.relation.delete utt 'actual-segment)
    (set! last-seg (car (last (utt.relation.items utt 'Segment))))
    (item.set_feat
     last-seg
     "end"
     (+ 0.200 (item.feat last-seg "p.end")))
    utt))

;;;
;;;   These are example uses of the above two functions
;;;

(define (voice_kal_impose)
  (voice_us1_mbrola)
  (set! postlex_vowel_reduce_table nil)
;  (voice_kal_diphone)
;  (voice_cmu_us_sls_diphone)
  (Parameter.set 'Duration_Method External_Duration)
  (Parameter.set 'Int_Target_Method External_Int_Targets)
)

(define (impose_all datafile)
  (mapcar
   (lambda (a)
     (let ((fileid (car a))
	   (text (car (cdr a))))
     (format t "%s\n" fileid)
     (utt.save.wave
      (impose_synth fileid text)
      (format nil "wav.us1c/%s.wav" fileid)
      'riff)))
   (load datafile t))
  t)

(define (impose_synth fileid text)
  "(impose_synth fileid text)
A little example imposition synthesis.  The utterances requires the
fileid to be set so this is included here in what would otherwise
be normal synthesis (except duration and F0 come from external sources)."
  (voice_kal_impose) ;; or whatever
  (let ((utt1 (eval (list 'Utterance 'Text text))))
    (utt.set_feat utt1 "fileid" fileid)
    (utt.synth utt1)
    utt1))


  
