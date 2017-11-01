;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
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

(defvar ofile "ehmm/etc/txt.phseq.data")

(define (getwdur item fp)
  (if item
   (begin
     (format fp "%s %f %s\n" (item.name item) (item.feat item 'word_duration) (item.feat item 'pos))
     (getwdur (item.next item) fp))))

(define (getssq item fp wnm fp1)
  (if item
   (begin
     (set! wnm1 (item.feat item "R:SylStructure.parent.parent.name"))

     (if (not (string-equal wnm1 wnm))
       (if (and (not (equal? wnm 0)) (not (equal? wnm1 0)))
	 (format fp "ssil ")))


     (if (not (string-equal wnm1 wnm))
       (begin 
         (format fp1 "\n")
         (format fp1 "%s " wnm1)))

;     ;(if (string-equal "pau" (item.name item))
;     ;    (begin 
;	   (if (and (item.prev item) (item.next item))
;             (format fp "ssil ")
;	     (format fp "pau ")))
;         (format fp "%s " (item.name item)))
;

     (format fp "%s " (item.name item))
     (format fp1 "%s " (item.name item))
     (set! wnm (item.feat item "R:SylStructure.parent.parent.name"))
     (getssq (item.next item) fp wnm fp1))))


(define (proc_file fid)

    (set! fd (fopen ofile "a"))
    (set! u1 (utt.load nil (format nil "tmp/%s.utt" fid)))
    (set! dfnm (format nil "tmp/%s.dict" fid))
    (set! fd1 (fopen dfnm "w"))
    (fclose fd1)
    (set! fd1 (fopen dfnm "a"))

    ;;(set! w1 (utt.relation.first u1 'Word))

    (set! w1 (utt.relation.first u1 'Segment))

    ;;(getwdur w1 fd)
    ;;(format fd "%s " bnm) 

    (format fd "%s " fid) 

    (set! wnm (item.feat w1 "R:SylStructure.parent.parent.name"))
    (format fd1 "%s " wnm)
    (getssq w1 fd wnm fd1)
    (format fd "\n") 
    (format fd1 "\n") 
    (fclose fd)
    (fclose fd1)
)

(define (phseq datafile outfile)
  (set! ofile outfile)
  (set! fd1 (fopen ofile "w"))  ;; making it empty
  (fclose fd1)
  (mapcar
   (lambda (f)
     (format t "phseq %s\n" (car f))
     (unwind-protect
      (proc_file (car f))
      nil)
     )
   (load datafile t))
  t)

;;;ESTDIR/../festival/bin/festival -b $EHMMDIR/bin/phseq.scm '(phseq "'$PROMPTFILE'" "ehmm/etc/txt.phseq.data")'

(define (genutt datafile ofile)
  (mapcar 
    (lambda (f) 
      (set! txt (car (cdr f)))
      (set! u1 (eval (list 'Utterance 'Text txt)))
      (utt.synth u1)
      (utt.save u1 ofile)
      )
    (load datafile t))
   t)


;;;; for word level processing..... added 20 March 2007

(define (getwsq item wnm fd1)
  (if item
   (begin
     (set! wnm1 (item.feat item "R:SylStructure.parent.parent.name"))

     (if (not (string-equal wnm1 wnm))
       (if (and (not (equal? wnm1 0)))
	 (format fd1 "%s\n"wnm1)))

     (set! wnm (item.feat item "R:SylStructure.parent.parent.name"))
     (getwsq (item.next item) wnm fd1))))


(define (wutt datafile ofile)
  (mapcar 
    (lambda (f) 
      (set! txt (car (cdr f)))
      (set! u1 (eval (list 'Utterance 'Text txt)))
      (utt.synth u1)   ;;; bad thing, but how else to get it work?
      (set! s1 (utt.relation.first u1 'Segment))
      (set! w1 (item.feat s1 "R:SylStructure.parent.parent.name"))

      (set! fd1 (fopen ofile "w"))  ;; making it empty
      (if (not (equal? w1 0))
        (format fd1 "%s\n",w1))
      (getwsq s1 w1 fd1)
      (fclose fd1)
      )
    (load datafile t))
   t)

(define (gen_phone_seq promptfile phseqfile)

  (set! ofd (fopen phseqfile "w"))
  ;; no need to generate a waveform here, so save some time
  (Parameter.set 'Synth_Method 'None)
  (mapcar
   (lambda (x)
     (set! utt1 (SynthText (car (cdr x))))
     (format ofd "%s " (car x)) ;; fileid
     (mapcar
      (lambda (seg)
        (format ofd "%s " (item.name seg))
        (if (and 
             (not (string-equal (item.name seg) "pau"))
             (not (string-equal (item.feat seg "n.name") "pau"))
             (null (item.relation.next seg 'SylStructure)) ;; end of syl
             (null (item.next 
                    (item.relation.parent seg 'SylStructure)))) ; end of word
            (format ofd "ssil "))
        )
      (utt.relation.items utt1 'Segment))
     (format ofd "\n"))
   (load promptfile t))
  (fclose ofd))
