;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                 and Alan W Black and Kevin A. Lenzo                 ;;;
;;;                         Copyright (c) 2000                          ;;;
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
;;; Modification of Festival saytime.sh program to run as loadable lisp ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (saytime)
  "(saytime)
Say the current time with the current voice."
  (saythetime (get-the-time)))

(define (saythistime time)
  "(saythistime)
Say the givens time, e.g. \"11:23\" with the current voice."
  (if (not (string-matches time "[012][0-9]?:[0-5][0-9]"))
      (error "not a valid date" time))
  (saythetime
   (read-from-string
    (format 
     nil
     "(%s %s)"
     (string-before time ":")
     (string-after time ":")))))

(define (saythetime actual-time)
  (let (round-time time-string)
;    (format t "%l\n" actual-time)
    (set! round-time (round-up-time actual-time))
    ;;; Construct the time expression
    (set! time-string
          (string-append
	   "The time is now, "
	   (approx actual-time)
	   (minute-string round-time)
	   (hour-string round-time)
	   (ampm-string round-time)))
    ;(format t "%s\n" time-string)
    ;;; Synthesize it
    (SayText time-string)
))

(define (savetime ofile)
  (let ((actual-time (get-the-time))
        round-time time-string)
    (set! round-time (round-up-time actual-time))
    ;;; Construct the time expression
    (set! time-string
          (string-append
           "The time is now, "
           (approx actual-time)
           (minute-string round-time)
           (hour-string round-time)
           (ampm-string round-time)))
    ;(format t "%s\n" time-string)
    ;;; Synthesize it
    (utt.save.wave
     (utt.synth 
      (eval (list 'Utterance 'Text time-string)))
     ofile
     'riff)
))

(define (get-the-time)
"Returns a list of hour and minute and second, for later processing"
 (let (date)
   (system "date | awk '{print $4}' | tr : ' ' >/tmp/saytime.tmp")
   (set! date (load "/tmp/saytime.tmp" t)) ;; loads the file unevaluated
   (system "rm /tmp/saytime.tmp")
   date)
)

(define (round-up-time time)
"Rounds time up/down to nearest five minute interval"
  (let ((hour (car time))
	(min (car (cdr time)))
	(sec (car (cdr (cdr time)))))
    (set! min (round-min (+ 2 min)))
    (if (> min 55)
	(if (> hour 23)
	    (list 1 0 sec)
	    (list (+ 1 hour) 0 sec))
	(list hour min sec))))

(define (round-min min)
"Returns minutes rounded down to nearest 5 minute interval"
  (cond
   ((< min 5)
    0)
   (t
    (+ 5 (round-min (- min 5))))))

(define (approx time)
"Returns a string stating the approximation of the time.
   exactly -- within a minute either side
   almost  -- 1-2 minutes before
   just after - 1-2 minutes after
   a little after 2-3 minutes after
"
 (let ((rm (round-min (car (cdr time))))
       (min (car (cdr time))))
   (cond
    ((or (< (- min rm) 1)
	 (> (- min rm) 3))
     "exactly ")
    ((< (- min rm) 2)
     "just after ")
    ((< (- min rm) 3)
     "a little after ")
    (t
     "almost "))))

(define (hour-string time)
"Return description of hour"
  (let ((hour (car time)))
    (if (> (car (cdr time)) 30)
	(set! hour (+ 1 hour)))
    (cond 
     ((or (eq hour 0) (eq hour 24))
      "midnight ")
     ((> hour 12)
      (string-append (- hour 12) ", "))
     (t
      (string-append hour ", ")))))

(define (minute-string time)
"Return description of minute"
  (let ((min (car (cdr time))))
    (cond
     ((or (eq min 0) (eq min 60)) " ")
     ((eq min 5) "five past ")
     ((eq min 10) "ten past ")
     ((eq min 15) "quarter past ")
     ((eq min 20) "twenty past ")
     ((eq min 25) "twenty-five past ")
     ((eq min 30) "half past ")
     ((eq min 35) "twenty-five to ")
     ((eq min 40) "twenty to ")
     ((eq min 45) "quarter to ")
     ((eq min 50) "ten to ")
     ((eq min 55) "five to ")
     (t
      "something else "))))

(define (ampm-string time)
"Return morning/afternoon or evening string"
  (let ((hour (car time)))
   (cond
    ((or (eq hour 0) (eq hour 12) (eq hour 24))
     " ")
    ((< hour 12)
     "in the morning. ")
    ((< hour 18)
     "in the afternoon. ")
    (t
     "in the evening. "))))

(provide 'time)

