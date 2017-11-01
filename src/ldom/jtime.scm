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
;;; A japanese talking clock sitting completely on top of an English    ;;;
;;; voice                                                               ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (jsaytime)
  "(jsaytime)
Say the current time with the current voice."
  (jsaythetime (jget-the-time)))

(define (jsaythistime time)
  "(jsaythistime)
Say the givens time, e.g. \"11:23\" with the current voice."
  (if (not (string-matches time "[012][0-9]?:[0-5][0-9]"))
      (error "not a valid date" time))
  (jsaythetime
   (read-from-string
    (format 
     nil
     "(%s %s)"
     (string-before time ":")
     (string-after time ":")))))

(define (jgettimestring actual-time)
  "retruns string of words saying time."
  (let (jround-time time-string approximator)
;    (format t "%l\n" actual-time)
    (set! jround-time (jround-up-time actual-time))
    (set! approximator (japprox actual-time))
    ;;; Construct the time expression
    (set! time-string
          (string-append
	   "tadaima, "
	   (if (string-equal approximator "chodo ") "chodo " "")
	   (jampm-string jround-time)
	   (jhour-string jround-time)
	   (jminute-string jround-time)
	   (if (not (string-equal approximator "chodo ")) approximator "")
	   "desu. "))))

(define (jsaythetime actual-time)
   (let ((time-string (jgettimestring actual-time)))
     (format t "%s\n" time-string)
    ;;; Synthesize it
    (SayText time-string)
))

(define (savetime ofile)
   (let ((time-string (jgettimestring (jget-the-time))))
    ;(format t "%s\n" time-string)
    ;;; Synthesize it
    (utt.save.wave
     (utt.synth 
      (eval (list 'Utterance 'Text time-string)))
     ofile
     'riff)
))

(define (jget-the-time)
"Returns a list of hour and minute and second, for later processing"
 (let (date)
   (system "date | awk '{print $4}' | tr : ' ' >/tmp/saytime.tmp")
   (set! date (load "/tmp/saytime.tmp" t)) ;; loads the file unevaluated
   (system "rm /tmp/saytime.tmp")
   date)
)

(define (jround-up-time time)
"Rounds time up/down to nearest five minute interval"
  (let ((hour (car time))
	(min (car (cdr time)))
	(sec (car (cdr (cdr time)))))
    (set! min (jround-min (+ 2 min)))
    (if (> min 55)
	(if (> hour 23)
	    (list 1 0 sec)
	    (list (+ 1 hour) 0 sec))
	(list hour min sec))))

(define (jround-min min)
"Returns minutes rounded down to nearest 5 minute interval"
  (cond
   ((< min 5)
    0)
   (t
    (+ 5 (jround-min (- min 5))))))

(define (japprox time)
"Returns a string stating the approximation of the time.
   chodo -- within a minute either side
   ni naru  -- 1-2 minutes before
   sugi - 1-3 minutes after
"
 (let ((rm (jround-min (car (cdr time))))
       (min (car (cdr time))))
   (cond
    ((or (< (- min rm) 1)
	 (> (- min rm) 4))
     "chodo ")
    ((< (- min rm) 3)
     "sugi ")
    (t
     "ni naru tokoro "))))

(define (jhour-string time)
"Return description of hour"
  (let ((hour (car time)))
    (if (> hour 12)
	(set! hour (- hour 12)))
    (cond 
     ((eq hour 1) "ichi ji ")
     ((eq hour 2) "ni ji ")
     ((eq hour 3) "san ji ")
     ((eq hour 4) "yo ji ")
     ((eq hour 5) "go ji ")
     ((eq hour 6) "roku ji ")
     ((eq hour 7) "shichi ji ")
     ((eq hour 8) "hachi ji ")
     ((eq hour 9) "ku ji ")
     ((eq hour 10) "ju ji ")
     ((eq hour 11) "ju ichi ji ")
     ((eq hour 12) "ju ni ji ")
     (t ;; hmm
      "ju ji "))))

(define (jminute-string time)
"Return description of minute"
  (let ((min (car (cdr time))))
    (cond
     ((or (eq min 0) (eq min 60)) " ")
     ((eq min 5) "go fun ")
     ((eq min 10) "ju pun ")
     ((eq min 15) "ju go fun ")
     ((eq min 20) "ni ju pun ")
     ((eq min 25) "ni ju go fun ")
     ((eq min 30) "han ")
     ((eq min 35) "san ju go fun ")
     ((eq min 40) "yon ju pun ")
     ((eq min 45) "yon ju go fun ")
     ((eq min 50) "go ju pun ")
     ((eq min 55) "go ju go fun ")
     (t
      " "))))

(define (jampm-string time)
"Return morning/afternoon or evening string"
  (let ((hour (car time)))
   (cond
    ((or (eq hour 0) (eq hour 12) (eq hour 24))
     " ")
    ((< hour 12)
     "gozen ")
    (t
     "gogo "))))

(provide 'jtime)

