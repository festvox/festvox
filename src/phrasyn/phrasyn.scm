;###########################################################################
;##                                                                       ##
;##                  Language Technologies Institute                      ##
;##                     Carnegie Mellon University                        ##
;##                       Copyright (c) 2010-2011                         ##
;##                        All Rights Reserved.                           ##
;##                                                                       ##
;##  Permission is hereby granted, free of charge, to use and distribute  ##
;##  this software and its documentation without restriction, including   ##
;##  without limitation the rights to use, copy, modify, merge, publish,  ##
;##  distribute, sublicense, and/or sell copies of this work, and to      ##
;##  permit persons to whom this work is furnished to do so, subject to   ##
;##  the following conditions:                                            ##
;##   1. The code must retain the above copyright notice, this list of    ##
;##      conditions and the following disclaimer.                         ##
;##   2. Any modifications must be clearly marked as such.                ##
;##   3. Original authors' names are not deleted.                         ##
;##   4. The authors' names are not used to endorse or promote products   ##
;##      derived from this software without specific prior written        ##
;##      permission.                                                      ##
;##                                                                       ##
;##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
;##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
;##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
;##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
;##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
;##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
;##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
;##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
;##  THIS SOFTWARE.                                                       ##
;##                                                                       ##
;###########################################################################
;##                                                                       ##
;##            Authors: Alok Parlikar                                     ##
;##            Email:   aup@cs.cmu.edu                                    ##
;##                                                                       ##
;###########################################################################
;##                                                                       ##
;##  Syntactic Phrasing Model                                             ##
;##                                                                       ##
;###########################################################################

(define (dump_scfg_corpus datafile corpusfile mode)
  (set! ofd (fopen corpusfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Phrasing Tree: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "festival/utts/%s.utt" (car x))))
     (format ofd "( ( ")
     (mapcar
      (lambda (w)
        ;(format ofd "%s " (item.feat w "phr_pos"))
	;(format ofd "%s " (item.feat w "pos"))
	;(format ofd "%s " (item.feat w "gpos"))
	(format ofd "%s " (item.feat w mode))
        (if (and (null (item.relation.next w "Phrase"))
                 (item.next w))
            (format ofd ") ( "))
        )
      (utt.relation.items utt1 'Word))
     (format ofd ") )\n"))
   (load datafile t))
  (format t "Done dumping trees %20s\n" "")  
  (fclose ofd))

(define (dump_lm_corpus datafile corpusfile)
  (set! ofd (fopen corpusfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "festival/utts/%s.utt" (car x))))
     (mapcar
      (lambda (w)
	(format ofd "%s " (item.feat w 'word_break)))
      (utt.relation.items utt1 'Word))
     (format ofd "\n"))
   (load datafile t))
  (format t "Done dumping LM corpus %20s\n" "")
  (fclose ofd))

(define (dump_breaks_nopunc datafile uttdir outputfile)
  (set! ofd (fopen outputfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "%s/%s.utt" uttdir (car x))))
     (mapcar
      (lambda (w)
	(if (string-equal "0" (has_punc w))
	    (begin
	      (if (string-equal "phrase" (item.feat w 'word_break))
		  (format ofd "1\n")
		  (format ofd "%s\n" (item.feat w 'word_break))))
	    ()))
      (utt.relation.items utt1 'Word)))
     (load datafile t))
  (format t "Done dumping breaks %20s\n" "")           
  (fclose ofd))

(define (dump_breaks datafile uttdir outputfile)
  (set! ofd (fopen outputfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "%s/%s.utt" uttdir (car x))))
     (mapcar
      (lambda (w)
	(if (string-equal "phrase" (item.feat w 'word_break))
	    (format ofd "1\n")
	    (format ofd "%s\n" (item.feat w 'word_break))))
      (utt.relation.items utt1 'Word)))
     (load datafile t))
  (format t "Done dumping breaks %20s\n" "")           
  (fclose ofd))

(define (dump_breaks_line_by_line datafile uttdir outputfile)
  (set! ofd (fopen outputfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "%s/%s.utt" uttdir (car x))))
     (mapcar
      (lambda (w)
	(if (string-equal "phrase" (item.feat w 'word_break))
	    (format ofd "1 ")
	    (format ofd "%s " (item.feat w 'word_break))))
      (utt.relation.items utt1 'Word))
     (format ofd "\n"))
     (load datafile t))
  (format t "Done dumping breaks %20s\n" "")           
  (fclose ofd))

(define (dump_breaks_oneline datafile uttdir outputfile)
  (set! ofd (fopen outputfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "%s/%s.utt" uttdir (car x))))
     (mapcar
      (lambda (w)
	(format ofd "%s " (item.feat w 'word_break)))
      (utt.relation.items utt1 'Word))
     (format ofd "\n"))
     (load datafile t))
  (format t "Done dumping breaks %20s\n" "")           
  (fclose ofd))

(define (dump_breaks_durations datafile uttdir outputfile)
(set! ofd (fopen outputfile "w"))
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\r" (car x))
     (set! utt1 (utt.load nil (format nil "%s/%s.utt" uttdir (car x))))
     (mapcar
      (lambda (w)
	(format ofd "%s %s %1.3f\n" (item.feat w 'word_break) (item.feat w 'R:SylStructure.daughtern.daughtern.R:Segment.n.name) (item.feat w 'R:SylStructure.daughtern.daughtern.R:Segment.n.segment_duration)))
      (utt.relation.items utt1 'Word)))
     (load datafile t))
  (format t "Done dumping breaks %20s\n" "")
  (fclose ofd))

(define (posmap utt mode)
  (mapcar
   (lambda (w)
     (item.set_feat
      w "phr_pos"
       (item.feat w mode)))
   (utt.relation.items utt 'Word))
  utt)


(define (utt_parse datafile outdir mode)
  (mapcar
   (lambda (a)
     (format t "utt_parsing %10s\r" (car a))
     (set! utt (utt.load nil (format nil "festival/utts/%s.utt" (car a))))
     (set! utt_actual (eval (list 'Utterance 'Text (cadr a))))
;     (format t "%l\n"
;             (mapcar
;              (lambda (a) (item.name a))
;              (utt.relation.items utt 'Word)))
     (posmap utt mode)
     (ProbParse utt)
     (utt.save utt (format nil "%s/%s.utt" outdir (car a))))
   (load datafile t)) 
  (format t "Done utt_parsing %20s\n" "")
  t)

(define (save_utt datafile uttdir)
  (mapcar
   (lambda (x)
     (format t "Dumping Utterance: %10s\n" (car x))
     (set! utt1 (eval (list 'Utterance 'Text (car (cdr x)))))
     (utt.synth utt1)
     (utt.save utt1 (format nil "%s/%s.utt" uttdir (car x))))
   (load datafile t))
  (format t "Done dumping utterances %20s\n" "")
  t)

(define (pppbreak x)
  (cond
   ((null (item.relation.next x 'Phrase))
    (item.feat x 'R:Phrase.parent.name))
   (t
    "NB")))

(define (dist_to_last_break w)
  (set! c 1)
  (set! ww (item.relation.prev w 'Word))
  (while (and ww
              (string-equal (item.feat ww "pbreak") "NB"))
    (set! c (+ 1 c))
    (set! ww (item.prev ww)))
  c
)

(define (dist_to_eos w)
  (set! c 0)
  (set! ww (item.relation.next w 'Word))
  (while ww
   (set! c (+ 1 c))
   (set! ww (item.next ww)))
  c
)

(define (scfg_end_brackets x)
  (set! c 0)
  (set! q x)
  (while (and q (null (item.relation.next q 'Syntax)))
     (set! c (+ 1 c))
     (set! q (item.relation.parent q 'Syntax)))
  c)

(define (scfg_start_brackets x)
  (set! c 0)
  (set! q x)
  (while (and q (null (item.relation.prev q 'Syntax)))
     (set! c (+ 1 c))
     (set! q (item.relation.parent q 'Syntax)))
  c)

(define (scfg_delta_brackets x)
  (- (scfg_end_brackets x)
     (item.feat x "n.lisp_scfg_start_brackets")))

(define (scfg_abs_delta_brackets x)
  (set! q (- (scfg_end_brackets x)
             (item.feat x "n.lisp_scfg_start_brackets")))
  (if (> q 0)
      q
      (* -1 q)))

(define (token_in_quote token)
  "(token_no_starting_quote TOKEN)
Check to see if a single quote (or backquote) appears as prepunctuation
in this token or any previous one in this utterance.  This is used to
disambiguate ending single quote as possessive or end quote."
  (cond
   ((null token)
    "0")
   ((string-matches (item.feat token "prepunctuation") ".*\".*")
    "1")
   ((string-matches (item.feat token "p.punc") ".*\".*")
    "0")  ;; there's a start quote
   (t
    (token_in_quote (item.relation.prev token "Token")))))

(define (has_punc w)
 (if (string-equal "punc" (item.feat w "R:Token.n.pos"))
     "1"
     "0"))

(define (ww_num_syls w)
 (length (car (cddr (lex.lookup (item.name w))))))

(define (lpunc w)
  (if (and
       (string-equal "punc" (item.feat w "pos"))
       (not (string-equal "punc" (item.feat w "n.pos"))))
      "1"
      "0"))
