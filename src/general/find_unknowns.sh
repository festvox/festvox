;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;-*-mode:scheme-*-
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
;;;  Analysis a number of text files to find words not in the lexicon   ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Because this is a --script type file it has to explicitly
;;; load the initfiles: init.scm and user's .festivalrc
(if (not (symbol-bound? 'caar))
    (begin
      (load (path-append libdir "init.scm"))
      (if (probe_file (format nil "%s/.festivalrc" (getenv "HOME")))
	  (load (format nil "%s/.festivalrc" (getenv "HOME"))))))


(define (find_unknowns_help)
  (format t "%s\n"
  "find_unknowns [options] textfile1 textfile2 ...
  Find words not in the lexicon in a set of text files.
  Options
  -eval <file>
             Load in scheme file with run specific code, if file name
             starts with a left parent the string itself is interpreted
             Often used to specify the desried voice/lexicon to cheack
             against
  -output <ofile>
             Output file to save unknown words in,  by default this is
             \"unknown.words\"")
  (quit))

(defvar files nil)
(defvar outputfile "unknown.words")

;;; Get options
(define (get_options)
  (let ((o argv))
    (if (or (member_string "-h" argv)
	    (member_string "-help" argv)
	    (member_string "--help" argv)
	    (member_string "-?" argv))
	(find_unknowns_help))
    (while o
      (begin
	(cond
	 ((string-equal "-output" (car o))
	  (if (not (cdr o))
	      (make_unknowns_error "no output file specified"))
	  (set! outputfile (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-eval" (car o))
	  (if (not (cdr o))
	      (make_unknowns_error "no file specified to load"))
	  (if (string-matches (car (cdr o)) "^(.*")
	      (eval (read-from-string (car (cdr o))))
	      (load (car (cdr o))))
	  (set! o (cdr o)))
	 (t
	  (set! files (cons (car o) files))))
	(set! o (cdr o))))))

(define (make_unknowns_error message)
  (format stderr "%s: %s\n" "make_unknowns" message)
  (make_unknowns_help))

(define (lex_user_unknown_word word feats)
"Dump this name to outfp"
  (if (string-matches word "[a-zA-Z]*")
      (begin
	(format outfp "%s\n" word)
	(set! num_unknown (+ 1 num_unknown))))
  '("unknown" n (((uh n) 1) ((n ou n) 1)) ((pos "K6%" "OA%"))))

(define (WordCount utt)
  (set! num_words (+ (length (utt.relation.items utt 'Word)) num_words))
  utt)

(define (find-words utt)
"Main function for processing TTS utterances.  Predicts POS and dows
lexical lookup only."
  (Token_POS utt)    ;; when utt.synth is called
  (Token utt)        
  (POS utt)
  (Word utt)
  (WordCount utt)
)

(set! num_unknown 0)
(set! num_words 0)

;;;
;;; Redefine what happens to utterances during text to speech 
;;;
(set! tts_hooks (list find-words))

(define (main)
  (get_options)
  ;; Override any the OOV function from any default voice specified
  (lex.set.lts.method 'lex_user_unknown_word)
  (setq outfp (fopen outputfile "w"))
  (if files
      (mapcar (lambda (f) (tts f nil)) files)
      (tts "-" nil))
  (format t  ";; Number of words %d\n" num_words)
  (format t  ";; Number of unknown %d\n" num_unknown)
  (fclose outfp))

(main)


