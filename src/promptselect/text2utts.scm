;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;-*-mode:scheme-*-
;;;                                                                     ;;;
;;;                  Language Technologies Institute                    ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2002                          ;;;
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
;;;             Author: Alan W Black (awb@cs.cmu.edu)                   ;;;
;;;               Date: December 2002                                   ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;; Take lots of text and generated .data files one utterance per line  ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;; Because this is a --script type file it has to explicitly
;;; load the initfiles: init.scm and user's .festivalrc
(load (path-append libdir "init.scm"))

;(set_backtrace t)

;;; Process command line arguments
(define (text2utts_help)
  (format t "%s\n"
  "text2utts [options] textfile(s)
  Convert textfile(s) to .data files
  Options
  -onlynice        Output only \"nice\" utterances
  -onlynicelatin   Output only \"nice\" utterances (assume latin charset)
  -all             Output all (not just nice) utterances
  -o ofile         File to save output to (default is stdout).
  -odir <string>   Output directory
  -dbname <string> Set dbname prefix (default is txt_)
  -num <int>       Set number to start from (default is 1)
  -level <string>  Which level to print out, Token (default), Segment
  -otype <string>  data (default) or raw, utts, wavs.
  -itype <string>  raw (default) or data, utts.
  -nopunc          don't output punctuation for Words (default nil).
  -parainfo        output paragraph position features (default nil).
  -upto <int>      Stop after finding <int> sentences
  -eval <string>   File or lisp s-expression to be evaluated before
                   processing.
")
  (quit))

;;; No gc messages
(gc-status nil)

;;; Default argument values
(defvar outfile "-")
(defvar odir ".")
(defvar text_files '("-"))
(defvar nice_utts_only nil)
(defvar nice_utts_only_latin nil)
(defvar nopunc nil)
(defvar dbname "txt")
(defvar txtnum 1)
(defvar level "Token")
(defvar otype "data")
(defvar raw nil)
(defvar parainfo nil)
(defvar itype "raw")
(defvar word_output_hook nil)
(defvar upto 0)

;; For keeping track of paragraph boundaries
(defvar utt_number 0)
(defvar para_number 0)
(defvar para_sentence_number 0)

;;; Get options
(define (get_options)
  (let ((files nil)
	(o argv))
    (if (or (member_string "-h" argv)
	    (member_string "-help" argv)
	    (member_string "--help" argv)
	    (member_string "-?" argv))
	(text2utts_help))
    (while o
      (begin
	(cond
	 ((string-equal "-o" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no output file specified"))
	  (set! outfile (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-onlynice" (car o))
          (set! nice_utts_only t))
	 ((string-equal "-onlynicelatin" (car o))
          (set! nice_utts_only t)
          (set! nice_utts_only_latin t))
	 ((string-equal "-all" (car o))
          (set! nice_utts_only nil))
	 ((string-equal "-nopunc" (car o))
          (set! nopunc t))
	 ((string-equal "-parainfo" (car o))
          (set! parainfo t))
	 ((string-equal "-otype" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no otype specified"))
	  (set! otype (car (cdr o)))
	  (if (string-equal otype "raw")
	      (set! raw t))
	  (set! o (cdr o)))
	 ((string-equal "-dbname" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no dbname specified"))
	  (set! dbname (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-level" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no level specified"))
	  (set! level (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-itype" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no itype specified"))
	  (set! itype (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-odir" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no itype specified"))
	  (set! odir (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-num" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no num specified"))
	  (set! txtnum (car (cdr o)))
	  (set! o (cdr o)))
	 ((string-equal "-upto" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no upto specified"))
	  (set! upto (parse-number (car (cdr o))))
	  (set! o (cdr o)))
	 ((string-equal "-eval" (car o))
	  (if (not (cdr o))
	      (text2utts_error "no file specified to load"))
	  (if (string-matches (car (cdr o)) "^(.*")
              (begin
                (format t "%l\n" (car (cdr o)))
                (eval (read-from-string (car (cdr o)))))
	      (load (car (cdr o))))
	  (set! o (cdr o)))
	 (t
	  (set! files (cons (car o) files))))
	(set! o (cdr o))))
    (if files
	(set! text_files (reverse files)))))

(define (text2utts_error message)
  (format stderr "%s: %s\n" "text2utts" message)
  (text2utts_help))

(define (no_duplicates words)
  (let ((no_dups t))

    (while (and words (cdr words) no_dups)
       (if (string-equal (item.name (car words)) (item.name (cadr words)))
	   (set! no_dups nil))
       (set! words (cdr words)))
    no_dups))

(define (all_in_lex_plus words)
  (let ((all_ok t)
        (lname "")
        (this_name "")
        (upcase_name "")
        )
    (cond
     ((or (not words)  ;; start with capital letter
          (not (string-matches 
                (set! upcase_name (item.name (car words))) 
                "[A-Z].*")))
        (set! all_ok nil))
     (t
        (set! this_name (downcase upcase_name))
        (while (and words all_ok)
           (cond
            ((not (lex.lookup_all this_name))
             (set! all_ok nil))
            ((string-equal lname this_name)
             (set! all_ok nil))
            ((string-matches upcase_name ".*[A-Z].*[A-Z].*")
             (set! all_ok nil))     ;; no words with two capital letters
            ((or (not (string-matches (item.feat (car words) "punc") "[0,.]*"))
                 (not (string-equal (item.feat (car words) "prepunctuation") "")))
             (set! all_ok nil))     ;; no weird punctuation 
            ((and (not (cdr words)) ;; last word must have period
                  (or (string-matches upcase_name "[A-Z].?")
                      (not (string-matches ;; Last word contains a vowel
                            this_name
                            ".*[aeiouàáâãäåæèéêëìíîïòóôõöøùúûüý].*"))
                      (not (string-equal (item.feat (car words) "punc") "."))))
             (set! all_ok nil))
            (t
             (set! lname this_name)
             (set! words (cdr words))
             (if words
                 (begin
                   (set! upcase_name (item.name (car words)))
                   (set! this_name (downcase upcase_name)))))))))
    all_ok))

(define (all_in_lex words)
  (let ((all_ok t)
        (lname "")
        (this_name "")
        )
    (while (and words all_ok)
       (set! this_name (item.name (car words)))
       (cond
        ((not (lex.lookup_all this_name))
;         (format t "word not in lex %s\n" this_name)
         (set! all_ok nil))
        ((string-equal lname this_name)
;         (format t "duplicate word %s\n" this_name)
         (set! all_ok nil))
        (t
         (set! lname this_name)
         (set! words (cdr words)))))
    all_ok))

(define (nice_utt utt)
  ;; We're lying, we're checking Tokens, not words
  (set! words (utt.relation.items utt 'Token))
  (set! num_words (length words))
;  (format t "num_words in sentence %d\n" num_words)
  (if (and (> num_words 4)
           (< num_words 20)
           (if nice_utts_only_latin
               (all_in_lex_plus words)
               (all_in_lex words))
           )
      t
      nil))

(define (escape_characters s)
  (let ((nn ""))
    (while (string-matches s ".*\\\\.*")
       (set! nn (string-append nn (string-before s "\\") "\\\\"))
       (set! s (string-after s "\\")))
    (set! s (string-append nn s))
    (set! nn "")
    (while (string-matches s ".*\".*")
       (set! nn (string-append nn (string-before s "\"") "\\\""))
       (set! s (string-after s "\"")))
    (set! s (string-append nn s))
    s))

(define (utt_set_uttname utt)
  (cond
   ((or (string-equal itype "utts")
	(string-equal itype "data"))
    (set! uttname (utt.feat utt "fileid")))
   (t
    (set! uttname (format nil "%s_%05d" dbname txtnum))
    (set! txtnum (+ 1 txtnum))
    (if (and (> upto 0)
             (> txtnum upto))
        ;; need to signal end
        (error (format t "stopping at %d utts\n" upto))
        )
    (utt.set_feat utt "fileid" uttname)))
  utt)

(define (utt_output_token utt)
  (let ((token (utt.relation.first utt 'Token)))
    (if (not raw)
	(format ofd "( %s \"" uttname))
    (format t "utt %l\n" uttname)
    (set! whitespace "")
    (while token
	 (format t ">%s<\n" (item.name token))
       (let ((punc (item.feat token "punc"))
	     (prepunctuation (item.feat token "prepunctuation")))
	 (set! name (item.name token))
         (set! actual_whitespace (item.feat token "whitespace"))
         (if (string-matches actual_whitespace ".*\n.*\n.*")
             ;; or newline and significant indentation ?
             (begin ;; a new paragraph
               (set! para_sentence_number 0)
               (set! para_number (+ 1 para_number))
               ))
	 (if (string-equal "0" punc) (set! punc ""))
	 (if (string-equal "0" prepunctuation) (set! prepunctuation ""))
	 (if (not raw)
	     (begin
	       (set! prepunctuation (escape_characters prepunctuation))
	       (set! punc (escape_characters punc))
	       (set! name (escape_characters name))))
	 (format ofd "%s%s%s%s"
		 whitespace
		 prepunctuation
		 name
		 punc)
	 (set! whitespace " ")
	 )
       (set! token (item.next token)))

    (if (not raw)
        (begin 
          (format ofd "\"")
          (if parainfo
              (format ofd " (utt_number %d) (para_number %d) (para_sentence_number %d)"
                      utt_number para_number para_sentence_number))
          (format ofd " )")))
    (format ofd "\n")
    (set! utt_number (+ 1 utt_number))
    (set! para_sentence_number (+ 1 para_sentence_number))
    ))

(define (utt_output_word utt)
  (let ((word (utt.relation.first utt 'Word)))
    (if (not raw)
	(format ofd "( %s \"" uttname))
    (set! whitespace "")
    (while word
       (let (punc prepunctuation)
	 (set! name (item.name word))
	 (if nopunc
	     (begin
	       (set! prepunctuation "")
	       (set! punc ""))
	     (begin
	       (set! punc 
		     (if (string-equal

                          (item.feat word "R:Token.parent.id")
                          (item.feat word "R:Word.n.R:Token.parent.id"))
			 "0"
			 (item.feat word "R:Token.parent.punc")))
	       (set! prepunctuation 
		     (if (string-equal
                          (item.feat word "R:Token.parent.id")
                          (item.feat word "R:Word.p.R:Token.parent.id"))
			 "0"
			 (item.feat word "R:Token.parent.prepunctuation")))
               (set! actual_whitespace
		     (if (string-equal
                          (item.feat word "R:Token.parent.id")
                          (item.feat word "R:Word.p.R:Token.parent.id"))
			 ""
			 (item.feat word "R:Token.parent.whitespace")))
               (if (string-matches actual_whitespace ".*\n.*\n.*")
                   ;; or newline and significant indentation ?
                   (begin ;; a new paragraph
                     (set! para_sentence_number 0)
                     (set! para_number (+ 1 para_number))
                     ))
	       (if (string-equal "0" punc) (set! punc ""))
	       (if (string-equal "0" prepunctuation) (set! prepunctuation ""))
	       (if (not raw)
		   (begin
		     (set! prepunctuation (escape_characters prepunctuation))
		     (set! punc (escape_characters punc))
		     (set! name (escape_characters name))))))
         (if (or (string-matches name "[^a-zA-Z0-9']")
                 (string-equal name "'s"))
             (set! whitespace ""))
	 (format ofd "%s%s%s%s"
		 whitespace
		 prepunctuation
		 name
		 punc)
	 (set! whitespace " ")
	 )
       (set! word (item.next word)))
    (if (not raw)
	(format ofd "\" "))
    (if word_output_hook
        (apply_hooks word_output_hook utt)
        )
    (if parainfo
        (format ofd " (utt_number %d) (para_number %d) (para_sentence_number %d)"
                utt_number para_number para_sentence_number))
    (if (not raw)
	(format ofd " )"))
    (format ofd "\n")
    (set! utt_number (+ 1 utt_number))
    (set! para_sentence_number (+ 1 para_sentence_number))
    ))

(define (utt_output_segment utt)
  (let ((segment (utt.relation.first utt 'Segment)))
    (if (not raw)
	(format ofd "( %s \"" uttname))
    (set! whitespace "")
    (while segment
       (set! name (item.name segment))
       (cond
	((string-equal name "pau")
	 (format ofd "%s%s" whitespace name))
	((string-matches name "[aeiou].*")
	 (format ofd "%s%s%s" whitespace name
		 (item.feat segment "R:SylStructure.parent.stress")))
	(t
;	 (format ofd "%s%s%s" whitespace name
;		 (item.feat segment "seg_onsetcoda"))
	 (format ofd "%s%s" whitespace name)
         ))
       (set! whitespace " ")
       (set! segment (item.next segment)))
    (if (not raw)
	(format ofd "\" )"))
    (format ofd "\n")))

(define (check_utt_output utt)
  (cond
   ((or (not nice_utts_only)
	(nice_utt utt))
    (utt_set_uttname utt)  ;; label the utt (its interesting)
    (cond
     ((string-equal otype "utts")
      (utt.save utt
		(format nil "%s/%s.utt" odir uttname)))
     ((string-equal otype "wavs")
      (utt.save.wave utt
		     (format nil "%s/%s.wav" odir uttname)))
     ((string-equal level "Token")
      (utt_output_token utt))
     ((string-equal level "Word")
      (utt_output_word utt))
     ((string-equal level "Segment")
      (utt_output_segment utt))
     (t
      (format stderr "Unknown level %s\n" level)
      (error))))
   (t  ;; not an interesting utt 
    t))
)

;;;
;;; Redefine what happens to utterances during text to speech 
;;;
(set! tts_hooks 
      (list 
       check_utt_output))

(define (utt.synth_toToken utt)
  utt)

(define (utt.synth_texttotoken utt)
  (Initialize utt)
  (Text utt))

(define (utt.synth_toWord utt)
  (Token_POS utt) 
  (Token utt)
  (POS utt)
  (Phrasify utt)
  (Word utt)
  (Pauses utt)
)

(define (utt.synth_toSegment utt)
  (Token_POS utt)    ;; when utt.synth is called
  (Token utt)        
  (POS utt)
  (Phrasify utt)
  (Word utt)
  (Pauses utt)
  (Intonation utt)
  (PostLex utt))

(define (main)
  (get_options)

  (cond 
   ((string-equal itype "utts")
    t)
   ((string-equal otype "wavs")
    (set! tts_hooks (cons utt.synth tts_hooks)))
   ((string-equal level "Token")
    (set! tts_hooks (cons utt.synth_toToken tts_hooks)))
   ((string-equal level "Word")
    (set! tts_hooks (cons utt.synth_toWord tts_hooks)))
   ((string-equal level "Segment")
    (set! tts_hooks (cons utt.synth_toSegment tts_hooks)))
   (t
    (set! tts_hooks (cons utt.synth tts_hooks))))

  (if (string-equal outfile "-")
    (set! ofd t)
    (set! ofd (fopen (path-append odir outfile) "w")))

  ;; do the synthesis
  (cond
   ((string-equal itype "raw")
    (unwind-protect
     (mapcar
      (lambda (f) 
        (tts_file f (tts_find_text_mode f auto-text-mode-alist)))
      text_files)
     t))
   ((string-equal itype "utts")
    (mapcar
     (lambda (f) 
       (let ((utt (utt.load nil f)))
	 (apply_hooks tts_hooks utt))
       text_files)))
   ((string-equal itype "data")
    (mapcar
     (lambda (f) 
       (mapcar
	(lambda (s)
	  (let ((utt (eval (list 'Utterance 'Text (cadr s)))))
	    (utt.set_feat utt "fileid" (car s))
	    (utt.synth_texttotoken utt)
	    (apply_hooks tts_hooks utt)))
	(load f t)))
     text_files))
   (t
    (format stderr "Unknown itype %s\n" itype)))

  (if (not (string-equal "-" outfile))
      (fclose ofd))

)

;;;  Do the work
(main)
