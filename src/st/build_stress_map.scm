;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2026                          ;;;
;;;                        All Rights Reserved.                         ;;;
;;;                                                                     ;;;
;;; Generate stress mapping for multi-pronunciation alignment           ;;;
;;; Maps WORD(n) variants to syllable stress patterns                   ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Extract stress pattern from Festival lexicon entry
;; Entry format: ("word" pos (((phones) stress) ((phones) stress) ...))
;; Returns list of stress values: (1 0) for REcord, (0 1) for reCORD
(define (get_stress_pattern entry)
  (mapcar (lambda (syl) (car (cdr syl))) (car (cdr (cdr entry)))))

;; Generate stress map for all words in corpus
;; Output format: WORD(n) stress1 stress2 ...
(define (build_stress_map datafile outfile)
  (let ((words_seen nil)
        (ofd (fopen outfile "w")))

    ;; Collect words from corpus
    (mapcar
     (lambda (p)
       (let ((utt (utt.load nil (format nil "prompt-utt/%s.utt" (car p)))))
         (mapcar
          (lambda (w)
            (let ((word (item.name w)))
              (if (not (member_string word words_seen))
                  (set! words_seen (cons word words_seen)))))
          (utt.relation.items utt 'Word))))
     (load datafile t))

    ;; For each word, get all pronunciations and their stress patterns
    (mapcar
     (lambda (word)
       (let ((entries (lex.lookup_all word))
             (nicename (make_nicename word))
             (n 1))
         (mapcar
          (lambda (entry)
            (let ((stress (get_stress_pattern entry)))
              ;; Write mapping: WORD(n) stress1 stress2 ...
              (if (eq? n 1)
                  (format ofd "%s %l\n" nicename stress)
                  (format ofd "%s(%d) %l\n" nicename n stress))
              (set! n (+ n 1))))
          entries)))
     words_seen)

    (fclose ofd)))

;; Helper to convert word to SphinxTrain format
(define (make_nicename word)
  (let ((ws (format nil "%s" word))
        (lets nil)
        (p 0))
    (while (< p (length ws))
      (let ((c (substring ws p 1)))
        (if (string-matches c "[A-Za-z0-9']")
            (set! lets (cons (upcase c) lets))
            (set! lets (cons "Q" lets))))
      (set! p (+ 1 p)))
    (apply string-append (reverse lets))))

(provide 'build_stress_map)
