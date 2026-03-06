;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                  and Alan W Black and Kevin Lenzo                   ;;;
;;;                      Copyright (c) 1998-2026                        ;;;
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
;;;
;;; Multi-pronunciation SphinxTrain setup
;;;
;;; Uses ALL pronunciations from the lexicon for each word, allowing
;;; the aligner to select the best variant acoustically. Stress can
;;; be recovered from the selected variant (e.g., RECORD(2)) by
;;; looking up the corresponding lexicon entry via lex.lookup_all.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (st_multipron_setup datafile vname)
  "Main entry point for multi-pronunciation SphinxTrain setup.
   Generates dictionary with all lexicon pronunciations for each word."
  (let ((words-seen nil)
        (phones-seen nil)
        (silence (car (cadr (car (PhoneSet.description '(silences))))))
        (pfd (fopen (format nil "st/etc/%s.phone" vname) "w"))
        (dfd (fopen (format nil "st/etc/%s.dic" vname) "w"))
        (ffd (fopen (format nil "st/etc/%s.filler" vname) "w"))
        (ifd (fopen (format nil "st/etc/%s.fileids" vname) "w"))
        (tfd (fopen (format nil "st/etc/%s.transcription" vname) "w")))

    ;; First pass: collect unique words and write transcripts/fileids
    (format t "Pass 1: Collecting words from corpus...\n")
    (mapcar
     (lambda (p)
       (let ((uttid (car p))
             (utt (utt.load nil (format nil "prompt-utt/%s.utt" (car p)))))
         (format t "%s\n" uttid)
         (format ifd "%s\n" uttid)
         (format tfd "<s>")
         (mapcar
          (lambda (w)
            (let ((word (item.name w))
                  (nicename (make_nicename (item.name w))))
              (format tfd " %s" nicename)
              (if (not (member_string word words-seen))
                  (set! words-seen (cons word words-seen)))))
          (utt.relation.items utt 'Word))
         (format tfd " </s> (%s)\n" uttid)))
     (load datafile t))
    (fclose tfd)
    (fclose ifd)

    ;; Second pass: get ALL lexicon pronunciations for each word
    (format t "\nPass 2: Getting all lexicon pronunciations for %d words...\n"
            (length words-seen))
    (mapcar
     (lambda (word)
       (let ((nicename (make_nicename word))
             (entries (lex.lookup_all word))
             (n 1))
         (mapcar
          (lambda (entry)
            (let ((pron (flatten_phones entry)))
              ;; Track phones for phone list
              (mapcar
               (lambda (ph)
                 (if (not (member_string ph phones-seen))
                     (set! phones-seen (cons ph phones-seen))))
               (string-split pron " "))
              ;; Write dictionary entry: WORD or WORD(n)
              (if (eq? n 1)
                  (format dfd "%s  %s\n" nicename pron)
                  (format dfd "%s(%d)  %s\n" nicename n pron))
              (set! n (+ n 1))))
          entries)))
     words-seen)
    (fclose dfd)

    ;; Write phone list
    (mapcar
     (lambda (ph)
       (if (not (string-equal silence ph))
           (format pfd "%s\n" ph)))
     phones-seen)
    (format pfd "SIL\n")
    (fclose pfd)

    ;; Write filler dictionary
    (format ffd "<s> SIL\n")
    (format ffd "</s> SIL\n")
    (format ffd "<sil> SIL\n")
    (fclose ffd)

    ;; Write silence phone name
    (let ((sfd (fopen "etc/mysilence" "w")))
      (format sfd "%s\n" silence)
      (fclose sfd))

    (format t "Done. Dictionary contains all lexicon pronunciations.\n")))

(define (make_nicename word)
  "Convert word to SphinxTrain-friendly uppercase name"
  (let ((ws (format nil "%s" word))
        (lets nil)
        (p 0))
    (while (< p (length ws))
      (let ((c (substring ws p 1)))
        (if (string-matches c "[A-Za-z0-9']")
            (set! lets (cons (upcase c) lets))
            (set! lets (cons "Q" lets))))
      (set! p (+ 1 p)))
    (if (> (length lets) 20)
        (set! lets (list "X" (length lets) "Q")))
    (apply string-append (reverse lets))))

(define (flatten_phones entry)
  "Flatten Festival lexicon entry to phone string.
   Entry format: (word pos (((ph1 ph2) stress) ((ph3) stress) ...))
   Returns space-separated phone string."
  (let ((phones nil))
    (mapcar
     (lambda (syl)
       (mapcar
        (lambda (ph)
          (set! phones (cons (format nil "%s" ph) phones)))
        (car syl)))
     (caddr entry))
    (apply string-append
           (cons (car (reverse phones))
                 (mapcar (lambda (p) (string-append " " p))
                         (cdr (reverse phones)))))))

(define (string-split str delim)
  "Split string by delimiter into list of strings"
  (let ((result nil)
        (current "")
        (i 0)
        (len (length str)))
    (while (< i len)
      (let ((c (substring str i 1)))
        (if (string-equal c delim)
            (begin
              (if (> (length current) 0)
                  (set! result (cons current result)))
              (set! current ""))
            (set! current (string-append current c))))
      (set! i (+ i 1)))
    (if (> (length current) 0)
        (set! result (cons current result)))
    (reverse result)))

(provide 'build_st_multipron)
