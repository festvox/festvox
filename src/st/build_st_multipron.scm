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
;;; - Dictionary has WORD and WORD(2) format for alternatives
;;; - Transcript has base word forms only
;;; - Aligner chooses best pronunciation
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (st_multipron_setup datafile vname)
  "Main entry point for multi-pronunciation SphinxTrain setup"
  (let ((words-seen nil)
        (phones-seen nil)
        (silence (car (cadr (car (PhoneSet.description '(silences))))))
        (pfd (fopen (format nil "st/etc/%s.phone" vname) "w"))
        (dfd (fopen (format nil "st/etc/%s.dic" vname) "w"))
        (ffd (fopen (format nil "st/etc/%s.filler" vname) "w"))
        (ifd (fopen (format nil "st/etc/%s.fileids" vname) "w"))
        (tfd (fopen (format nil "st/etc/%s.transcription" vname) "w")))

    ;; Process each utterance
    (mapcar
     (lambda (p)
       (let ((uttid (car p))
             (utt (utt.load nil (format nil "prompt-utt/%s.utt" (car p)))))
         (format t "%s\n" uttid)

         ;; Write file ID
         (format ifd "%s\n" uttid)

         ;; Write transcript with base word forms only
         (format tfd "<s>")
         (mapcar
          (lambda (w)
            (let ((word (item.name w))
                  (nicename (make_nicename (item.name w))))
              ;; Add word to transcript (base form, uppercase)
              (format tfd " %s" nicename)
              ;; Collect word and its pronunciation for dictionary
              (set! words-seen (add_word_pron words-seen nicename
                                              (get_word_phones w silence)
                                              phones-seen))
              (set! phones-seen (cadr words-seen))
              (set! words-seen (car words-seen))))
          (utt.relation.items utt 'Word))
         (format tfd " </s> (%s)\n" uttid)))
     (load datafile t))

    (fclose tfd)
    (fclose ifd)

    ;; Write dictionary with multiple pronunciations
    (write_multipron_dict dfd words-seen)
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
    (format ffd "<sil> SIL\n")  ;; For optional silence insertion
    (fclose ffd)

    ;; Write silence phone name
    (let ((sfd (fopen "etc/mysilence" "w")))
      (format sfd "%s\n" silence)
      (fclose sfd))))

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

(define (get_word_phones word silence)
  "Extract phone sequence for a word from its segments"
  (let ((phones nil))
    (mapcar
     (lambda (seg)
       (let ((ph (item.name seg)))
         (if (not (string-equal ph silence))
             (set! phones (cons (make_nicephone seg) phones)))))
     (item.daughters (item.relation.daughter1 word 'SylStructure)))
    (reverse phones)))

(define (make_nicephone seg)
  "Convert phone name for SphinxTrain (handle case issues)"
  (let ((ph (item.name seg)))
    (if (string-matches ph ".*[A-Z].*")
        (string-append "CAP" ph)
        ph)))

(define (add_word_pron words-seen nicename phones phones-seen)
  "Add word and pronunciation to collection, tracking alternatives"
  ;; Update phones-seen with any new phones
  (mapcar
   (lambda (ph)
     (if (not (member_string ph phones-seen))
         (set! phones-seen (cons ph phones-seen))))
   phones)

  (let ((pron-str (apply string-append
                         (cons (car phones)
                               (mapcar (lambda (p) (string-append " " p))
                                       (cdr phones)))))
        (entry (assoc_string nicename words-seen)))
    (if entry
        ;; Word exists - check if this is a new pronunciation
        (if (not (member_string pron-str (cdr entry)))
            (set-cdr! entry (cons pron-str (cdr entry))))
        ;; New word
        (set! words-seen (cons (list nicename pron-str) words-seen))))
  (list words-seen phones-seen))

(define (write_multipron_dict fd words)
  "Write dictionary with WORD(n) format for multiple pronunciations"
  (mapcar
   (lambda (entry)
     (let ((word (car entry))
           (prons (cdr entry))
           (n 1))
       (mapcar
        (lambda (pron)
          (if (eq? n 1)
              (format fd "%s  %s\n" word pron)
              (format fd "%s(%d)  %s\n" word n pron))
          (set! n (+ n 1)))
        (reverse prons))))  ;; Reverse to keep first-seen as base
   words))

;; Also dump ALL pronunciations from the lexicon for words in corpus
(define (st_dump_lexicon_prons datafile vname)
  "Dump all lexicon pronunciations for words in corpus"
  (let ((words-seen nil)
        (dfd (fopen (format nil "st/etc/%s.fulldict" vname) "w")))

    ;; Collect all words from utterances
    (mapcar
     (lambda (p)
       (let ((utt (utt.load nil (format nil "prompt-utt/%s.utt" (car p)))))
         (mapcar
          (lambda (w)
            (let ((word (item.name w)))
              (if (not (member_string word words-seen))
                  (set! words-seen (cons word words-seen)))))
          (utt.relation.items utt 'Word))))
     (load datafile t))

    ;; For each word, get ALL pronunciations from lexicon
    (mapcar
     (lambda (word)
       (let ((nicename (make_nicename word))
             (entries (lex.lookup_all word))
             (n 1))
         (mapcar
          (lambda (entry)
            (let ((phones (flatten_phones entry)))
              (if (eq? n 1)
                  (format dfd "%s  %s\n" nicename phones)
                  (format dfd "%s(%d)  %s\n" nicename n phones))
              (set! n (+ n 1))))
          entries)))
     words-seen)

    (fclose dfd)))

(define (flatten_phones entry)
  "Flatten Festival lexicon entry to phone string"
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

(provide 'build_st_multipron)
