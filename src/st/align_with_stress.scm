;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2026                          ;;;
;;;                        All Rights Reserved.                         ;;;
;;;                                                                     ;;;
;;; Align utterances with SphinxTrain labels and correct stress         ;;;
;;; based on acoustically-selected pronunciation variants               ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Global stress map: maps WORD and WORD(n) to stress patterns
(defvar *stress_map* nil)

;; Load stress map from file
;; Format: WORD (stress1 stress2 ...)
(define (load_stress_map filename)
  (set! *stress_map* nil)
  (let ((fd (fopen filename "r"))
        line word stress)
    (while (set! line (readfp fd))
      ;; Parse: WORD followed by stress list
      (if (and (consp line) (> (length line) 1))
          (let ((word (car line))
                (stress (cadr line)))
            (set! *stress_map* (cons (list word stress) *stress_map*)))))
    (fclose fd))
  (format t "Loaded %d stress mappings\n" (length *stress_map*)))

;; Look up stress pattern for a word variant
(define (get_stress word_variant)
  (let ((entry (assoc_string word_variant *stress_map*)))
    (if entry
        (cadr entry)
        nil)))

;; Parse SphinxTrain alignoutput line to get word->variant mapping
;; Format: <s> <s> WORD1 WORD2(2) WORD3 <sil> WORD4 </s> </s> (uttid)
(define (parse_alignoutput_line line)
  (let ((result nil)
        (words (string-split line " ")))
    ;; Skip <s> markers and extract words
    (mapcar
     (lambda (w)
       (if (and (not (string-matches w "<.*>"))
                (not (string-matches w "\\(.*\\)")))
           (set! result (cons w result))))
     words)
    (reverse result)))

;; Get the selected variant for a word from alignoutput
;; Returns the variant string (e.g., "READ" or "READ(2)")
(define (get_selected_variant word alignoutput_words)
  (let ((found nil))
    (mapcar
     (lambda (w)
       ;; Match base word (with or without (n) suffix)
       (if (or (string-equal (upcase word) w)
               (string-matches w (string-append (upcase word) "\\([0-9]+\\)")))
           (set! found w)))
     alignoutput_words)
    found))

;; Update syllable stress for a word based on selected variant
(define (update_word_stress word_item variant)
  (let ((stress_pattern (get_stress variant)))
    (if stress_pattern
        (let ((syls (item.daughters
                     (item.relation.parent word_item 'SylStructure)))
              (i 0))
          (mapcar
           (lambda (syl)
             (if (< i (length stress_pattern))
                 (begin
                   (item.set_feat syl "stress"
                                  (nth i stress_pattern))
                   (set! i (+ i 1)))))
           syls)
          (if (not (eq? i (length stress_pattern)))
              (format t "Warning: stress mismatch for %s (%d syls, %d stress)\n"
                      variant i (length stress_pattern)))))))

;; Main alignment function with stress correction
(define (align_utt_with_stress name lab_file alignoutput_line)
  (let ((utt (utt.load nil (format nil "festival/utts/%s.utt" name)))
        (silence (car (cadr (car (PhoneSet.description '(silences))))))
        (alignoutput_words (parse_alignoutput_line alignoutput_line))
        segments actual-segments)

    ;; Load lab file
    (utt.relation.load utt 'actual-segment lab_file)
    (set! segments (utt.relation.items utt 'Segment))
    (set! actual-segments (utt.relation.items utt 'actual-segment))

    ;; Merge phone timing (standard alignment)
    (while (and segments actual-segments)
      (cond
       ;; Lab has extra silence - insert
       ((and (not (string-equal (item.name (car segments))
                                (item.name (car actual-segments))))
             (or (string-equal (item.name (car actual-segments)) silence)
                 (string-equal (item.name (car actual-segments)) "ssil")))
        (item.insert
         (car segments)
         (list silence (list (list "end" (item.feat
                                          (car actual-segments) "end"))))
         'before)
        (set! actual-segments (cdr actual-segments)))
       ;; UTT has extra silence - delete
       ((and (not (string-equal (item.name (car segments))
                                (item.name (car actual-segments))))
             (string-equal (item.name (car segments)) silence))
        (item.delete (car segments))
        (set! segments (cdr segments)))
       ;; Match - copy timing
       ((string-equal (item.name (car segments))
                      (item.name (car actual-segments)))
        (item.set_feat (car segments) "end"
                       (item.feat (car actual-segments) "end"))
        (set! segments (cdr segments))
        (set! actual-segments (cdr actual-segments)))
       (t
        (format t "Align mismatch: %s vs %s\n"
                (item.name (car segments))
                (item.name (car actual-segments)))
        (set! segments nil))))

    ;; Update stress based on selected variants
    (mapcar
     (lambda (word_item)
       (let ((word (item.name word_item))
             (variant (get_selected_variant word alignoutput_words)))
         (if (and variant (string-matches variant ".*\\([0-9]+\\)"))
             (begin
               (format t "Updating stress for %s -> %s\n" word variant)
               (update_word_stress word_item variant)))))
     (utt.relation.items utt 'Word))

    utt))

;; Process all utterances
(define (align_all_with_stress datafile lab_dir alignoutput_file stressmap_file)
  (load_stress_map stressmap_file)

  ;; Load alignoutput into hash
  (let ((alignoutput_hash (make-hash-table))
        (fd (fopen alignoutput_file "r"))
        line)
    (while (set! line (fgets fd))
      (if (string-matches line ".*\\(arctic_.*\\)")
          (let ((uttid (string-after (string-before line ")") "(")))
            (hash-set! alignoutput_hash uttid line))))
    (fclose fd)

    ;; Process each utterance
    (mapcar
     (lambda (p)
       (let ((uttid (car p))
             (alignline (hash-ref alignoutput_hash (car p))))
         (format t "Processing %s\n" uttid)
         (if alignline
             (align_utt_with_stress uttid
                                    (format nil "%s/%s.lab" lab_dir uttid)
                                    alignline)
             (format t "Warning: no alignoutput for %s\n" uttid))))
     (load datafile t))))

(provide 'align_with_stress)
