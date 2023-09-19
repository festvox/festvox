;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
;;;
;;; Lexicon, LTS and Postlexical rules for lvl_is
;;;

;;; Load any necessary files here

(define (lvl_is_addenda)
  "(lvl_is_addenda)
Basic lexicon should (must ?) have basic letters, symbols and punctuation."

;;; Pronunciation of letters in the alphabet
;(lex.add.entry '("a" nn (((a) 0))))
;(lex.add.entry '("b" nn (((b e) 0))))
;(lex.add.entry '("c" nn (((th e) 0))))
;(lex.add.entry '("d" nn (((d e) 0))))
;(lex.add.entry '("e" nn (((e) 0))))
; ...

;;; Symbols ...
;(lex.add.entry 
; '("*" n (((a s) 0) ((t e) 0) ((r i1 s) 1)  ((k o) 0))))
;(lex.add.entry 
; '("%" n (((p o r) 0) ((th i e1 n) 1) ((t o) 0))))

;; Basic punctuation must be in with nil pronunciation
(lex.add.entry '("." punc (((pau) 0)) ))
;(lex.add.entry '("." nn (((p u1 n) 1) ((t o) 0))))
(lex.add.entry '("'" punc nil))
(lex.add.entry '(":" punc (((pau) 0)) ))
(lex.add.entry '(";" punc (((pau) 0)) ))
(lex.add.entry '("," punc (((pau) 0)) ))
;(lex.add.entry '("," nn (((k o1) 1) ((m a) 0))))
(lex.add.entry '("-" punc (((pau) 0)) ))
(lex.add.entry '("\"" punc nil))
(lex.add.entry '("`" punc nil))
(lex.add.entry '("?" punc (((pau) 0)) ))
(lex.add.entry '("!" punc (((pau) 0)) ))
(lex.add.entry '("/" punc (((pau) 0)) ))
)

(require 'lts)

;;;  Function called when word not found in lexicon
;;;  and you've trained letter to sound rules
(define (lvl_is_lts_function word features)
  "(lvl_is_lts_function WORD FEATURES)
Return pronunciation of word not in lexicon."

  ;; If you have nothing ...
  (format t "Unknown word %s\n" word)
  (list word features nil)

  ;; If you have lts rules (trained or otherwise)
;  (if (not boundp 'lvl_is_lts_rules)
;      (require 'lvl_is_lts_rules))
;  (let ((dword (downcase word)) (phones) (syls))
;    (set! phones (lts_predict dword lvl_is_lts_rules))
;    (set! syls (lvl_is_lex_syllabify_phstress phones))
;    (list word features syls))
  )

(define (lvl_is_map_modify ps)
  (cond
   ((null ps) nil)
   ((null (cdr ps)) ps)
   ((assoc_string (string-append (car ps) (cadr ps))
                   INST_is_VOX_char_phone_map)
    (cons
     (string-append (car ps) (cadr ps))
     (lvl_is_map_modify (cddr ps))))
   (t
    (cons
     (car ps)
     (lvl_is_map_modify (cdr ps))))))

(define (lvl_is_map_phones p)
  (cond
   ((null p) nil)
   (t
    (let ((a (assoc_string (car p) INST_is_VOX_char_phone_map)))
      (cond
       (a (cons (cadr a) (lvl_is_map_phones (cdr p))))
       (t (lvl_is_map_phones (cdr p))))))))

(define (lvl_is_is_vowel x)
  (string-equal "+" (phone_feature x "vc")))

(define (lvl_is_contains_vowel l)
  (member_string
   t
   (mapcar (lambda (x) (lvl_is_is_vowel x)) l)))

(define (lvl_is_lex_sylbreak currentsyl remainder)
  "(lvl_is_lex_sylbreak currentsyl remainder)
t if this is a syl break, nil otherwise."
  (cond
   ((not (lvl_is_contains_vowel remainder))
    nil)
   ((not (lvl_is_contains_vowel currentsyl))
    nil)
   (t
    ;; overly naive, I mean wrong
    t))
)

(define (lvl_is_lex_syllabify_phstress phones)
 (let ((syl nil) (syls nil) (p phones) (stress 0))
    (while p
     (set! syl nil)
     (set! stress 0)
     (while (and p (not (lvl_is_lex_sylbreak syl p)))
       (if (string-matches (car p) "xxxx")
           (begin
             ;; whatever you do to identify stress
             (set! stress 1)
             (set syl (cons (car p-stress) syl)))
           (set! syl (cons (car p) syl)))
       (set! p (cdr p)))
     (set! syls (cons (list (reverse syl) stress) syls)))
    (reverse syls)))

(if (probe_file (path-append INST_is_VOX::dir "festvox/lex_lts_rules.scm"))
    (begin
      (load (path-append INST_is_VOX::dir "festvox/lex_lts_rules.scm"))
      (set! lvl_is_lts_rules lex_lts_rules)))

    ;; utf8-sampa map based on unitran 
(if (probe_file (path-append INST_is_VOX::dir "festvox/INST_is_VOX_char_phone_map.scm"))
    (begin
      (set! INST_is_VOX_char_phone_map
            (load (path-append INST_is_VOX::dir 
                               "festvox/INST_is_VOX_char_phone_map.scm") t))
	(load (path-append INST_is_VOX::dir 
                           "festvox/unicode_sampa_mapping.scm"))

    ;; utf8-indic-sampa letter based one
    (define (lvl_is_lts_function word features)
      "(lvl_is_lts_function WORD FEATURES)
Return pronunciation of word not in lexicon."
      (let ((dword word) (phones) (syls) (aphones))
        (if (boundp 'lvl_is_lts_rules)
            (set! phones (lts_predict (utf8explode dword) lvl_is_lts_rules))
            (begin
              (set! aphones (lvl_is_map_modify (utf8explode dword)))
              (set! phones (lvl_is_map_phones aphones))
              (set! phones (sampa_lookup phones))))
;        (set! phones (indic_unicode_lts sphones))
        (set! syls (lvl_is_lex_syllabify_phstress phones))
        (list word features syls)))
    ))

(define (sampa_lookup gphones)
  (let ((phlist nil) (sp nil))
    (mapcar 
     (lambda (gg)
       (set! sp (assoc_string gg unicode_sampa_mapping))
       (if sp
           (set! phlist (append (car (cadr sp)) phlist))
           (set! phlist (cons gg phlist))))
     gphones)
    (reverse phlist)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Postlexical Rules 
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;(define (lvl_is::postlex_rule1 utt)
;  "(lvl_is::postlex_rule1 utt)
;A postlexical rule form correcting phenomena over word boundaries."
;  (mapcar
;   (lambda (s)
;     ;; do something
;     )
;   (utt.relation.items utt 'Segment))
;   utt)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Lexicon definition
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(lex.create "lvl_is")
(lex.set.phoneset "lvl_is")
(lex.set.lts.method 'lvl_is_lts_function)
(lex.set.compile.file "festvox/lexicon.scm")
(lvl_is_addenda)
(if (probe_file (path-append INST_is_VOX::dir "festvox/lvl_is_addenda.scm"))
    (load (path-append INST_is_VOX::dir "festvox/lvl_is_addenda.scm")))


; (set! str_phone_map
;   (load "festvox/lvl_is_v0_sampa.scm" t)
; )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Lexicon setup
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (INST_is_VOX::select_lexicon)
  "(INST_is_VOX::select_lexicon)
Set up the lexicon for lvl_is."
  (lex.select "lvl_is")

  ;; Post lexical rules
  ;(set! postlex_rules_hooks (list lvl_is::postlex_rule1))
)

(define (INST_is_VOX::reset_lexicon)
  "(INST_is_VOX::reset_lexicon)
Reset lexicon information."
  t
)

(provide 'INST_is_VOX_lexicon)
