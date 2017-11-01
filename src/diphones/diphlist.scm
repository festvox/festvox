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
;;;                                                                       ;;
;;;  Simple example of code for generating nonsense words containing      ;;
;;;  all diphones in a particular phone set.  Inspired by Steve Isard's   ;;
;;;  diphone schema's from CSTR, University of Edinburgh                  ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; You want to also include a definition of diphone-gen-list
;;; for the phone set you are building for
;;; This should be in festvox/LANG_scheme.scm

(defvar diph_do_db_boundaries t)

(define (remove-list l1 l2)
  "(remove-list l1 l2)
Remove all members of l2 from l1."
  (if l2
      (remove-list (remove (car l2) l1) (cdr l2))
      l1))

(define (remove-all i l)
  (cond
   ((null l) l)
   ((string-equal i (car l))
    (remove-all i (cdr l)))
   (t
    (cons (car l) (remove-all i (cdr l))))))
   

(define (synth-nonsense-word phones)
  "(synth-nonsense-word phones)
Synthesize given phone list.  It is assumed that the current voice
is appropriate."
  (let ((utt (eval (list 'Utterance 'Phones (remove-all '- phones)))))
    (Initialize utt)
    (Fixed_Prosody utt)
    (if (boundp 'Diphone_Prompt_Word)
	(Diphone_Prompt_Word utt))
    (Wave_Synth utt)
    utt))

(define (synth-nonsense-word-1 phones)
  "(synth-nonsense-word-1 phones)
Synthesize given phone list.  It is assumed that the current voice
is appropriate. Synthesis up to phones and prosody"
  (let ((utt (eval (list 'Utterance 'Phones (remove-all '- phones)))))
    (Initialize utt)
    (Fixed_Prosody utt)
    utt))

(define (synth-nonsense-word-2 utt)
  "(synth-nonsense-word phones)
Carry out mapping and waveform synthesis.  This is split into two
stages as when doing cross-language prompting we need to dump the
unmapped Segment stream to the label file."
  (if (boundp 'Diphone_Prompt_Word)
      (Diphone_Prompt_Word utt))
  (Wave_Synth utt)
  utt)

(define (change_underbars words_as_symbol)
  (apply
   string-append
   (mapcar
    (lambda (l)
      (if (string-equal l "_") 
	  " "
	  l))
    (symbolexplode words_as_symbol))))

(define (synth-carrier-word words_as_symbol)
  "(synth-carrier-word words)
Synthesize given words list.  The words are underscore separated."
  (let ((words_as_string (change_underbars words_as_symbol))
	(utt))
    (set! utt (eval (list 'Utterance 'Text words_as_string)))
    (utt.synth utt)
    utt))

(define (save_segs_plus_db utt filename)
  "(save_segs_plus_db utt filename)
Saves the segment times, interspersed with diphone boundary placements
if available in utterance.  If the utterance was synthesized with
UniSyn, diphone boundary places will be available through the Unit
relation, these are saved to help find best diphone boundary
placements in the new nonsense words to be labelled. (Steve Isard 
suggested this)"
  (let ((fd (fopen filename "w"))
	source_end end l_end l_source_end)
    (format fd "#\n")
    (if (and 
	 diph_do_db_boundaries
	 (member_string "Unit" (utt.relationnames utt))
	 (equal? (length (utt.relation.items utt "Segment"))
		 (+ 1 (length (utt.relation.items utt "Unit")))))
	(mapcar 
	 (lambda (seg unit)
	   (if (and unit
		    (or (string-equal (item.name seg)
				  (string-after (item.name unit) "-"))
			(string-equal (item.name seg)
				  (string-after (item.name unit) "-_"))))
	       (begin
		 ;; need to calculate where the boundary from the
		 ;; source diphones will be in the target waveform
		 (set! end (item.feat seg "end"))
		 (set! source_end (item.feat seg "source_end"))
		 (set! source_d_end (item.feat unit "end"))
		 (set! l_end (item.feat seg "p.end"))
		 (set! l_source_end (item.feat seg "p.source_end"))
		 (set! d_end (+ l_end
				(* (/ (- source_d_end l_source_end)
				      (- source_end l_source_end))
				   (- end l_end))))
		 (format fd "%2.4f 100 DB\n" d_end)
		 ))
	   (format fd "%2.4f 100 %s\n"  
		   (item.feat seg "segment_end")
		   (item.name seg)))
	 (utt.relation.items utt "Segment")
	 (cons nil (utt.relation.items utt "Unit")))
	;; else
	(mapcar
	 (lambda (seg)
	   (format fd "%2.4f 100 %s\n"  
		   (item.feat seg "segment_end")
		   (item.name seg)))
	 (utt.relation.items utt "Segment")))
    (fclose fd)))

(define (diphone-gen-waves wavedir labdir dlistfile)
  "(diphone-gen-waves wavedir labdir dlistfile)
Synthesis each nonsense word with fixed duration and fixed pitch
saving the waveforms for playing as prompts (as autolabelling)."
  (if (boundp 'Diphone_Prompt_Setup)
      (Diphone_Prompt_Setup))
  (if (consp dlistfile)
      (set! dlist dlistfile)
      (set! dlist (load dlistfile t)))
  (mapcar
   (lambda (d)
     (let ((utt (synth-nonsense-word-1 
		 (read-from-string
		  (string-append "(" (car (cdr d)) ")")))))
       (format t "%l\n" d)
       (if (and labdir (not diph_do_db_boundaries))
	   (save_segs_plus_db utt (format nil "%s/%s.lab" labdir (car d))))
       (synth-nonsense-word-2 utt)
       (if (and labdir diph_do_db_boundaries)
	   (save_segs_plus_db utt (format nil "%s/%s.lab" labdir (car d))))
       (utt.save.wave utt (format nil "%s/%s.wav" wavedir (car d)))
       t))
   dlist)
  t
)

(define (diphone-gen-waves-words wavedir labdir dlist)
  "(diphone-gen-waves wavedir labdir dlist)
Synthesis each nonsense word with fixed duration and fixed pitch
saving the waveforms for playing as prompts (as autolabelling)."
  (if (boundp 'Diphone_Prompt_Setup)
      (Diphone_Prompt_Setup))
  (mapcar
   (lambda (d)
     (let ((utt (synth-carrier-word (car (cdr (cdr (cdr d)))))))
       (format t "%l\n" d)
       (utt.save.wave utt (format nil "%s/%s.wav" wavedir (car d)))
       (if labdir
	   (utt.save.segs utt (format nil "%s/%s.lab" labdir (car d))))
       t))
   dlist)
  t
)

(define (diphone-gen-schema name outfile)
  "(diphone-gen-schema name outfile)
Saves a complete diphone schema in named file.  Filenames for
each diphone are generated as name_XXXX."
  (let ((ofd (fopen outfile "w"))
	(dlist (diphone-gen-list))
	(n 0))
    ;; Name them
    (set! dlist-all
	  (mapcar
	   (lambda (d)
	     (cons (format nil "%s_%04d" name (set! n (+ 1 n))) d))
	   dlist))
    ;; save the info to file
    (mapcar
     (lambda (d)
       (format ofd "( %s %l %l )\n"
	       (car d) 
	       (diph-stringify (car (cdr (cdr d))))
	       (car (cdr d))))
     dlist-all)
    (fclose ofd)
    t))

(define (diphone-synth-waves dlistfile n)
  "(diphone-synth-waves dlistfile)
Synthesis each nonsense word with fixed duration and fixed pitch
and play them to find errors."
  (let ((c 1))
    (if (consp dlistfile)
	(set! dlist dlistfile)
	(set! dlist (load dlistfile t)))
    (mapcar
     (lambda (d)
       (let ((utt (synth-nonsense-word-1 
		   (read-from-string
		    (string-append "(" (car (cdr d)) ")")))))
	 (if (>= c n)
	     (begin
	       (format t "%l\n" d)
	       (synth-nonsense-word-2 utt)
	       (utt.play utt)))
	 (set! c (+ 1 c))
	 t))
     dlist)
    t)
)

(define (diph-stringify plist)
  "(diph-stringify plist)
Change the list of phones into a string of space separated phones"
  (let ((s ""))
    (mapcar
     (lambda (p)
       (if (string-equal "" s)
	   (set! s p)
	   (set! s (string-append s " " p))))
     plist)
    s))

(provide 'diphlist)
