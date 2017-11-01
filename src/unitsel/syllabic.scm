;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                  and Alan W Black and Kevin Lenzo                   ;;;
;;;                      Copyright (c) 2000-2001                        ;;;
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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;; Code to add to a INST_LANG_VOX_clunits.scm                          ;;;
;;;                                                                     ;;;
;;; sorry you've got to do this by hand at present                      ;;;
;;;                                                                     ;;;
;;; This allows multi-resolution unit selection, allowing a             ;;;
;;; a demi-syllable, diphone and phone index to be built over the       ;;;
;;; same db.  It may or may not work for you and ist still just an      ;;;
;;; experimental idea.                                                  ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;; Add to near top of INST_LANG_VOX_clunits.scm file after settig
;;; of INST_LANG_VOX::closest_voice
(defvar INST_LANG_VOX::build_stage nil)

;;; Remove the definition of INST_LANG_VOX:clunit_name and
;;; add the folowing functions

(define (INST_LANG_VOX::get_syllable s)
  (apply
   string-append
   (mapcar
    (lambda (seg)
      (string-append "_" (item.name seg)))
    (item.relation.daughters s 'SylStructure))))

(define (INST_LANG_VOX::get_onset s)
  (let ((seenvowel nil) (dd))
    (apply
     string-append
     (mapcar
      (lambda (seg)
	(if (not seenvowel)
	    (set! dd (string-append "_" (item.name seg)))
	    (set! dd ""))
	(if (string-equal "+" (item.feat seg "ph_vc"))
	    (set! seenvowel t))
	dd)
      (item.relation.daughters s 'SylStructure)))))
     
(define (INST_LANG_VOX::get_coda s)
  (let ((seenvowel nil) (dd))
    (apply
     string-append
     (mapcar
      (lambda (seg)
	(if (string-equal "+" (item.feat seg "ph_vc"))
	    (set! seenvowel t))
	(if (not seenvowel)
	    (set! dd "")
	    (set! dd (string-append "_" (item.name seg))))

	dd)
      (item.relation.daughters s 'SylStructure)))))

(define (INST_LANG_VOX::syllabic i)
  "(INST_LANG_VOX::syllabic i)
Returns the syllabic identifier for this segment, which consisys of
A position an value.  The positions are _O (onset) _N (nucleus)
_C (coda) and _S (final s). The value is the demi syllable
value."
  (let ((oc (item.feat i "seg_onsetcoda")))
    (cond
     ((null (item.relation.parent i 'SylStructure))
      (string-append
       "_nosyl_"
       (item.feat i "n.name")))
     ((string-equal "onset" oc)
	  (string-append
       "_O"
       (INST_LANG_VOX::get_onset (item.relation.parent i 'SylStructure))))
     ((string-equal "+" (item.feat i "ph_vc"))
      (string-append
       "_N"
       (INST_LANG_VOX::get_coda (item.relation.parent i 'SylStructure))))
     ((and (member_string (item.name i) '("s"))
	   (string-equal "-" (item.feat i "p.ph_cvox")))
      (string-append
       "_S_"
       "s"))
     (t  ;; simple in the coda
      (string-append
       "_C"
       (INST_LANG_VOX::get_coda (item.relation.parent i 'SylStructure))))))
)

(define (INST_LANG_VOX::clunit_name i)
  "(INST_LANG_VOX::clunit_name i)
Defines the unit name for unit selection for us.  The can be modified
changes the basic claissifncation of unit for the clustering.  By default
this does segment plus word, which is reasonable for many ldom domains."
  (let ((phone_name (item.name i))
	(diphone_name 
	 (string-append
	  (item.name i)
	  "_"
	  (if (string-equal "0" (item.feat i "p.name"))
	      (car (cadr (car (PhoneSet.description '(silences)))))
	      (item.feat i "p.name"))))
	(syllabic_name 
	 (string-append
	  (item.name i)
	  (INST_LANG_VOX::syllabic i))))
    (cond
     ((string-equal INST_LANG_VOX::build_stage 'syllable)
      syllabic_name)
     ((string-equal INST_LANG_VOX::build_stage 'phone)
      phone_name)
     ((string-equal INST_LANG_VOX::build_stage 'diphone)
      diphone_name)
     ((and (not (string-equal phone_name 
		 (car (cadr (car (PhoneSet.description '(silences)))))))
	   (assoc_string syllabic_name INST_LANG_VOX::clunits_clunit_selection_trees))
      syllabic_name)
     ((assoc_string diphone_name INST_LANG_VOX::clunits_clunit_selection_trees)
      diphone_name)
     ((assoc_string phone_name INST_LANG_VOX::clunits_clunit_selection_trees)
      phone_name)
     (t   ;; should never get here, but maybe
      "sh")))
)
