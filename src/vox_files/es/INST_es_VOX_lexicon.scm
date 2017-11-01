;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                  and Alan W Black and Kevin Lenzo                   ;;;
;;;                  and Ariadna Font Llitjos                           ;;;
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
;;; Lexicon, LTS and Postlexical rules for Spain Spanish (cmu_es_sp)
;;;
;;; Taken from old Spanish voice and modified by Ariadna Font Llitjos 
;;; (aria@cs) to make it consistent with Spanish spoken in Spain (in particluar;;; in cities like Salamanca and Valladolid).
;;; Authors of old version: Alistair Conkie, Borja Etxebarria and Alan W Black
;;;
;;; Letter to sounds rules and functions to produce stressed syllabified
;;; pronunciations for Spanish words.
;;;
;;; There is some history in one set of the LTS rules back to
;;; Rob van Gerwen, University of Nijmegen.
;;;

(define (cmu_es_addenda)
  "(cmu_es_addenda)
Basic lexicon should have basic letters, symbols and punctuation."

;;; Pronunciation of letters in the alphabet
;(lex.add.entry '("a" nn (((a) 0))))
(lex.add.entry '("b" nn (((b e) 0))))
(lex.add.entry '("c" nn (((th e) 0))))
(lex.add.entry '("ç" nn (((th e) 0)((th e) 0)((d iS) 1)((y a) 0)))) ;; ari
(lex.add.entry '("d" nn (((d e) 0))))
;(lex.add.entry '("e" nn (((e) 0))))
(lex.add.entry '("f" nn (((eS) 1)((f e) 0))))
(lex.add.entry '("g" nn (((j e) 0)))) ;; ari: previously /ge/
(lex.add.entry '("h" nn (((aS) 1)((ch e) 0))))
;(lex.add.entry '("i" nn (((i) 0))))
(lex.add.entry '("j" nn (((j oS) 1)((t a) 0)))) ;; ari: previously /xota/
(lex.add.entry '("k" nn (((k a) 0))))
(lex.add.entry '("l" nn (((eS) 1)((l e) 0))))
(lex.add.entry '("m" nn (((eS) 1)((m e) 0))))
(lex.add.entry '("n" nn (((eS) 1)((n e) 0))))
(lex.add.entry '("~n" nn (((eS) 1)((ny e) 0))))
(lex.add.entry '("ñ" nn (((eS) 1)((ny e) 0))))
;(lex.add.entry '("o" nn (((o) 0))))
(lex.add.entry '("p" nn (((p e) 0))))
(lex.add.entry '("q" nn (((k u) 0))))
(lex.add.entry '("r" nn (((eS) 1)((rr e) 0))))
(lex.add.entry '("s" nn (((eS) 1) ((s e) 0))))
(lex.add.entry '("t" nn (((t e) 0))))
;(lex.add.entry '("u" nn (((u) 0))))
(lex.add.entry '("v" nn (((uS) 1)((b e) 0)))) ;; ari: arguably /ube/ 
(lex.add.entry '("w" nn (((u) 0) ((b e) 0) ((d oS) 1) ((b l e) 0))))
(lex.add.entry '("x" nn (((eS) 1)((k i s) 0))))
;(lex.add.entry '("y" nn (((i) 0)((g r i eS) 1))((g a) 0)))   ;; doubt: stres
(lex.add.entry '("z" nn (((th eS) 1)((t a) 0))))
;(lex.add.entry '("á" nn (((a) 0))))
;(lex.add.entry '("é" nn (((e) 0))))
;(lex.add.entry '("í" nn (((i) 0))))
;(lex.add.entry '("ó" nn (((o) 0))))
;(lex.add.entry '("ú" nn (((u) 0))))
;(lex.add.entry '("ü" nn (((u) 0))))
;; ari: 
;; Need to include all the Nahuatl words as a lexical entry, since in such
;; words 'x' is pronounces /j/ instead of /ks/
(lex.add.entry '("méxico" n (((m eS) 1) ((j i) 0) ((k o) 0))))
(lex.add.entry '("mexico" n (((m eS) 1) ((j i) 0) ((k o) 0))))
(lex.add.entry '("mexicano" n (((m e) 0) ((j i) 0) ((k aS) 1) ((n o) 0))))
(lex.add.entry '("mexicana" n (((m e) 0) ((j i) 0) ((k aS) 1) ((n a) 0))))
(lex.add.entry '("oaxaca" n (((uW a) 0) ((j aS) 1) ((k a) 0)))) 
(lex.add.entry '("oaxaqueño" n (((uW a) 0) ((j a) 0) ((k eS) 1) ((ny o) 0))))
(lex.add.entry '("texas" n (((t eS) 1) ((j a s) 0))))
(lex.add.entry '("texano" n (((t e) 0) ((j aS) 1) ((n o) 0))))
(lex.add.entry '("texana" n (((t e) 0) ((j aS) 1) ((n a) 0))))
;; ari: added after recording the Spanish data (afp and infosel)
;; + foreign words 
 (lex.add.entry '("Deneuve" nn (((d e) 0) ((n eS f) 1))))
 (lex.add.entry '("Mihail" nn (((m i) 0) ((j a iS l) 1))))  
 (lex.add.entry '("Bochum" nn  (((b oS) 1) ((j u m) 0))))
 (lex.add.entry '("hit" nil (((j iS t) 1)))) ;; usually used as nn in Spanish
 (lex.add.entry '("bach" nil (((b aS k) 1)))) ;; sometimes (((b aS j) 1)))
 (lex.add.entry '("hardware" nil (((j a r) 0) ((uW aS r) 1))))
 (lex.add.entry '("software" nil (((s o f t) 0) ((uW aS r) 1))))
 (lex.add.entry '("tech" nil (((t eS k) 1))))
 (lex.add.entry '("high tech" nil (((j a iW) 0) ((t eS k) 1))))
 (lex.add.entry '("hi-tech" nil (((j a iW) 0) ((t eS k) 1))))
 (lex.add.entry '("hi tech" nil (((j a iW) 0) ((t eS k) 1))))
 ;; ari: very specific, but appeared in the prompts
 (lex.add.entry '("logitech" nil (((l o) 0) ((g i) 0) ((t eS k) 1))))

;;; Abreviations
;;; months
;(lex.add.entry '("en." nil (((e) 0) ((n eS) 1) ((r o) 0))))
(lex.add.entry '("Feb" nil (((f e) 0) ((b r eS) 1) ((r o) 0))))
(lex.add.entry '("Abr" nil (((a) 0) ((b r iS l) 1))))
(lex.add.entry '("Jun" nil (((j uS) 1) ((n iW o) 0))))
(lex.add.entry '("Jul" nil (((j uS) 1) ((l iW o) 0))))
(lex.add.entry '("Ag" nil (((a) 0) ((g oS s) 1) ((t o) 0))))
(lex.add.entry '("Sept" nil (((s e) 0) ((t iW eS m) 1) ((b r e) 0))))
(lex.add.entry '("Oct" nil (((o k) 0) ((t uS) 1) ((b r e) 0))))
(lex.add.entry '("Nov" nil (((n o) 0) ((b iW eS m) 1) ((b r e) 0))))
(lex.add.entry '("Dic" nil (((d i) 0) ((th iW eS m) 1) ((b r e) 0))))
;;; days of the week: what are the most common abr.: ls ma mie ju vi sab dom ?

;; ari: added after recording the Spanish data (afp and infosel)
 (lex.add.entry '("Dr." nil (((d o k) 0) ((t oS r) 1))))
 (lex.add.entry '("Dra." nil (((d o k) 0) ((t oS) 1) ((r a) 0))))
 (lex.add.entry '("Sr." nil (((s e) 0) ((ny oS r) 1))))
 (lex.add.entry '("Sra." nil (((s e) 0) ((ny oS) 1) ((r a) 0))))
 (lex.add.entry '("Sres." nil (((s e) 0) ((ny oS) 1) ((r e s) 0))))
 (lex.add.entry '("Ud." nil (((u s) 0) ((t eS d) 1))))
 (lex.add.entry '("Uds." nil (((u s) 0) ((t eS) 1) ((d e s) 0))))
 (lex.add.entry '("Udes." nil (((u s) 0) ((t eS) 1) ((d e s) 0))))
 (lex.add.entry '("Ud" nil (((u s) 0) ((t eS d) 1))))
 (lex.add.entry '("Uds" nil (((u s) 0) ((t eS) 1) ((d e s) 0))))
 (lex.add.entry '("Udes" nil (((u s) 0) ((t eS) 1) ((d e s) 0))))
 (lex.add.entry '("Mr." nil  (((m iS s) 1) ((t e r) 0))))
 (lex.add.entry '("Mrs." nil (((m iS) 1) ((s i s) 0))))
 (lex.add.entry '("Miss." nil (((m iS s ) 1))))
 (lex.add.entry '("Av." nil (((a) 0) ((b e) 0) ((n iS) 1) ((d a) 0))))
 (lex.add.entry '("etc." nil (((e t) 0) ((th eS) 1) ((t e) 0) ((r a) 0))))
 (lex.add.entry '("etc" nil (((e t) 0) ((th eS) 1) ((t e) 0) ((r a) 0))))
 (lex.add.entry '("Kg." nil (((k i) 0) ((l o) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("Kgs." nil (((k i) 0) ((l o) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("Kg" nil (((k i) 0) ((l o) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("Kgs" nil (((k i) 0) ((l o) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("mg" nil (((m i) 0) ((l i) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("mg." nil (((m i) 0) ((l i) 0) ((g r aS) 1) ((m o s) 0))))
 (lex.add.entry '("mm" nil (((m i) 0) ((l iS) 1) ((m e) 0) ((t r o s) 0))))
 (lex.add.entry '("mm." nil (((m i) 0) ((l iS) 1) ((m e) 0) ((t r o s) 0))))
 (lex.add.entry '("seg." nil (((s e) 0) ((g uS n) 1) ((d o s) 0))))
 (lex.add.entry '("min." nil (((m i) 0) ((n uS) 1) ((t o s) 0))))
 (lex.add.entry '("mín." nil (((m iS) 1) ((n i) 0) ((m o) 0))))
 (lex.add.entry '("máx." nil (((m aS) 1) ((k s i) 0) ((m o) 0))))
 (lex.add.entry '("km" nil (((k i) 0) ((l oS) 1) ((m e) 0) ((t r o) 0))))
 (lex.add.entry '("h." nil (((oS) 1) ((r a s) 0))))
 (lex.add.entry '("km / h" nil (((k i) 0) ((l oS) 1) ((m e) 0) ((t r o) 0) ((p o) 0) ((r oS) 1) ((r a) 0))))
 (lex.add.entry '("II" nil (((d oS s) 1)))) ;; could also be: (((s e) 0) ((g uS n) 1) ((d o) 0)))
 (lex.add.entry '("III" nil (((t r eS s) 1))))
 ;(lex.add.entry '("IV" nil (((k uW aS) 1) ((t r o) 0))))
 ;(lex.add.entry '("V" nil (((th iS n) 1) ((k o) 0))))
 (lex.add.entry '("XX" nil (((b eS iW n) 1) ((t e) 0))))
 (lex.add.entry '("XXI" nil (((b e iW n) 0) ((t iW uS) 1) ((n o) 0))))
 (lex.add.entry '("EEUU" nil (((e s) 0) ((t aS) 1) ((d o s) 0) ((u n iS) 1) ((d o s) 0))))
 (lex.add.entry '("E.E.U.U." nil (((e s) 0) ((t aS) 1) ((d o s) 0) ((u n iS) 1) ((d o s) 0))))
 (lex.add.entry '("AyD" nil (((a) 1) ((i) 0) ((d eS) 1))))

;;; Symbols ...
(lex.add.entry 
 '("*" n (((a s) 0) ((t e) 0) ((r iS s) 1)  ((k o) 0))))
(lex.add.entry 
 '("%" n (((p o r) 0) ((th i eS n) 1) ((t o) 0))))
;; ari: in English "ampersand", but I there seems to be no translation into 
;; Spanish, other than it's just moslty pronounced "an"
(lex.add.entry 
 '("&" n (((aS m) 1) ((p e r) 0) ((s a n) 0)))) 
(lex.add.entry 
 '("&" punc (((aS n) 1)))) ;; ari
(lex.add.entry 
 '("$" n (((d oS) 1) ((l a r) 0))))
(lex.add.entry 
 '("#" n (((a l) 0) ((m u a) 0) ((d iS) 1) ((ll a) 0))))
(lex.add.entry 
 '("@" n (((a) 0) ((rr oS) 1) ((b a) 0))))
;; ari: Alan?????????
(lex.add.entry 
 '("+" n (((m a s) 0)) ((pos "K7%" "OA%" "T-%"))))
(lex.add.entry 
 '("^" n (((k aS) 1) ((r e t) 0)) ((pos "K6$")))) ;; ari: also "and" or "triangulito" (Mex.: testa triangular)
(lex.add.entry 
 '("~" n (((t iS l) 1) ((d e) 0)) ((pos "K6$"))))
(lex.add.entry 
 '("=" n (((i) 0) ((g u aS l) 1))))
;(lex.add.entry 
; '("/" n (((eS n ) 1) ((t r e) 0))))  ;; division, etc.
;; ari: RAE says / should be called 'barra', and I guess it's more general than
;;      'entre' which in many cases it will be just wrong: 50 km/hora
(lex.add.entry 
 '("/" n (((b aS) 1) ((rr a) 1))))
(lex.add.entry 
 '("\\" n (((b aS) 1) ((rr a) 1))))
(lex.add.entry 
 '("_" n (((s u b) 0) ((rr a) 0) ((ll aS) 1) ((d o) 0)) ))
(lex.add.entry 
 '("|" n (((b aS) 1) ((rr a) 0))))
(lex.add.entry 
 '(">" n ((( m a ) 0) ((ll oS r) 1) ((k e) 0))))
(lex.add.entry 
 '("<" n ((( m e ) 0) ((n oS r) 1) ((k e) 0))))
(lex.add.entry 
 '("[" n ((( a) 0) ((b r iS r) 1) ((k o r) 0)((ch eS) 1)((t e) 0))))
(lex.add.entry 
 '("]" n (((th e) 0) ((rr aS r) 1) ((k o r) 0)((ch eS) 1)((t e) 0))))
(lex.add.entry 
 '(" " n (((e s) 0)((p aS) 1)((th i o) 0))))
(lex.add.entry 
 '("\t" n (((t a) 0) ((b u) 0) ((l a) 0) ((d oS r) 1))))
(lex.add.entry 
 '("\n" n (((n u eS) 1) ((b a) 0)((l i) 1) ((n e a) 0))))

;; Basic punctuation must be in with nil pronunciation
;; ari: commented out the nn pronunciations since I need to align prompts
;;      with phones 
(lex.add.entry '("." punc nil))
;(lex.add.entry '("." nn (((p uS n) 1) ((t o) 0))))
(lex.add.entry '("'" punc nil))
;(lex.add.entry '("'" nn (((a) 0) ((p oS s) 1) ((t r o) 0) ((f o) 0)))) ;; ari
(lex.add.entry '(":" punc nil))
;(lex.add.entry '(":" nn (((d o s) 0) ((p uS n) 1) ((t o s) 0)))) ;; ari
(lex.add.entry '(";" punc nil))
;(lex.add.entry '(";" nn (((p uS n) 1) ((t o iW) 0) ((k oS) 1) ((m a) 0)))) ;;ari
(lex.add.entry '("," punc nil))
;(lex.add.entry '("," nn (((k oS) 1) ((m a) 0)))) ;; ari
(lex.add.entry '("-" punc nil))
;(lex.add.entry '("-" nn (((g iW oS n) 1)))) ;; ari
(lex.add.entry '("\"" punc nil))
;(lex.add.entry '("\"" nn (((k o) 0) ((m iS) 1) ((ll a s) 0)))) ;; ari 
(lex.add.entry '("`" punc nil)) 
(lex.add.entry '("?" punc nil)) 
;(lex.add.entry '("?" nn (((i n) 0) ((t e) 0) ((rr o) 0) ((g aS n) 1) ((t e) 0)))) ;; ari
(lex.add.entry '("!" punc nil))
;(lex.add.entry '("!" nn (((e s k l a) 0) ((m a) 0) ((th iS o n) 1)))) ;; ari
;; ari: should add the ones for ? and ! upside down

) ;; end of addenda

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Down cases with accents
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(lts.ruleset
 cmu_spanish_downcase
 ( )
 (
  ( [ a ] = a )
  ( [ e ] = e )
  ( [ i ] = i )
  ( [ o ] = o )
  ( [ u ] = u )
  ( [ á ] = á )
  ( [ é ] = é )
  ( [ í ] = í )
  ( [ ó ] = ó )
  ( [ ú ] = ú )
  ( [ ü ] = ü ) 
  ( [ b ] = b )
  ( [ c ] = c )
  ( [ "ç" ] = s )
  ( [ d ] = d )
  ( [ f ] = f )
  ( [ g ] = g )
  ( [ h ] = h )
  ( [ j ] = j )
  ( [ k ] = k )
  ( [ l ] = l )
  ( [ m ] = m )
  ( [ n ] = n )
  ( [ ñ ] =  ñ )
  ( [ p ] = p )
  ( [ q ] = q )
  ( [ r ] = r )
  ( [ s ] = s )
  ( [ t ] = t )
  ( [ v ] = v )
  ( [ w ] = w )
  ( [ x ] = x )
  ( [ y ] = y )
  ( [ z ] = z )
  ( [ "\'" ] = "\'" )
  ( [ : ] = : )
  ( [ ~ ] = ~ )
  ( [ "\"" ] = "\"" )
  ( [ A ] = a )
  ( [ E ] = e )
  ( [ I ] = i )
  ( [ O ] = o )
  ( [ U ] = u )
  ( [ Á ] = á )
  ( [ É ] = é )
  ( [ Í ] = í )
  ( [ Ó ] = ó )
  ( [ Ú ] = ú )
  ( [ Ü ] = ü ) 
  ( [ B ] = b )
  ( [ C ] = c )
  ( [ "Ç" ] = s )
  ( [ D ] = d )
  ( [ F ] = f )
  ( [ G ] = g )
  ( [ H ] = h )
  ( [ J ] = j )
  ( [ K ] = k )
  ( [ L ] = l )
  ( [ M ] = m )
  ( [ N ] = n )
  ( [ Ñ ] =  ñ )
  ( [ P ] = p )
  ( [ Q ] = q )
  ( [ R ] = r )
  ( [ S ] = s )
  ( [ T ] = t )
  ( [ V ] = v )
  ( [ W ] = w )
  ( [ X ] = x )
  ( [ Y ] = y )
  ( [ Z ] = z )
))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Hand written letter to sound rules
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;
;;;  Function to turn word into lexical entry for Spanish 
;;;  (taken from old Spanish files)
;;;
;;;  First uses lts to get phoneme string then assigns stress if
;;;  there is no stress and then uses a third set of rules to
;;;  mark syllable boundaries, finally converting that list
;;;  to the bracket structure festival requires
;;;

(define (cmu_es_lts_function word features)
  "(cmu_es_lts_function WORD FEATURES)
Using various letter to sound rules build a Spanish pronunciation of 
WORD."
  (let (phones syl stresssyl dword weakened)
    (if (lts.in.alphabet word 'cmu_spanish_downcase)
	(set! dword (spanish_downcase word))
	(set! dword (spanish_downcase "equis")))
    (set! phones (lts.apply dword 'cmu_es))
    (set! syl (lts.apply phones 'cmu_spanish_syl))
    (if (spanish_is_a_content_word 
	 (apply string-append dword)
	 spanish_guess_pos)
	(set! stresssyl (lts.apply syl 'cmu_spanish.stress))
	(set! stresssyl syl))  ;; function words leave as is
    (set! weakened (lts.apply stresssyl 'cmu_spanish_weak_vowels))
    (list word
	  nil
	  (spanish_tosyl_brackets weakened))))

(define (spanish_is_a_content_word word poslist)
  "(spanish_is_a_content_word WORD POSLIST)
Check explicit list of function words and return t if this is not
listed."
  (cond
   ((null poslist)
    t)
   ((member_string word (cdr (car poslist)))
    nil)
   (t
    (spanish_is_a_content_word word (cdr poslist)))))

(define (spanish_downcase word)
  "(spanish_downcase WORD)
Downs case word by letter to sound rules becuase or accented form
this can't use the builtin downcase function."
  (lts.apply word 'cmu_spanish_downcase))

(define (spanish_tosyl_brackets phones)
   "(spanish_tosyl_brackets phones)
Takes a list of phones containing - as syllable boundary.  Construct the
Festival bracket structure."
 (let ((syl nil) (syls nil) (p phones) (stress 0))
    (while p
     (set! syl nil)
     (set! stress 0)
     (while (and p (not (eq? '- (car p))))
       (set! syl (cons (car p) syl))
       (if (string-matches (car p) ".*S")
           (set! stress 1))
       (set! p (cdr p)))
     (set! p (cdr p))  ;; skip the syllable separator
     (set! syls (cons (list (reverse syl) stress) syls)))
    (reverse syls)))   

;;

;; this is the one given by default, does the default syllabification (bad
;; even for English)
; ;;;  Function called when word not found in lexicon
; (define (cmu_es_lts_function word features)
;   "(cmu_es_lts_function WORD FEATURES)
; Return pronunciation of word not in lexicon."
;   (let ((dword (downcase word)))
;     ;; Note you may need to use a letter to sound rule set to do
;     ;; casing if the language has non-ascii characters in it.
;     (if (lts.in.alphabet word 'cmu_es)
; 	(list
; 	 word
; 	 features
; 	 ;; This syllabification is almost certainly wrong for
; 	 ;; this language (its not even very good for English)
; 	 ;; but it will give you something to start off with
; 	 (lex.syllabify.phstress
; 	   (lts.apply word 'cmu_es)))
; 	(begin
; 	  (format stderr "unpronouncable word %s\n" word)
; 	  ;; Put in a word that means "unknown" with its pronunciation
; 	  '("nepoznat" nil (((N EH P) 0) ((AO Z) 0) ((N AA T) 0))))))
; )



;; from old voice
; borja: some rules updated or deleted.
; Rules for directly accented vowels, are typed using 
; the sun character set and codepage ISO 8859/1 Latin 1. This 
; matches the one on Linux and Windows for our purposes, so 
; almost everybody happy.
; Umlaut (dieresis) management. I have considered
; three diferent ways to include the umlaut for spanish in 
; festival, using <:> or <">. example:  ping:uino  ping"uino,
; and of course, directly typing the weird thing (ü).
; Accented vowels can be typed both directly (á) or as a
; quote preceding the plain vowel ('a). example: cami'on  camión

;; ari: Letter -> Phone
(lts.ruleset
 cmu_es
;  Sets used in the rules
(
  (LNS l n s )
  (DNSR d n s r )
  (EI e i é í)  ; note that accented vowels are included in this set
  (AEIOUt  á é í ó ú )
  (V a e i o u )
  (C b c d f g h j k l m n ñ ~ p q r s t v w x y z )
)
;  Rules
(

 ; these weird rule, to break dipthongs at end of words like atribuid atribuido,...
 ( "'" V* C* u [ i ] DNSR SIL = i ) 
 ( AEIOUt V* C* u [ i ] DNSR SIL = i )   ;; $$$  ~n and so, what will do?
 ( u [ i ] DNSR SIL = iS ) 
 ( "'" V* C* u [ i ] d V SIL = i ) 
 ( AEIOUt V* C* u [ i ] d V SIL = i ) 
 ( u [ i ] d AEIOUt SIL = i )   ;; not sure about these two
 ( u [ i ] d V SIL = iS )       ;; 


 ( [ a ] = a )
 ( [ e ] = e )
 ( [ i ] = i )
 ( [ o ] = o )
 ( [ u ] = u )
 ( [ "'" a ] = aS )
 ( [ "'" e ] = eS )
 ( [ "'" i ] = iS )
 ( [ "'" o ] = oS )
 ( [ "'" u ] = uS )
 ( [ á ] = aS )
 ( [ é ] = eS )
 ( [ í ] = iS )
 ( [ ó ] = oS )
 ( [ ú ] = uS )
 ( [ ":" u ] = u )      ; umlaut (u dieresis) (should not happen, only with g, and already removed)
 ( [ "\"" u ] = u )  
 ( [ ü ] = u ) 

 ( [ b ] = b )
 ( [ v ] = b )
 ( [ c ] "'" EI = th )
 ( [ c ] EI = th )
 ( [ c h ] = ch )
 ( [ c ] = k )
 ( [ d ] = d )
 ( [ f ] = f )
 ( [ g ] "'" EI = j ) ;; ari: /x/ in sampa
 ( [ g ] EI = j ) ;; ari: coger, Gijon   (/x/ in sampa)
 ( [ g u ] "'" EI = g )
 ( [ g u ] EI = g )

 ( [ g ":" u ] EI = g u )      ; umlaut (u dieresis)
 ( [ g ":" u ] "'" EI = g u ) 
 ( [ g "\"" u ] EI = g u )  
 ( [ g "\"" u ] "'" EI = g u ) 
 ( [ g ü ] EI = g u ) 
 ( [ g ü ] "'" EI = g u ) 

 ( [ g ] = g )
 ( [ h ] =  ) ;; ari: even though a few people say /guevo/, this is not 
              ;;      considered the correct form in Spain (unlike Mexico)
 ( [ j ] = j )
 ( [ k ] = k )
 ( [ l l ] SIL = l ) ;; ari: palabra acabada en 'll'
 ( [ l l ] = y ) ;; ari: calle, llorar, ya,  (ye'ismo)
 ( [ l ] = l )
 ( [ m ] = m )
 ( [ "~" n ] = ny )
 ( [ ñ ] = ny )
 ( [ n ] = n )
 ( [ p ] = p )
 ( [ p h ] = f )  ;; to speak a bit of greek.
 ( [ q u ] a = k u )  ;; no castillian word uses this, but it would be pronounced this way in greek and foreign words (aquarium, quo, etc)
 ( [ q u ] = k )
 ( [ q ] = k ) ;; should't happend, but if you type it...
 ( [ r r ] = rr )
 ( SIL [ r ] = rr )
 ( LNS [ r ] = rr )
 ( [ r ] = r )
 ( [ s ] = s )
 ( SIL [ s ] C = e s )
 ( SIL [ s ] "'" C = e s )
;; ( SIL [ s ] ":" C = e s );; ari: don't know what an umlaut is supposed to do with a consonant
;; ( SIL [ s ] "\"" C = e s )
;; ari: adding /sh/ phoneme for foreign names
;;( [ s h ]  = sh ) ;; Shirley, sasha
;;( [ s c h ] = sh ) ;; Schubert
 ( [ t ] = t )
 ( [ w ] = u ) ;; ari: whisky -> /g:uiski/
;; ari:
;; In general, when 'x' is between vowels or at the end of word, it's 
;; pronounced /ks/, and when it's at the beginning of the word or it's
;; followed by a consonant, is pronounced /s/.
;; However this varies a lot from region to region, and sometimes, these words
;; get pronounced the other way around (ex: examen -> /esamen/, 
;; sexto -> /seksto/ in Sevilla to distinguish it from cesto (/sesto/), 
;; exacto -> /esacto/, etc...)
 ( V [ x ] V = k s ) ;; ari: ex: examen, taxi, exhibir, 
 ( [ x ] SIL = k s ) ;; ari: relax  (what about Marx?)
 ( [ x ] C = s ) ;; ari ex: explicar, excavar, exclamacion, extra, exponer, 
 ( SIL [ x ] = s ) ;; ari ex: xilograf'ia, xilofono
 ( [ x ] = k s ) ;; ari: default /ks/
 ( [ y ] SIL = i )
 ( [ y ] C = i )
 ( [ y ] "'" C = i )
 ( [ y ] ":" C = i )
 ( [ y ] "\"" C = i )
 ( [ y ] = y ) ;; ya, yo, iogurt?
 ( [ z ] = th )

  ; quotes are used for vowel accents in foreign keyboards (i.e. cami'on).
  ; remove those that were not before a vowel. same with other signs.
 ( [ "'" ] = )  
 ( [ ":" ] = )  
 ( [ "\"" ] = )  
 ( [ "~" ] = )  
))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Spanish sylabification by rewrite rules
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; ari: Phone -> Phone
(lts.ruleset
   cmu_spanish_syl
   (  (V aS iS uS eS oS a i u e o )
      (IUT iS uS )
      (C b ch d f g j k l m n ny p r rr s t th y)
   )
   ;; Rules will add - at syllable boundary
   (
      ;; valid CC groups
      ( V C * [ b l ] V = - b l )
      ( V C * [ b r ] V = - b r )
      ( V C * [ k l ] V = - k l )
      ( V C * [ k r ] V = - k r )
      ( V C * [ k s ] V = - k s ) ; for words with "x"
      ( V C * [ d r ] V = - d r )
      ( V C * [ f l ] V = - f l )
      ( V C * [ f r ] V = - f r )
      ( V C * [ g l ] V = - g l )
      ( V C * [ g r ] V = - g r )
      ( V C * [ p l ] V = - p l )
      ( V C * [ p r ] V = - p r )
      ( V C * [ t l ] V = - t l )
      ( V C * [ t r ] V = - t r )

      ;; triptongs
      ( [ i a i ] = i a i )  
      ( [ i a u ] = i a u )  
      ( [ u a i ] = u a i )  
      ( [ u a u ] = u a u )  
      ( [ i e i ] = i e i )  
      ( [ i e u ] = i e u )  
      ( [ u e i ] = u e i )  
      ( [ u e u ] = u e u )  
      ( [ i o i ] = i o i )  
      ( [ i o u ] = i o u )  
      ( [ u o i ] = u o i )  
      ( [ u o u ] = u o u )  
      ( [ i aS i ] = i aS i )  
      ( [ i aS u ] = i aS u )  
      ( [ u aS i ] = u aS i )  
      ( [ u aS u ] = u aS u )  
      ( [ i eS i ] = i eS i )  
      ( [ i eS u ] = i eS u )  
      ( [ u eS i ] = u eS i )  
      ( [ u eS u ] = u eS u )  
      ( [ i oS i ] = i oS i )  
      ( [ i oS u ] = i oS u )  
      ( [ u oS i ] = u oS i )  
      ( [ u oS u ] = u oS u )  

      ;; break invalid triptongs
      ( IUT [ i a ]  = - i a )
      ( IUT [ i e ]  = - i e )
      ( IUT [ i o ]  = - i o )
      ( IUT [ u a ]  = - u a )
      ( IUT [ u e ]  = - u e )
      ( IUT [ u o ]  = - u o )
      ( IUT [ a i ]  = - a i )
      ( IUT [ e i ]  = - e i )
      ( IUT [ o i ]  = - o i )
      ( IUT [ a u ]  = - a u )
      ( IUT [ e u ]  = - e u )
      ( IUT [ o u ]  = - o u )
      ( IUT [ i u ]  = - i u )
      ( IUT [ u i ]  = - u i )
      ( IUT [ i aS ]  = - i aS )
      ( IUT [ i eS ]  = - i eS )
      ( IUT [ i oS ]  = - i oS )
      ( IUT [ u aS ]  = - u aS )
      ( IUT [ u eS ]  = - u eS )
      ( IUT [ u oS ]  = - u oS )
      ( IUT [ aS i ]  = - aS i )
      ( IUT [ eS i ]  = - eS i )
      ( IUT [ oS i ]  = - oS i )
      ( IUT [ aS u ]  = - aS u )
      ( IUT [ eS u ]  = - eS u )
      ( IUT [ oS u ]  = - oS u )
      ( IUT [ i uS ]  = - i uS )
      ( IUT [ u iS ]  = - u iS )

      ;; diptongs
      ( [ i a ]  = i a )
      ( [ i e ]  = i e )
      ( [ i o ]  = i o )
      ( [ u a ]  = u a )
      ( [ u e ]  = u e )
      ( [ u o ]  = u o )
      ( [ a i ]  = a i )
      ( [ e i ]  = e i )
      ( [ o i ]  = o i )
      ( [ a u ]  = a u )
      ( [ e u ]  = e u )
      ( [ o u ]  = o u )
      ( [ i u ]  = i u )
      ( [ u i ]  = u i )
      ( [ i aS ]  = i aS )
      ( [ i eS ]  = i eS )
      ( [ i oS ]  = i oS )
      ( [ u aS ]  = u aS )
      ( [ u eS ]  = u eS )
      ( [ u oS ]  = u oS )
      ( [ aS i ]  = aS i )
      ( [ eS i ]  = eS i )
      ( [ oS i ]  = oS i )
      ( [ aS u ]  = aS u )
      ( [ eS u ]  = eS u )
      ( [ oS u ]  = oS u )
      ( [ uS i ]  = uS i )
      ( [ iS u ]  = iS u )
    
      ;; Vowels preceeded by vowels are syllable breaks
      ;; triptongs and diptongs are dealt with above
      ( V [ a ]  = - a )
      ( V [ i ]  = - i )
      ( V [ u ]  = - u )
      ( V [ e ]  = - e )
      ( V [ o ]  = - o )
      ( V [ aS ]  = - aS )
      ( V [ eS ]  = - eS )
      ( V [ iS ]  = - iS )
      ( V [ oS ]  = - oS )
      ( V [ uS ]  = - uS )

      ;; If any consonant is followed by a vowel and there is a vowel
      ;; before it, its a syl break
      ;; the consonant cluster are dealt with above
      ( V C * [ b ] V = - b )
      ( V C * [ ch ] V = - ch )
      ( V C * [ d ] V = - d )
      ( V C * [ f ] V = - f )
      ( V C * [ g ] V = - g )
      ( V C * [ j ] V = - j )
      ( V C * [ k ] V = - k )
      ( V C * [ l ] V = - l )
      ( V C * [ m ] V = - m )
      ( V C * [ n ] V = - n )
      ( V C * [ ny ] V = - ny )
      ( V C * [ p ] V = - p )
      ( V C * [ r ] V = - r )
      ( V C * [ rr ] V = - rr )
      ( V C * [ s ] V = - s )
      ( V C * [ t ] V = - t )
      ( V C * [ y ] V = - y )
      ( V C * [ th ] V = - th )


      ;; Catch all consonants on their own (at end of word)
      ;; and vowels not preceded by vowels are just written as it
      ( [ b ] = b )
      ( [ ch ] = ch )
      ( [ d ] = d )
      ( [ f ] = f )
      ( [ g ] = g )
      ( [ j ] = j )
      ( [ k ] = k )
      ( [ l ] = l )
      ( [ m ] = m )
      ( [ n ] = n )
      ( [ ny ] = ny )
      ( [ p ] = p )
      ( [ r ] = r )
      ( [ rr ] = rr )
      ( [ s ] = s )
      ( [ t ] = t )
      ( [ y ] = y )
      ( [ th ] = th )
      ( [ a ]  =  a )
      ( [ i ]  =  i )
      ( [ u ]  =  u )
      ( [ e ]  =  e )
      ( [ o ]  =  o )
      ( [ aS ]  =  aS )
      ( [ iS ]  =  iS )
      ( [ uS ]  =  uS )
      ( [ eS ]  =  eS )
      ( [ oS ]  =  oS )
   )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Stress assignment in unstress words by rewrite rules
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; ari: Phone -> Phone
(lts.ruleset
 ;; Assign stress to a vowel when non-exists
 cmu_spanish.stress
 (
  (UV a i u e o)
  (V aS iS uS eS oS  a i u e o)
  (V1 aS iS uS eS oS)
  (VNS n s a i u e o)
  (C b ch d f g j k l m n ny p r rr s t th y )
  (VC b ch d f g j k l m n ny p r rr s t th y aS iS uS eS oS a i u e o)
  (ANY b ch d f g j k l m n ny p r rr s t th y - aS iS uS eS oS  a i u e o)
  (notNS b ch d f g j k l m ny p r rr t th y )
  (iu i u )
  (aeo a e o)
  )
 (
  ;; consonants to themselves
  ( [ b ] = b )
  ( [ d ] = d )
  ( [ ch ] = ch )
  ( [ f ] = f )
  ( [ g ] = g )
  ( [ j ] = j )
  ( [ k ] = k )
  ( [ l ] = l )
  ( [ m ] = m )
  ( [ n ] = n )
  ( [ ny ] = ny )
  ( [ p ] = p )
  ( [ r ] = r )
  ( [ rr ] = rr )
  ( [ s ] = s )
  ( [ t ] = t )
  ( [ th ] = th )
  ( [ y ] = y )
  ( [ - ] = - )
  ;; stressed vowels to themselves
  ( [ aS ] = aS )
  ( [ iS ] = iS )
  ( [ uS ] = uS )
  ( [ eS ] = eS )
  ( [ oS ] = oS )

  ( V1 ANY * [ a ] = a )
  ( V1 ANY * [ e ] = e )
  ( V1 ANY * [ i ] = i )
  ( V1 ANY * [ o ] = o )
  ( V1 ANY * [ u ] = u )
  ( [ a ] ANY * V1 = a )
  ( [ e ] ANY * V1 = e )
  ( [ i ] ANY * V1 = i )
  ( [ o ] ANY * V1 = o )
  ( [ u ] ANY * V1 = u )
  
  ;; We'll only get here when the vowel is in an unstressed word
  ;; two more syllables so don't worry about it yet
  ( [ a ] VC * -  VC * - = a )
  ( [ e ] VC * -  VC * - = e )
  ( [ i ] VC * -  VC * - = i )
  ( [ o ] VC * -  VC * - = o )
  ( [ u ] VC * -  VC * - = u )

  ( [ a ] ANY * - VC * aeo i SIL = a )
  ( [ e ] ANY * - VC * aeo i SIL = e )
  ( [ i ] ANY * - VC * aeo i SIL = i )
  ( [ o ] ANY * - VC * aeo i SIL = o )
  ( [ u ] ANY * - VC * aeo i SIL = u )

  ( [ a ] VC * - VC * VNS SIL = aS )
  ( [ e ] VC * - VC * VNS SIL = eS )
  ( [ o ] VC * - VC * VNS SIL = oS )
  ( [ i ] aeo C * - VC * VNS SIL = i )
  ( [ u ] aeo C * - VC * VNS SIL = u )
  ( aeo [ i ] C * - VC * VNS SIL = i )
  ( aeo [ u ] C * - VC * VNS SIL = u )
  ( [ u ] C * - VC * VNS SIL = uS )
  ( [ i ] C * - VC * VNS SIL = iS )

  ( [ a ] i SIL = aS )
  ( [ e ] i SIL = eS )
  ( [ o ] i SIL = oS )

  ;; stress on previous syllable
  ( - VC * [ a ] VC * VNS SIL = a )
  ( - VC * [ e ] VC * VNS SIL = e )
  ( - VC * [ i ] VC * VNS SIL = i )
  ( - VC * [ o ] VC * VNS SIL = o )
  ( - VC * [ u ] VC * VNS SIL = u )
  ( - VC * [ a ] SIL = a )
  ( - VC * [ e ] SIL = e )
  ( - VC * [ i ] SIL = i )
  ( - VC * [ o ] SIL = o )
  ( - VC * [ u ] SIL = u )

  ;; stress on final syllable
  ( [ a ] VC * SIL = aS )
  ( [ e ] VC * SIL = eS )
  ( [ o ] VC * SIL = oS )
  ( aeo [ i ] VC * SIL = i )
  ( aeo [ u ] VC * SIL = u )
  ( [ i ] aeo VC * SIL = i )
  ( [ u ] aeo VC * SIL = u )
  ( [ i ] VC * SIL = iS )
  ( [ u ] VC * SIL = uS )

  ( [ a ] = a )
  ( [ e ] = e )
  ( [ i ] = i )
  ( [ o ] = o )
  ( [ u ] = u )
  
))

;; ari: Phone -> Phone
(lts.ruleset
 ;; reduce i and u in diphthongs to uW iW
 cmu_spanish_weak_vowels
 (
  (aeo a e o aS eS oS iS uS )
  )
 (
  ;; consonants to themselves
  ( [ b ] = b )
  ( [ d ] = d )
  ( [ ch ] = ch )
  ( [ f ] = f )
  ( [ g ] = g )
  ( [ j ] = j )
  ( [ k ] = k )
  ( [ l ] = l )
  ( [ m ] = m )
  ( [ n ] = n )
  ( [ ny ] = ny )
  ( [ p ] = p )
  ( [ r ] = r )
  ( [ rr ] = rr )
  ( [ s ] = s )
  ( [ t ] = t )
  ( [ th ] = th )
  ( [ y ] = y )
  ( [ - ] = - )
  ;; stressed vowels to themselves
  ( [ aS ] = aS )
  ( [ iS ] = iS )
  ( [ uS ] = uS )
  ( [ eS ] = eS )
  ( [ oS ] = oS )

  ( aeo [ i ] = iW )
  ( [ i ] aeo = iW )
  ( aeo [ u ] = uW )
  ( [ u ] aeo = uW )

  ( [ a ] = a )
  ( [ i ] = i )
  ( [ u ] = u )
  ( [ e ] = e )
  ( [ o ] = o )
))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Postlexical Rules 
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (cmu_es::postlex_rule1 utt)
  "(cmu_es::postlex_ruleS utt)
A postlexical rule form correcting phenomena over word boundaries."
  (mapcar
   (lambda (s)
     ;; do something
     )
   (utt.relation.items utt 'Segment))
   utt)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Lexicon definition
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(lex.create "cmu_es")
(lex.set.phoneset "cmu_es")
(lex.set.lts.method 'cmu_es_lts_function)
;;; If you have a compiled lexicon uncomment this
;(lex.set.compile.file (path-append INST_es_VOX_dir "cmu_es_lex.out"))
(cmu_es_addenda)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Lexicon setup
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (INST_es_VOX::select_lexicon)
  "(INST_es_VOX::select_lexicon)
Set up the lexicon for cmu_es."
  (lex.select "cmu_es")

  ;; Post lexical rules
  (set! postlex_rules_hooks (list cmu_es::postlex_rule1))
)

(define (INST_es_VOX::reset_lexicon)
  "(INST_es_VOX::reset_lexicon)
Reset lexicon information."
  t
)

(provide 'INST_es_VOX_lexicon)








