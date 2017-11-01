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
;;; Lexicon, LTS and Postlexical rules for Mexican Spanish (cmu_es)
;;;
;;; Taken from old Spanish voice and modified by Ariadna Font Llitjos 
;;; (aria@cs) to make it consistent with Mexican Spanish.
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
(lex.add.entry '("c" nn (((s e) 0)))) ;; ari: in Spain (th e)
(lex.add.entry '("ç" nn (((s e) 0)((s e) 0)((d i1) 1)((y a) 0)))) ;; ari
(lex.add.entry '("d" nn (((d e) 0))))
;(lex.add.entry '("e" nn (((e) 0))))
(lex.add.entry '("f" nn (((e1) 1)((f e) 0))))
(lex.add.entry '("g" nn (((j e) 0)))) 
(lex.add.entry '("h" nn (((a1) 1)((ch e) 0))))
;(lex.add.entry '("i" nn (((i) 0))))
(lex.add.entry '("j" nn (((j o1) 1)((t a) 0)))) ;; ari: replaced /x/ by /j/
(lex.add.entry '("k" nn (((k a) 0))))
(lex.add.entry '("l" nn (((e1) 1)((l e) 0))))
(lex.add.entry '("m" nn (((e1) 1)((m e) 0))))
(lex.add.entry '("n" nn (((e1) 1)((n e) 0))))
(lex.add.entry '("~n" nn (((e1) 1)((ny e) 0))))
(lex.add.entry '("ñ" nn (((e1) 1)((ny e) 0))))
;(lex.add.entry '("o" nn (((o) 0))))
(lex.add.entry '("p" nn (((p e) 0))))
(lex.add.entry '("q" nn (((k u) 0))))
(lex.add.entry '("r" nn (((e1) 1)((rr e) 0))))
(lex.add.entry '("s" nn (((e1) 1) ((s e) 0))))
(lex.add.entry '("t" nn (((t e) 0))))
;(lex.add.entry '("u" nn (((u) 0))))
(lex.add.entry '("v" nn (((u1) 1)((b e) 0))))
(lex.add.entry '("w" nn (((u) 0) ((b e) 0) ((d o1) 1) ((b l e) 0))))
(lex.add.entry '("x" nn (((e1) 1)((k i s) 0))))

; ari: Alan should be uncomment the 'y'?
;(lex.add.entry '("y" nn (((i) 0)((g r i e1) 1))((g a) 0))) 
(lex.add.entry '("z" nn (((s e1) 1)((t a) 0)))) ;; ari: /th/ for Spain Spanish
;(lex.add.entry '("á" nn (((a) 0))))
;(lex.add.entry '("é" nn (((e) 0))))
;(lex.add.entry '("í" nn (((i) 0))))
;(lex.add.entry '("ó" nn (((o) 0))))
;(lex.add.entry '("ú" nn (((u) 0))))
;(lex.add.entry '("ü" nn (((u) 0))))

;; ari: 
;; Need to include all the Nahuatl words as a lexical entry, since in such
;; words 'x' is pronounces /j/ instead of /ks/
(lex.add.entry '("méxico" n (((m e1) 1) ((j i) 0) ((k o) 0))))
(lex.add.entry '("mexico" n (((m e1) 1) ((j i) 0) ((k o) 0))))
(lex.add.entry '("mexicano" n (((m e) 0) ((j i) 0) ((k a1) 1) ((n o) 0))))
(lex.add.entry '("mexicana" n (((m e) 0) ((j i) 0) ((k a1) 1) ((n a) 0))))
(lex.add.entry '("oaxaca" n (((u a) 0) ((j a1) 1) ((k a) 0)))) 


;;; Abreviations
;;; months
;(lex.add.entry '("en." nil (((e) 0) ((n e1) 1) ((r o) 0))))
(lex.add.entry '("feb" nil (((f e) 0) ((b r e1) 1) ((r o) 0))))
(lex.add.entry '("abr" nil (((a) 0) ((b r i1 l) 1))))
(lex.add.entry '("jun" nil (((j u1) 1) ((n i0 o) 0))))
(lex.add.entry '("jul" nil (((j u1) 1) ((l i0 o) 0))))
(lex.add.entry '("ag" nil (((a) 0) ((g o1 s) 1) ((t o) 0))))
(lex.add.entry '("sept" nil (((s e) 0) ((t i0 e1 m) 1) ((b r e) 0))))
(lex.add.entry '("oct" nil (((o k) 0) ((t u1) 1) ((b r e) 0))))
(lex.add.entry '("nov" nil (((n o) 0) ((v i0 e1 m) 1) ((b r e) 0))))
(lex.add.entry '("dic" nil (((d i) 0) ((th i0 e1 m) 1) ((b r e) 0))))
(lex.add.entry '("Feb" nil (((f e) 0) ((b r e1) 1) ((r o) 0))))
(lex.add.entry '("Abr" nil (((a) 0) ((b r i1 l) 1))))
(lex.add.entry '("Jun" nil (((j u1) 1) ((n i0 o) 0))))
(lex.add.entry '("Jul" nil (((j u1) 1) ((l i0 o) 0))))
(lex.add.entry '("Ag" nil (((a) 0) ((g o1 s) 1) ((t o) 0))))
(lex.add.entry '("Sept" nil (((s e) 0) ((t i0 e1 m) 1) ((b r e) 0))))
(lex.add.entry '("Oct" nil (((o k) 0) ((t u1) 1) ((b r e) 0))))
(lex.add.entry '("Nov" nil (((n o) 0) ((v i0 e1 m) 1) ((b r e) 0))))
(lex.add.entry '("Dic" nil (((d i) 0) ((th i0 e1 m) 1) ((b r e) 0))))
;;; days of the week
;; most common abreviations: ls ma mie ju vi sab dom ?


;;; Symbols ...
(lex.add.entry 
 '("*" n (((a s) 0) ((t e) 0) ((r i1 s) 1)  ((k o) 0))))
(lex.add.entry 
 '("%" n (((p o r) 0) ((s i e1 n) 1) ((t o) 0))))
;; ari: in English "ampersand", but I there seems to be no translation into 
;; Spanish, other than it's just moslty pronounced "an"
(lex.add.entry 
 '("&" n (((a1 m) 1) ((p e r) 0) ((s a n) 0)))) 
(lex.add.entry 
 '("&" punc (((a1 n) 1)))) ;; ari
(lex.add.entry 
 '("$" n (((d o1) 1) ((l a r) 0))))
(lex.add.entry 
 '("#" n (((a l) 0) ((m u a) 0) ((d i1) 1) ((y a) 0))))
(lex.add.entry 
 '("@" n (((a) 0) ((rr o1) 1) ((b a) 0))))
;; ari: Alan, what do all these symbols mean?
(lex.add.entry 
 '("+" n (((m a s) 0)) ((pos "K7%" "OA%" "T-%"))))
(lex.add.entry 
 '("^" n (((k a1) 1) ((r e t) 0)) ((pos "K6$")))) ;; ari: also "and" or "triangulito" (Mex.: testa triangular)
(lex.add.entry 
 '("~" n (((t i1 l) 1) ((d e) 0)) ((pos "K6$"))))
(lex.add.entry 
 '("=" n (((i) 0) ((g u a1 l) 1))))
;(lex.add.entry 
; '("/" n (((e1 n ) 1) ((t r e) 0))))  ;; division, etc.
;; ari: RAE says / should be called 'barra', and I guess it's more general than
;;      'entre' which in many cases it will be just wrong: 50 km/hora
(lex.add.entry 
 '("/" n (((b a1) 1) ((rr a) 1))))
(lex.add.entry 
 '("\\" n (((b a1) 1) ((rr a) 1))))
(lex.add.entry 
 '("_" n (((s u b) 0) ((rr a) 0) ((y a1) 1) ((d o) 0)) ))
(lex.add.entry 
 '("|" n (((b a1) 1) ((rr a) 0))))
(lex.add.entry 
 '(">" n ((( m a ) 0) ((y o1 r) 1) ((k e) 0))))
(lex.add.entry 
 '("<" n ((( m e ) 0) ((n o1 r) 1) ((k e) 0))))
(lex.add.entry 
 '("[" n ((( a) 0) ((b r i1 r) 1) ((k o r) 0)((ch e1) 1)((t e) 0))))
(lex.add.entry 
 '("]" n (((c e) 0) ((rr a1 r) 1) ((k o r) 0)((ch e1) 1)((t e) 0))))
(lex.add.entry 
 '(" " n (((e s) 0)((p a1) 1)((s i o) 0))))
(lex.add.entry 
 '("\t" n (((t a) 0) ((b u) 0) ((l a) 0) ((d o1 r) 1))))
(lex.add.entry 
 '("\n" n (((n u e1) 1) ((b a) 0)((l i) 1) ((n e a) 0))))

;; Basic punctuation must be in with nil pronunciation
(lex.add.entry '("." punc nil))
(lex.add.entry '("." nn (((p u1 n) 1) ((t o) 0))))
(lex.add.entry '("'" punc nil))
(lex.add.entry '("'" nn (((a) 0) ((p o1 s) 1) ((t r o) 0) ((f o) 0)))) ;; ari
(lex.add.entry '(":" punc nil))
(lex.add.entry '(":" nn (((d o s) 0) ((p u1 n) 1) ((t o s) 0)))) ;; ari
(lex.add.entry '(";" punc nil))
(lex.add.entry '(";" nn (((p u1 n) 1) ((t o i0) 0) ((k o1) 1) ((m a) 0)))) ;;ari
(lex.add.entry '("," punc nil))
(lex.add.entry '("," nn (((k o1) 1) ((m a) 0)))) ;; ari
(lex.add.entry '("-" punc nil))
(lex.add.entry '("-" nn (((g i0 o1 n) 1)))) ;; ari
(lex.add.entry '("\"" punc nil))
(lex.add.entry '("\"" nn (((k o) 0) ((m i1) 1) ((y a s) 0)))) ;; ari 
(lex.add.entry '("`" punc nil)) 
(lex.add.entry '("?" punc nil)) 
(lex.add.entry '("?" nn (((i n) 0) ((t e) 0) ((rr o) 0) ((g a1 n) 1) ((t e) 0)))) ;; ari
(lex.add.entry '("!" punc nil))
(lex.add.entry '("!" nn (((e s k l a) 0) ((m a) 0) ((s i1 o n) 1)))) ;; ari
;; ari: should add the ones for ? and ! upside down
) ;; end of addenda

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Down cases with accents
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(lts.ruleset
 spanish_downcase
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
    (if (lts.in.alphabet word 'spanish_downcase)
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
  (lts.apply word 'spanish_downcase))

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
       (if (string-matches (car p) ".*1")
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
 ( "'" V* C* u [ i ] DNSR # = i ) 
 ( AEIOUt V* C* u [ i ] DNSR # = i )   ;; $$$  ~n and so, what will do?
 ( u [ i ] DNSR # = i1 ) 
 ( "'" V* C* u [ i ] d V # = i ) 
 ( AEIOUt V* C* u [ i ] d V # = i ) 
 ( u [ i ] d AEIOUt # = i )   ;; not sure about these two
 ( u [ i ] d V # = i1 )       ;; 


 ( [ a ] = a )
 ( [ e ] = e )
 ( [ i ] = i )
 ( [ o ] = o )
 ( [ u ] = u )
 ( [ "'" a ] = a1 )
 ( [ "'" e ] = e1 )
 ( [ "'" i ] = i1 )
 ( [ "'" o ] = o1 )
 ( [ "'" u ] = u1 )
 ( [ á ] = a1 )
 ( [ é ] = e1 )
 ( [ í ] = i1 )
 ( [ ó ] = o1 )
 ( [ ú ] = u1 )
 ( [ ":" u ] = u )      ; umlaut (u dieresis) (should not happen, only with g, and already removed)
 ( [ "\"" u ] = u )  
 ( [ ü ] = u ) 

 ( [ b ] = b )
 ( [ v ] = b );; ari: in Spain: ( [ v ] = v )
 ( [ c ] "'" EI = s ) ;; ari: in Spain it would be th  
 ( [ c ] EI = s ) ;; ari: in Spain it would be th (ex: cebolla)
 ( [ c h ] = ch )
 ( [ c ] = k )
 ( [ d ] = d )
 ( [ f ] = f )
 ( [ g ] "'" EI = j ) ;; ari: previously /x/
 ( [ g ] EI = j ) ;; ari: previously /x/
 ( [ g u ] "'" EI = g )
 ( [ g u ] EI = g )

 ( [ g ":" u ] EI = g u )      ; umlaut (u dieresis)
 ( [ g ":" u ] "'" EI = g u ) 
 ( [ g "\"" u ] EI = g u )  
 ( [ g "\"" u ] "'" EI = g u ) 
 ( [ g ü ] EI = g u ) 
 ( [ g ü ] "'" EI = g u ) 

 ( [ g ] = g )
 ( [ h ] =  ) ;; ari: in Mexico sometime 'h' will be pronounced /g/ ex: huevo
;; ari: not sure where to include that... add huevo in the lexicon?
 ( [ j ] = j )
 ( [ k ] = k )
 ( [ l l ] # = l ) ;; ari: palabra acabada en 'll'
 ( [ l l ] = y ) ;; ari: ye'ismo 
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
 ( # [ r ] = rr )
 ( LNS [ r ] = rr )
 ( [ r ] = r )
 ( [ s ] = s )
 ( # [ s ] C = e s )
 ( # [ s ] "'" C = e s )
;; ( # [ s ] ":" C = e s )  ;; ari: don't know what an umlaut is supposed to do with a consonant
;; ( # [ s ] "\"" C = e s )
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
 ( [ x ] # = k s ) ;; ari: relax  (what about Marx?)
 ( [ x ] C = s ) ;; ari ex: explicar, excavar, exclamacion, extra, exponer, 
 ( # [ x ] = s ) ;; ari ex: xilograf'ia, xilofono
 ( [ x ] = s ) ;; ari: default different than in Spain 
 ( [ y ] # = i )
 ( [ y ] C = i )
 ( [ y ] "'" C = i )
;; ( [ y ] ":" C = i ) ; don't know what an umlaut is supposed to do with a consonant
;; ( [ y ] "\"" C = i )
 ( [ y ] = y ) ;; ari ex: ya, yo, collar, llorar 
 ( [ z ] = s )  ;; ari: in Spain it would be th

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

(lts.ruleset
   cmu_spanish_syl
   (  (V a1 i1 u1 e1 o1 a i u e o )
      (IUT i1 u1 )
      (C b ch d f g j k l m n ny p r rr s t y)
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
      ( [ i a1 i ] = i a1 i )  
      ( [ i a1 u ] = i a1 u )  
      ( [ u a1 i ] = u a1 i )  
      ( [ u a1 u ] = u a1 u )  
      ( [ i e1 i ] = i e1 i )  
      ( [ i e1 u ] = i e1 u )  
      ( [ u e1 i ] = u e1 i )  
      ( [ u e1 u ] = u e1 u )  
      ( [ i o1 i ] = i o1 i )  
      ( [ i o1 u ] = i o1 u )  
      ( [ u o1 i ] = u o1 i )  
      ( [ u o1 u ] = u o1 u )  

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
      ( IUT [ i a1 ]  = - i a1 )
      ( IUT [ i e1 ]  = - i e1 )
      ( IUT [ i o1 ]  = - i o1 )
      ( IUT [ u a1 ]  = - u a1 )
      ( IUT [ u e1 ]  = - u e1 )
      ( IUT [ u o1 ]  = - u o1 )
      ( IUT [ a1 i ]  = - a1 i )
      ( IUT [ e1 i ]  = - e1 i )
      ( IUT [ o1 i ]  = - o1 i )
      ( IUT [ a1 u ]  = - a1 u )
      ( IUT [ e1 u ]  = - e1 u )
      ( IUT [ o1 u ]  = - o1 u )
      ( IUT [ i u1 ]  = - i u1 )
      ( IUT [ u i1 ]  = - u i1 )

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
      ( [ i a1 ]  = i a1 )
      ( [ i e1 ]  = i e1 )
      ( [ i o1 ]  = i o1 )
      ( [ u a1 ]  = u a1 )
      ( [ u e1 ]  = u e1 )
      ( [ u o1 ]  = u o1 )
      ( [ a1 i ]  = a1 i )
      ( [ e1 i ]  = e1 i )
      ( [ o1 i ]  = o1 i )
      ( [ a1 u ]  = a1 u )
      ( [ e1 u ]  = e1 u )
      ( [ o1 u ]  = o1 u )
      ( [ u1 i ]  = u1 i )
      ( [ i1 u ]  = i1 u )
    
      ;; Vowels preceeded by vowels are syllable breaks
      ;; triptongs and diptongs are dealt with above
      ( V [ a ]  = - a )
      ( V [ i ]  = - i )
      ( V [ u ]  = - u )
      ( V [ e ]  = - e )
      ( V [ o ]  = - o )
      ( V [ a1 ]  = - a1 )
      ( V [ e1 ]  = - e1 )
      ( V [ i1 ]  = - i1 )
      ( V [ o1 ]  = - o1 )
      ( V [ u1 ]  = - u1 )

      ;; If any consonant is followed by a vowel and there is a vowel
      ;; before it, its a syl break
      ;; the consonant cluster are dealt with above
      ( V C * [ b ] V = - b )
      ( V C * [ v ] V = - v )
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

      ;; Catch all consonants on their own (at end of word)
      ;; and vowels not preceded by vowels are just written as it
      ( [ b ] = b )
      ( [ v ] = v )
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
      ( [ x ] = x ) 
      ( [ a ]  =  a )
      ( [ i ]  =  i )
      ( [ u ]  =  u )
      ( [ e ]  =  e )
      ( [ o ]  =  o )
      ( [ a1 ]  =  a1 )
      ( [ i1 ]  =  i1 )
      ( [ u1 ]  =  u1 )
      ( [ e1 ]  =  e1 )
      ( [ o1 ]  =  o1 )
   )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Stress assignment in unstress words by rewrite rules
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(lts.ruleset
 ;; Assign stress to a vowel when non-exists
 cmu_spanish.stress
 (
  (UV a i u e o)
  (V a1 i1 u1 e1 o1  a i u e o)
  (V1 a1 i1 u1 e1 o1)
  (VNS n s a i u e o)
  (C b ch d f g j k l m n ny p r rr s t v x y)
  (VC v b ch d f g j k l m n ny p r rr s t x y a1 i1 u1 e1 o1  a i u e o)
  (ANY v b ch d f g j k l m n ny p r rr s t x y- a1 i1 u1 e1 o1  a i u e o)
  (notNS v b ch d f g j k l m ny p r rr t x y)
  (iu i u )
  (aeo a e o)
  )
 (
  ;; consonants to themselves
  ( [ b ] = b )
  ( [ v ] = v )
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
  ( [ y ] = y )
  ( [ x ] = x )
  ( [ - ] = - )
  ;; stressed vowels to themselves
  ( [ a1 ] = a1 )
  ( [ i1 ] = i1 )
  ( [ u1 ] = u1 )
  ( [ e1 ] = e1 )
  ( [ o1 ] = o1 )

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

  ( [ a ] ANY * - VC * aeo i # = a )
  ( [ e ] ANY * - VC * aeo i # = e )
  ( [ i ] ANY * - VC * aeo i # = i )
  ( [ o ] ANY * - VC * aeo i # = o )
  ( [ u ] ANY * - VC * aeo i # = u )

  ( [ a ] VC * - VC * VNS # = a1 )
  ( [ e ] VC * - VC * VNS # = e1 )
  ( [ o ] VC * - VC * VNS # = o1 )
  ( [ i ] aeo C * - VC * VNS # = i )
  ( [ u ] aeo C * - VC * VNS # = u )
  ( aeo [ i ] C * - VC * VNS # = i )
  ( aeo [ u ] C * - VC * VNS # = u )
  ( [ u ] C * - VC * VNS # = u1 )
  ( [ i ] C * - VC * VNS # = i1 )

  ( [ a ] i # = a1 )
  ( [ e ] i # = e1 )
  ( [ o ] i # = o1 )

  ;; stress on previous syllable
  ( - VC * [ a ] VC * VNS # = a )
  ( - VC * [ e ] VC * VNS # = e )
  ( - VC * [ i ] VC * VNS # = i )
  ( - VC * [ o ] VC * VNS # = o )
  ( - VC * [ u ] VC * VNS # = u )
  ( - VC * [ a ] # = a )
  ( - VC * [ e ] # = e )
  ( - VC * [ i ] # = i )
  ( - VC * [ o ] # = o )
  ( - VC * [ u ] # = u )

  ;; stress on final syllable
  ( [ a ] VC * # = a1 )
  ( [ e ] VC * # = e1 )
  ( [ o ] VC * # = o1 )
  ( aeo [ i ] VC * # = i )
  ( aeo [ u ] VC * # = u )
  ( [ i ] aeo VC * # = i )
  ( [ u ] aeo VC * # = u )
  ( [ i ] VC * # = i1 )
  ( [ u ] VC * # = u1 )

  ( [ a ] = a )
  ( [ e ] = e )
  ( [ i ] = i )
  ( [ o ] = o )
  ( [ u ] = u )
  
))

(lts.ruleset
 ;; reduce i and u in diphthongs to u0 i0
 cmu_spanish_weak_vowels
 (
  (aeo a e o a1 e1 o1 i1 u1 )
  )
 (
  ;; consonants to themselves
  ( [ b ] = b )
  ( [ v ] = v )
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
  ( [ y ] = y )
  ( [ x ] = x )
  ( [ - ] = - )
  ;; stressed vowels to themselves
  ( [ a1 ] = a1 )
  ( [ i1 ] = i1 )
  ( [ u1 ] = u1 )
  ( [ e1 ] = e1 )
  ( [ o1 ] = o1 )

  ( aeo [ i ] = i0 )
  ( [ i ] aeo = i0 )
  ( aeo [ u ] = u0 )
  ( [ u ] aeo = u0 )

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
  "(cmu_es::postlex_rule1 utt)
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








