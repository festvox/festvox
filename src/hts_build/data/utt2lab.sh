#!/bin/tcsh -f
#  ----------------------------------------------------------------  #
#      The HMM-Based Speech Synthesis System (HTS): version 1.1      #
#                        HTS Working Group                           #
#                                                                    #
#                   Department of Computer Science                   #
#                   Nagoya Institute of Technology                   #
#                                and                                 #
#    Interdisciplinary Graduate School of Science and Engineering    #
#                   Tokyo Institute of Technology                    #
#                      Copyright (c) 2001-2003                       #
#                        All Rights Reserved.                        #
#                                                                    #
#  Permission is hereby granted, free of charge, to use and          #
#  distribute this software and its documentation without            #
#  restriction, including without limitation the rights to use,      #
#  copy, modify, merge, publish, distribute, sublicense, and/or      #
#  sell copies of this work, and to permit persons to whom this      #
#  work is furnished to do so, subject to the following conditions:  #
#                                                                    #
#    1. The code must retain the above copyright notice, this list   #
#       of conditions and the following disclaimer.                  #
#                                                                    #
#    2. Any modifications must be clearly marked as such.            #
#                                                                    #     
#  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,   #
#  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM     #
#  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL        #
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT    #
#  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF         #
#  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE     #
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY         #
#  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,   #
#  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS    #
#  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR           #
#  PERFORMANCE OF THIS SOFTWARE.                                     #
#                                                                    #
#  ----------------------------------------------------------------  # 
#    utt2lab.sh  : convert festival utt file into context-dependent  #
#                  label and context-independent segment label file  #
#                  for HMM-based speech synthesis                    #
#                                                                    #
#                                    2003/05/09 by Heiga Zen         #
#  ----------------------------------------------------------------  #

set basedir   = CURRENTDIR 
set estdir    = FESTIVALDIR
set speaker   = SPEAKER
set dumpfeats = $estdir/examples/dumpfeats
set scpdir    = $basedir/scripts

set src  = $basedir/utts
set mono = $basedir/labels/monophone
set full = $basedir/labels/fullcontext

cat <<EOF >! $scpdir/label.feats
;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SEGMENT

;; {p, c, n}.name
    p.name                                                  ;  1 
    name                                                    ;  2 
    n.name                                                  ;  3

;; position in syllable (segment)
    pos_in_syl                                              ;  4

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SYLLABLE

;; {p, c, n}.stress
    R:SylStructure.parent.R:Syllable.p.stress               ;  5
    R:SylStructure.parent.R:Syllable.stress                 ;  6
    R:SylStructure.parent.R:Syllable.n.stress               ;  7

;; {p, c, n}.accent
    R:SylStructure.parent.R:Syllable.p.accented             ; 8
    R:SylStructure.parent.R:Syllable.accented               ; 9
    R:SylStructure.parent.R:Syllable.n.accented             ; 10

;; {p, c, n}.length (segment)
    R:SylStructure.parent.R:Syllable.p.syl_numphones        ; 11
    R:SylStructure.parent.R:Syllable.syl_numphones          ; 12
    R:SylStructure.parent.R:Syllable.n.syl_numphones        ; 13

;; position in word (syllable)
    R:SylStructure.parent.R:Syllable.pos_in_word            ; 14

;; position in phrase (syllable)
    R:SylStructure.parent.R:Syllable.syl_in                 ; 15
    R:SylStructure.parent.R:Syllable.syl_out                ; 16

;; position in phrase (stressed syllable)
    R:SylStructure.parent.R:Syllable.ssyl_in                ; 17
    R:SylStructure.parent.R:Syllable.ssyl_out               ; 18

;; position in phrase (accented syllable)
    R:SylStructure.parent.R:Syllable.asyl_in                ; 19
    R:SylStructure.parent.R:Syllable.asyl_out               ; 20

;; distance from stressed syllable                 
    R:SylStructure.parent.R:Syllable.lisp_distance_to_p_stress   ; 21
    R:SylStructure.parent.R:Syllable.lisp_distance_to_n_stress   ; 22

;; distance to accented syllable (syllable)                 
    R:SylStructure.parent.R:Syllable.lisp_distance_to_p_accent   ; 23
    R:SylStructure.parent.R:Syllable.lisp_distance_to_n_accent   ; 24  

;; name of the vowel of syllable
    R:SylStructure.parent.R:Syllable.syl_vowel              ; 25

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; WORD

;; {p, c, n}.gpos
    R:SylStructure.parent.parent.R:Word.p.gpos              ; 26
    R:SylStructure.parent.parent.R:Word.gpos                ; 27
    R:SylStructure.parent.parent.R:Word.n.gpos              ; 28

;; {p, c, n}.length (syllable)
    R:SylStructure.parent.parent.R:Word.p.word_numsyls      ; 29
    R:SylStructure.parent.parent.R:Word.word_numsyls        ; 30
    R:SylStructure.parent.parent.R:Word.n.word_numsyls      ; 31

;; position in phrase (word)
    R:SylStructure.parent.parent.R:Word.pos_in_phrase       ; 32
    R:SylStructure.parent.parent.R:Word.words_out           ; 33

;; position in phrase (content word)
    R:SylStructure.parent.parent.R:Word.content_words_in    ; 34
    R:SylStructure.parent.parent.R:Word.content_words_out   ; 35

;; distance to content word (word)
    R:SylStructure.parent.parent.R:Word.lisp_distance_to_p_content ; 36
    R:SylStructure.parent.parent.R:Word.lisp_distance_to_n_content ; 37

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; PHRASE

;; {p, c, n}.length (syllable)
    R:SylStructure.parent.parent.R:Phrase.parent.p.lisp_num_syls_in_phrase ; 38
    R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase   ; 39
    R:SylStructure.parent.parent.R:Phrase.parent.n.lisp_num_syls_in_phrase ; 40

;; {p, c, n}.length (word)
    R:SylStructure.parent.parent.R:Phrase.parent.p.lisp_num_words_in_phrase; 41
    R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase  ; 42
    R:SylStructure.parent.parent.R:Phrase.parent.n.lisp_num_words_in_phrase; 43

;; position in major phrase (phrase)
    R:SylStructure.parent.R:Syllable.sub_phrases            ; 44

;; type of end tone of this phrase
    R:SylStructure.parent.parent.R:Phrase.parent.daughtern.R:SylStructure.daughtern.tobi_endtone ; 45

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; UTTERANCE

;; length (syllable)
    lisp_total_syls                                         ; 46

;; length (word)
    lisp_total_words                                        ; 47

;; length (phrase)
    lisp_total_phrases                                      ; 48

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; for "pau"

    p.R:SylStructure.parent.R:Syllable.stress               ; 49
    n.R:SylStructure.parent.R:Syllable.stress               ; 50

    p.R:SylStructure.parent.R:Syllable.accented             ; 51
    n.R:SylStructure.parent.R:Syllable.accented             ; 52

    p.R:SylStructure.parent.R:Syllable.syl_numphones        ; 53
    n.R:SylStructure.parent.R:Syllable.syl_numphones        ; 54

    p.R:SylStructure.parent.parent.R:Word.gpos              ; 55
    n.R:SylStructure.parent.parent.R:Word.gpos              ; 56

    p.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 57
    n.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 58

    p.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase ; 59
    n.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase ; 60

    p.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase ; 61
    n.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase ; 62

;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; additional feature

;; for quinphone

    p.p.name                                                ; 63
    n.n.name                                                ; 64

;; boundary

    segment_start                                           ; 65
    segment_end                                             ; 66

EOF
 
cat <<EOF >! $scpdir/label-full.awk
{
##############################
###  SEGMENT

#  boundary
    printf "%10.0f %10.0f ", 1e7 * \$65, 1e7 * \$66

#  pp.name
    printf "%s",  \$63 == "0" ? "x" : \$63
#  p.name
    printf "^%s", \$1  == "0" ? "x" : \$1
#  c.name
    printf "-%s", \$2
#  n.name
    printf "+%s", \$3  == "0" ? "x" : \$3
#  nn.name
    printf "=%s", \$64 == "0" ? "x" : \$64 

#  position in syllable (segment)
    printf "@"
    printf "%s",  \$2 == "pau" ? "x" : \$4 + 1
    printf "_%s", \$2 == "pau" ? "x" : \$12 - \$4

##############################
###  SYLLABLE

## previous syllable

#  p.stress
    printf "/A:%s", \$2 == "pau" ? \$49 : \$5
#  p.accent
    printf "_%s", \$2 == "pau" ? \$51 : \$8
#  p.length
    printf "_%s", \$2 == "pau" ? \$53 : \$11

## current syllable

#  c.stress
    printf "/B:%s", \$2 == "pau" ? "x" : \$6
#  c.accent
    printf "-%s", \$2 == "pau" ? "x" : \$9
#  c.length
    printf "-%s", \$2 == "pau" ? "x" : \$12

#  position in word (syllable)
    printf "@%s", \$2 == "pau" ? "x" : \$14 + 1
    printf "-%s", \$2 == "pau" ? "x" : \$30 - \$14

#  position in phrase (syllable)
    printf "&%s", \$2 == "pau" ? "x" : \$15 + 1
    printf "-%s", \$2 == "pau" ? "x" : \$16 + 1

#  position in phrase (stressed syllable)
    printf "#%s", \$2 == "pau" ? "x" : \$17 + 1
    printf "-%s", \$2 == "pau" ? "x" : \$18 + 1

#  position in phrase (accented syllable)
    printf  "\$"
    printf "%s", \$2 == "pau" ? "x" : \$19 + 1
    printf "-%s", \$2 == "pau" ? "x" : \$20 + 1

#  distance from stressed syllable
    printf "!%s", \$2 == "pau" ? "x" : \$21
    printf "-%s", \$2 == "pau" ? "x" : \$22

#  distance from accented syllable 
    printf ";%s", \$2 == "pau" ? "x" : \$23
    printf "-%s", \$2 == "pau" ? "x" : \$24

#  name of the vowel of current syllable
    printf "|%s", \$2 == "pau" ? "x" : \$25

## next syllable

#  n.stress
    printf "/C:%s", \$2 == "pau" ? \$50 : \$7
#  n.accent
    printf "+%s", \$2 == "pau" ? \$52 : \$10
#  n.length
    printf "+%s", \$2 == "pau" ? \$54 : \$13

##############################
#  WORD

##################
## previous word

#  p.gpos
    printf "/D:%s", \$2 == "pau" ? \$55 : \$26
#  p.lenght (syllable)
    printf "_%s", \$2 == "pau" ? \$57 : \$29

#################
## current word

#  c.gpos
    printf "/E:%s", \$2 == "pau" ? "x" : \$27
#  c.lenght (syllable)
    printf "+%s", \$2 == "pau" ? "x" : \$30

#  position in phrase (word)
    printf "@%s", \$2 == "pau" ? "x" : \$32 + 1
    printf "+%s", \$2 == "pau" ? "x" : \$33

#  position in phrase (content word)
    printf "&%s", \$2 == "pau" ? "x" : \$34 + 1
    printf "+%s", \$2 == "pau" ? "x" : \$35

#  distance from content word in phrase
    printf "#%s", \$2 == "pau" ? "x" : \$36
    printf "+%s", \$2 == "pau" ? "x" : \$37

##############
## next word

#  n.gpos
    printf "/F:%s", \$2 == "pau" ? \$56 : \$28
#  n.lenghte (syllable)
    printf "_%s", \$2 == "pau" ? \$58 : \$31

##############################
#  PHRASE

####################
## previous phrase

#  length of previous phrase (syllable)
    printf "/G:%s", \$2 == "pau" ? \$59 : \$38

#  length of previous phrase (word)
    printf "_%s"  , \$2 == "pau" ? \$61 : \$41

####################
## current phrase

#  length of current phrase (syllable)
    printf "/H:%s", \$2 == "pau" ? "x" : \$39

#  length of current phrase (word)
    printf "=%s",   \$2 == "pau" ? "x" : \$42

#  position in major phrase (phrase)
    printf "@";
    printf "%s", \$44 + 1
    printf "=%s", \$48 - \$44

#  type of tobi endtone of current phrase
    printf "|%s",  \$45

####################
## next phrase

#  length of next phrase (syllable)
    printf "/I:%s", \$2 == "pau" ? \$60 : \$40

#  length of next phrase (word)
    printf "=%s",   \$2 == "pau" ? \$62 : \$43

##############################
#  UTTERANCE

#  length (syllable)
    printf "/J:%s", \$46

#  length (word)
    printf "+%s", \$47

#  length (phrase)
    printf "-%s", \$48

    printf "\n"
}
EOF



cat <<EOF >! $scpdir/label-mono.awk
{
##############################
###  SEGMENT

#  boundary
    printf "%10.0f %10.0f ", 1e7 * \$65, 1e7 * \$66

#  c.name
    printf "%s", \$2
    printf "\n"
}
EOF




foreach utt ($src/*.utt)

$dumpfeats -eval $scpdir/extra_feats.scm -relation Segment -feats $scpdir/label.feats \
    -output %s.tmp $utt

awk -f $scpdir/label-full.awk $utt:t:r.tmp >! $full/$speaker/$utt:t:r.lab
awk -f $scpdir/label-mono.awk $utt:t:r.tmp >! $mono/$speaker/$utt:t:r.lab

'rm' $utt:t:r.tmp

end
