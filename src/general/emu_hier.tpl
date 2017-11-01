level Utterance
level Phrase      	Utterance               
level Word      	Phrase
level Syllable		Word                    
level Phoneme      	Syllable

!label Syllable 	Pitch_Accent 	

labfile Phoneme :format ESPS :type SEGMENT :mark END :extension lab :time-factor 1000
labfile Syllable :format ESPS :type SEGMENT :mark END :extension syl :time-factor 1000
labfile Word :format ESPS :type SEGMENT :mark END :extension wrd :time-factor 1000
labfile Phrase :format ESPS :type SEGMENT :mark END :extension phr :time-factor 1000

! location of files, here we just look in the current directory, 
! modify this if you install this template file somewhere other than
! in the database directory
path phr phr
path wrd wrd
path syl syl
path lab lab
path hlb emu/lab_hlb
path wav wav

! definition of associations between tracks and file extensions
track samples	wav

!set HierarchyViewLevels Intonational Intermediate Word Syllable Phonetic
set SignalDisplayLevels Phrase Word Syllable Phoneme

set PrimaryExtension lab
! set LabelTracks samples
