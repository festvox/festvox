
                  "Building Voices in Festival"
     Alan W Black (awb@cs.cmu.edu) and see ACKNOWLEDGEMENTS
                      http://www.festvox.org

For full details about voice building see the document itself

    http://festvox.org/bsv/

The included documentation, scripts and examples should be sufficient
for an interested person to build their own synthetic voices in
currently supported languages or new languages in the University of
Edinburgh's Festival Speech Synthesis System.  The quality of the
result depends much on the time and skill of the builder.  For English
it may be possible to build a new voice in a couple of days work, a
new language may take months or years to build.  It should be noted
that even the best voices in Festival (or any other speech synthesis
system for that matter) are still nowhere near perfect quality.

This distribution includes

    Support for designing, recording and autolabelling statistical parametric
        synthesis voices
    Support for designing, recording and autolabelling diphone databases
    Support for designing, recording and autolabelling unit selection dbs
    Building simple limited domain synthesis engines
    Support for building rule driven and data driven prosody models
       (duration, intonation and phrasing)
    Support for building rule driven and data driven text analysis
    Lexicon and building Letter to Sound rule support
    Predefined scripts for building new US (and UK) English voices
    Predefined scripts for building grapheme(++) voices for any language
    Scripts for designing and selecting prompts to record for
       arbitrary languages

New in 2.8

    https://github.com/festvox/festival/
    Grapheme built voices can be converted to .flitevox files for android
    Database size reduction for random forest clustergen voices
    Random Forests for F0 prediction too
    18 English voices, and 13 Indic voices

New in 2.7

    Random forest models building for spectrum and duration in clustergen
    Grapheme based synthesizers (with specific support for large number
      of unicode writing systems)
    Clustergen state and stop value optimization
    Wavesurfer label support
    SPAM F0 support
    Phrase break support
    Support for SPTK's mgc parameterization

New in 2.3
    
    Support for cygwin tools under Windows
    Substantially improved CLUSTERGEN support with mlpg and mlsb

WARNING
-------

This is not a pointy/clicky plug and play program to build new voices.
It is instructions with discussion on the problems and an attempt to
document the expertise we have gained in building other voices.
Although we have tried to automate the task as much as possible this
is no substitute for careful correction and understanding of the
processes involved.  There are significant pointers into the
literature throughout the document that allow for more detailed study
and further reading.

REQUIREMENTS
------------

A Unix Machine

    although there is nothing inheritantly Unix about the scripts, no
    attempt has yet been made about porting this to other platforms
    
Festival and Speech Tools

    This uses speech tools programs and festival itself at various
    stages in builidng voices as well as (of course) for the final
    voices.  Festival and the Edinburgh Speech Tools are available from
    
       http://www.cstr.ed.ac.uk/projects/festival/
       
    or
    
       http://www.festvox.org/festival

    or

       https://github.com/festvox
       
    It is recommended that you compile your own versions of these
    as you will need the libraries and include files to build some
    programs in this festvox.
    
Wavesurfer

    To display waveforms, spectragrams and phoneme labels.
    
Patience and understanding

    Building a new voice is a lot of work, and something will probably
    go wrong which may require the repetition of some long boring and
    tedious process.  Even with lots of care a new voice still might 
    just not work.  In distributing this document we hope to increase the
    basic knowledge of synthesis out there and hopefully find people 
    who can improve on this making the processing easier and more reliable
    in the future.

INSTALLATION
-----------

You must have the Edinburgh Speech Tools and Festival instllation
before you can build the tools in the festvox distribution.

Unpack festvox-2.8-release.tar.gz or clone it from github

    git clone https://github.com/festvox/festvox
    cd festvox
    ./configure
    make

The configuration basically tries to find your version of 
the Edinburgh Speech Tools and uses its configuration to set
compiler type etc.  So you must have that installed.  If configure
fails try expliciting setting your ESTDIR environment variable
to point ot your compiled version of the Speech Tools.

A pre-generated version of the document in html and postscript are
distributed in the html/ directory

If you need to build the document itself, you will need a working
version of the docbook tools, which may (or may not) already
be installed on your system
    
To build the documenation
   
    cd docbook
    make doc

Note that even if the documentation build fails you can still use all
the scripts and programs.  

To use the scripts and programs in the festvox distribution each
user is expected to have the environment variables ESTDIR and
FESTVOXDIR set for example as (if you use bash, zsh, ksh or sh)

    export ESTDIR=/home/awb/projects/speech_tools
    export FESTVOXDIR=/home/awb/projects/festvox
    export FLITEDIR=/home/awb/projects/flite
    export SPTKDIR=/home/awb/projects/SPTK

Or if you use csh or tcsh

    setenv ESTDIR /home/awb/projects/speech_tools
    setenv FESTVOXDIR /home/awb/projects/festvox
    setenv FLITEDIR /home/awb/projects/flite
    setenv SPTKDIR /home/awb/projects/SPTK

Remember to set these to where *your* installations are, not *ours*.

