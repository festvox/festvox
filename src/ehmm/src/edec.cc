/*************************************************************************/
/*                                                                       */
/*                     Carnegie Mellon University                        */
/*                         Copyright (c) 2005                            */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*                                                                       */
/*            Authors: Kishore Prahallad                                 */
/*            Email:   skishore@cs.cmu.edu                               */
/*            Date Created: 05/25/2005                                   */
/*            Last Modified: 05/25/2005                                  */
/*            Purpose: A state class for HMM implementation              */
/*                                                                       */
/*       Additional Documentation added by Prasanna in May 2015          */
/*                                                                       */
/*************************************************************************/

#include <string.h>

#include "./header.h"
#include "./hmmstate.h"
#include "./hmmword.h"

#ifndef FESTVOX_NO_THREADS
#include "./threading.h"
#endif


using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::ios;


wrdC *hwrd; //Word Class defined in hmmword.h
//Note: what Kishore calls words are actually phone(me)s
stC  *hst;  //State Class defined in hmmstate.h
double **trw; //transition weights: Matrix that's [number of words x number of words]

int _tst;
int _now = 0; // number of words
int _dim = 0; // number of dimensions (of MFCCs)
int _noc = 0; // number of connections
int _nog = 0 ; // number of Gaussians

int **tarUL;
int *lenU;
char **wavF;
int _nsen;

int _git;
int _seqF = 0;
int _ndeF = 0;  // non-deterministic ending.... flag
int _pauID;  // 17 July 2005
int _spauID;
char *_labD;  // lab directory...

int _numthreads;

int next_sentence_number_to_process;

ofstream fp_log100;

// Feature Extraction Parameters;
double _shift = 80;
double _sf    = 16000;
double _size = 160;

char *phF, *prmF, *fDir, *tFext, *mF;

#ifndef FESTVOX_NO_THREADS
Mutex *global_queue_mutex;
Mutex *global_logfile_mutex;
Mutex *global_stdout_mutex;
#endif

#ifndef FESTVOX_NO_THREADS
void* ThreadStart(void* userdata);
#endif

void ProcessSentence(int sent_num);
void BuildEnv(char *fnm, wrdC*& hwrd, stC*& hst,
              int& now, int& tst, int& dim, ifstream& fp_md);
void LoadPromptFile(char *fnm, int**& tarUL, int*& lenU, char**&wavF, int& nos);
void LoadFeatFile(char *fnm, double**& feat, int&r, int&c);
void LoadBinaryFeatFile(char *filename, double*** feats,
                        int* num_rows, int* num_cols);
void Emissions(wrdC *hwrd, stC *hst, double**& emt, int& er, int& ec,
               double **ft, int fr, int fc, int *tar, int ltar, int*& stMap);
void PrintMatrix(double **arr, int r, int c, char *nm);
void FillStateTrans(double **arcW, int *tar, int ltar, int *nullI);
void FillWordTrans(double **arcW, int *tar, int ltar, double **trw,
                   int *bI, int *eI, int**& fwM, int**& bwM, int er);
void FillWordTrans_Seq(double **arcW, int *tar, int ltar, double **trw,
                       int *bI, int *eI, int**& fwM, int**& bwM, int er);
int Viterbi(double **emt, int r, int c, double **trp,
            double **alp, double **bet, double *nrmF, int *bI,
            int *eI, int *path, int **fwM, int **bwM, int *nullI,
            int **pbuf, int& lastS);
int Viterbi_Seq(double **emt, int r, int c,
                double **trp, double **alp, double **bet,
                double *nrmF, int *bI, int *eI, int *path,
                int **fwM, int **bwM, int *nullI, int **pbuf, int &lastS);
int Viterbi_Seq_nde(double **emt, int r, int c,
                    double **trp, double **alp, double **bet,
                    double *nrmF, int *bI, int *eI, int *path,
                    int **fwM, int **bwM, int *nullI, int **pbuf, int &lastS);
void ReadWordTran(double **trw, int nw, ifstream& fp_md);
void PostProcess(int *path, int *stMap, int ns, int nt, ofstream& fp_log,
                 char *fnm, int *list, int ltar, int *wrdB);
void Get_FrameRate(char *file, double& shft, double& sf, double& size);
int IsShortPause(char *nm);
int IsSilence(char *nm);
void NewProcess(int *path, int **pubf, int *wrdB,
                int *nullI, int *stMap, stC *hst, int ns, int nt, int lastS);


int main(int argc, char *argv[]) {
  char *mstring1;
  char *mstring2;

#ifndef FESTVOX_NO_THREADS
  if (argc < 11) {
    cout << "Usage: ./a.out "
         << "<ph-list.int> <prompt-file> "
         << "<seq-flag> <feat-dir> <extn> "
         << "<settings-file> <mod-dir> <nde-flag> <labD> <numThreads>" << endl;
    exit(1);
  }
#else
  if (argc < 10) {
    cout << "Usage: ./a.out "
         << "<ph-list.int> <prompt-file> "
         << "<seq-flag> <feat-dir> <extn> "
         << "<settings-file> <mod-dir> <nde-flag> <labD>" << endl;
    exit(1);
  }
#endif

#ifndef FESTVOX_NO_THREADS
  // Initialize global mutex
  global_queue_mutex = new PthreadMutex();
  global_logfile_mutex = new PthreadMutex();
  global_stdout_mutex = new PthreadMutex();

#endif


  Get_FrameRate(argv[6], _shift, _sf, _size); //argv[6] is the settings file


  int max_string_length = strlen(argv[7])+20; //argv[7] is the path to the directory 
                                              //where the HMM models are. 


  //The next three lines basically construct the filename with the full path
  //The log101.txt file seems to contain a list of filenames
  mstring1 = new char[max_string_length];
  snprintf(mstring1, max_string_length, "%s/%s", argv[7], "log101.txt");
  fp_log100.open(mstring1, ios::out);

  phF  = argv[1]; // ehmm/etc/ph_list.int file that lists all the phonemes and
                  // how many states they each have

  prmF = argv[2]; // ehmm/etc/txt.phseq.data.int file that lists the HMM states
                  // in each utterance

  _seqF = atoi(argv[3]); // Sequential flag. (Turns off ergodic transitions).
  _ndeF = atoi(argv[8]); // NDE flag. No idea what NDE is ??
  _labD = argv[9];       // Directory where the label files should be written out to.

  _numthreads = 1;

#ifndef FESTVOX_NO_THREADS
  if (argc > 10)
    _numthreads = atoi(argv[10]);
#endif

  // cout << "Sequential Processing Flag: " << _seqF << endl;
  // cout << "non-deterministic Processing Flag: " << _ndeF << endl;

  fDir = argv[4]; // Directory with the feature files (binfeats)
  tFext = argv[5];// File extension of the files in the above mentioned directory

  // cout << "Feature Path: " << fDir << endl;
  // cout << "Feature exten: " << tFext << endl;


  // The next few lines read in the model101.txt file from the ehmm/mod directory.
  // This file contains all the HMMs that were trained in the bw step.
  mstring2 = new char[max_string_length]; 
  snprintf(mstring2, max_string_length, "%s/%s", argv[7], "model101.txt");
  mF = mstring2;
  FileExist(mF);
  ifstream fp_md;
  fp_md.open(mF, ios::in);



  // This function reads in the list of phonemes, the number of states associated
  // with each phoneme, the HMM model parameters, etc..
  BuildEnv(phF, hwrd, hst, _now, _tst, _dim, fp_md);
  // Arguments being passed in: 
  // (phF="ehmm/etc/ph_list.int", This is the file described earlier
  // hwrd is an empty HMM word object
  // hst is an empty HMM state object
  // _now is the number of words
  // _tst: no idea
  // _dim: number of dimensions of the MCEPs
  // fp_md: File pointer to the models file


  Alloc2f(trw, _now, _now);  // Allocate 2D array for Transition weights matrix
  ReadWordTran(trw, _now, fp_md);
  // PrintMatrix(trw, _now, _now, "WORD-TRANS");
  fp_md.close();

  // This function reads in the HMM state numbers corresponding to the 
  // text in each utterance of the promptfile. 
  LoadPromptFile(prmF, tarUL, lenU, wavF, _nsen);  // Load Prompt file....
  // Arguments passed in:
  // prmF="ehmm/etc/txt.phseq.data.int"
  // tarUL: ??
  // lenU: Length of utterance?
  // wavF: List of wav file names
  // _nsen: Probably number of sentences

  next_sentence_number_to_process = 0;

#ifndef FESTVOX_NO_THREADS
    Thread** threadList = new Thread*[_numthreads];
    for (int i = 0; i < _numthreads; i++) {
      Thread *newthread   = StartJoinableThread(ThreadStart);
      threadList[i] = newthread;
    }

    for (int i = 0; i < _numthreads; i++) {
      threadList[i]->Join();
      delete threadList[i];
    }
    delete [] threadList;
#else
    for (int i = 0; i < _nsen; i++) {
      next_sentence_number_to_process = i;
      ProcessSentence(next_sentence_number_to_process);
    }
#endif  // FESTVOX_NO_THREADS

#ifndef FESTVOX_NO_THREADS
  delete global_queue_mutex;
  delete global_stdout_mutex;
  delete global_logfile_mutex;
#endif

  Delete2f(trw, _now);
  fp_log100.close();
}

#ifndef FESTVOX_NO_THREADS
void* ThreadStart(void* /*unused*/) {
  int current_sentence_number_to_process;

  while (1) {
    // See if the next sentence number to be processed exists
    {
      // Scope to define a mutex lock
      ScopedLock sl(global_queue_mutex);
      if (next_sentence_number_to_process >= _nsen)
        break;  // We are done. Thread can exit.

      current_sentence_number_to_process = next_sentence_number_to_process;
      next_sentence_number_to_process++;
      // End of mutex lock scope
    }

    // Don't lock this function with the mutex!
    ProcessSentence(current_sentence_number_to_process);
  }
  return NULL;
}

#endif  // FESTVOX_NO_THREADS


void ProcessSentence(int sent_num) {
  int p = sent_num;
  char tF[kNmLimit * 2];

    double **ft;
  int fr;
  int fc;

  double **emt;
  double **arcW;
  int er;
  int ec;

  double **alp;
  double **bet;
  double *nrmF;

  int *bI;
  int *eI;
  int *stMap;
  int *path;

  int **fwM;
  int **bwM;
  int *nullI;
  int **pubf;
  int *wrdB;
  int lastS;

  snprintf(tF, kNmLimit*2, "%s/%s.%s", fDir, wavF[p], tFext);

  {
#ifndef FESTVOX_NO_THREADS
    ScopedLock sl(global_stdout_mutex);
#endif
    cout << "Aligning " << tF << endl;
  }
  // cout << "LOAD FILE....." << endl;
  //LoadFeatFile(tF, ft, fr, fc);
  LoadBinaryFeatFile(tF, &ft, &fr, &fc);
  // tF = filename
  // ft = pointer to the features
  // fr = number of rows
  // fc = number of columns

  // PrintMatrix(ft, fr, fc, "FEAT-FILE");

  // cout << "ENTERING EMISSIONS..." << endl;
  
  // Looks like the purpose of this function is to generate the Emissions matrix
  // 'emt'. This matrix is (number of states x number of MFCC frames). Each entry
  // in the matrix tells you the probability of that particular state emitting 
  // that particular MFCC frame. Since emission probabilities are modeled using
  // GMMs, each entry in the matrix is generated by plugging in the frame into
  // the GMM corresponding to each state.
  Emissions(hwrd, hst, emt, er, ec, ft, fr, fc, tarUL[p], lenU[p], stMap);
  // hwrd : HMM words (phonemes)
  // hst : HMM states
  // emt : Emission matrix
  // er : Number of rows of emission matrix
  // ec : Number of columns of emission matrix
  // ft : Pointer to the feats
  // fr : Number of rows of features
  // fc : Number of columns of features
  // tarUL[p] : Array containing list of phones in utterance 'p'
  // lenU[p] : Number of phones in utterance 'p'
  // stMap : List of states corresponding to this utterance
  

  Alloc2f(arcW, er, er); //arcW: Matrix that's number of states x number of states
  Alloc1d(bI, er);
  Alloc1d(eI, er);
  // bI and eI matrices indicate which state are beginner and enders
  // of the words This info is need to intialize the alpha at 0th
  // column and n-1th for beta
  Alloc1d(nullI, er);


  // cout << "ENTERING STATE TRANSITIONS..." << endl;

  // The 'tar' array in the second argument of this function contains
  // a list of phones in the utterance. This function fills the arcW 
  // matrix with transition probabilites of going from each HMM state
  // in the utterance to any other state. The arcW matrix is probably
  // block-diagonal
  FillStateTrans(arcW, tarUL[p], lenU[p], nullI);
  // arcW: Empty matrix
  // tarUL[p]: Array containing list of phones in utterance 'p'
  // lenU[p]: Number of phones in utterance 'p'
  // nullI: Empty matrix

  if (1 == _seqF) {
    // cout << "ENTERING SEQ. WORD TRANSITIONS..." << endl;

    // This function updates the arcW matrix to include the transition
    // probabilities across phones too. It also generates fwM and bwM
    // matrices that keep track of the forward and backward connections
    // of each state.
    FillWordTrans_Seq(arcW, tarUL[p], lenU[p], trw, bI, eI, fwM, bwM, er);
    // arcW: Transition probabilities matrix
    // tarUL[p]: Array containing list of phones in utterance 'p'
    // lenU[p]: Number of phones in utterance 'p'
    // trw: Transition probability matrix between phones (not states)
    // bI: described earlier
    // eI: described earlier
    // fwM: Matrix that keeps track of forward connections of each state
    // bwM: Matrix that keeps track of backward connections of each state
    // er: Number of rows of emission matrix (number of states)
  } else {
    // cout << "ENTERING ERG. WORD TRANSITIONS..." << endl;
    FillWordTrans(arcW, tarUL[p], lenU[p], trw, bI, eI, fwM, bwM, er);
  }
  // PrintMatrix(arcW, er, er, "ARC-MAT");

  Alloc2f(alp, er, ec);
  Alloc2f(bet, er, ec);
  Alloc1f(nrmF, ec);
  Alloc1d(path, ec);
  Alloc2d(pubf, er, ec);
  Alloc1d(wrdB, ec);

  int nanF = 0;

  if (1 == _seqF) {
    // cout << "ENTERING Viterbi SEQ..." << endl;
    if (0 == _ndeF) {
      // Viterbi of course
      nanF = Viterbi_Seq(emt, er, ec, arcW, alp, bet, nrmF, bI, eI,
                         path, fwM, bwM, nullI, pubf, lastS);
      // emt : Matrix containing the emission probabilities
      // er : Number of rows of emt (number of states)
      // ec : Number of columns of emt (number of MFCC frames)
      // arcW: Transition probabilities between states
      // alp : Alphas. Parameters calculated during Viterbi
      // bet : Empty matrix. Unused. Probably a leftover from the baum-welch code
      // nrmF: Empty vector
      // bI : Matrix indicating beginning of phones
      // eI : Matrix indicating end of phones
      // path: Presumably the vector that tells you the optimal state path
      // fwM : Matrix that keeps track of forward connections of each state
      // bwM : Matrix that keeps track of backward connections of each state
      // nullI: Keeps track of null states (beginning and end of each phone)
      // pubf : Empty matrix
      // lastS : Last state
    } else {
      nanF = Viterbi_Seq_nde(emt, er, ec, arcW, alp, bet, nrmF,
                             bI, eI, path, fwM, bwM, nullI, pubf, lastS);
    }
    // nde stands for non deterministic end..
  } else {
    // cout << "ENTERING Viterbi ERG..." << endl;
    nanF = Viterbi(emt, er, ec, arcW, alp, bet, nrmF, bI, eI,
                   path, fwM, bwM, nullI, pubf, lastS);
  }

  if (1 == nanF) {
#ifndef FESTVOX_NO_THREADS
    ScopedLock sl(global_stdout_mutex);
#endif
    cout << "SKIPPING... " << wavF[p] << " (NAN problem)" << endl;
  } else {
    // cout << "ENTERING NEW PROCESS..." << endl;
    NewProcess(path, pubf, wrdB, nullI, stMap, hst, er, ec, lastS);
    // cout << "ENTERING POST PROCESS..." << endl;
    PostProcess(path, stMap, er, ec, fp_log100, wavF[p],
                tarUL[p], lenU[p], wrdB);
    // path: Viterbi path through the states
    // stMap: List of states corresponding to this utterance
    // er: Number of states
    // ec: Number of MFCC frames
    // fp_log100: List of filenames to run edec on
    // wavF[p]: Filename of current file
    // tarUL[p] : List of phones in current utterance
    // lenU[p]: Number of phones in current utterance
    // wrdB: Word (actually phone) beginning
  }
    Delete2f(ft, fr);
    Delete2f(emt, er);
    Delete2f(arcW, er);
    Delete2f(alp, er);
    Delete2f(bet, er);
    Delete1f(nrmF);
    Delete1d(bI);
    Delete1d(eI);
    Delete1d(stMap);
    Delete1d(path);
    Delete2d(fwM, er);
    Delete2d(bwM, er);
    Delete1d(nullI);
    Delete2d(pubf, er);
    Delete1d(wrdB);
}

// Reads in the settings file and sets variables for shift, sampling freq, etc..
void Get_FrameRate(char *file, double& shft, double& sf, double& size) {
  FileExist(file);

  // WaveDir: /home/awb/data/arctic/cmu_us_bdl_arctic/wav/
  // HeaderBytes: 44
  // SamplingFreq: 16000
  // FrameSize: 160
  // FrameShift: 80
  // Lporder: 12
  // CepsNum: 16
  // FeatDir: ./feat
  // Ext: .wav

  char tstr[kNmLimit];
  double t;
  ifstream fp_in;
  fp_in.open(file, ios::in);

  fp_in >> tstr >> tstr;  // path
  fp_in >> tstr >> t;     // header

  fp_in >> tstr >> sf;    // sf

  fp_in >> tstr >> size;  // size

  fp_in >> tstr >> shft;  // shift

  fp_in.close();
}

void NewProcess(int *path, int **pbuf, int *wrdB, int *nullI, int *stMap,
                stC *hst, int ns, int nt, int lastS) {
  (void) ns;
  int mas, sid, t;
  t = nt - 1;
  mas = -1;

  mas = lastS;  // Choose the last state...

  for (t = nt - 1; t > 0; t--) {
    // cout << "TIME: " << nt-1 << endl;
    while (1 == nullI[mas]) {
      mas = pbuf[mas][t];
    }
    path[t] = mas;
    mas     = pbuf[mas][t];
    sid = stMap[mas];
    if (hst[sid].getbegS() >= 0) {
      wrdB[t] = 1;
    }
  }
  path[0] = mas;
}

// This function writes out the label file based on the path. I didn't 
// read this function very carefully because file I/O is never that
// interesting and not very relevant for my task anyway.
void PostProcess(int *path, int *stMap, int ns, int nt, ofstream& fp_log,
                 char *fnm, int *list, int ltar, int *wrdB) {
  // path: Viterbi path through the states
  // stMap: List of states corresponding to this utterance
  // er: Number of states
  // ec: Number of MFCC frames
  // fp_log100: List of filenames to run edec on
  // wavF[p]: Filename of current file
  // tarUL[p] : List of phones in current utterance
  // lenU[p]: Number of phones in current utterance
  // wrdB: Word (actually phone) beginning

  // int prev = list[0]; //Assign the first word-id of list[0]...
  (void) ns;
  (void) list;
  (void)ltar;

  int sid;
  int wid;
  int s;
  double tim = 0;
  // double pT = 0;
  // char labD[] = "lab/";

  char myfile[kNmLimit];
  char stfile[kNmLimit];

  int pid;
  int ps;
  int pwd = 0;

  ofstream fp_out;
  ofstream fp_st;

  snprintf(myfile, kNmLimit, "%s/%s.lab", _labD, fnm);
  snprintf(stfile, kNmLimit, "%s/%s.sl", _labD, fnm);

  fp_out.open(myfile, ios::out);
  if (!fp_out.is_open()) {
    cout << "Cannot open " << myfile << endl;
    cout << "pls create a directory called: lab/ (in the current folder)\n";
    exit(1);
  }

  fp_st.open(stfile, ios::out);
  fp_out << "#" << endl;
  fp_st << "#" << endl;
  {
#ifndef FESTVOX_NO_THREADS
    ScopedLock sl(global_logfile_mutex);
#endif
    fp_log << fnm << endl;
  }

  s = path[0];
  ps = s;
  pid = stMap[ps];
  pwd = hst[pid].getwid();

  // double crc = (_size - _shift) / _sf;
  double sfT = _shift / _sf;

  for (int t = 0; t < nt; t++) {
    s = path[t];
    sid = stMap[s];
    wid = hst[sid].getwid();

    //  **tim = ((t + 1) * sfT)  + crc;
    tim = t * sfT;

    if (s != ps) {
      fp_st << tim << " 125 " << pid << " " << pwd << " " << ps << endl;
      // " " << t << " " << wrdB[t] << endl;
      ps = s;

      // if (hst[sid].getbegS() == wid) {
      if (1 == wrdB[t]) {
        // begS is an indicator for the word begining, and its value
        // is the word id itself...
        fp_out << tim << " 125 " << pwd << endl;
      }

      pid = sid;
      pwd = wid;
    }
  }

  wid = -1;
  s   = -1;

  if (s != ps) {
    fp_st << tim << " 125 " << pid << " " << pwd << " " << ps << endl;
    ps = s;
    fp_out << tim << " 125 " << pwd << endl;
    pid = sid;
    pwd = wid;
  }

  fp_out.close();
  fp_st.close();
}

// This function reads in the phoneme to phoneme transition probabilities
// that are at the end of the model file. The values seem to be set to be uniform.
// The probabilites are 1/(num of states * num of phonemes)
void ReadWordTran(double **trw, int nw, ifstream& fp_md) {
  // Arguments are:
  // trw: Transition weights matrix
  // nw: Number of words
  // fp_md: File pointer to the model101.txt file
  int tw;
  fp_md >> tw >> tw;

  if ( tw != nw ) {
    cout << "NO of words: do not match: TW: " << tw << " " << nw << endl;
    exit(1);
  }

  for (int i = 0; i < nw; i++) {
    for (int j = 0; j < nw; j++) {
      fp_md >> trw[i][j];
    }
  }
}

// Initializes the setup
void BuildEnv(char *fnm, wrdC*& hwrd, stC*& hst, int& now,
              int& tst, int& dim, ifstream& fp_md) {
  // Arguments being passed in: 
  // fnm="ehmm/etc/ph_list.int", This is the file described earlier
  // hwrd is an empty HMM word object
  // hst is an empty HMM state object
  // now is the number of words
  // tst: no idea (possibly temp string)
  // dim: number of dimensions of the MCEPs
  // fp_md: File pointer to the models file

  char tstr[kNmLimit];

  FileExist(fnm);

  ifstream fp_in;
  fp_in.open(fnm, ios::in);

  // Read tstr and no-of-words
  fp_in >> tstr >> now;

  // Read tstr and no-of-states
  fp_in >> tstr >> tst;

  // Read tstr and feature dimension
  fp_in >> tstr >> dim;
  // cout << tstr << " " << dim << endl;

  hwrd = new wrdC[now];
  hst  = new stC[tst];

  char nm[kNmLimit]; //name of phone
  int  nst; //number of states
  int  bst; //begin state
  int  est; //end state
  int  wid = 0; //word id
  int  st  = 0; //state id
  int  noc; //number of connections
  int  nog; //number of Gaussians

  for (int i = 0; i < now; i++) {
    // Though noc & nog are read here, they are actually read from the
    // fp_md (model file)

    fp_in >> wid >> nm >> nst >> noc >> nog;

    bst = st;
    est = st + (nst - 1);

    hwrd[wid].init(bst, est, nst, wid, nm);

    // Get SIL ID and short pause id from here....
    if (IsSilence(nm)) {
      _pauID  = wid;
    } else if (IsShortPause(nm)) {
      _spauID = wid;
    }
    st = st + nst;
  }
  fp_in.close();

  for (int i = 0; i < tst; i++) {
    hst[i].ReadModel(fp_md);
  }
}

// This function reads in the HMM state numbers corresponding to the 
// text in each utterance of the promptfile. 
void LoadPromptFile(char *fnm, int**& tarUL, int*& lenU,
                    char**&wavF, int& nos) {
  // Arguments passed in:
  // fnm="ehmm/etc/txt.phseq.data.int"
  // tarUL: List of phones corresponding to each utterance
  // lenU: Length of utterance?
  // wavF: List of wav file names
  // nos: Probably number of sentences

  FileExist(fnm);

  ifstream fp_in;
  fp_in.open(fnm, ios::in);

  fp_in >> nos;  // Read no-of-sentences....

  wavF = new char*[nos];   // wavF contains wave file names
  tarUL = new int*[nos];    // tarUL[sen][words]
  lenU = new int[nos];     // lenU[sen] - tells the length of each sentence

  int tl;

  for (int i = 0; i < nos; i++) {
    wavF[i] = new char[kNmLimit];
    fp_in >> wavF[i];    // Read the wave/feat file prefix

    fp_in >> lenU[i];  // ReadLength
    tl = lenU[i];

    tarUL[i] = new int[tl];

    for (int j = 0; j < tl; j++) {
      fp_in >> tarUL[i][j];   // Read the units - words/phones/units
    }
  }
  fp_in.close();
}

void LoadFeatFile(char *fnm, double**& feat, int&r, int&c) {
  FileExist(fnm);
  ifstream fp_in;
  fp_in.open(fnm, ios::in);

  fp_in >> r >> c;
  Alloc2f(feat, r, c);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      fp_in >> feat[i][j];
    }
  }
  fp_in.close();
}

void LoadBinaryFeatFile(char *filename, double*** feats_ptr,
                        int* num_rows, int* num_cols) {
  FileExist(filename);

  FILE* input_file;
  double** feats;
  double feat;

  if (*num_cols < 0) {
    cout << "Error: negative number of columns on: "
         << filename << endl;
    cout << "Aborting." << endl;
    exit(-1);
  }
  input_file = fopen(filename, "rb");
  if (input_file == NULL) {
    cout << "Error opening file: "
         << filename << endl;
    cout << "Aborting." << endl;
    exit(-1);
  }

  if (fread(num_rows, sizeof(*num_rows), 1, input_file) != 1)
  {
      fprintf(stderr, "Can't read num_rows\n");
      exit(1);
  }
  if (fread(num_cols, sizeof(*num_cols), 1, input_file) != 1)
  {
      fprintf(stderr, "Can't read num_cols\n");
      exit(1);
  }

  feats = new double*[*num_rows];
  for (int row = 0; row < *num_rows; row++) {
    feats[row] = new double[*num_cols];
    if (fread(feats[row], sizeof(feat), *num_cols, input_file) != (unsigned int)*num_cols)
    {
        fprintf(stderr, "Can't read enough feats\n");
        exit(1);
    }

  }
  *feats_ptr = feats;
  fclose(input_file);
}

// Looks like the purpose of this function is to generate the Emissions matrix
// 'emt'. This matrix is (number of states x number of MFCC frames). Each entry
// in the matrix tells you the probability of that particular state emitting 
// that particular MFCC frame. Since emission probabilities are modeled using
// GMMs, each entry in the matrix is generated by plugging in the frame into
// the GMM corresponding to each state.
void Emissions(wrdC *hwrd, stC *hst, double**& emt, int& er, int& ec,
               double **ft, int fr, int fc, int *tar, int ltar, int*& stMap) {
  // hwrd : HMM words (phonemes)
  // hst : HMM states
  // emt : Emission matrix
  // er : Number of rows of emission matrix (number of states)
  // ec : Number of columns of emission matrix (number of MFCC frames)
  // ft : Pointer to the feats
  // fr : Number of rows of features
  // fc : Number of columns of features
  // tar : Array containing list of phones in utterance
  // ltar : Number of phones in utterance
  // stMap : List of statenames corresponding to this utterance (I think)

  (void)fc;
  int tw;
  int bs;
  int sid;
  int rn;

  ec = fr;
  er = 0;
  // Counts total number of states for this utterance
  for (int i = 0; i < ltar; i++) {
    tw = tar[i];
    er += hwrd[tw].nst;
  }
  // Creates emissions matrix that is number of states x number of MCEP frames
  Alloc2f(emt, er, ec);
  // Creates an array to store the frames that correspond to each state
  Alloc1d(stMap, er);

  rn = 0;
  for (int i = 0; i < ltar; i++) {
    tw = tar[i];
    bs = hwrd[tw].bst;
    for (int j = 0; j < hwrd[tw].nst; j++) {
      sid = bs + j;
      stMap[rn] = sid;
      // for (int k = 0; k < ec; k++) {
      //  emt[rn][k] = hst[sid].GauProb(ft[k]);
      // }
      rn++;
    }
  }

  if (rn != er) {
    cout << "ER AND RN " << er << " " << rn << " are expected to match...";
    cout << "Possible error... aborting..." << endl;
    exit(1);
  }

  int ts;
  double *cshP;
  Alloc1f(cshP, _tst);
  // k iterates over each MFCC frame
  for (int k = 0; k < ec; k++) {
    // Calculate Gaussian for all states.....
    for (int s = 0; s < _tst; s++) {
      cshP[s] = hst[s].GauProb(ft[k]);
    }
    // j iterates over each HMM state
    for (int j = 0; j < er; j++) {
      ts = stMap[j];
      emt[j][k] = cshP[ts];
    }
  }
  Delete1f(cshP);
}

void PrintMatrix(double **arr, int r, int c, char *nm) {
  cout << "Printing " << nm << endl;
  for (int i = 0; i < r; i++) {
    cout << " I = " << i << " ";
    for (int j = 0; j < c; j++) {
      cout << arr[i][j] << " ";
    }
    cout << endl;
  }
}

// The 'tar' array in the second argument of this function contains
// a list of phones in the utterance. This function fills the arcW 
// matrix with transition probabilites of going from each HMM state
// in the utterance to any other state. The arcW matrix is probably
// block-diagonal
void FillStateTrans(double **arcW, int *tar, int ltar, int *nullI) {
  // arcW: Empty matrix
  // tar: Array containing list of phones in utterance
  // ltar: Number of phones in utterance 
  // nullI: Empty matrix

  int rn;   // row number
  int tw;   // temporary word
  int bs;   // begin state
  int sid;  // state id
  int nxt;  // next state

  int srn;   // store row number...
  int barn;  // begining arc row number;;

  rn = 0;
  // Loops through each phone
  for (int i = 0; i < ltar; i++) {
    tw = tar[i];
    bs = hwrd[tw].bst;

    srn = rn;  // store the row number of the word begining state...

    // Loops through each state in the phone
    for (int j = 0; j < hwrd[tw].nst; j++) {
      sid = bs + j;

      if (hst[sid].getconBS() == bs + 1) {
        // Useful for fully connected cases....
        barn = srn + 1;
      } else {
        barn = rn;  // rn indicates current row number...
      }

      nullI[rn] = hst[sid].getnullF();
      // 1 or 0, depending on null state or not.

      // Fills the arcW[i][j] matrix with probabilities of transition from
      // state i to state j
      for (int k = 0; k < hst[sid].getnc(); k++) {
        nxt = barn + k;
        arcW[rn][nxt] = hst[sid].trans(k);
      }
      rn++;
    }
  }
}

void FillWordTrans(double **arcW, int *tar, int ltar, double **trw,
                   int *bI, int *eI, int**& fwM, int**& bwM, int er) {
  (void)trw;
  int rn;
  int tw;
  int bs;
  int es;

  int *esa;
  int *bsa;
  int *wa;

  int nos;

  double estrn;
  double bias;

  Alloc1d(esa, ltar);
  Alloc1d(bsa, ltar);
  Alloc1d(wa, ltar);

  rn = 0;
  nos = 0;

  for (int i = 0; i < ltar; i++) {
    tw = tar[i];
    bsa[i] = rn;
    bI[rn] = 1;      // bI[st] = 1; useful for initialization of trellis....

    esa[i] = rn + (hwrd[tw].nst - 1);
    eI[esa[i]] = 1;  // eI[st] = 1; useful for initialization of trellis...

    wa[i]  = tw;
    rn += hwrd[tw].nst;

    nos += hwrd[tw].nst;
  }

  for (int i = 0; i < ltar; i++) {
    es = esa[i];

    /*This code stops transition leaks, if any.. */
    // cw = wa[i];
    // sum = 0;     //Compute the summations of the remianing trans.
    // for (int j = 0; j < ltar; j++) {
    //    bs = bsa[j];
    // nw = wa[j];
    //  sum += trw[cw][nw];
    //  }

    estrn = arcW[es][es];  // Take end state transition
    if (0 != estrn) {
      cout << "ESTRN SHOULD HAVE BEEN ZERO ALWAYS..." << endl;
      exit(1);
    }

    // bias  = (1.0 - (estrn + sum )) / (ltar + 1);  // estrn + sum should be 1
    bias  = 1.0 / ltar;     // estrn + sum should be 1
    // if not get the bias to be distributed..
    // ltar + 1 is for self-transition.
    for (int j = 0; j < ltar; j++) {
      bs = bsa[j];
      //      nw = wa[j];
      // arcW[es][bs] = trw[cw][nw] + bias;
      arcW[es][bs] = bias;
    }
    // arcW[es][es] += bias;  // Adding bias to the self transition tooo...
  }

  fwM = new int*[er];
  bwM = new int*[er];

  for (int i = 0; i < er; i++) {
    int cn = 0;
    for (int j = 0; j < er; j++) {
      if (arcW[i][j] > 0) {
        cn++;
      }
    }

    if (cn == 0 && i != er - 1) {
      cout << "State number is " << i
           << " : It does not seem to have any connections from it.. " << endl;
      cout << "Max states are: " << er << endl;
      exit(1);
    }

    fwM[i] = new int[cn + 1];
    fwM[i][0] = cn;
    // cout << "I: " << i << " CN: " << fwM[i][0] << " = " << cn << endl;

    int mi = 1;
    for (int j = 0; j < er; j++) {
      if (arcW[i][j] > 0) {
        fwM[i][mi] = j;
        mi++;
      }
    }
  }

  for (int j = 0; j < er; j++) {
    int cn = 0;
    for (int i = 0; i < er; i++) {
      if (arcW[i][j] > 0) {
        cn++;
      }
    }
    if (cn == 0 && j != 0) {
      cout << "State number is " << j
           << " : It does not seem to have any connections to it.. " << endl;
      cout << "Max states are: " << er << endl;
      exit(1);
    }

    bwM[j] = new int[cn + 1];
    bwM[j][0] = cn;

    int mj = 1;
    for (int i = 0; i < er; i++) {
      if (arcW[i][j] > 0) {
        bwM[j][mj] = i;
        mj++;
      }
    }
  }


  Delete1d(esa);
  Delete1d(bsa);
  Delete1d(wa);
}

int Viterbi(double **emt, int r, int c, double **trp,
            double **alp, double **bet, double *nrmF,
            int *bI, int *eI, int *path, int **fwM, int **bwM,
            int *nullI, int **pbuf, int& lastS) {
  (void)bet;
  int ns;
  int nt;
  int t;
  int s;
  int tm1;
  double maxV;

  // int **pbuf;
  int mas;
  int pst;
  double maxCon;
  double val;
  int nanF = 0;

  // Alloc2d(pbuf, r, c);


  ns = r;
  nt = c;
  t = 0;
  s = 0;
  maxV = -1.0e+32;
  int myc, p, n;
  int tit;

  // Initialize all the possible word beginners......
  for (s = 0; s < ns; s++) {
    if (1 == bI[s] && 1 == nullI[s]) {
      // cout << "S: " << s << endl;
      // cout << "FWM " << fwM[s][0] << endl;
      for (myc = 0; myc < fwM[s][0]; myc++) {
        int n = fwM[s][myc + 1];
        alp[n][t] = emt[n][t] * trp[s][n];
        pbuf[n][t] = s;
        if (maxV < alp[n][t]) {
          maxV = alp[n][t];
        }
      }
    }
  }

  // PULL THE NULL STATES.....
  for (s = 0; s < ns; s++) {
    tit = t;
    if (1 == nullI[s]) {
      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc + 1];
        val = alp[p][tit] * trp[p][s];
        if (alp[s][t] < val) { alp[s][t] = val;
          pbuf[s][t] = p;
        }
      }

      for (myc = 0; myc < fwM[s][0]; myc++) {
        n = fwM[s][myc + 1];
        if (1 == nullI[n] && n < s) {
          val = alp[s][t] * trp[s][n];
          if (alp[n][t] < val) {
            alp[n][t] = val;
            pbuf[n][t] = s;
          }
          if (maxV < alp[n][t]) {
            maxV = alp[n][t];
          }
        }
      }
    }
    if (maxV < alp[s][t]) { maxV = alp[s][t];
    }
  }

  ////////////////////////////////////////////
  nrmF[t] = maxV;
  if (nrmF[t] == 0) {
    nanF = 1;
    return(nanF);
  }
  for (s = 0; s < ns; s++) {
    alp[s][t] /= nrmF[t];
  }

  /////////////////////////////////////
  for (t = 1; t < nt; t++) {
    tm1 = t - 1;
    maxV = -1.0e+32;
    ////////////////////////////////////
    for (s = 0; s < ns; s++) {
      maxCon = -1.0e+32;
      pst  = -1;

      if (1 == nullI[s]) {
        tit = t;
      } else {
        tit = tm1;
      }

      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc+1];
        val = alp[p][tit] * trp[p][s];
        if (maxCon < val) { maxCon = val;
          pst = p;
        }
      }
      alp[s][t] = maxCon * emt[s][t];
      pbuf[s][t] = pst;

      if (1 == nullI[s]) {
        for (myc = 0; myc < fwM[s][0]; myc++) {
          n = fwM[s][myc + 1];
          if (1 == nullI[n] && n < s) {
            val = alp[s][t] * trp[s][n];
            if (alp[n][t] < val) {
              alp[n][t] = val;
              pbuf[n][t] = s;
            }
            if (maxV < alp[n][t]) {
              maxV = alp[n][t];
            }
          }
        }
      }

      if (maxV < alp[s][t]) { maxV = alp[s][t];
      }
    }  // ///////////////////////////////////////////

    nrmF[t] = maxV;
    if (nrmF[t] == 0) {
      nanF = 1;
      return(nanF);
    }
    for (s = 0; s < ns; s++) {
      alp[s][t] /= nrmF[t];
    }
  }  // //////////////////////////////////////////////////////////

  // Find the max ending state.....

  t = nt - 1;
  mas = -1;
  maxV = -1.0e+32;
  for (s = 0; s < ns; s++) {
    if (1 == eI[s]) {
      if (maxV < alp[s][t]) {
        maxV = alp[s][t];
        mas = s;
      }
    }
  }

  lastS = mas;

  for (int t = nt - 1; t >= 0; t--) {
    path[t] = mas;
    mas     = pbuf[mas][t];
  }

  // Delete2d(pbuf, r);
  return nanF;
}

// Vanilla Viterbi search
int Viterbi_Seq(double **emt, int r, int c, double **trp,
                double **alp, double **bet, double *nrmF,
                int *bI, int *eI, int *path, int **fwM, int **bwM,
                int *nullI, int **pbuf, int &lastS) {
  // emt : Matrix containing the emission probabilities
  // r : Number of rows of emt (number of states)
  // c : Number of columns of emt (number of MFCC frames)
  // trp: Transition probabilities between states
  // alp : Alphas. Parameters calculated during Viterbi search
  // bet : Betas. Unused. Probably a vestigial remnant of the baum-welch code
  // nrmF: Empty vector
  // bI : Matrix indicating beginning of phones
  // eI : Matrix indicating end of phones
  // path: Presumably the vector that tells you the optimal state path
  // fwM : Matrix that keeps track of forward connections of each state
  // bwM : Matrix that keeps track of backward connections of each state
  // nullI: Keeps track of null states (beginning and end of each phone)
  // pbuf : Empty matrix
  // lastS : Last state

  (void)bet;
  (void) bI;
  (void) eI;
  int ns;
  int nt;
  int t;
  int s;
  int tm1;

  double maxV;

  // int **pbuf;
  int mas;
  int pst;
  double maxCon;
  double val;
  int nanF = 0;

  // Alloc2d(pbuf, r, c);

  ns = r; //number of states
  nt = c; //number of frames
  t = 0;
  s = 0;
  maxV = -1.0e+32;

  int myc, p, n;
  int tit;

  alp[s][t] = 0;  // s = 0; and it is a null state..
  // This for loop probably initializes the first row of the alpha matrix
  for (int myc = 0; myc < fwM[s][0]; myc++) {
    int n = fwM[s][myc + 1];
    alp[n][t] = emt[n][t] * trp[s][n];
    // pbuf[n][t] = s;
    if (maxV < alp[n][t]) {
      maxV = alp[n][t];
    }
  }

  // Do stuff for the first frame
  for (s = 0; s < ns; s++) {
    tit = t;
    if (1 == nullI[s]) {
      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc + 1];
        val = alp[p][tit] * trp[p][s];
        if (alp[s][t] < val) { alp[s][t] = val;
          // pbuf[s][t] = p;
        }
      }

      for (myc = 0; myc < fwM[s][0]; myc++) {
        n = fwM[s][myc + 1];
        if (1 == nullI[n] && n < s) {
          val = alp[s][t] * trp[s][n];
          if (alp[n][t] < val) {
            alp[n][t] = val;
            // pbuf[n][t] = s;
          }
          if (maxV < alp[n][t]) {
            maxV = alp[n][t];
          }
        }
      }
    }
    if (maxV < alp[s][t]) { maxV = alp[s][t];
    }
  }

  nrmF[t] = maxV;
  if (nrmF[t] == 0) {
    nanF = 1;
    return(nanF);
  }

  for (s = 0; s < ns; s++) {
    alp[s][t] /= nrmF[t];
  }

  // Standard Viterbi from here on:
  for (t = 1; t < nt; t++) {
    tm1 = t - 1;
    maxV = -1.0e+32;

    for (s = 0; s < ns; s++) {
      maxCon = -1.0e+32;
      pst  = -1;

      if (1 == nullI[s]) {
        tit = t;
      } else {
        tit = tm1;
      }

      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc+1];
        val = alp[p][tit] * trp[p][s];
        if (maxCon < val) { maxCon = val;
          pst = p;
        }
      }
      alp[s][t] = maxCon * emt[s][t];
      pbuf[s][t] = pst;

      if (1 == nullI[s]) {
        for (myc = 0; myc < fwM[s][0]; myc++) {
          n = fwM[s][myc + 1];
          if (1 == nullI[n] && n < s) {
            val = alp[s][t] * trp[s][n];
            if (alp[n][t] < val) {
              alp[n][t] = val;
              pbuf[n][t] = s;
            }
            if (maxV < alp[n][t]) {
              maxV = alp[n][t];
            }
          }
        }
      }

      if (maxV < alp[s][t]) { maxV = alp[s][t];
      }
    }

    nrmF[t] = maxV;

    if (nrmF[t] == 0) {
      nanF = 1;
      return(nanF);
    }

    for (s = 0; s < ns; s++) {
      alp[s][t] /= nrmF[t];
    }
  }

  // Find the max ending state.....

  t = nt - 1;
  mas = -1;
  maxV = -1.0e+32;
  mas = ns - 1;  // Choose the last state...
  lastS = mas;

  for (int t = nt - 1; t >= 0; t--) {
    path[t] = mas;
    mas     = pbuf[mas][t];
  }

  // Delete2d(pbuf, r);

  return nanF;
}

int Viterbi_Seq_nde(double **emt, int r, int c, double **trp,
                    double **alp, double **bet, double *nrmF,
                    int *bI, int *eI, int *path, int **fwM, int **bwM,
                    int *nullI, int **pbuf, int &lastS) {
  (void) eI;
  (void) bI;
  (void) bet;
  int ns;
  int nt;
  int t;
  int s;
  int tm1;

  double maxV;

  // int **pbuf;
  int mas;
  int pst;
  double maxCon;
  double val;
  int nanF = 0;

  // Alloc2d(pbuf, r, c);


  ns = r;
  nt = c;
  t = 0;
  s = 0;
  maxV = -1.0e+32;

  int myc, p, n;
  int tit;

  alp[s][t] = 0;  // s = 0; and it is a null state..
  for (int myc = 0; myc < fwM[s][0]; myc++) {
    int n = fwM[s][myc + 1];
    alp[n][t] = emt[n][t] * trp[s][n];
    // pbuf[n][t] = s;
    if (maxV < alp[n][t]) {
      maxV = alp[n][t];
    }
  }

  for (s = 0; s < ns; s++) {
    tit = t;
    if (1 == nullI[s]) {
      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc + 1];
        val = alp[p][tit] * trp[p][s];
        if (alp[s][t] < val) { alp[s][t] = val;
          // pbuf[s][t] = p;
        }
      }

      for (myc = 0; myc < fwM[s][0]; myc++) {
        n = fwM[s][myc + 1];
        if (1 == nullI[n] && n < s) {
          val = alp[s][t] * trp[s][n];
          if (alp[n][t] < val) {
            alp[n][t] = val;
            // pbuf[n][t] = s;
          }
          if (maxV < alp[n][t]) {
            maxV = alp[n][t];
          }
        }
      }
    }
    if (maxV < alp[s][t]) { maxV = alp[s][t];
    }
  }

  nrmF[t] = maxV;
  if (nrmF[t] == 0) {
    nanF = 1;
    return(nanF);
  }

  for (s = 0; s < ns; s++) {
    alp[s][t] /= nrmF[t];
  }


  for (t = 1; t < nt; t++) {
    tm1 = t - 1;
    maxV = -1.0e+32;

    for (s = 0; s < ns; s++) {
      maxCon = -1.0e+32;
      pst  = -1;

      if (1 == nullI[s]) {
        tit = t;
      } else {
        tit = tm1;
      }

      for (myc = 0; myc < bwM[s][0]; myc++) {
        p = bwM[s][myc+1];
        val = alp[p][tit] * trp[p][s];
        if (maxCon < val) { maxCon = val;
          pst = p;
        }
      }
      alp[s][t] = maxCon * emt[s][t];
      pbuf[s][t] = pst;

      if (1 == nullI[s]) {
        for (myc = 0; myc < fwM[s][0]; myc++) {
          n = fwM[s][myc + 1];
          if (1 == nullI[n] && n < s) {
            val = alp[s][t] * trp[s][n];
            if (alp[n][t] < val) {
              alp[n][t] = val;
              pbuf[n][t] = s;
            }
            if (maxV < alp[n][t]) {
              maxV = alp[n][t];
            }
          }
        }
      }

      if (maxV < alp[s][t]) { maxV = alp[s][t];
      }
    }

    nrmF[t] = maxV;

    if (nrmF[t] == 0) {
      nanF = 1;
      return(nanF);
    }

    for (s = 0; s < ns; s++) {
      alp[s][t] /= nrmF[t];
    }
  }

  // Find the max ending state.....

  t = nt - 1;
  maxV = alp[0][t];
  mas = 0;
  for (s = 1; s < ns; s++) {
    if (alp[s][t] > maxV) {
      maxV = alp[s][t];
      mas = s;
    }
  }

  // mas = ns - 1; //Choose the last state...
  lastS = mas;

  for (int t = nt - 1; t >= 0; t--) {
    path[t] = mas;
    mas     = pbuf[mas][t];
  }

  // Delete2d(pbuf, r);

  return nanF;
}

// This function updates the arcW matrix to include the transition
// probabilities across phones too. It also generates fwM and bwM
// matrices that keep track of the forward and backward connections
// of each state.
void FillWordTrans_Seq(double **arcW, int *tar, int ltar,
                       double **trw, int *bI, int *eI,
                       int**& fwM, int**& bwM, int er) {
  // arcW: Transition probabilities matrix
  // tar: Array containing list of phones in utterance 'p'
  // ltar: Number of phones in utterance 'p'
  // trw: Transition probability matrix between phones (not states)
  // bI: described earlier
  // eI: described earlier
  // fwM: Forward connection Matrix for each state
  // bwM: Backward connection Matrix for each state
  // er: Number of rows of emission matrix (number of states)

  // There is no need of trw matrix here....
  (void) trw;
  int rn;
  int tw;
  int bs;
  int es;

  int *esa;
  int *bsa;
  int *wa;

  int nos;

  double estrn;
  double rest;
  int e1s;
  int b1s;
  double val;

  Alloc1d(esa, ltar);
  Alloc1d(bsa, ltar);
  Alloc1d(wa, ltar);

  rn = 0;
  nos = 0;

  for (int i = 0; i < ltar; i++) {
    tw = tar[i];
    bsa[i] = rn;
    bI[rn] = 1;      // bI[st] = 1; useful for initialization of trellis....

    esa[i] = rn + (hwrd[tw].nst - 1);
    eI[esa[i]] = 1;  // eI[st] = 1; useful for initialization of trellis...

    wa[i]  = tw;
    rn += hwrd[tw].nst;

    nos += hwrd[tw].nst;
  }

  for (int i = 0; i < ltar; i++) {
    es = esa[i];
    bs = bsa[i];

    estrn = 0;
    for (int z = bs; z <= es; z++) {
      estrn += arcW[es][z];  // Sum the transitions of the last state....
    }
    // estrn = arcW[es][es];  // Take end state transition

    rest  = 1.0 - estrn;  // estrn + sum should be 1

    // cout << "Pause ID is " << _pauID << endl;
    // cout << "Short Pause ID is " << _spauID << endl;

    // Break rest - remaining into half, if the wid is PAU - July 17 2005
    if (tar[i] == _pauID && i != ltar - 1) {
      rest = rest/2.0;
    }
    //

    if (i == ltar - 1) {
      // arcW[es][es] += rest;
      // The last state of the sentence will have a 1.0 self transition.
      // The Last state is a null state, so no need to add any transition
    } else {
      bs = bsa[i+1];
      arcW[es][bs] = rest;
    }

    // if tar[i] matches with PAU, then assign the second half of the
    // rest to self-word transition - 17 July 2005
    if (tar[i] == _pauID) {
      bs = bsa[i];
      arcW[es][bs] += rest;

    } else if (tar[i] == _spauID) {
      bs = bsa[i];
      if ((i + 1 >= ltar)  || (i - 1 < 0)) {
        cout << "Cannot have a Short Pause at the end/begin of a sentence\n";
        exit(1);
      }
      e1s = esa[i-1];  // ending state of previous word
      b1s = bsa[i+1];  // begining state of next word

      val = arcW[e1s][bs] / 2.0;
      arcW[e1s][bs] = val;   // connect previous word and current word
      arcW[e1s][b1s] = val;  // connect previous word and next word
      // cout << "States at Short Pause are: SSIL bs: " << bs
      //      << " e1s " << e1s << " b1s " << b1s << endl;
    }
  }

  fwM = new int*[er]; //fwM and bwM seem to keep track of the forward and backward
  bwM = new int*[er]; //connections between each state and the others.

  for (int i = 0; i < er; i++) {
    int cn = 0;
    for (int j = 0; j < er; j++) {
      if (arcW[i][j] > 0) {
        cn++;
      }
    }
    if (cn == 0 && i != er - 1) {
      cout << "State number is " << i
           << " : It does not seem to have any connections from it.. " << endl;
      cout << "Max states are: " << er << endl;
      exit(1);
    }

    fwM[i] = new int[cn + 1];
    fwM[i][0] = cn;

    int mi = 1;
    for (int j = 0; j < er; j++) {
      if (arcW[i][j] > 0) {
        fwM[i][mi] = j;
        mi++;
      }
    }
  }

  for (int j = 0; j < er; j++) {
    int cn = 0;
    for (int i = 0; i < er; i++) {
      if (arcW[i][j] > 0) {
        cn++;
      }
    }
    if (cn == 0 && j != 0) {
      cout << "State number is " << j
           << " : It does not seem to have any connections to it.. " << endl;
      cout << "Max states are: " << er << endl;
      exit(1);
    }

    bwM[j] = new int[cn + 1];
    bwM[j][0] = cn;

    int mj = 1;
    for (int i = 0; i < er; i++) {
      if (arcW[i][j] > 0) {
        bwM[j][mj] = i;
        mj++;
      }
    }
  }

  Delete1d(esa);
  Delete1d(bsa);
  Delete1d(wa);
}


int IsSilence(char *nm) {
  int rv = 0;
  if (0 == strcasecmp("sil", nm) ||
      0 == strcasecmp("pau", nm)) {
    rv = 1;
  }
  return rv;
}

int IsShortPause(char *nm) {
  int rv = 0;
  if (0 == strcasecmp("ssil", nm)) {
    rv = 1;
  }
  return rv;
}

