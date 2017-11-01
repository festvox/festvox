/*************************************************************************/
/*                                                                       */
/*                     Carnegie Mellon University                        */
/*                         Copyright (c) 2007                            */
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
/*                                                                       */
/*************************************************************************/
#include "my_header.h"

#include <string.h>

class wavC {

 private: 

  typedef struct MS_WavHeader_struct{
    char riff[5];           //4 = "RIFF"  
    int  remainingBytes;    //8 length of packet to follow = filesize-8
    char waveFmt[9];        //16 = "WAVEfmt "
    int  lenFormatChunk;    //20 length of format chunk = 0x10
    char unDefined[3];      //22 always 01,00
    char channels[3];       //24 will be 01,00 in our case
    int  sampleRate;        //28 sampling frequency 16000 for us
    int  bytesPerSec;       //32 sample_rate * bytes_per_sample = 32000
    char bytesPerSample[3]; //34 for our case = 02,00
    char bitsPerSample[3];  //36 for our case = 16,00
    char data[5];           //40 = "data"
    int  dataByteCount;     //44 length of data to follow in bytes
  } MS_WavHeader; 

  MS_WavHeader wavH;
 int _sF;  //samp frequency
 int _frmSz; //frame size;
 int _frmSft; //frame shift
 int _hsize;  //header size;
 int _bps;  //bytes per sample
 int maxSmp; //maximum no. of samples;;
 float totdur; //total duration of the speech file;

 string spchF; //speech file name

 float  *sblk; //speech block;
 int    nsmp;  //no. of samples;
 float  _durB;  //duration of the block;
 float  olpD;  //overlapping duration;
 float  _sfrac; //segmentation fraction (actually inverse of it is used).
 float  bT;    //begin time;
 float  eT;    //end time;
 int    bB;    //begin byte;

 string etcD;  //etc dir
 string blkF;  //temporary block file, where data is stored.
 string tempD; //temporary directory..

 float cb_dur; //current block duration/length

 public: 
   wavC();
   wavC(string, float, float);
   ~wavC();
  
   void read_wave_header();
   void print_header();

   void cal_begin_end_byte(); 
   void dump_speech_bytes(string blkF);

};

wavC::wavC() {
}
wavC::~wavC() {
}

wavC::wavC(string sFile, float begT, float endT)  {
  
  spchF = sFile;

  FileExist((char*)sFile.c_str());

  read_wave_header();
  //print_header();
  
  _hsize = 44; //wave header 

  _sF = wavH.sampleRate;

  //cal bytes per sample;
  _bps = wavH.bytesPerSec / _sF;

  //cal max. samples;
  maxSmp = wavH.dataByteCount / _bps;
  
  //total duration
  totdur = (float) maxSmp / (float) _sF;

  cout<<"bytes per Sample "<<_bps<<endl;
  cout<<"samp freq: "<<_sF<<endl;
  cout<<"max Samples are: "<<maxSmp<<endl;
  cout<<"total duration: "<<totdur<<" S "<<endl;

  bT = begT;
  eT = endT;
  nsmp = 0;
  //Alloc1f(sblk, 10); //dummy allocation of 10 bytes..

}

void wavC::read_wave_header() {

 /* typedef struct MS_WavHeader{
   char riff[5];           //4 = "RIFF"  
   int  remainingBytes;    //8 length of packet to follow = filesize-8
   char waveFmt[9];        //16 = "WAVEfmt "
   int  lenFormatChunk;    //20 length of format chunk = 0x10
   char unDefined[3];      //22 always 01,00
   char channels[3];       //24 will be 01,00 in our case
   int  sampleRate;        //28 sampling frequency 16000 for us
   int  bytesPerSec;       //32 sample_rate * bytes_per_sample = 32000
   char bytesPerSample[3]; //34 for our case = 02,00
   char bitsPerSample[3];  //36 for our case = 16,00
   char data[5];           //40 = "data"
   int  dataByteCount;     //44 length of data to follow in bytes
 }; */

 wavH.riff[4] = '\0';
 wavH.waveFmt[8] = '\0';
 wavH.unDefined[2] = '\0';
 wavH.channels[2] = '\0';
 wavH.bytesPerSample[2] = '\0';
 wavH.bitsPerSample[2] = '\0';
 wavH.data[4] = '\0';

 ifstream fp_bin;
   fp_bin.open((char*)spchF.c_str(), ios::in | ios::binary);

   fp_bin.read((char*)wavH.riff, 4);   //4
   fp_bin.read((char*)&wavH.remainingBytes, 4); //8
   fp_bin.read((char*)wavH.waveFmt, 8); //16
   fp_bin.read((char*)&wavH.lenFormatChunk, 4); //20
   fp_bin.read((char*)wavH.unDefined, 2); //22
   fp_bin.read((char*)wavH.channels, 2); //24
   fp_bin.read((char*)&wavH.sampleRate, 4); //28
   fp_bin.read((char*)&wavH.bytesPerSec, 4); //32
   fp_bin.read((char*)wavH.bytesPerSample, 2); //34
   fp_bin.read((char*)wavH.bitsPerSample, 2); //36
   fp_bin.read((char*)wavH.data, 4); //40
   fp_bin.read((char*)&wavH.dataByteCount, 4); //44
 fp_bin.close();

 if (strcmp(wavH.riff, "RIFF") != 0) {
   cout<<"wavFile: "<<spchF<<" is not in MS RIFF format..."<<endl;
   cout<<"wavH.riff: "<<wavH.riff<<endl;
   cout<<"Aborting.... "<<endl;
   exit(1);
 }
}

void wavC::print_header() {

  cout<<"File Name: "<<spchF<<endl;

  cout<<"Riff (4): "<<wavH.riff<<endl;
  cout<<"Remaining Bytes (8): "<<wavH.remainingBytes<<endl;
  cout<<"waveFmt (16): "<<wavH.waveFmt<<endl;
  cout<<"LenFormatChunk (20): "<<wavH.lenFormatChunk<<endl;
  cout<<"unDefined (22): "<<wavH.unDefined<<endl;
  cout<<"Channels (24): "<<wavH.channels<<endl;
  cout<<"SampleRate (28): "<<wavH.sampleRate<<endl;
  cout<<"bytesPerSec (32): "<<wavH.bytesPerSec<<endl;
  cout<<"bytesPerSample (34): "<<wavH.bytesPerSample<<endl;
  cout<<"bitsPerSample (36): "<<wavH.bitsPerSample<<endl;
  cout<<"Data (40): "<<wavH.data<<endl;
  cout<<"dataByteCount (44): "<<wavH.dataByteCount<<endl;
}

void wavC::cal_begin_end_byte() {
 
 int bS; //begin sample;
 bS = (int) (bT * (float)_sF);
 bB = bS * _bps;
 bB = _hsize + bB;

 _durB = eT - bT;

 nsmp = (int) (_durB * (float)_sF);

 if (bS + nsmp >= maxSmp) {  //adjust to the last block.....
                             //or handle files less than 1 min;
   nsmp = maxSmp - bS;
 }

 cb_dur = (float) nsmp  / (float)_sF;

 cout<<"Processing from: "<<bS<<" to "<<bS + nsmp<<" max: "<<maxSmp<<" cb_dur: "<<cb_dur<<endl;
}

void wavC::dump_speech_bytes(string blkF) {

  cal_begin_end_byte();

  ifstream fp_in;
  ofstream fp_out;
  short int temp;

  fp_in.open((char*)spchF.c_str(), ios::in | ios::binary);
  fp_out.open((char*)blkF.c_str(), ios::out | ios::binary);

  fp_in.seekg(bB);
  cout<<"bB: "<<bB<<" nsmp: "<<nsmp<<endl;

  for (int i = 0; i < nsmp; i++) {
    fp_in.read((char*)&temp, _bps);
    fp_out.write((char*)&temp, _bps);
  } 
  fp_in.close();
  fp_out.close();
}

int main(int argc, char *argv[]) {
  
  if (argc != 5) { 
    cout<<"pass <speech-file> <begT> <endT> <outF>\n";
    exit(1);
  }

  string spchF;
  string outF;
  float begT;
  float endT;

  spchF = argv[1];
  begT = atof(argv[2]);
  endT = atof(argv[3]);
  outF = argv[4];

  wavC lslice(spchF, begT, endT); 

  lslice.dump_speech_bytes(outF);
}
