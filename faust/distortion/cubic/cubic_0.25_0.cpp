//----------------------------------------------------------
// name: "cubic_0.25_0"
//
// Code generated with Faust 0.9.95 (http://faust.grame.fr)
//----------------------------------------------------------

/* link with  */
#ifndef FAUSTPOWER
#define FAUSTPOWER
#include <cmath>
template <int N> inline int faustpower(int x)              { return faustpower<N/2>(x) * faustpower<N-N/2>(x); } 
template <> 	 inline int faustpower<0>(int x)            { return 1; }
template <> 	 inline int faustpower<1>(int x)            { return x; }
template <> 	 inline int faustpower<2>(int x)            { return x*x; }
template <int N> inline float faustpower(float x)            { return faustpower<N/2>(x) * faustpower<N-N/2>(x); } 
template <> 	 inline float faustpower<0>(float x)          { return 1; }
template <> 	 inline float faustpower<1>(float x)          { return x; }
template <> 	 inline float faustpower<2>(float x)          { return x*x; }
#endif
/************************************************************************

	IMPORTANT NOTE : this file contains two clearly delimited sections :
	the ARCHITECTURE section (in two parts) and the USER section. Each section
	is governed by its own copyright and license. Please check individually
	each section for license and copyright information.
*************************************************************************/

/*******************BEGIN ARCHITECTURE SECTION (part 1/2)****************/

/************************************************************************
    FAUST Architecture File
	Copyright (C) 2003-2011 GRAME, Centre National de Creation Musicale
    ---------------------------------------------------------------------
    This Architecture section is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 3 of
	the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
	along with this program; If not, see <http://www.gnu.org/licenses/>.

	EXCEPTION : As a special exception, you may create a larger work
	that contains this FAUST architecture section and distribute
	that work under terms of your choice, so long as this FAUST
	architecture section is not modified.


	************************************************************************
	************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <sndfile.h>
#include <vector>
#include <stack>
#include <string>
#include <map>
#include <iostream>

#include "faust/gui/console.h"
#include "faust/gui/FUI.h"
#include "faust/dsp/dsp.h"
#include "faust/misc.h"

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif  

#define READ_SAMPLE sf_readf_float
//#define READ_SAMPLE sf_readf_double

/******************************************************************************
*******************************************************************************

VECTOR INTRINSICS

*******************************************************************************
*******************************************************************************/


/********************END ARCHITECTURE SECTION (part 1/2)****************/

/**************************BEGIN USER SECTION **************************/

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif  


#ifndef FAUSTCLASS 
#define FAUSTCLASS mydsp
#endif

class mydsp : public dsp {
  private:
	float 	fVec0[2];
	float 	fRec0[2];
	int fSamplingFreq;

  public:
	virtual void metadata(Meta* m) { 
		m->declare("name", "cubic_0.25_0");
		m->declare("misceffect.lib/name", "Faust Math Library");
		m->declare("misceffect.lib/version", "2.0");
		m->declare("filter.lib/name", "Faust Filter Library");
		m->declare("filter.lib/version", "2.0");
	}

	virtual int getNumInputs() { return 1; }
	virtual int getNumOutputs() { return 1; }
	static void classInit(int samplingFreq) {
	}
	virtual void instanceConstants(int samplingFreq) {
		fSamplingFreq = samplingFreq;
	}
	virtual void instanceResetUserInterface() {
	}
	virtual void instanceClear() {
		for (int i=0; i<2; i++) fVec0[i] = 0;
		for (int i=0; i<2; i++) fRec0[i] = 0;
	}
	virtual void init(int samplingFreq) {
		classInit(samplingFreq);
		instanceInit(samplingFreq);
	}
	virtual void instanceInit(int samplingFreq) {
		instanceConstants(samplingFreq);
		instanceResetUserInterface();
		instanceClear();
	}
	virtual mydsp* clone() {
		return new mydsp();
	}
	virtual int getSampleRate() {
		return fSamplingFreq;
	}
	virtual void buildUserInterface(UI* ui_interface) {
		ui_interface->openVerticalBox("0x00");
		ui_interface->closeBox();
	}
	virtual void compute (int count, FAUSTFLOAT** input, FAUSTFLOAT** output) {
		FAUSTFLOAT* input0 = input[0];
		FAUSTFLOAT* output0 = output[0];
		for (int i=0; i<count; i++) {
			float fTemp0 = max((float)-1, min((float)1, (3.1622777f * (float)input0[i])));
			float fTemp1 = (fTemp0 * (1 - (0.33333334f * faustpower<2>(fTemp0))));
			fVec0[0] = fTemp1;
			fRec0[0] = ((fVec0[0] + (0.995f * fRec0[1])) - fVec0[1]);
			output0[i] = (FAUSTFLOAT)fRec0[0];
			// post processing
			fRec0[1] = fRec0[0];
			fVec0[1] = fVec0[0];
		}
	}
};



/***************************END USER SECTION ***************************/

/*******************BEGIN ARCHITECTURE SECTION (part 2/2)***************/

mydsp	DSP;

class Separator
{
  int		fNumFrames;
  int		fNumInputs;
  int		fNumOutputs;

  FAUSTFLOAT*	fInput;
  FAUSTFLOAT*	fOutputs[256];

public:

  Separator(int numFrames, int numInputs, int numOutputs)
  {
    fNumFrames 	= numFrames;
    fNumInputs 	= numInputs;
    fNumOutputs = max(numInputs, numOutputs);

    // allocate interleaved input channel
    fInput = (FAUSTFLOAT*) calloc(fNumFrames * fNumInputs, sizeof(FAUSTFLOAT));

    // allocate separate output channels
    for (int i = 0; i < fNumOutputs; i++) {
      fOutputs[i] = (FAUSTFLOAT*) calloc (fNumFrames, sizeof(FAUSTFLOAT));
    }
  }

  ~Separator()
  {
    // free interleaved input channel
    free(fInput);

    // free separate output channels
    for (int i = 0; i < fNumOutputs; i++) {
      free(fOutputs[i]);
    }
  }

  FAUSTFLOAT*	input()		{ return fInput; }

  FAUSTFLOAT** outputs()	{ return fOutputs; }

  void 	separate()
  {
    for (int s = 0; s < fNumFrames; s++) {
      for (int c = 0; c < fNumInputs; c++) {
        fOutputs[c][s] = fInput[c + s*fNumInputs];
      }
    }
  }
};

class Interleaver
{
  int fNumFrames;
  int fNumChans;

  FAUSTFLOAT* fInputs[256];
  FAUSTFLOAT* fOutput;

public:

  Interleaver(int numFrames, int numChans)
  {
    fNumFrames = numFrames;
    fNumChans  = numChans;

    // allocate separate input channels
    for (int i = 0; i < fNumChans; i++) {
      fInputs[i] = (FAUSTFLOAT*) calloc (fNumFrames, sizeof(FAUSTFLOAT));
    }

    // allocate interleaved output channel
    fOutput = (FAUSTFLOAT*) calloc(fNumFrames * fNumChans, sizeof(FAUSTFLOAT));

  }

  ~Interleaver()
  {
    // free separate input channels
    for (int i = 0; i < fNumChans; i++) {
      free(fInputs[i]);
    }

    // free interleaved output channel
    free(fOutput);
  }

  FAUSTFLOAT**	inputs()		{ return fInputs; }

  FAUSTFLOAT* 	output()		{ return fOutput; }

  void interleave()
  {
    for (int s = 0; s < fNumFrames; s++) {
      for (int c = 0; c < fNumChans; c++) {
        fOutput[c + s*fNumChans] = fInputs[c][s];
      }
    }
  }
};

#define kFrames 512

// loptrm : Scan command-line arguments and remove and return long int value when found
long loptrm(int *argcP, char *argv[], const char* longname, const char* shortname, long def)
{
  int argc = *argcP;
  for (int i=2; i<argc; i++) {
    if (strcmp(argv[i-1], shortname) == 0 || strcmp(argv[i-1], longname) == 0) {
      int optval = atoi(argv[i]);
      for (int j=i-1; j<argc-2; j++) {  // make it go away for sake of "faust/gui/console.h"
        argv[j] = argv[j+2];
      }
      *argcP -= 2;
      return optval;
    }
  }
  return def;
}

int main(int argc, char *argv[])
{
  SNDFILE*	in_sf;
  SNDFILE*	out_sf;
  SF_INFO	in_info;
  SF_INFO	out_info;
  unsigned int nAppend = 0; // number of frames to append beyond input file

  if (argc < 3) {
    fprintf(stderr,"*** USAGE: %s input_soundfile output_soundfile\n",argv[0]);
    exit(1);
  }

  nAppend = loptrm(&argc, argv, "--continue", "-c", 0);
    
  CMDUI* interface = new CMDUI(argc, argv);
  DSP.buildUserInterface(interface);
  interface->process_command();

  // open input file
  in_info.format = 0;
  in_sf = sf_open(interface->input_file(), SFM_READ, &in_info);
  if (in_sf == NULL) {
    fprintf(stderr,"*** Input file not found.\n");
    sf_perror(in_sf); 
    exit(1); 
  }

  // open output file
  out_info = in_info;
  out_info.format = in_info.format;
  out_info.channels = DSP.getNumOutputs();
  out_sf = sf_open(interface->output_file(), SFM_WRITE, &out_info);
  if (out_sf == NULL) { 
    fprintf(stderr,"*** Cannot write output file.\n");
    sf_perror(out_sf); 
    exit(1); 
  }

  // create separator and interleaver
  Separator   sep(kFrames, in_info.channels, DSP.getNumInputs());
  Interleaver ilv(kFrames, DSP.getNumOutputs());

  // init signal processor
  DSP.init(in_info.samplerate);
  //DSP.buildUserInterface(interface);
  interface->process_init();

  // process all samples
  int nbf;
  do {
    nbf = READ_SAMPLE(in_sf, sep.input(), kFrames);
    sep.separate();
    DSP.compute(nbf, sep.outputs(), ilv.inputs());
    ilv.interleave();
    sf_writef_float(out_sf, ilv.output(), nbf);
    //sf_write_raw(out_sf, ilv.output(), nbf);
  } while (nbf == kFrames);

  sf_close(in_sf);

  // compute tail, if any
  if (nAppend>0) {
    FAUSTFLOAT *input = (FAUSTFLOAT*) calloc(nAppend * DSP.getNumInputs(), sizeof(FAUSTFLOAT));
    FAUSTFLOAT *inputs[1] = { input };
    Interleaver ailv(nAppend, DSP.getNumOutputs());
    DSP.compute(nAppend, inputs, ailv.inputs());
    ailv.interleave();
    sf_writef_float(out_sf, ailv.output(), nAppend);
  }

  sf_close(out_sf);
}

/********************END ARCHITECTURE SECTION (part 2/2)****************/
