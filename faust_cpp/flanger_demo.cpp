//----------------------------------------------------------
// name: "flanger_demo"
//
// Code generated with Faust 0.9.95 (http://faust.grame.fr)
//----------------------------------------------------------

/* link with  */
#include <math.h>
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
	int 	iVec0[2];
	FAUSTFLOAT 	fslider0;
	float 	fConst0;
	float 	fConst1;
	float 	fRec1[2];
	float 	fRec2[2];
	FAUSTFLOAT 	fbargraph0;
	FAUSTFLOAT 	fcheckbox0;
	FAUSTFLOAT 	fslider1;
	FAUSTFLOAT 	fslider2;
	int 	IOTA;
	float 	fVec1[2048];
	FAUSTFLOAT 	fslider3;
	FAUSTFLOAT 	fslider4;
	float 	fRec0[2];
	FAUSTFLOAT 	fslider5;
	FAUSTFLOAT 	fcheckbox1;
	float 	fVec2[2048];
	float 	fRec3[2];
	int fSamplingFreq;

  public:
	virtual void metadata(Meta* m) { 
		m->declare("name", "flanger_demo");
		m->declare("basic.lib/name", "Faust Basic Element Library");
		m->declare("basic.lib/version", "0.0");
		m->declare("route.lib/name", "Faust Signal Routing Library");
		m->declare("route.lib/version", "0.0");
		m->declare("signal.lib/name", "Faust Signal Routing Library");
		m->declare("signal.lib/version", "0.0");
		m->declare("miscoscillator.lib/name", "Faust Oscillator Library");
		m->declare("miscoscillator.lib/version", "0.0");
		m->declare("filter.lib/name", "Faust Filter Library");
		m->declare("filter.lib/version", "2.0");
		m->declare("math.lib/name", "Faust Math Library");
		m->declare("math.lib/version", "2.0");
		m->declare("math.lib/author", "GRAME");
		m->declare("math.lib/copyright", "GRAME");
		m->declare("math.lib/license", "LGPL with exception");
		m->declare("phafla.lib/name", "Faust Phaser and Flanger Library");
		m->declare("phafla.lib/version", "0.0");
		m->declare("delay.lib/name", "Faust Delay Library");
		m->declare("delay.lib/version", "0.0");
	}

	virtual int getNumInputs() { return 2; }
	virtual int getNumOutputs() { return 2; }
	static void classInit(int samplingFreq) {
	}
	virtual void instanceConstants(int samplingFreq) {
		fSamplingFreq = samplingFreq;
		fConst0 = min(1.92e+05f, max(1.0f, (float)fSamplingFreq));
		fConst1 = (6.2831855f / fConst0);
	}
	virtual void instanceResetUserInterface() {
		fslider0 = 0.5f;
		fcheckbox0 = 0.0;
		fslider1 = 0.0f;
		fslider2 = 0.0f;
		fslider3 = 1e+01f;
		fslider4 = 1.0f;
		fslider5 = 1.0f;
		fcheckbox1 = 0.0;
	}
	virtual void instanceClear() {
		for (int i=0; i<2; i++) iVec0[i] = 0;
		for (int i=0; i<2; i++) fRec1[i] = 0;
		for (int i=0; i<2; i++) fRec2[i] = 0;
		IOTA = 0;
		for (int i=0; i<2048; i++) fVec1[i] = 0;
		for (int i=0; i<2; i++) fRec0[i] = 0;
		for (int i=0; i<2048; i++) fVec2[i] = 0;
		for (int i=0; i<2; i++) fRec3[i] = 0;
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
		ui_interface->declare(0, "tooltip", "Reference: https://ccrma.stanford.edu/~jos/pasp/Flanging.html");
		ui_interface->openVerticalBox("FLANGER");
		ui_interface->declare(0, "0", "");
		ui_interface->openHorizontalBox("0x00");
		ui_interface->declare(&fcheckbox0, "0", "");
		ui_interface->declare(&fcheckbox0, "tooltip", "When this is checked, the flanger    has no effect");
		ui_interface->addCheckButton("Bypass", &fcheckbox0);
		ui_interface->declare(&fcheckbox1, "1", "");
		ui_interface->addCheckButton("Invert Flange Sum", &fcheckbox1);
		ui_interface->declare(&fbargraph0, "2", "");
		ui_interface->declare(&fbargraph0, "style", "led");
		ui_interface->declare(&fbargraph0, "tooltip", "Display sum of flange delays");
		ui_interface->addHorizontalBargraph("Flange LFO", &fbargraph0, -1.5f, 1.5f);
		ui_interface->closeBox();
		ui_interface->declare(0, "1", "");
		ui_interface->openHorizontalBox("0x00");
		ui_interface->declare(&fslider0, "1", "");
		ui_interface->declare(&fslider0, "style", "knob");
		ui_interface->declare(&fslider0, "unit", "Hz");
		ui_interface->addHorizontalSlider("Speed", &fslider0, 0.5f, 0.0f, 1e+01f, 0.01f);
		ui_interface->declare(&fslider5, "2", "");
		ui_interface->declare(&fslider5, "style", "knob");
		ui_interface->addHorizontalSlider("Depth", &fslider5, 1.0f, 0.0f, 1.0f, 0.001f);
		ui_interface->declare(&fslider2, "3", "");
		ui_interface->declare(&fslider2, "style", "knob");
		ui_interface->addHorizontalSlider("Feedback", &fslider2, 0.0f, -0.999f, 0.999f, 0.001f);
		ui_interface->closeBox();
		ui_interface->declare(0, "2", "");
		ui_interface->openHorizontalBox("Delay Controls");
		ui_interface->declare(&fslider3, "1", "");
		ui_interface->declare(&fslider3, "style", "knob");
		ui_interface->declare(&fslider3, "unit", "ms");
		ui_interface->addHorizontalSlider("Flange Delay", &fslider3, 1e+01f, 0.0f, 2e+01f, 0.001f);
		ui_interface->declare(&fslider4, "2", "");
		ui_interface->declare(&fslider4, "style", "knob");
		ui_interface->declare(&fslider4, "unit", "ms");
		ui_interface->addHorizontalSlider("Delay Offset", &fslider4, 1.0f, 0.0f, 2e+01f, 0.001f);
		ui_interface->closeBox();
		ui_interface->declare(0, "3", "");
		ui_interface->openHorizontalBox("0x00");
		ui_interface->declare(&fslider1, "unit", "dB");
		ui_interface->addHorizontalSlider("Flanger Output Level", &fslider1, 0.0f, -6e+01f, 1e+01f, 0.1f);
		ui_interface->closeBox();
		ui_interface->closeBox();
	}
	virtual void compute (int count, FAUSTFLOAT** input, FAUSTFLOAT** output) {
		float 	fSlow0 = (fConst1 * float(fslider0));
		float 	fSlow1 = cosf(fSlow0);
		float 	fSlow2 = sinf(fSlow0);
		float 	fSlow3 = (0 - fSlow2);
		int 	iSlow4 = int(float(fcheckbox0));
		float 	fSlow5 = powf(10,(0.05f * float(fslider1)));
		float 	fSlow6 = float(fslider2);
		float 	fSlow7 = (0.0005f * float(fslider3));
		float 	fSlow8 = (0.001f * float(fslider4));
		float 	fSlow9 = float(fslider5);
		float 	fSlow10 = ((int(float(fcheckbox1)))?(0 - fSlow9):fSlow9);
		FAUSTFLOAT* input0 = input[0];
		FAUSTFLOAT* input1 = input[1];
		FAUSTFLOAT* output0 = output[0];
		FAUSTFLOAT* output1 = output[1];
		for (int i=0; i<count; i++) {
			iVec0[0] = 1;
			fRec1[0] = ((fSlow2 * fRec2[1]) + (fSlow1 * fRec1[1]));
			fRec2[0] = (((fSlow1 * fRec2[1]) + (fSlow3 * fRec1[1])) + (1 - iVec0[1]));
			fbargraph0 = (fRec1[0] + fRec2[0]);
			float fTemp0 = (float)input0[i];
			float fTemp1 = (fSlow5 * ((iSlow4)?0:fTemp0));
			float fTemp2 = ((fSlow6 * fRec0[1]) - fTemp1);
			fVec1[IOTA&2047] = fTemp2;
			float fTemp3 = (fConst0 * (fSlow8 + (fSlow7 * (fRec1[0] + 1))));
			int iTemp4 = int(fTemp3);
			float fTemp5 = floorf(fTemp3);
			fRec0[0] = ((fVec1[(IOTA-int((iTemp4 & 2047)))&2047] * (fTemp5 + (1 - fTemp3))) + ((fTemp3 - fTemp5) * fVec1[(IOTA-int((int((iTemp4 + 1)) & 2047)))&2047]));
			output0[i] = (FAUSTFLOAT)((iSlow4)?fTemp0:(0.5f * (fTemp1 + (fSlow10 * fRec0[0]))));
			float fTemp6 = (float)input1[i];
			float fTemp7 = (fSlow5 * ((iSlow4)?0:fTemp6));
			float fTemp8 = ((fSlow6 * fRec3[1]) - fTemp7);
			fVec2[IOTA&2047] = fTemp8;
			float fTemp9 = (fConst0 * (fSlow8 + (fSlow7 * (fRec2[0] + 1))));
			int iTemp10 = int(fTemp9);
			float fTemp11 = floorf(fTemp9);
			fRec3[0] = ((fVec2[(IOTA-int((iTemp10 & 2047)))&2047] * (fTemp11 + (1 - fTemp9))) + ((fTemp9 - fTemp11) * fVec2[(IOTA-int((int((iTemp10 + 1)) & 2047)))&2047]));
			output1[i] = (FAUSTFLOAT)((iSlow4)?fTemp6:(0.5f * (fTemp7 + (fSlow10 * fRec3[0]))));
			// post processing
			fRec3[1] = fRec3[0];
			fRec0[1] = fRec0[0];
			IOTA = IOTA+1;
			fRec2[1] = fRec2[0];
			fRec1[1] = fRec1[0];
			iVec0[1] = iVec0[0];
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
