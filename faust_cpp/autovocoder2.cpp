//----------------------------------------------------------
// name: "autovocoder2"
//
// Code generated with Faust 0.9.95 (http://faust.grame.fr)
//----------------------------------------------------------

/* link with  */
#include <math.h>
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
	float 	fConst0;
	float 	fConst1;
	float 	fConst2;
	float 	fConst3;
	float 	fConst4;
	float 	fConst5;
	float 	fConst6;
	float 	fConst7;
	float 	fRec2[3];
	float 	fConst8;
	float 	fRec1[2];
	float 	fRec0[2];
	float 	fConst9;
	float 	fConst10;
	float 	fConst11;
	float 	fConst12;
	float 	fConst13;
	float 	fRec5[3];
	float 	fConst14;
	float 	fRec4[2];
	float 	fRec3[2];
	float 	fConst15;
	float 	fConst16;
	float 	fConst17;
	float 	fConst18;
	float 	fConst19;
	float 	fRec8[3];
	float 	fConst20;
	float 	fRec7[2];
	float 	fRec6[2];
	float 	fConst21;
	float 	fConst22;
	float 	fConst23;
	float 	fConst24;
	float 	fConst25;
	float 	fRec11[3];
	float 	fConst26;
	float 	fRec10[2];
	float 	fRec9[2];
	float 	fConst27;
	float 	fConst28;
	float 	fConst29;
	float 	fConst30;
	float 	fConst31;
	float 	fRec14[3];
	float 	fConst32;
	float 	fRec13[2];
	float 	fRec12[2];
	float 	fConst33;
	float 	fConst34;
	float 	fConst35;
	float 	fConst36;
	float 	fConst37;
	float 	fRec17[3];
	float 	fConst38;
	float 	fRec16[2];
	float 	fRec15[2];
	float 	fConst39;
	float 	fConst40;
	float 	fConst41;
	float 	fConst42;
	float 	fConst43;
	float 	fRec20[3];
	float 	fConst44;
	float 	fRec19[2];
	float 	fRec18[2];
	float 	fConst45;
	float 	fConst46;
	float 	fConst47;
	float 	fConst48;
	float 	fConst49;
	float 	fRec23[3];
	float 	fConst50;
	float 	fRec22[2];
	float 	fRec21[2];
	int fSamplingFreq;

  public:
	virtual void metadata(Meta* m) { 
		m->declare("name", "autovocoder2");
		m->declare("vaeffect.lib/name", "Faust Virtual Analog Filter Effect Library");
		m->declare("vaeffect.lib/version", "0.0");
		m->declare("analyzer.lib/name", "Faust Analyzer Library");
		m->declare("analyzer.lib/version", "0.0");
		m->declare("signal.lib/name", "Faust Signal Routing Library");
		m->declare("signal.lib/version", "0.0");
		m->declare("basic.lib/name", "Faust Basic Element Library");
		m->declare("basic.lib/version", "0.0");
		m->declare("math.lib/name", "Faust Math Library");
		m->declare("math.lib/version", "2.0");
		m->declare("math.lib/author", "GRAME");
		m->declare("math.lib/copyright", "GRAME");
		m->declare("math.lib/license", "LGPL with exception");
		m->declare("filter.lib/name", "Faust Filter Library");
		m->declare("filter.lib/version", "2.0");
	}

	virtual int getNumInputs() { return 1; }
	virtual int getNumOutputs() { return 1; }
	static void classInit(int samplingFreq) {
	}
	virtual void instanceConstants(int samplingFreq) {
		fSamplingFreq = samplingFreq;
		fConst0 = min(1.92e+05f, max(1.0f, (float)fSamplingFreq));
		fConst1 = expf((0 - (1e+01f / fConst0)));
		fConst2 = expf((0 - (2.0f / fConst0)));
		fConst3 = tanf((40212.387f / fConst0));
		fConst4 = (2 * (1 - (1.0f / faustpower<2>(fConst3))));
		fConst5 = (1.0f / fConst3);
		fConst6 = (((fConst5 + -0.541498f) / fConst3) + 1);
		fConst7 = (1.0f / (((fConst5 + 0.541498f) / fConst3) + 1));
		fConst8 = (0 - fConst5);
		fConst9 = tanf((18437.46f / fConst0));
		fConst10 = (2 * (1 - (1.0f / faustpower<2>(fConst9))));
		fConst11 = (1.0f / fConst9);
		fConst12 = (((fConst11 + -0.541498f) / fConst9) + 1);
		fConst13 = (1.0f / (((fConst11 + 0.541498f) / fConst9) + 1));
		fConst14 = (0 - fConst11);
		fConst15 = tanf((8453.613f / fConst0));
		fConst16 = (2 * (1 - (1.0f / faustpower<2>(fConst15))));
		fConst17 = (1.0f / fConst15);
		fConst18 = (((fConst17 + -0.541498f) / fConst15) + 1);
		fConst19 = (1.0f / (((fConst17 + 0.541498f) / fConst15) + 1));
		fConst20 = (0 - fConst17);
		fConst21 = tanf((3875.9985f / fConst0));
		fConst22 = (2 * (1 - (1.0f / faustpower<2>(fConst21))));
		fConst23 = (1.0f / fConst21);
		fConst24 = (((fConst23 + -0.541498f) / fConst21) + 1);
		fConst25 = (1.0f / (((fConst23 + 0.541498f) / fConst21) + 1));
		fConst26 = (0 - fConst23);
		fConst27 = tanf((1777.1532f / fConst0));
		fConst28 = (2 * (1 - (1.0f / faustpower<2>(fConst27))));
		fConst29 = (1.0f / fConst27);
		fConst30 = (((fConst29 + -0.541498f) / fConst27) + 1);
		fConst31 = (1.0f / (((fConst29 + 0.541498f) / fConst27) + 1));
		fConst32 = (0 - fConst29);
		fConst33 = tanf((814.8283f / fConst0));
		fConst34 = (2 * (1 - (1.0f / faustpower<2>(fConst33))));
		fConst35 = (1.0f / fConst33);
		fConst36 = (((fConst35 + -0.541498f) / fConst33) + 1);
		fConst37 = (1.0f / (((fConst35 + 0.541498f) / fConst33) + 1));
		fConst38 = (0 - fConst35);
		fConst39 = tanf((373.60043f / fConst0));
		fConst40 = (2 * (1 - (1.0f / faustpower<2>(fConst39))));
		fConst41 = (1.0f / fConst39);
		fConst42 = (((fConst41 + -0.541498f) / fConst39) + 1);
		fConst43 = (1.0f / (((fConst41 + 0.541498f) / fConst39) + 1));
		fConst44 = (0 - fConst41);
		fConst45 = tanf((171.29655f / fConst0));
		fConst46 = (2 * (1 - (1.0f / faustpower<2>(fConst45))));
		fConst47 = (1.0f / fConst45);
		fConst48 = (((fConst47 + -0.541498f) / fConst45) + 1);
		fConst49 = (1.0f / (((fConst47 + 0.541498f) / fConst45) + 1));
		fConst50 = (0 - fConst47);
	}
	virtual void instanceResetUserInterface() {
	}
	virtual void instanceClear() {
		for (int i=0; i<3; i++) fRec2[i] = 0;
		for (int i=0; i<2; i++) fRec1[i] = 0;
		for (int i=0; i<2; i++) fRec0[i] = 0;
		for (int i=0; i<3; i++) fRec5[i] = 0;
		for (int i=0; i<2; i++) fRec4[i] = 0;
		for (int i=0; i<2; i++) fRec3[i] = 0;
		for (int i=0; i<3; i++) fRec8[i] = 0;
		for (int i=0; i<2; i++) fRec7[i] = 0;
		for (int i=0; i<2; i++) fRec6[i] = 0;
		for (int i=0; i<3; i++) fRec11[i] = 0;
		for (int i=0; i<2; i++) fRec10[i] = 0;
		for (int i=0; i<2; i++) fRec9[i] = 0;
		for (int i=0; i<3; i++) fRec14[i] = 0;
		for (int i=0; i<2; i++) fRec13[i] = 0;
		for (int i=0; i<2; i++) fRec12[i] = 0;
		for (int i=0; i<3; i++) fRec17[i] = 0;
		for (int i=0; i<2; i++) fRec16[i] = 0;
		for (int i=0; i<2; i++) fRec15[i] = 0;
		for (int i=0; i<3; i++) fRec20[i] = 0;
		for (int i=0; i<2; i++) fRec19[i] = 0;
		for (int i=0; i<2; i++) fRec18[i] = 0;
		for (int i=0; i<3; i++) fRec23[i] = 0;
		for (int i=0; i<2; i++) fRec22[i] = 0;
		for (int i=0; i<2; i++) fRec21[i] = 0;
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
			float fTemp0 = (float)input0[i];
			fRec2[0] = (fTemp0 - (fConst7 * ((fConst6 * fRec2[2]) + (fConst4 * fRec2[1]))));
			float fTemp1 = fabsf((fConst7 * ((fConst5 * fRec2[0]) + (fConst8 * fRec2[2]))));
			float fTemp2 = ((int((fRec0[1] > fTemp1)))?fConst2:fConst1);
			fRec1[0] = ((fRec1[1] * fTemp2) + (fTemp1 * (1.0f - fTemp2)));
			fRec0[0] = fRec1[0];
			fRec5[0] = (fTemp0 - (fConst13 * ((fConst12 * fRec5[2]) + (fConst10 * fRec5[1]))));
			float fTemp3 = fabsf((fConst13 * ((fConst11 * fRec5[0]) + (fConst14 * fRec5[2]))));
			float fTemp4 = ((int((fRec3[1] > fTemp3)))?fConst2:fConst1);
			fRec4[0] = ((fRec4[1] * fTemp4) + (fTemp3 * (1.0f - fTemp4)));
			fRec3[0] = fRec4[0];
			fRec8[0] = (fTemp0 - (fConst19 * ((fConst18 * fRec8[2]) + (fConst16 * fRec8[1]))));
			float fTemp5 = fabsf((fConst19 * ((fConst17 * fRec8[0]) + (fConst20 * fRec8[2]))));
			float fTemp6 = ((int((fRec6[1] > fTemp5)))?fConst2:fConst1);
			fRec7[0] = ((fRec7[1] * fTemp6) + (fTemp5 * (1.0f - fTemp6)));
			fRec6[0] = fRec7[0];
			fRec11[0] = (fTemp0 - (fConst25 * ((fConst24 * fRec11[2]) + (fConst22 * fRec11[1]))));
			float fTemp7 = fabsf((fConst25 * ((fConst23 * fRec11[0]) + (fConst26 * fRec11[2]))));
			float fTemp8 = ((int((fRec9[1] > fTemp7)))?fConst2:fConst1);
			fRec10[0] = ((fRec10[1] * fTemp8) + (fTemp7 * (1.0f - fTemp8)));
			fRec9[0] = fRec10[0];
			fRec14[0] = (fTemp0 - (fConst31 * ((fConst30 * fRec14[2]) + (fConst28 * fRec14[1]))));
			float fTemp9 = fabsf((fConst31 * ((fConst29 * fRec14[0]) + (fConst32 * fRec14[2]))));
			float fTemp10 = ((int((fRec12[1] > fTemp9)))?fConst2:fConst1);
			fRec13[0] = ((fRec13[1] * fTemp10) + (fTemp9 * (1.0f - fTemp10)));
			fRec12[0] = fRec13[0];
			fRec17[0] = (fTemp0 - (fConst37 * ((fConst36 * fRec17[2]) + (fConst34 * fRec17[1]))));
			float fTemp11 = fabsf((fConst37 * ((fConst35 * fRec17[0]) + (fConst38 * fRec17[2]))));
			float fTemp12 = ((int((fRec15[1] > fTemp11)))?fConst2:fConst1);
			fRec16[0] = ((fRec16[1] * fTemp12) + (fTemp11 * (1.0f - fTemp12)));
			fRec15[0] = fRec16[0];
			fRec20[0] = (fTemp0 - (fConst43 * ((fConst42 * fRec20[2]) + (fConst40 * fRec20[1]))));
			float fTemp13 = fabsf((fConst43 * ((fConst41 * fRec20[0]) + (fConst44 * fRec20[2]))));
			float fTemp14 = ((int((fRec18[1] > fTemp13)))?fConst2:fConst1);
			fRec19[0] = ((fRec19[1] * fTemp14) + (fTemp13 * (1.0f - fTemp14)));
			fRec18[0] = fRec19[0];
			fRec23[0] = (fTemp0 - (fConst49 * ((fConst48 * fRec23[2]) + (fConst46 * fRec23[1]))));
			float fTemp15 = fabsf((fConst49 * ((fConst47 * fRec23[0]) + (fConst50 * fRec23[2]))));
			float fTemp16 = ((int((fRec21[1] > fTemp15)))?fConst2:fConst1);
			fRec22[0] = ((fRec22[1] * fTemp16) + (fTemp15 * (1.0f - fTemp16)));
			fRec21[0] = fRec22[0];
			output0[i] = (FAUSTFLOAT)((((((((fConst49 * ((fRec23[2] * (0 - (fConst47 * fRec21[0]))) + (fConst47 * (fRec23[0] * fRec21[0])))) + (fConst43 * ((fRec20[2] * (0 - (fConst41 * fRec18[0]))) + (fConst41 * (fRec20[0] * fRec18[0]))))) + (fConst37 * ((fRec17[2] * (0 - (fConst35 * fRec15[0]))) + (fConst35 * (fRec17[0] * fRec15[0]))))) + (fConst31 * ((fRec14[2] * (0 - (fConst29 * fRec12[0]))) + (fConst29 * (fRec14[0] * fRec12[0]))))) + (fConst25 * ((fRec11[2] * (0 - (fConst23 * fRec9[0]))) + (fConst23 * (fRec11[0] * fRec9[0]))))) + (fConst19 * ((fRec8[2] * (0 - (fConst17 * fRec6[0]))) + (fConst17 * (fRec8[0] * fRec6[0]))))) + (fConst13 * ((fRec5[2] * (0 - (fConst11 * fRec3[0]))) + (fConst11 * (fRec5[0] * fRec3[0]))))) + (fConst7 * ((fRec2[2] * (0 - (fConst5 * fRec0[0]))) + (fConst5 * (fRec2[0] * fRec0[0])))));
			// post processing
			fRec21[1] = fRec21[0];
			fRec22[1] = fRec22[0];
			fRec23[2] = fRec23[1]; fRec23[1] = fRec23[0];
			fRec18[1] = fRec18[0];
			fRec19[1] = fRec19[0];
			fRec20[2] = fRec20[1]; fRec20[1] = fRec20[0];
			fRec15[1] = fRec15[0];
			fRec16[1] = fRec16[0];
			fRec17[2] = fRec17[1]; fRec17[1] = fRec17[0];
			fRec12[1] = fRec12[0];
			fRec13[1] = fRec13[0];
			fRec14[2] = fRec14[1]; fRec14[1] = fRec14[0];
			fRec9[1] = fRec9[0];
			fRec10[1] = fRec10[0];
			fRec11[2] = fRec11[1]; fRec11[1] = fRec11[0];
			fRec6[1] = fRec6[0];
			fRec7[1] = fRec7[0];
			fRec8[2] = fRec8[1]; fRec8[1] = fRec8[0];
			fRec3[1] = fRec3[0];
			fRec4[1] = fRec4[0];
			fRec5[2] = fRec5[1]; fRec5[1] = fRec5[0];
			fRec0[1] = fRec0[0];
			fRec1[1] = fRec1[0];
			fRec2[2] = fRec2[1]; fRec2[1] = fRec2[0];
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
