//----------------------------------------------------------
// name: "freeverb_demo"
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
	FAUSTFLOAT 	fslider0;
	float 	fConst0;
	float 	fConst1;
	float 	fRec9[2];
	FAUSTFLOAT 	fslider1;
	float 	fConst2;
	FAUSTFLOAT 	fslider2;
	int 	IOTA;
	float 	fVec0[8192];
	int 	iConst3;
	float 	fRec8[2];
	float 	fRec11[2];
	float 	fVec1[8192];
	int 	iConst4;
	float 	fRec10[2];
	float 	fRec13[2];
	float 	fVec2[8192];
	int 	iConst5;
	float 	fRec12[2];
	float 	fRec15[2];
	float 	fVec3[8192];
	int 	iConst6;
	float 	fRec14[2];
	float 	fRec17[2];
	float 	fVec4[8192];
	int 	iConst7;
	float 	fRec16[2];
	float 	fRec19[2];
	float 	fVec5[8192];
	int 	iConst8;
	float 	fRec18[2];
	float 	fRec21[2];
	float 	fVec6[8192];
	int 	iConst9;
	float 	fRec20[2];
	float 	fRec23[2];
	float 	fVec7[8192];
	int 	iConst10;
	float 	fRec22[2];
	float 	fVec8[1024];
	int 	iConst11;
	int 	iConst12;
	float 	fRec6[2];
	float 	fVec9[1024];
	int 	iConst13;
	int 	iConst14;
	float 	fRec4[2];
	float 	fVec10[1024];
	int 	iConst15;
	int 	iConst16;
	float 	fRec2[2];
	float 	fVec11[1024];
	int 	iConst17;
	int 	iConst18;
	float 	fRec0[2];
	float 	fRec33[2];
	float 	fVec12[8192];
	FAUSTFLOAT 	fslider3;
	float 	fConst19;
	float 	fRec32[2];
	float 	fRec35[2];
	float 	fVec13[8192];
	float 	fRec34[2];
	float 	fRec37[2];
	float 	fVec14[8192];
	float 	fRec36[2];
	float 	fRec39[2];
	float 	fVec15[8192];
	float 	fRec38[2];
	float 	fRec41[2];
	float 	fVec16[8192];
	float 	fRec40[2];
	float 	fRec43[2];
	float 	fVec17[8192];
	float 	fRec42[2];
	float 	fRec45[2];
	float 	fVec18[8192];
	float 	fRec44[2];
	float 	fRec47[2];
	float 	fVec19[8192];
	float 	fRec46[2];
	float 	fVec20[1024];
	float 	fRec30[2];
	float 	fVec21[1024];
	float 	fRec28[2];
	float 	fVec22[1024];
	float 	fRec26[2];
	float 	fVec23[1024];
	float 	fRec24[2];
	int fSamplingFreq;

  public:
	virtual void metadata(Meta* m) { 
		m->declare("name", "freeverb_demo");
		m->declare("math.lib/name", "Faust Math Library");
		m->declare("math.lib/version", "2.0");
		m->declare("math.lib/author", "GRAME");
		m->declare("math.lib/copyright", "GRAME");
		m->declare("math.lib/license", "LGPL with exception");
		m->declare("reverb.lib/name", "Faust Reverb Library");
		m->declare("reverb.lib/version", "0.0");
		m->declare("filter.lib/name", "Faust Filter Library");
		m->declare("filter.lib/version", "2.0");
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
		fConst1 = (1.764e+04f / fConst0);
		fConst2 = (12348.0f / fConst0);
		iConst3 = int((0.036666665f * fConst0));
		iConst4 = int((0.035306122f * fConst0));
		iConst5 = int((0.033809524f * fConst0));
		iConst6 = int((0.0322449f * fConst0));
		iConst7 = int((0.030748298f * fConst0));
		iConst8 = int((0.028956916f * fConst0));
		iConst9 = int((0.026938776f * fConst0));
		iConst10 = int((0.025306122f * fConst0));
		iConst11 = int((0.0126077095f * fConst0));
		iConst12 = int((int((iConst11 + -1)) & 1023));
		iConst13 = int((0.01f * fConst0));
		iConst14 = int((int((iConst13 + -1)) & 1023));
		iConst15 = int((0.0077324263f * fConst0));
		iConst16 = int((int((iConst15 + -1)) & 1023));
		iConst17 = int((0.0051020407f * fConst0));
		iConst18 = int((int((iConst17 + -1)) & 1023));
		fConst19 = (0.0010430838f * fConst0);
	}
	virtual void instanceResetUserInterface() {
		fslider0 = 0.5f;
		fslider1 = 0.5f;
		fslider2 = 0.3333f;
		fslider3 = 0.5f;
	}
	virtual void instanceClear() {
		for (int i=0; i<2; i++) fRec9[i] = 0;
		IOTA = 0;
		for (int i=0; i<8192; i++) fVec0[i] = 0;
		for (int i=0; i<2; i++) fRec8[i] = 0;
		for (int i=0; i<2; i++) fRec11[i] = 0;
		for (int i=0; i<8192; i++) fVec1[i] = 0;
		for (int i=0; i<2; i++) fRec10[i] = 0;
		for (int i=0; i<2; i++) fRec13[i] = 0;
		for (int i=0; i<8192; i++) fVec2[i] = 0;
		for (int i=0; i<2; i++) fRec12[i] = 0;
		for (int i=0; i<2; i++) fRec15[i] = 0;
		for (int i=0; i<8192; i++) fVec3[i] = 0;
		for (int i=0; i<2; i++) fRec14[i] = 0;
		for (int i=0; i<2; i++) fRec17[i] = 0;
		for (int i=0; i<8192; i++) fVec4[i] = 0;
		for (int i=0; i<2; i++) fRec16[i] = 0;
		for (int i=0; i<2; i++) fRec19[i] = 0;
		for (int i=0; i<8192; i++) fVec5[i] = 0;
		for (int i=0; i<2; i++) fRec18[i] = 0;
		for (int i=0; i<2; i++) fRec21[i] = 0;
		for (int i=0; i<8192; i++) fVec6[i] = 0;
		for (int i=0; i<2; i++) fRec20[i] = 0;
		for (int i=0; i<2; i++) fRec23[i] = 0;
		for (int i=0; i<8192; i++) fVec7[i] = 0;
		for (int i=0; i<2; i++) fRec22[i] = 0;
		for (int i=0; i<1024; i++) fVec8[i] = 0;
		for (int i=0; i<2; i++) fRec6[i] = 0;
		for (int i=0; i<1024; i++) fVec9[i] = 0;
		for (int i=0; i<2; i++) fRec4[i] = 0;
		for (int i=0; i<1024; i++) fVec10[i] = 0;
		for (int i=0; i<2; i++) fRec2[i] = 0;
		for (int i=0; i<1024; i++) fVec11[i] = 0;
		for (int i=0; i<2; i++) fRec0[i] = 0;
		for (int i=0; i<2; i++) fRec33[i] = 0;
		for (int i=0; i<8192; i++) fVec12[i] = 0;
		for (int i=0; i<2; i++) fRec32[i] = 0;
		for (int i=0; i<2; i++) fRec35[i] = 0;
		for (int i=0; i<8192; i++) fVec13[i] = 0;
		for (int i=0; i<2; i++) fRec34[i] = 0;
		for (int i=0; i<2; i++) fRec37[i] = 0;
		for (int i=0; i<8192; i++) fVec14[i] = 0;
		for (int i=0; i<2; i++) fRec36[i] = 0;
		for (int i=0; i<2; i++) fRec39[i] = 0;
		for (int i=0; i<8192; i++) fVec15[i] = 0;
		for (int i=0; i<2; i++) fRec38[i] = 0;
		for (int i=0; i<2; i++) fRec41[i] = 0;
		for (int i=0; i<8192; i++) fVec16[i] = 0;
		for (int i=0; i<2; i++) fRec40[i] = 0;
		for (int i=0; i<2; i++) fRec43[i] = 0;
		for (int i=0; i<8192; i++) fVec17[i] = 0;
		for (int i=0; i<2; i++) fRec42[i] = 0;
		for (int i=0; i<2; i++) fRec45[i] = 0;
		for (int i=0; i<8192; i++) fVec18[i] = 0;
		for (int i=0; i<2; i++) fRec44[i] = 0;
		for (int i=0; i<2; i++) fRec47[i] = 0;
		for (int i=0; i<8192; i++) fVec19[i] = 0;
		for (int i=0; i<2; i++) fRec46[i] = 0;
		for (int i=0; i<1024; i++) fVec20[i] = 0;
		for (int i=0; i<2; i++) fRec30[i] = 0;
		for (int i=0; i<1024; i++) fVec21[i] = 0;
		for (int i=0; i<2; i++) fRec28[i] = 0;
		for (int i=0; i<1024; i++) fVec22[i] = 0;
		for (int i=0; i<2; i++) fRec26[i] = 0;
		for (int i=0; i<1024; i++) fVec23[i] = 0;
		for (int i=0; i<2; i++) fRec24[i] = 0;
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
		ui_interface->openHorizontalBox("Freeverb");
		ui_interface->declare(0, "0", "");
		ui_interface->openVerticalBox("0x00");
		ui_interface->declare(&fslider0, "0", "");
		ui_interface->declare(&fslider0, "style", "knob");
		ui_interface->declare(&fslider0, "tooltip", "Somehow control the    density of the reverb.");
		ui_interface->addVerticalSlider("Damp", &fslider0, 0.5f, 0.0f, 1.0f, 0.025f);
		ui_interface->declare(&fslider1, "1", "");
		ui_interface->declare(&fslider1, "style", "knob");
		ui_interface->declare(&fslider1, "tooltip", "The room size    between 0 and 1 with 1 for the largest room.");
		ui_interface->addVerticalSlider("RoomSize", &fslider1, 0.5f, 0.0f, 1.0f, 0.025f);
		ui_interface->declare(&fslider3, "2", "");
		ui_interface->declare(&fslider3, "style", "knob");
		ui_interface->declare(&fslider3, "tooltip", "Spatial    spread between 0 and 1 with 1 for maximum spread.");
		ui_interface->addVerticalSlider("Stereo Spread", &fslider3, 0.5f, 0.0f, 1.0f, 0.01f);
		ui_interface->closeBox();
		ui_interface->declare(&fslider2, "1", "");
		ui_interface->declare(&fslider2, "tooltip", "The amount of reverb applied to the signal    between 0 and 1 with 1 for the maximum amount of reverb.");
		ui_interface->addVerticalSlider("Wet", &fslider2, 0.3333f, 0.0f, 1.0f, 0.025f);
		ui_interface->closeBox();
	}
	virtual void compute (int count, FAUSTFLOAT** input, FAUSTFLOAT** output) {
		float 	fSlow0 = (fConst1 * float(fslider0));
		float 	fSlow1 = (1 - fSlow0);
		float 	fSlow2 = ((fConst2 * float(fslider1)) + 0.7f);
		float 	fSlow3 = float(fslider2);
		float 	fSlow4 = (0.1f * fSlow3);
		float 	fSlow5 = (1 - fSlow3);
		int 	iSlow6 = int((fConst19 * float(fslider3)));
		int 	iSlow7 = int((iConst3 + iSlow6));
		int 	iSlow8 = int((iConst4 + iSlow6));
		int 	iSlow9 = int((iConst5 + iSlow6));
		int 	iSlow10 = int((iConst6 + iSlow6));
		int 	iSlow11 = int((iConst7 + iSlow6));
		int 	iSlow12 = int((iConst8 + iSlow6));
		int 	iSlow13 = int((iConst9 + iSlow6));
		int 	iSlow14 = int((iConst10 + iSlow6));
		int 	iSlow15 = (iSlow6 + -1);
		int 	iSlow16 = int((int((iConst11 + iSlow15)) & 1023));
		int 	iSlow17 = int((int((iConst13 + iSlow15)) & 1023));
		int 	iSlow18 = int((int((iConst15 + iSlow15)) & 1023));
		int 	iSlow19 = int((int((iConst17 + iSlow15)) & 1023));
		FAUSTFLOAT* input0 = input[0];
		FAUSTFLOAT* input1 = input[1];
		FAUSTFLOAT* output0 = output[0];
		FAUSTFLOAT* output1 = output[1];
		for (int i=0; i<count; i++) {
			fRec9[0] = ((fSlow1 * fRec8[1]) + (fSlow0 * fRec9[1]));
			float fTemp0 = (float)input1[i];
			float fTemp1 = (float)input0[i];
			float fTemp2 = (fSlow4 * (fTemp1 + fTemp0));
			fVec0[IOTA&8191] = (fTemp2 + (fSlow2 * fRec9[0]));
			fRec8[0] = fVec0[(IOTA-iConst3)&8191];
			fRec11[0] = ((fSlow1 * fRec10[1]) + (fSlow0 * fRec11[1]));
			fVec1[IOTA&8191] = (fTemp2 + (fSlow2 * fRec11[0]));
			fRec10[0] = fVec1[(IOTA-iConst4)&8191];
			fRec13[0] = ((fSlow1 * fRec12[1]) + (fSlow0 * fRec13[1]));
			fVec2[IOTA&8191] = (fTemp2 + (fSlow2 * fRec13[0]));
			fRec12[0] = fVec2[(IOTA-iConst5)&8191];
			fRec15[0] = ((fSlow1 * fRec14[1]) + (fSlow0 * fRec15[1]));
			fVec3[IOTA&8191] = (fTemp2 + (fSlow2 * fRec15[0]));
			fRec14[0] = fVec3[(IOTA-iConst6)&8191];
			fRec17[0] = ((fSlow1 * fRec16[1]) + (fSlow0 * fRec17[1]));
			fVec4[IOTA&8191] = (fTemp2 + (fSlow2 * fRec17[0]));
			fRec16[0] = fVec4[(IOTA-iConst7)&8191];
			fRec19[0] = ((fSlow1 * fRec18[1]) + (fSlow0 * fRec19[1]));
			fVec5[IOTA&8191] = (fTemp2 + (fSlow2 * fRec19[0]));
			fRec18[0] = fVec5[(IOTA-iConst8)&8191];
			fRec21[0] = ((fSlow1 * fRec20[1]) + (fSlow0 * fRec21[1]));
			fVec6[IOTA&8191] = (fTemp2 + (fSlow2 * fRec21[0]));
			fRec20[0] = fVec6[(IOTA-iConst9)&8191];
			fRec23[0] = ((fSlow1 * fRec22[1]) + (fSlow0 * fRec23[1]));
			fVec7[IOTA&8191] = (fTemp2 + (fSlow2 * fRec23[0]));
			fRec22[0] = fVec7[(IOTA-iConst10)&8191];
			float fTemp3 = (fRec22[0] + (fRec20[0] + (fRec18[0] + (fRec16[0] + (fRec14[0] + (fRec12[0] + (fRec10[0] + ((0.5f * fRec6[1]) + fRec8[0]))))))));
			fVec8[IOTA&1023] = fTemp3;
			fRec6[0] = fVec8[(IOTA-iConst12)&1023];
			float 	fRec7 = (0 - (0.5f * fVec8[IOTA&1023]));
			float fTemp4 = (fRec7 + ((0.5f * fRec4[1]) + fRec6[1]));
			fVec9[IOTA&1023] = fTemp4;
			fRec4[0] = fVec9[(IOTA-iConst14)&1023];
			float 	fRec5 = (0 - (0.5f * fVec9[IOTA&1023]));
			float fTemp5 = (fRec5 + ((0.5f * fRec2[1]) + fRec4[1]));
			fVec10[IOTA&1023] = fTemp5;
			fRec2[0] = fVec10[(IOTA-iConst16)&1023];
			float 	fRec3 = (0 - (0.5f * fVec10[IOTA&1023]));
			float fTemp6 = (fRec3 + ((0.5f * fRec0[1]) + fRec2[1]));
			fVec11[IOTA&1023] = fTemp6;
			fRec0[0] = fVec11[(IOTA-iConst18)&1023];
			float 	fRec1 = (0 - (0.5f * fVec11[IOTA&1023]));
			output0[i] = (FAUSTFLOAT)(fRec1 + ((fSlow5 * fTemp1) + fRec0[1]));
			fRec33[0] = ((fSlow1 * fRec32[1]) + (fSlow0 * fRec33[1]));
			fVec12[IOTA&8191] = (fTemp2 + (fSlow2 * fRec33[0]));
			fRec32[0] = fVec12[(IOTA-iSlow7)&8191];
			fRec35[0] = ((fSlow1 * fRec34[1]) + (fSlow0 * fRec35[1]));
			fVec13[IOTA&8191] = (fTemp2 + (fSlow2 * fRec35[0]));
			fRec34[0] = fVec13[(IOTA-iSlow8)&8191];
			fRec37[0] = ((fSlow1 * fRec36[1]) + (fSlow0 * fRec37[1]));
			fVec14[IOTA&8191] = (fTemp2 + (fSlow2 * fRec37[0]));
			fRec36[0] = fVec14[(IOTA-iSlow9)&8191];
			fRec39[0] = ((fSlow1 * fRec38[1]) + (fSlow0 * fRec39[1]));
			fVec15[IOTA&8191] = (fTemp2 + (fSlow2 * fRec39[0]));
			fRec38[0] = fVec15[(IOTA-iSlow10)&8191];
			fRec41[0] = ((fSlow1 * fRec40[1]) + (fSlow0 * fRec41[1]));
			fVec16[IOTA&8191] = (fTemp2 + (fSlow2 * fRec41[0]));
			fRec40[0] = fVec16[(IOTA-iSlow11)&8191];
			fRec43[0] = ((fSlow1 * fRec42[1]) + (fSlow0 * fRec43[1]));
			fVec17[IOTA&8191] = (fTemp2 + (fSlow2 * fRec43[0]));
			fRec42[0] = fVec17[(IOTA-iSlow12)&8191];
			fRec45[0] = ((fSlow1 * fRec44[1]) + (fSlow0 * fRec45[1]));
			fVec18[IOTA&8191] = (fTemp2 + (fSlow2 * fRec45[0]));
			fRec44[0] = fVec18[(IOTA-iSlow13)&8191];
			fRec47[0] = ((fSlow1 * fRec46[1]) + (fSlow0 * fRec47[1]));
			fVec19[IOTA&8191] = (fTemp2 + (fSlow2 * fRec47[0]));
			fRec46[0] = fVec19[(IOTA-iSlow14)&8191];
			float fTemp7 = (fRec46[0] + (fRec44[0] + (fRec42[0] + (fRec40[0] + (fRec38[0] + (fRec36[0] + (fRec34[0] + ((0.5f * fRec30[1]) + fRec32[0]))))))));
			fVec20[IOTA&1023] = fTemp7;
			fRec30[0] = fVec20[(IOTA-iSlow16)&1023];
			float 	fRec31 = (0 - (0.5f * fVec20[IOTA&1023]));
			float fTemp8 = (fRec31 + ((0.5f * fRec28[1]) + fRec30[1]));
			fVec21[IOTA&1023] = fTemp8;
			fRec28[0] = fVec21[(IOTA-iSlow17)&1023];
			float 	fRec29 = (0 - (0.5f * fVec21[IOTA&1023]));
			float fTemp9 = (fRec29 + ((0.5f * fRec26[1]) + fRec28[1]));
			fVec22[IOTA&1023] = fTemp9;
			fRec26[0] = fVec22[(IOTA-iSlow18)&1023];
			float 	fRec27 = (0 - (0.5f * fVec22[IOTA&1023]));
			float fTemp10 = (fRec27 + ((0.5f * fRec24[1]) + fRec26[1]));
			fVec23[IOTA&1023] = fTemp10;
			fRec24[0] = fVec23[(IOTA-iSlow19)&1023];
			float 	fRec25 = (0 - (0.5f * fVec23[IOTA&1023]));
			output1[i] = (FAUSTFLOAT)(fRec25 + ((fSlow5 * fTemp0) + fRec24[1]));
			// post processing
			fRec24[1] = fRec24[0];
			fRec26[1] = fRec26[0];
			fRec28[1] = fRec28[0];
			fRec30[1] = fRec30[0];
			fRec46[1] = fRec46[0];
			fRec47[1] = fRec47[0];
			fRec44[1] = fRec44[0];
			fRec45[1] = fRec45[0];
			fRec42[1] = fRec42[0];
			fRec43[1] = fRec43[0];
			fRec40[1] = fRec40[0];
			fRec41[1] = fRec41[0];
			fRec38[1] = fRec38[0];
			fRec39[1] = fRec39[0];
			fRec36[1] = fRec36[0];
			fRec37[1] = fRec37[0];
			fRec34[1] = fRec34[0];
			fRec35[1] = fRec35[0];
			fRec32[1] = fRec32[0];
			fRec33[1] = fRec33[0];
			fRec0[1] = fRec0[0];
			fRec2[1] = fRec2[0];
			fRec4[1] = fRec4[0];
			fRec6[1] = fRec6[0];
			fRec22[1] = fRec22[0];
			fRec23[1] = fRec23[0];
			fRec20[1] = fRec20[0];
			fRec21[1] = fRec21[0];
			fRec18[1] = fRec18[0];
			fRec19[1] = fRec19[0];
			fRec16[1] = fRec16[0];
			fRec17[1] = fRec17[0];
			fRec14[1] = fRec14[0];
			fRec15[1] = fRec15[0];
			fRec12[1] = fRec12[0];
			fRec13[1] = fRec13[0];
			fRec10[1] = fRec10[0];
			fRec11[1] = fRec11[0];
			fRec8[1] = fRec8[0];
			IOTA = IOTA+1;
			fRec9[1] = fRec9[0];
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
