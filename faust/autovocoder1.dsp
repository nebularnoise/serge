// Author: Thibault Geoffroy
import("stdfaust.lib");
nbands = 4;
atk = 0.1;
rel = 0.1;
BWRatio = 0.5;
process(e) = ve.vocoder(nbands, atk, rel, BWRatio, e, e);
