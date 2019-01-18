// Author: Thibault Geoffroy
import("stdfaust.lib");
nbands = 8;
atk = 0.1;
rel = 0.5;
BWRatio = 1;
process(e) = ve.vocoder(nbands, atk, rel, BWRatio, e, e);
