// Author: Thibault Geoffroy
import("stdfaust.lib");
res = 0.75;
freq = 3200;
process = ve.moog_vcf_2bn(res, freq);
