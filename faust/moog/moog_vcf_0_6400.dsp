// Author: Thibault Geoffroy
import("stdfaust.lib");
res = 0;
freq = 6400;
process = ve.moog_vcf_2bn(res, freq);
