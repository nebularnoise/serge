// Author: Thibault Geoffroy
import("stdfaust.lib");
drive = 1;
offset = 0.5;
process = ef.cubicnl_nodc(drive,offset);
