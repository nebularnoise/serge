// Author: Thibault Geoffroy
import("stdfaust.lib");
drive = 0.25;
offset = 0;
process = ef.cubicnl_nodc(drive,offset);
