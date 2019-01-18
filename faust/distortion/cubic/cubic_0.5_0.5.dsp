// Author: Thibault Geoffroy
import("stdfaust.lib");
drive = 0.5;
offset = 0.5;
process = ef.cubicnl_nodc(drive,offset);
