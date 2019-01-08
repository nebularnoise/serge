import("stdfaust.lib");
drive = 0.75;
offset = 0;
process = ef.cubicnl_nodc(drive,offset);
