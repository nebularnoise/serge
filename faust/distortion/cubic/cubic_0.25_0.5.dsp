import("stdfaust.lib");
drive = 0.25;
offset = 0.5;
process = ef.cubicnl_nodc(drive,offset);
