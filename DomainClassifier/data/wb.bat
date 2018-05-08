@echo off

CutUI 0 "\t" train_utf16.txt $t1
CutUI 1 "\t" train_utf16.txt $t2
WBUI_2 lex_utf16.txt $t2 $t3
ReplaceUI "\t" " " $t3 $t4
LineMergeUI $t1 $t4 train_wb_utf16.csv ","
cmd /C TYPE train_wb_utf16.csv > train_wb_utf8.csv

CutUI 0 "\t" valid_utf16.txt $t1
CutUI 1 "\t" valid_utf16.txt $t2
WBUI_2 lex_utf16.txt $t2 $t3
ReplaceUI "\t" " " $t3 $t4
LineMergeUI $t1 $t4 valid_wb_utf16.csv ","
cmd /C TYPE valid_wb_utf16.csv > valid_wb_utf8.csv

CutUI 0 "\t" test_utf16.txt $t1
CutUI 1 "\t" test_utf16.txt $t2
WBUI_2 lex_utf16.txt $t2 $t3
ReplaceUI "\t" " " $t3 $t4
LineMergeUI $t1 $t4 test_wb_utf16.csv ","
cmd /C TYPE test_wb_utf16.csv > test_wb_utf8.csv

del $t1 $t2 $t3 $t4