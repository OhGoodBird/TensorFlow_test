#!/bin/bash

# input 1: input/data.txt
# format:
# domain_intent,sentence
# $$Weather,一周天氣
# $$Weather,今天天氣怎樣

# input 2: input/lex.txt
# format:
# 的
# 在

mkdir -p output
#tmpdir=$(mktemp -d)
#trap 'rm -rf "${tmpdir}"' EXIT
mkdir -p tmp
tmpdir='tmp'


# get csv header
head -n1 input/data.txt > $tmpdir/head
# data without csv header
tail -n +2 input/data.txt > $tmpdir/data_wo_head
# data labels
cut -d ',' -f 1 $tmpdir/data_wo_head > $tmpdir/t1
# data sentences
cut -d ',' -f 2 $tmpdir/data_wo_head > $tmpdir/t2
# append feff unicode bom
sed -i '1s/^\(\xff\xfe\)\?/\xff\xfe/' $tmpdir/t2
tools/wbUI_2 $input/lex. $tmpdir/t2 