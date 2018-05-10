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

input_dir=input
output_dir=output

mkdir -p $output_dir
tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT
#rm -rf tmp
#mkdir -p tmp
#tmpdir='tmp'

test_portion=0.1
valid_portion=0.2
train_portion=$(python -c "print(1.0-$test_portion-$valid_portion)")

    # Do word segmentation for input/data.txt and get label list
# prepare utf-16le lexicon file
iconv -f utf8 -t utf-16le $input_dir/lex.txt > $tmpdir/lex_utf16.txt
sed -i '1s/^\(\xff\xfe\)\?/\xff\xfe/' $tmpdir/lex_utf16.txt
# prepare data for word segmentation
head -n1 $input_dir/data.csv > $tmpdir/head
tail -n +2 $input_dir/data.csv > $tmpdir/data_wo_head
cut -d ',' -f 1 $tmpdir/data_wo_head > $tmpdir/t1
sort -u $tmpdir/t1 > $output_dir/label_list.txt
cut -d ',' -f 2 $tmpdir/data_wo_head > $tmpdir/t2
iconv -f utf8 -t utf-16le $tmpdir/t2 > $tmpdir/t2_utf16
sed -i '1s/^\(\xff\xfe\)\?/\xff\xfe/' $tmpdir/t2_utf16
# do word segmentation
tools/wbUI_2 $tmpdir/lex_utf16.txt $tmpdir/t2_utf16 $tmpdir/t2_wb_utf16
# get output data file
iconv -f utf-16le $tmpdir/t2_wb_utf16 -t utf8 > $tmpdir/t2_wb
dos2unix $tmpdir/t2_wb 2> /dev/null
sed -i 's/\t/ /g' $tmpdir/t2_wb
paste -d ',' $tmpdir/t1 $tmpdir/t2_wb > $tmpdir/t1_t2wb
cat $tmpdir/head > $output_dir/data.csv
cat $tmpdir/t1_t2wb >> $output_dir/data.csv
rm $tmpdir/*


    # Shuffle data
head -n1 $output_dir/data.csv > $tmpdir/head
tail -n +2 $output_dir/data.csv > $tmpdir/data_wo_head
shuf $tmpdir/data_wo_head > $tmpdir/data_shuffled
cat $tmpdir/head > $output_dir/data_shuffled.csv
cat $tmpdir/data_shuffled >> $output_dir/data_shuffled.csv
rm $tmpdir/*


    # Split train, valid and test data
head -n1 $output_dir/data_shuffled.csv > $tmpdir/head
tail -n +2 $output_dir/data_shuffled.csv > $tmpdir/data_wo_head
total_num=`wc -l $tmpdir/data_wo_head | awk '{print $1}'`
test_num=$(python -c "print(int($total_num * $test_portion))")
valid_num=$(python -c "print(int($total_num * $valid_portion))")
train_num=$(python -c "print(int($total_num - ($test_num + $valid_num)))")
echo "data total number = $total_num"
echo "train data number = $train_num ($train_portion)"
echo "valid data number = $valid_num ($valid_portion)"
echo "test  data number = $test_num ($test_portion)"
head -n $train_num $tmpdir/data_wo_head > $tmpdir/data_train
tail -n +$(($train_num+1)) $tmpdir/data_wo_head | head -n $valid_num > $tmpdir/data_valid
tail -n $test_num $tmpdir/data_wo_head > $tmpdir/data_test

cat $tmpdir/head > $output_dir/data_train.csv
cat $tmpdir/data_train >> $output_dir/data_train.csv
cat $tmpdir/head > $output_dir/data_valid.csv
cat $tmpdir/data_valid >> $output_dir/data_valid.csv
cat $tmpdir/head > $output_dir/data_test.csv
cat $tmpdir/data_test >> $output_dir/data_test.csv

echo "Preprocess Data Finish ($SECONDS sec.)"

