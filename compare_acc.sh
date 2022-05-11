#!/bin/bash

# Compare accuracy curves created in batch mode and incremental mode

inst_no=100000
unit_no=10
drift_pos=5
outfile=batch_acc.csv

moa="java -Xmx40g -cp /cyberange/apps/moa-release-2021.07.0/lib/moa.jar -javaagent:/cyberange/apps/moa-release-2021.07.0/lib/sizeofag-1.0.4.jar moa.DoTask"
wek="java -Xmx40g -cp /cyberange/apps/weka-3-8-6/weka.jar"

touch $outfile && rm $outfile
full=unit$unit_no.arff

echo -e "\n=== Create full dataset file: $full ===\n"
$moa "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3) \
  -d (generators.SEAGenerator -f 2) -p $((inst_no * drift_pos / unit_no)) -w 1) \
  -f $full -m $inst_no"

echo -e "\n=== Split dataset ===\n"
for d in $(seq $((unit_no - 1))); do
  $wek weka.filters.unsupervised.instance.RemovePercentage -P ${d}0 -V -i $full -o unit$d.arff -decimal 15
done

echo -e "\n\n=== Training models on each dataset and calc acc: ===\n\n"
for d in $(seq $unit_no); do
  echo -e "\n=== Evaluate accuracy of unit $d ===\n"
  acc=$($wek weka.classifiers.trees.J48 -force-batch-training -split-percentage 80 -t unit$d.arff | \
    grep '=== Error on test split ===' -A 2 | \
    grep '^Correctly Classified Instances' | awk '{print $5}')
  echo -e "$((d * inst_no / unit_no)), $acc" >> $outfile
done
echo -e "\n=== Batch Accuracy records saved to $outfile ===\n"

echo -e "\n=== Calc acc in incremental mode: ===\n"
$moa "EvaluatePrequential -s (ArffFileStream -f $full) -i $inst_no -f $((inst_no / unit_no))" > incr_acc.csv

echo -e "\n=== Compare acc in batch and imcremental modes: ===\n"
gnuplot -p -e 'plot "batch_acc.csv" using 1:2 with lines title "Batch Mode",
                    "incr_acc.csv" using 1:5 with lines title "Incremental Mode"'
