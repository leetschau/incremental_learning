#!/bin/bash

# Compare accuracy curves created in batch mode and incremental mode

inst_no=100000
unit_no=10
drift_pos=5

base_dir=out
batch_out=$base_dir/batch_moa.csv
incr_out=$base_dir/incr_moa.csv

river_raw_out=$base_dir/river_out.log
river_metrics=$base_dir/river_metrics.csv
river_script=$base_dir/river_script.py

ddm_river_script=$base_dir/ddm_river.py
ddm_moa=$base_dir/ddm_moa.csv

plot_file=$base_dir/compare_acc.gp
errfile=$base_dir/err.log

moa="java -Xmx40g -cp /cyberange/apps/moa-release-2021.07.0/lib/moa.jar -javaagent:/cyberange/apps/moa-release-2021.07.0/lib/sizeofag-1.0.4.jar moa.DoTask"
wek="java -Xmx40g -cp /cyberange/apps/weka-3-8-6/weka.jar"

mkdir -p $base_dir
touch $batch_out && rm $batch_out
full=$base_dir/unit$unit_no.arff


echo -e "Create full dataset file: $full"
$moa "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3) \
  -d (generators.SEAGenerator -f 2) -p $((inst_no * drift_pos / unit_no)) -w 1) \
  -f $full -m $inst_no" 2> $errfile
sed -i '/@data/{n;d}' $full  # fix River bug #925


echo -e "\nExtract dataset to $unit_no batch sets ...\n"
for d in $(seq $((unit_no - 1))); do
  $wek weka.filters.unsupervised.instance.RemovePercentage \
    -P ${d}0 -V -i $full -o $base_dir/unit$d.arff -decimal 15 2> $errfile
done


echo -e "Training models on each dataset and calculate accuracy:"
for d in $(seq $unit_no); do
  echo -e "Evaluate accuracy of unit $d ..."
  acc=$($wek weka.classifiers.trees.J48 -force-batch-training -split-percentage 80 \
        -t $base_dir/unit$d.arff 2> $errfile | \
    grep '=== Error on test split ===' -A 2 | \
    grep '^Correctly Classified Instances' | awk '{print $5}')
  echo -e "$((d * inst_no / unit_no)), $acc" >> $batch_out
done
echo -e "Batch Accuracy records saved to $batch_out\n"


echo -e "Evaluate accuracy of MOA incremental model ...\n"
$moa "EvaluateInterleavedTestThenTrain -l trees.HoeffdingTree -s (ArffFileStream -f $full) \
    -i $inst_no -f $((inst_no / unit_no))" > $incr_out 2> $errfile


echo -e "Evaluate accuracy of River incremental model ...\n"
cat << EOF > $river_script
import river
with open('$river_raw_out', 'w') as f:
    river.evaluate.progressive_val_score(
            model=river.tree.HoeffdingTreeClassifier(),
            dataset=river.stream.iter_arff("$full", target='class'),
            metric=river.metrics.Accuracy(),
            print_every=$((inst_no / unit_no)),
            file = f)
EOF

pdm run python $river_script
cat $river_raw_out | tr -d , | awk '{print substr($1, 2, length($1)-2)", " substr($3, 1, length($3)-1)}' > $river_metrics


echo -e "Compare accuracy for batch model, MOA imcremental model, and River incremental model:\n"
cat << EOF > $plot_file
plot "$batch_out" using 1:2 with linespoints title "Batch Model", \
     "$incr_out" using 1:5 with lines title "MOA Incremental Model", \
     "$river_metrics" using 1:2 with lines title "River Incremental Model"
EOF

gnuplot -p $plot_file 2> /dev/null


echo -e "Drift Detection with MOA:\n"
$moa "EvaluateInterleavedTestThenTrain \
    -l (drift.SingleClassifierDrift -d EDDM -l trees.HoeffdingTree) \
    -s (ArffFileStream -f $full) \
    -i $inst_no -f $((inst_no / unit_no))" > $ddm_moa 2>$errfile


echo -e "Drift Detection with River:\n"
cat << EOF > $ddm_river_script
from river import stream, drift, tree, metrics

inp = stream.iter_arff("$full", target='class')
model = tree.HoeffdingTreeClassifier()
ddm = drift.EDDM()

for i, (xi, yi) in enumerate(inp):
    y_pred = model.predict_one(xi)
    model.learn_one(xi, yi)
    in_drift, in_warning = ddm.update(0 if y_pred == yi else 1)
    if in_drift:
        print(f'Change detected at index {i}')
EOF

pdm run python $ddm_river_script

