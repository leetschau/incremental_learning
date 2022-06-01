# Compare accuracy curves created in batch mode and incremental mode

SHELL := bash

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

inst_no = 100000
unit_no = 10
drift_pos = 5

base_dir = build
batch_acc = $(base_dir)/batch_moa.csv
hoeffding_moa = $(base_dir)/hoeffding_moa.csv
full = $(base_dir)/unit$(unit_no).arff

hoeffding_river_acc_raw = $(base_dir)/hoeffding_river_acc.log
hoeffding_river = $(base_dir)/hoeffding_river_metrics.csv
hoeffding_river_script = $(base_dir)/hoeffding_river_script.py

ddm_river_script = $(base_dir)/ddm_river.py
ddm_moa = $(base_dir)/ddm_moa.csv

errfile = $(base_dir)/err.log
plot_batch_incr = $(base_dir)/compare_batch_incr.gp
plot_incr = $(base_dir)/compare_incr.gp

java = $(HOME)/.asdf/installs/java/zulu-11.56.19/bin/java -Xmx2g

moa = $(java) -cp $(HOME)/.local/moa-release-2021.07.0/lib/moa.jar -javaagent:/home/leo/.local/moa-release-2021.07.0/lib/sizeofag-1.0.4.jar moa.DoTask
wek = $(java) -cp $(HOME)/.local/weka-3-8-6/weka.jar

hoeffding_river_tpl = hoeffding_template.py
plot_template = plot_template.gp

$(base_dir):
	mkdir $@
init: $(base_dir)

$(full): $(base_dir)
	$(moa) "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3) \
	        -d (generators.SEAGenerator -f 2) \
	        -p $$(( $(inst_no) * $(drift_pos) / $(unit_no) )) -w 1) \
	    -f $(full) -m $(inst_no)" 2> $(errfile)
	sed -i '/@data/{n;d}' $(full)  # fix River bug #925
fulldata: $(full)

extract-subset: $(full)
	for d in $$(seq $$(($(unit_no) - 1))); do
	    $(wek) weka.filters.unsupervised.instance.RemovePercentage \
	      -P $$(($$d * 10)) -V -i $(full) -o $(base_dir)/unit$$d.arff -decimal 15 2> $(errfile)
	done

$(batch_acc): extract-subset
	echo -n > $(batch_acc)
	for d in $$(seq $(unit_no)); do
	  echo -e "Evaluate accuracy of unit $$d ..."
	  acc=$$($(wek) weka.classifiers.trees.J48 -force-batch-training -split-percentage 80 \
	        -t $(base_dir)/unit$$d.arff 2> $(errfile) | \
	    grep '=== Error on test split ===' -A 2 | \
	    grep '^Correctly Classified Instances' | awk '{print $$5}')
	  echo -e "$$(($$d * $(inst_no) / $(unit_no))), $$acc" >> $(batch_acc)
	done
	echo -e "Batch Accuracy records saved to $(batch_acc)\n"

$(hoeffding_moa): $(full)
	$(moa) "EvaluateInterleavedTestThenTrain -l trees.HoeffdingTree -s (ArffFileStream -f $(full)) \
	    -i $(inst_no) -f $$(( $(inst_no) / $(unit_no) ))" > $@ 2> $(errfile)
	echo -e "Acc of HoeffdingTree saved to $(hoeffding_moa)\n"

$(hoeffding_river): $(hoeffding_river_tpl) $(full)
	logfile=$(hoeffding_river_acc_raw) data=$(full) interval=$$(( $(inst_no) / $(unit_no) )) \
	    envsubst < $< > $(hoeffding_river_script)
	pdm run python $(hoeffding_river_script)
	cat $(hoeffding_river_acc_raw) | tr -d , | awk '{print substr($$1, 2, length($$1)-2)", " substr($$3, 1, length($$3)-1)}' > $@

$(plot_batch_incr): $(plot_template) $(batch_acc) $(hoeffding_moa) $(hoeffding_river)
	batch_acc=$(batch_acc) hoeffding_moa=$(hoeffding_moa) hoeffding_river=$(hoeffding_river) \
	    envsubst < $< > $@
	gnuplot -p $@ 2> /dev/null

clean:
	rm -rf $(base_dir)

test:
	eval "$$RIVER_SCRIPT"

