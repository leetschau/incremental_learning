# Compare accuracy curves created in batch mode and incremental mode

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

inst_no = 100000
unit_no = 10
drift_pos = 5

base_dir = out
batch_out = $(base_dir)/batch_moa.csv
incr_out = $(base_dir)/incr_moa.csv
full = $(base_dir)/unit$(unit_no).arff

river_raw_out = $(base_dir)/river_out.log
river_metrics = $(base_dir)/river_metrics.csv
river_script = $(base_dir)/river_script.py

ddm_river_script = $(base_dir)/ddm_river.py
ddm_moa = $(base_dir)/ddm_moa.csv

plot_file = $(base_dir)/compare_acc.gp
errfile = $(base_dir)/err.log

java = $(HOME)/.asdf/installs/java/zulu-11.56.19/bin/java -Xmx2g

moa = $(java) -cp $(HOME)/.local/moa-release-2021.07.0/lib/moa.jar -javaagent:/home/leo/.local/moa-release-2021.07.0/lib/sizeofag-1.0.4.jar moa.DoTask
wek = $(java) -cp $(HOME)/.local/weka-3-8-6/weka.jar

.PHONY: clean

river: clean fullset river-eval
batch: clean fullset extract-subset

clean:
	touch $(base_dir) && rm -rf $(base_dir) && mkdir $(base_dir)

fullset: clean
	$(moa) "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3) \
	        -d (generators.SEAGenerator -f 2) \
	        -p $$(( $(inst_no) * $(drift_pos) / $(unit_no) )) -w 1) \
	    -f $(full) -m $(inst_no)" 2> $(errfile)
	sed -i '/@data/{n;d}' $(full)  # fix River bug #925

extract-subset: fullset
	for d in $$(seq $$(($(unit_no) - 1))); do
	    $(wek) weka.filters.unsupervised.instance.RemovePercentage \
	      -P $$(($$d * 10)) -V -i $(full) -o $(base_dir)/unit$$d.arff -decimal 15 2> $(errfile)
	done

define RIVER_SCRIPT
cat << EOF > $(river_script)
import river
with open('$(river_raw_out)', 'w') as f:
    river.evaluate.progressive_val_score(
        model=river.tree.HoeffdingTreeClassifier(),
        dataset=river.stream.iter_arff("$(full)", target='class'),
        metric=river.metrics.Accuracy(),
        print_every=$$(( $(inst_no) / $(unit_no) )),
        file = f)
EOF
endef

export RIVER_SCRIPT

river-eval: fullset
	eval "$$RIVER_SCRIPT"
	pdm run python $(river_script)
	cat $(river_raw_out) | tr -d , | awk '{print substr($$1, 2, length($$1)-2)", " substr($$3, 1, length($$3)-1)}' > $(river_metrics)

test:
	eval "$$RIVER_SCRIPT"

