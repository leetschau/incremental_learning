import river
with open('$logfile', 'w') as f:
    river.evaluate.progressive_val_score(
        model=river.tree.HoeffdingTreeClassifier(),
        dataset=river.stream.iter_arff("$data", target='class'),
        metric=river.metrics.Accuracy(),
        print_every=$interval,
        file = f)
