plot "$batch_acc" using 1:2 with linespoints title "Batch Model", \
     "$hoeffding_moa" using 1:5 with lines title "HoeffdingTree with MOA", \
     "$hoeffding_river" using 1:2 with lines title "HoeffdingTree with River"

