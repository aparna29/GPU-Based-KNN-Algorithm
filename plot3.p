set terminal png
set output 'compare3.png'

set multiplot layout 1,1
set style data lines
set autoscale

#set xtics 10
#set ytics 20

#set xrange [0:100]
#set yrange [0:220]

set xlabel "Number of dimensions"
set ylabel "Execution time (miliseconds)"

plot "dim-serial.txt" u 1:2 title " Serial - Quick Sort", "dim-parallel.txt" u 1:2 title "Parallel"
unset multiplot
set terminal x11
set output
replot

