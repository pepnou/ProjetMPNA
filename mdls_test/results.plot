set terminal png size 1200,800
set output "results.png"

set pm3d
set pm3d interpolate 20,20
unset surface

set cntrparam levels disc 0.5,0.7,1,7,9
set contour surface

set grid
set xlabel "number of nodes"
set ylabel "sqrt of matrix side size"
set zlabel "cubicroot of time in sec"
set title "performance of matrix multiplication"

set datafile missing 'NAN'

splot "./results.dat" w l lw 3

