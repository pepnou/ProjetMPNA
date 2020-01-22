set terminal png size 1200,800
set output "acc.png"

set pm3d
set pm3d interpolate 20,20

unset surface

set cntrparam levels disc 1
set contour surface

set view map

set xlabel "number of nodes"
set ylabel "sqrt of matrix side size"
set zlabel "acceleration"
set title "acceleration of matrix multiplication"

set datafile missing 'NAN'

splot "./acc.dat" w l lw 3

