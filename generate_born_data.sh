# This script repeatedly runs the command
# python test.py --run_id born_series_XXX
# where XXX is a number from 1 to N, to generate the born series data.
# It is single-threaded, so it is not very efficient.

N=1000
for i in `seq 1 $N`; do
    echo "Running test.py --run_id born_series_$i"
    python test.py --run_id born_series_$i
done
