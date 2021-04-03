bagsize=$1
n_train=$2
n_test=$3
trial=$4
beta=$5
reuse=$6
python ./meta/generate_meta.py -ntr $n_train -nte $n_test -bs $bagsize --trial=$trial --beta $beta --reuse $reuse > "./meta/reuse=$reuse.n=$n_train.bag=$bagsize.beta=$beta.stdout.txt"


