
bagsize=$1
echo "Bagsize is " $bagsize
nsample=$2
echo "N sample is " $nsample
batchsize=$3
echo "batch size is " $batchsize
epoch=$4
echo "Number of epochs is " $epoch
reuse=$5
echo "Reuse is " $reuse
CUDA_DEVICES=$6
echo "Cuda devices to be used are " $CUDA_DEVICES
beta=$7
echo "Sparsity control is " $beta
binary_loss_weight=$8
echo "Binary loss weight is " $binary_loss_weight
entropy_loss_weight=$9
echo "entropy loss weight is " $entropy_loss_weight
count_loss_weight=${10}
echo "count loss weight is " $count_loss_weight
regularize=${11}
echo "Regularization param " $regularize
loss_type=${12}
echo "Loss type under use " $loss_type
entropy_weighted=${13}
echo "weighted entropy is " $entropy_weighted
lr=${14}
echo "Learning rate is " $lr
base_classifier=${15}
echo "Base classifier deployed is " $base_classifier 
#loss_type="l1"
#loss_type="l2"
#train



for trial in 0 1 2 3 4
do
  t=$trial
  echo "Run Resue Bag dataset"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python semi_weak_main.py \
    --comment "uniform.reg=$regularize.loss=$loss_type.n=$nsample.bag=$bagsize.trial=$t.beta=$beta.lr=$lr.bs=$batchsize.ew=$entropy_weighted.blw=$binary_loss_weight.elw=$entropy_loss_weight.clw=$count_loss_weight" \
    --meta-path="./meta/reuse=$reuse.n=$nsample.bag=$bagsize.beta=$beta.uniform" \
    -bs $batchsize \
    -e $epoch \
    -t=$t \
    --loss-type $loss_type \
    --binary-loss-weight=$binary_loss_weight \
    --entropy-loss-weight=$entropy_loss_weight \
    --entropy-weighted $entropy_weighted \
    --regularize $regularize \
    --count-loss-weight $count_loss_weight \
    --bag-size $bagsize \
    --lr $lr \
    --base_classifier $base_classifier
done

#evaluate
#for trial in 0
#do
#  t=$trial
#  echo "Run Resue Bag dataset"
#    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python semi_weak_main.py \
#    --comment "" \
#    --meta-path="./meta/reuse=$reuse.n=$nsample.bag=$bagsize.beta=$beta" \
#    -bs $batchsize \
#    -e $epoch \
#    -t=$t \
#    --loss-type $loss_type \
#    --binary-loss-weight=$binary_loss_weight \
#    --entropy-loss-weight=$entropy_loss_weight \
#    --entropy-weighted $entropy_weighted \
#    --regularize $regularize \
#    --count-loss-weight $count_loss_weight \
#    --bag-size $bagsize \
#    --lr $lr \
#    -eval
#done
