#metadata=./meta/reuse=2.n=10000.bag=8.beta=1.2
#metadata=./meta/reuse=2.n=10000.bag=8.beta=0.8.poisson
#train

metadata=./meta/reuse=2.n=10000.bag=8.beta=1.2.poisson
CUDA_VISIBLE_DEVICES=0,1,2,3 python fully-supervised-upperbound.py --comment "train_ResNet18" -bs 128 -e 100 --lr 0.1 --base_classifier "Resnet18" --meta-path=$metadata
CUDA_VISIBLE_DEVICES=0,1,2,3 python fully-supervised-upperbound.py --comment "train_ResNet34" -bs 128 -e 100 --lr 0.1 --base_classifier "Resnet34" --meta-path=$metadata
CUDA_VISIBLE_DEVICES=0,1,2,3 python fully-supervised-upperbound.py --comment "train_ResNet50" -bs 128 -e 100 --lr 0.1 --base_classifier "Resnet50" --meta-path=$metadata
CUDA_VISIBLE_DEVICES=0,1,2,3 python fully-supervised-upperbound.py --comment "train_Mobilenet_v2" -bs 128 -e 100 --lr 0.1 --base_classifier "Mobilenet_v2" --meta-path=$metadata

# evaluate on fully supervised data
#model_name=fully-supervised-ckpt_resnet18_0.1_256.pth
#CUDA_VISIBLE_DEVICES=2,3 python fully-supervised-upperbound.py --comment "evaluate_on_fully_supervised_data" -bs 64 -eval 2 --base_classifier "resnet18" --meta-path=$metadata --model-name $model_name


#CUDA_VISIBLE_DEVICES=0 python fully-supervised-upperbound.py --comment "evaluate_on_fully_supervised_data" -bs 128 -eval 2 --base_classifier "Resnet18" --meta-path=$metadata --model-name $model_name
#CUDA_VISIBLE_DEVICES=0 python fully-supervised-upperbound.py --comment "evaluate_on_fully_supervised_data" -bs 128 -eval 2 --base_classifier "Resnet34" --meta-path=$metadata --model-name $model_name
#CUDA_VISIBLE_DEVICES=0 python fully-supervised-upperbound.py --comment "evaluate_on_fully_supervised_data" -bs 128 -eval 2 --base_classifier "Resnet50" --meta-path=$metadata --model-name $model_name

# evaluate on semi-supervised data
#CUDA_VISIBLE_DEVICES=0,1,2,3 python fully-supervised-upperbound.py --comment "evaluate_on_semi-supervised_data" -bs 64 -eval 0 --meta-path=$metadata
