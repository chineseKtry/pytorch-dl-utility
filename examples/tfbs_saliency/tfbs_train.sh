arch="32x2_16_dna";
eps=1.0;
data_dir=(/cluster/zeng/research/recomb/generic/saber/*/);
shopt -s extglob
for dir in "${data_dir[@]}" 
do
	tfbs_exp=${dir%%+(/)}
	tfbs_exp=${tfbs_exp##*/}
	echo "$/cluster/zeng/research/recomb/generic/saber/$tfbs_exp/CV0/data/"
	
        # baseline (w/o adversarial training)
        CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/main.py -adv "fit" -epsilon $eps -seq "dna" -f model_adversarial_class.py -r "saliency/tfbs-results/baseline/$tfbs_exp/" -d "/cluster/zeng/research/recomb/generic/saber/$tfbs_exp/CV0/data/" -bs 128 -ep 3

        # adversarial training
        CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/main.py -adv "normal" -epsilon $eps -seq "dna" -f model_adversarial_class.py -r "saliency/tfbs-results/adversarial/$tfbs_exp/" -d "/cluster/zeng/research/recomb/generic/saber/$tfbs_exp/CV0/data/" -bs 128 -ep 3
done

