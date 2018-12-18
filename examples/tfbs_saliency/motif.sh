# CUDA_VISIBLE_DEVICES=6 python /cluster/alexwu/pytorch-dl-utility/saliency.py
# CUDA_VISIBLE_DEVICES=6 python /cluster/alexwu/pytorch-dl-utility/run_saliency.py -r "/cluster/alexwu/adversarial/test-adv/saliency/tfbs-results/adversarial/wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk/" -d "/cluster/zeng/research/recomb/generic/saber/wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk/CV0/data/"

arch="32x2_16_dna";
data_dir=(/cluster/zeng/research/recomb/generic/saber/*/);
shopt -s extglob
for dir in "${data_dir[@]}"
do
        tfbs_exp=${dir%%+(/)}
        tfbs_exp=${tfbs_exp##*/}
	echo $tfbs_exp

        # calculate saliency scores & create PWMs (baseline training)
        CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/run_saliency.py -r "saliency/tfbs-results/baseline/$tfbs_exp/" -d "/cluster/zeng/research/recomb/generic/saber/$tfbs_exp/CV0/data/" -bs 1024

        # calculate saliency scores & create PWMs (adversarial training)
	CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/run_saliency.py -r "saliency/tfbs-results/adversarial/$tfbs_exp/" -d "/cluster/zeng/research/recomb/generic/saber/$tfbs_exp/CV0/data/" -bs 1024

done
