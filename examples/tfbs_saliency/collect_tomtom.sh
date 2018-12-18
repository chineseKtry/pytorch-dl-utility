arch="32x2_16_dna";
eps=1.0;
data_dir=(/cluster/zeng/research/recomb/generic/saber/*/);
shopt -s extglob

# create headers
head -1 "saliency/tfbs-results/baseline/wgEncodeAwgTfbsBroadDnd41CtcfUniPk/match_result80/tomtom.tsv" > baseline.tomtom
head -1 "saliency/tfbs-results/baseline/wgEncodeAwgTfbsBroadDnd41CtcfUniPk/match_result80/tomtom.tsv" > adversarial.tomtom

# aggregate test statistics from TOMTOM results (for both baseline & adversarial training)
for dir in "${data_dir[@]}" 
do
	tfbs_exp=${dir%%+(/)}
	tfbs_exp=${tfbs_exp##*/}
	head "saliency/tfbs-results/baseline/$tfbs_exp/match_result_weighted/tomtom.tsv" | sed -n 2p >> "baseline.tomtom"
	head "saliency/tfbs-results/adversarial/$tfbs_exp/match_result_weighted/tomtom.tsv" | sed -n 2p >> "adversarial.tomtom"
done

