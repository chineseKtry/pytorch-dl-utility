arch="64x1_16";
epsilon_vals=(0.001 0.01 0.1);
antibody_arr=("Avastin_a" "Avastin_b" "Enbrel_a" "Enbrel_b" "Herceptin_a" "Herceptin_b" "Lucentis_a" "Lucentis_b" "Mock_a" "Mock_b" "rHSA_A_a" "rHSA_A_b" "rHSA_G_a" "rHSA_G_b");
shopt -s extglob
for antibody in "${antibody_arr[@]}"
do
	for eps in "${epsilon_vals[@]}"
	do
		CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/main.py -adv "normal" -epsilon $eps -f model_adversarial_reg.py -r "adversarial-reg-results/$antibody/$arch/eps=$eps" -d "/cluster/geliu/novartis_may/Easy_classification_0605_freq_reg/embed/$antibody/CV0/" -bs 128 -ep 4
	done
done

