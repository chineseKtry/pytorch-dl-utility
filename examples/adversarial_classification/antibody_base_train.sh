arch="64x1_16";
antibody_arr=("Avastin_a" "Avastin_b" "Enbrel_a" "Enbrel_b" "Herceptin_a" "Herceptin_b" "Lucentis_a" "Lucentis_b" "Mock_a" "Mock_b" "rHSA_A_a" "rHSA_A_b" "rHSA_G_a" "rHSA_G_b");
shopt -s extglob
for antibody in "${antibody_arr[@]}"
do
	CUDA_VISIBLE_DEVICES=6 network=$arch python /cluster/alexwu/pytorch-dl-utility/main.py -adv "fit" -f model_adversarial_class.py -r "baseline-results/$antibody/$arch/" -d "/cluster/geliu/novartis_may/Easy_classification_0604_freq/embed/$antibody/CV0/" -bs 32 -ep 10
done

