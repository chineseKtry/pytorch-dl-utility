# run TOMTOM using Saber's script (baseline training results)
python /cluster/alexwu/adversarial/for_alex/match_pwm.py "saliency/tfbs-results/baseline" "best_config/numpy.weighted.pwm"

# run TOMTOM using Saber's scrip (adversarial training results)
python /cluster/alexwu/adversarial/for_alex/match_pwm.py "saliency/tfbs-results/adversarial" "best_config/numpy.weighted.pwm"
