# Call CNN_exp
# /home/ytliu/.conda/envs/multispec/bin/python main.py --net CNN_exp --n_conv 6 --use_mixer 1 --n_fc 0 --n_mixer 7 --train --device cuda:0
# /home/ytliu/.conda/envs/multispec/bin/python main.py --test --net CNN_exp --n_conv 6 --use_mixer 1 --n_fc 0 --n_mixer 7 --device cuda:0 --test_checkpoint checkpoints/qm9s_raman/CNN_exp/2024-10-14_09_29/191_f1_2895.pth

# /home/ytliu/.conda/envs/multispec/bin/python main.py --net ResPeak --use_mixer 0 --n_fc 0 --n_mixer 8 --train --device cuda:2 --ds fcgformer_ir
# # /home/ytliu/.conda/envs/multispec/bin/python main.py --net CNN_SE --test --depth 24 --use_se 1 --use_res 1 --use_mixer 0 --n_fc 0 --n_mixer 2 --test_checkpoint checkpoints/qm9s_raman/CNN_SE/2024-09-29_08_47/190_f1_26.580400.pth --device cuda:2

/home/ytliu/.conda/envs/multispec/bin/python main.py --train --net MLPMixer --depth 0 --use_mixer 1 --use_res 1 --use_se 1 --n_fc 0 --n_mixer 12 --device cuda:0
# /home/ytliu/.conda/envs/multispec/bin/python main.py --test --net MLPMixer --depth 6 --use_mixer 0 --n_mixer 28 --use_se 1 --use_res 1 --n_fc 0 --device cuda:2 --test_checkpoint checkpoints/qm9s_raman/MLPMixer/2024-10-15_15_15mixer28_layer6/89_f1_2495.pth

# Call ResPeak
# /home/ytliu/.conda/envs/multispec/bin/python main.py --train --n_fc 0 --net ResPeak --device cuda:0
# /home/ytliu/.conda/envs/multispec/bin/python main.py --test --net ResPeak --use_mixer 0 --n_fc 0 --n_mixer 8 --test --device cuda:2 --test_checkpoint checkpoints/qm9s_raman/ResPeak/2024-10-15_08_31/188_f1_2764.pth