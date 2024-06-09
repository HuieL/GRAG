for seed in 3 4 5 6
do

# a) grag
python train.py --dataset expla_graphs --model_name graph_llm --seed $seed
python train.py --dataset webqsp --model_name graph_llm --seed $seed

# b) grag + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --seed $seed
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --seed $seed
done
