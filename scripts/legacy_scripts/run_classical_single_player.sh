python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4\
    --command "poetry run python ppo_classicalNNSineActor_atari.py --backbone_out_dim 12" \
    --num-seeds 1 \
    --workers 12 \

python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4\
    --command "poetry run python ppo_classicalNNSinelessActor_atari.py --backbone_out_dim 12" \
    --num-seeds 1 \
    --workers 12 \