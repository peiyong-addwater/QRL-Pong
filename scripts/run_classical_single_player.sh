python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4 FreewayNoFrameskip-v4 BoxingNoFrameskip-v4 TennisNoFrameskip-v4\
    --command "poetry run python ppo_classicalNN_atari.py --backbone_out_dim 18" \
    --num-seeds 3 \
    --workers 12 \

python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4 FreewayNoFrameskip-v4 BoxingNoFrameskip-v4 TennisNoFrameskip-v4\
    --command "poetry run python ppo_classicalNNSineless_atari.py --backbone_out_dim 18" \
    --num-seeds 3 \
    --workers 12 \