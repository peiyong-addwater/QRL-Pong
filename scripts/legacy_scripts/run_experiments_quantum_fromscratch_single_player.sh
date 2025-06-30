python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4 FreewayNoFrameskip-v4 BoxingNoFrameskip-v4 TennisNoFrameskip-v4\
    --command "poetry run python ppo_separable_unclamped_atari_from_scratch.py --backbone_out_dim 18" \
    --num-seeds 1 \
    --workers 12 \

python benchmark_utils.py \
    --env-ids PongNoFrameskip-v4 FreewayNoFrameskip-v4 BoxingNoFrameskip-v4 TennisNoFrameskip-v4\
    --command "poetry run python ppo_entangled_unclamped_atari_from_scratch.py --backbone_out_dim 18" \
    --num-seeds 1 \
    --workers 12 \