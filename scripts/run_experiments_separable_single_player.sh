python benchmark_utils.py \
    --env-ids ALE/Pacman-v5 PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 AirRaidNoFrameskip-v4 AmidarNoFrameskip-v4 AssaultNoFrameskip-v4 BowlingNoFrameskip-v4 ALE/Tetris-v5\
    --command "poetry run python ppo_separable_unclamped_atari_from_scratch.py" \
    --num-seeds 1 \
    --workers 9 \