python benchmark_utils.py \
    --env-ids MsPacmanNoFrameskip-v4 PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 AirRaidNoFrameskip-v4 AmidarNoFrameskip-v4 AssaultNoFrameskip-v4 BowlingNoFrameskip-v4 SkiingNoFrameskip-v4 FreewayNoFrameskip-v4 AsterixNoFrameskip-v4 SpaceInvadersNoFrameskip-v4\
    --command "poetry run python ppo_separable_unclamped_atari_from_scratch.py" \
    --num-seeds 1 \
    --workers 9 \