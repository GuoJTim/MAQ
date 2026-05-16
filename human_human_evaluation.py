import argparse
import csv
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    gym = types.ModuleType("gym")
    gym.make = lambda env_id: (_ for _ in ()).throw(ModuleNotFoundError("gym is not installed"))
    gym_wrappers = types.ModuleType("gym.wrappers")
    gym_wrappers.TimeLimit = object
    gym.wrappers = gym_wrappers
    sys.modules["gym"] = gym
    sys.modules["gym.wrappers"] = gym_wrappers

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

try:
    import moviepy.video.io.ImageSequenceClip  # noqa: F401
except ModuleNotFoundError:
    moviepy = types.ModuleType("moviepy")
    moviepy_video = types.ModuleType("moviepy.video")
    moviepy_video_io = types.ModuleType("moviepy.video.io")
    image_sequence_clip = types.ModuleType("moviepy.video.io.ImageSequenceClip")
    image_sequence_clip.ImageSequenceClip = object
    moviepy.video = moviepy_video
    moviepy_video.io = moviepy_video_io
    moviepy_video_io.ImageSequenceClip = image_sequence_clip
    sys.modules["moviepy"] = moviepy
    sys.modules["moviepy.video"] = moviepy_video
    sys.modules["moviepy.video.io"] = moviepy_video_io
    sys.modules["moviepy.video.io.ImageSequenceClip"] = image_sequence_clip

try:
    import d4rl  # noqa: F401 - imported for environment registration
except Exception:
    d4rl = None

try:
    from human_similarity_utils import (
        calculate_dtw_distances,
        calculate_min_dtw_distances,
        calculate_state_action_occupancy_mmd,
        calculate_trajectory_quality_diversity,
        calculate_wasserstein_distances,
        write_results_to_csv,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing a dependency required by human_similarity_utils.py. "
        "Run this script in the project evaluation environment with fastdtw, scipy, "
        "scikit-learn, POT/ot, and the existing evaluation dependencies installed. "
        f"Original error: {exc}"
    ) from exc


METRIC_FNS = {
    "dtw": calculate_dtw_distances,
    "wasserstein": calculate_wasserstein_distances,
    "min_dtw": calculate_min_dtw_distances,
    "occupancy": calculate_state_action_occupancy_mmd,
    "traj_qd": calculate_trajectory_quality_diversity,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate held-out human trajectories against reference human trajectories."
    )
    parser.add_argument("--env", required=True, help="Gym/D4RL environment id, e.g. door-human-v1.")
    parser.add_argument("--ratio", default="0.5", help="Dataset split ratio tag used in offline_data filenames.")
    parser.add_argument("--seeds", nargs="+", default=["1", "10", "100"], help="Split seeds to evaluate.")
    parser.add_argument("--offline-dir", default="offline_data", help="Directory containing split pickle files.")
    parser.add_argument("--output-prefix", default="agent_performance", help="CSV filename prefix.")
    parser.add_argument("--force", action="store_true", help="Append rows even when the agent_type already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print results without writing CSV rows.")
    args = parser.parse_args()

    env = make_env(args.env)
    try:
        for seed in args.seeds:
            agent_type = f"human_human_ratio{args.ratio}_seed{seed}"
            output_csv = Path("evaluation_results") / f"{args.output_prefix}_{args.env}.csv"
            if not args.force and row_exists(output_csv, agent_type):
                print(f"Skipping {agent_type}: row already exists in {output_csv}. Use --force to append anyway.")
                continue

            train_path, test_path = split_paths(args.offline_dir, args.env, seed, args.ratio)
            train_trajectories = load_trajectories(train_path)
            test_trajectories = load_trajectories(test_path)
            result = evaluate_human_human_split(
                env=env,
                env_id=args.env,
                seed=seed,
                ratio=args.ratio,
                train_path=train_path,
                test_path=test_path,
                train_trajectories=train_trajectories,
                test_trajectories=test_trajectories,
            )

            print_summary(agent_type, result)
            if not args.dry_run:
                write_results_to_csv(result, args.env, filename_prefix=args.output_prefix)
                print(f"Wrote {agent_type} to {output_csv}")
    finally:
        if env is not None:
            env.close()


def make_env(env_id):
    try:
        return gym.make(env_id)
    except Exception as exc:
        print(f"Warning: could not create env {env_id}; normalized scores will be NaN. Error: {exc}")
        return None


def split_paths(offline_dir, env_id, seed, ratio):
    root = Path(offline_dir)
    train_path = root / f"{env_id}_train_seed{seed}_ratio{ratio}.pkl"
    test_path = root / f"{env_id}_test_seed{seed}_ratio{ratio}.pkl"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split file(s): {', '.join(missing)}")
    return train_path, test_path


def load_trajectories(path):
    with open(path, "rb") as file:
        trajectories = pickle.load(file)
    if not trajectories:
        raise ValueError(f"{path} does not contain any trajectories.")
    return trajectories


def evaluate_human_human_split(
    env,
    env_id,
    seed,
    ratio,
    train_path,
    test_path,
    train_trajectories,
    test_trajectories,
):
    reference_states = [np.asarray(traj["observations"]) for traj in train_trajectories]
    reference_actions = [np.asarray(traj["actions"]) for traj in train_trajectories]
    episode_results = []

    for index, trajectory in enumerate(test_trajectories, start=1):
        episode_data = human_episode_result(env, env_id, trajectory)
        custom_metrics = calculate_custom_metrics(
            episode_data["states"],
            episode_data["actions"],
            reference_states,
            reference_actions,
        )
        episode_data.update(custom_metrics)
        episode_results.append(episode_data)
        print(
            f"  Human test trajectory {index}/{len(test_trajectories)} | "
            f"Score: {episode_data['normalized_score']:.2f} | "
            f"Len: {episode_data['length']} | Raw Reward: {episode_data['reward']:.2f}"
        )

    aggregate_metrics = aggregate_episode_results(episode_results)
    return {
        "env_id": env_id,
        "agent_type": f"human_human_ratio{ratio}_seed{seed}",
        "model_path": f"{train_path};{test_path}",
        "eval_episodes": len(test_trajectories),
        "base_seed": seed,
        **aggregate_metrics,
    }


def human_episode_result(env, env_id, trajectory):
    rewards = np.asarray(trajectory["rewards"])
    actions = np.asarray(trajectory["actions"])
    states = np.asarray(trajectory["observations"])
    total_raw_reward = float(np.sum(rewards))
    normalized_score = np.nan
    if env is not None and hasattr(env, "get_normalized_score"):
        try:
            normalized_score = float(env.get_normalized_score(total_raw_reward) * 100)
        except Exception as exc:
            print(f"Warning: failed to normalize score for {env_id}: {exc}")

    return {
        "states": states,
        "actions": actions,
        "reward": total_raw_reward,
        "normalized_score": normalized_score,
        "length": int(len(actions)),
        "success": True,
        "early_termination": False,
        "default_horizon": default_horizon(env),
    }


def default_horizon(env):
    if env is None:
        return 1000
    return getattr(env, "_max_episode_steps", getattr(env, "_default_horizon", 1000))


def calculate_custom_metrics(agent_states, agent_actions, reference_states, reference_actions):
    episode_custom_metrics = {}
    for name, func in METRIC_FNS.items():
        try:
            metric_results = func(agent_states, agent_actions, reference_states, reference_actions)
        except Exception as exc:
            print(f"Warning: error calculating human-human metric '{name}': {exc}")
            continue
        for key, value in metric_results.items():
            episode_custom_metrics[f"{name}_{key}"] = value
    return episode_custom_metrics


def aggregate_episode_results(episode_results):
    aggregate = {}
    scores = finite_values([res["normalized_score"] for res in episode_results])
    raw_rewards = [res["reward"] for res in episode_results]
    lengths = [res["length"] for res in episode_results]
    successes = [res.get("success", False) for res in episode_results]
    early_terminations = [res.get("early_termination", False) for res in episode_results]
    default_horizon_value = episode_results[0].get("default_horizon", 1000)

    aggregate["normalized_score_mean"] = mean_or_nan(scores)
    aggregate["normalized_score_std"] = std_or_nan(scores)
    aggregate["raw_reward_mean"] = float(np.mean(raw_rewards))
    aggregate["raw_reward_std"] = float(np.std(raw_rewards))
    aggregate["length_mean"] = float(np.mean(lengths))
    aggregate["length_std"] = float(np.std(lengths))
    aggregate["success_rate"] = float(np.mean(successes))
    aggregate["early_termination_rate"] = float(np.mean(early_terminations))
    aggregate["horizon"] = None
    aggregate["default_horizon"] = default_horizon_value

    successful_episodes = [res for res in episode_results if res.get("success", False)]
    non_successful_episodes = [res for res in episode_results if not res.get("success", False)]
    add_length_stats(aggregate, "success", successful_episodes, default_horizon_value)
    add_length_stats(aggregate, "non_success", non_successful_episodes, default_horizon_value)
    aggregate["total_success_episodes"] = len(successful_episodes)
    aggregate["total_non_success_episodes"] = len(non_successful_episodes)

    metric_keys = custom_metric_keys(episode_results)
    add_condition_metric_stats(aggregate, "success", successful_episodes, metric_keys)
    add_condition_metric_stats(aggregate, "non_success", non_successful_episodes, metric_keys)
    add_overall_metric_stats(aggregate, episode_results, metric_keys)
    return aggregate


def add_length_stats(aggregate, prefix, episodes, default_horizon_value):
    if not episodes:
        aggregate[f"{prefix}_length_mean"] = np.nan
        aggregate[f"{prefix}_length_std"] = np.nan
        aggregate[f"{prefix}_length_max"] = np.nan
        aggregate[f"{prefix}_length_min"] = np.nan
        aggregate[f"{prefix}_exceed_default_pct"] = np.nan if prefix == "non_success" else 0.0
        aggregate[f"{prefix}_exceed_default_count"] = 0
        return

    lengths = [res["length"] for res in episodes]
    exceed_default = [length for length in lengths if length > default_horizon_value]
    aggregate[f"{prefix}_length_mean"] = float(np.mean(lengths))
    aggregate[f"{prefix}_length_std"] = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    aggregate[f"{prefix}_length_max"] = float(np.max(lengths))
    aggregate[f"{prefix}_length_min"] = float(np.min(lengths))
    aggregate[f"{prefix}_exceed_default_pct"] = (len(exceed_default) / len(lengths)) * 100 if lengths else 0.0
    aggregate[f"{prefix}_exceed_default_count"] = len(exceed_default)


def add_condition_metric_stats(aggregate, prefix, episodes, metric_keys):
    for key in metric_keys:
        values = finite_values([res.get(key, np.nan) for res in episodes])
        if values:
            aggregate[f"{prefix}_{key}_mean"] = float(np.mean(values))
            aggregate[f"{prefix}_{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
            aggregate[f"{prefix}_{key}_max"] = float(np.max(values))
            aggregate[f"{prefix}_{key}_min"] = float(np.min(values))
        else:
            aggregate[f"{prefix}_{key}_mean"] = np.nan
            aggregate[f"{prefix}_{key}_std"] = np.nan
            aggregate[f"{prefix}_{key}_max"] = np.nan
            aggregate[f"{prefix}_{key}_min"] = np.nan


def add_overall_metric_stats(aggregate, episode_results, metric_keys):
    for key in metric_keys:
        values = finite_values([res.get(key, np.nan) for res in episode_results])
        aggregate[f"{key}_agg_mean"] = float(np.mean(values)) if values else np.nan
        if len(values) > 1:
            aggregate[f"{key}_agg_std"] = float(np.std(values))
        elif values:
            aggregate[f"{key}_agg_std"] = 0.0
        else:
            aggregate[f"{key}_agg_std"] = np.nan


def custom_metric_keys(episode_results):
    keys = []
    for result in episode_results:
        for key in result:
            if any(key.startswith(metric_name) for metric_name in METRIC_FNS):
                if key not in keys:
                    keys.append(key)
    return keys


def finite_values(values):
    return [float(value) for value in values if is_finite_number(value)]


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_or_nan(values):
    return float(np.mean(values)) if values else np.nan


def std_or_nan(values):
    return float(np.std(values)) if values else np.nan


def row_exists(csv_path, agent_type):
    if not csv_path.exists():
        return False
    with open(csv_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("agent_type") == agent_type:
                return True
    return False


def print_summary(agent_type, result):
    print(f"\n--- Human-Human Result: {agent_type} ---")
    for key in [
        "eval_episodes",
        "normalized_score_mean",
        "raw_reward_mean",
        "length_mean",
        "success_rate",
        "dtw_action_dtw_mean_agg_mean",
        "dtw_state_dtw_mean_agg_mean",
        "wasserstein_action_w2_dist_agg_mean",
        "wasserstein_state_w2_dist_agg_mean",
        "occupancy_state_action_mmd_agg_mean",
    ]:
        if key in result:
            print(f"  {key}: {result[key]}")


if __name__ == "__main__":
    main()
