import sys
import unittest
from collections import deque
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from seed_utils import evaluation_seed
from train import (
    EpisodeConsoleCallback,
    MasteryCurriculum,
    TrainingProfilerCallback,
    TrainingProgressCallback,
    apply_bridging_resume_safety,
    _profile_mean,
    _profile_percentile,
    close_training_environment,
    configure_torch_threads,
    initialize_bed_v2_action_biases,
    make_environment_factory,
    neutralize_bed_v2_tool_head,
    parse_args,
    resolve_algorithm_kwargs,
    resolve_training_device,
)
from task_registry import get_task_preset


class VectorEnvironmentTest(unittest.TestCase):
    def test_training_cli_defaults_to_measured_lean_backend_and_cpu(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--task", "bed_defense_break_v2"]):
            args = parse_args()
            self.assertEqual(args.vec_env, "lean-shared")
            self.assertEqual(args.device, "cpu")

    def test_auto_training_device_uses_measured_cpu_for_structured_mlps(self) -> None:
        self.assertEqual(resolve_training_device("auto"), "cpu")
        self.assertEqual(resolve_training_device("cpu"), "cpu")
        self.assertEqual(resolve_training_device("cuda"), "cuda")

    def test_bed_v2_fresh_policy_biases_stable_aim_and_attack_exploration(self) -> None:
        action_space = spaces.MultiDiscrete([9, 9, 3, 2])
        action_net = torch.nn.Linear(4, int(np.sum(action_space.nvec)))
        model = SimpleNamespace(
            action_space=action_space,
            policy=SimpleNamespace(action_net=action_net),
        )

        initialize_bed_v2_action_biases(model)
        bias = action_net.bias.detach()
        yaw = torch.softmax(bias[0:9], dim=0)
        pitch = torch.softmax(bias[9:18], dim=0)
        tools = torch.softmax(bias[18:21], dim=0)
        attack = torch.softmax(bias[21:23], dim=0)

        self.assertEqual(int(torch.argmax(yaw)), 4)
        self.assertEqual(int(torch.argmax(pitch)), 4)
        self.assertGreater(float(yaw[4]), 0.35)
        self.assertGreater(float(pitch[4]), 0.35)
        np.testing.assert_allclose(tools.numpy(), [1 / 3] * 3, rtol=1e-6)
        self.assertGreater(float(attack[1]), 0.73)

    def test_bed_v2_tool_neutralization_changes_only_tool_output_rows(self) -> None:
        action_space = spaces.MultiDiscrete([9, 9, 3, 2])
        action_net = torch.nn.Linear(4, int(np.sum(action_space.nvec)))
        optimizer = torch.optim.Adam(action_net.parameters(), lr=1e-3)
        with torch.no_grad():
            action_net.weight.fill_(1.0)
            action_net.bias.fill_(1.0)
        action_net(torch.ones((1, 4))).sum().backward()
        optimizer.step()
        before_weight = action_net.weight.detach().clone()
        before_bias = action_net.bias.detach().clone()
        model = SimpleNamespace(
            action_space=action_space,
            policy=SimpleNamespace(action_net=action_net, optimizer=optimizer),
        )

        neutralize_bed_v2_tool_head(model)

        torch.testing.assert_close(action_net.weight[:18], before_weight[:18])
        torch.testing.assert_close(action_net.weight[21:], before_weight[21:])
        torch.testing.assert_close(action_net.bias[:18], before_bias[:18])
        torch.testing.assert_close(action_net.bias[21:], before_bias[21:])
        torch.testing.assert_close(
            action_net.weight[18:21], torch.zeros_like(action_net.weight[18:21])
        )
        torch.testing.assert_close(
            action_net.bias[18:21], torch.zeros_like(action_net.bias[18:21])
        )
        for parameter in (action_net.weight, action_net.bias):
            state = optimizer.state[parameter]
            torch.testing.assert_close(
                state["exp_avg"][18:21],
                torch.zeros_like(state["exp_avg"][18:21]),
            )
            torch.testing.assert_close(
                state["exp_avg_sq"][18:21],
                torch.zeros_like(state["exp_avg_sq"][18:21]),
            )

    def test_training_profiler_statistics_and_interval_validation(self) -> None:
        self.assertEqual(_profile_mean([]), 0.0)
        self.assertEqual(_profile_mean([1.0, 3.0]), 2.0)
        self.assertEqual(_profile_percentile([4.0, 1.0, 2.0, 3.0], 0.95), 4.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingProfilerCallback(0)

    def test_torch_threads_are_bounded(self) -> None:
        intraop, interop = configure_torch_threads(1)
        self.assertEqual((intraop, interop), (1, 1))

    def test_torch_thread_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            configure_torch_threads(0)

    def test_vector_rollout_is_scaled_and_batch_aligned(self) -> None:
        defaults = {"n_steps": 2048, "batch_size": 256}
        expected = {1: 2048, 4: 512, 7: 256, 10: 256}
        for num_envs, n_steps in expected.items():
            kwargs = resolve_algorithm_kwargs(defaults, num_envs=num_envs)
            self.assertEqual(kwargs["n_steps"], n_steps)
            self.assertEqual((n_steps * num_envs) % kwargs["batch_size"], 0)

    def test_explicit_rollout_override_is_preserved(self) -> None:
        kwargs = resolve_algorithm_kwargs(
            {"n_steps": 2048, "batch_size": 256},
            num_envs=10,
            n_steps_override=128,
        )
        self.assertEqual(kwargs["n_steps"], 128)

    def test_bridging_resume_uses_conservative_ppo_settings(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=2.5e-4)
        model = SimpleNamespace(
            learning_rate=2.5e-4,
            lr_schedule=lambda _: 2.5e-4,
            n_epochs=5,
            clip_range=lambda _: 0.2,
            target_kl=None,
            policy=SimpleNamespace(optimizer=optimizer),
        )

        apply_bridging_resume_safety(model)

        self.assertEqual(model.learning_rate, 1e-4)
        self.assertEqual(model.lr_schedule(1.0), 1e-4)
        self.assertEqual(model.n_epochs, 4)
        self.assertEqual(model.clip_range(1.0), 0.15)
        self.assertEqual(model.target_kl, 0.02)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_curriculum_requires_sustained_stage_success(self) -> None:
        curriculum = MasteryCurriculum(min_episodes=10)
        for _ in range(9):
            self.assertFalse(curriculum.observe("one_guided", True))
        self.assertTrue(curriculum.observe("one_guided", True))
        self.assertEqual(curriculum.stage, "two_guided")

        # Results from episodes that began in the previous stage cannot count
        # toward mastery of the newly selected stage.
        self.assertFalse(curriculum.observe("one_guided", True))
        self.assertEqual(len(curriculum.successes), 0)

    def test_curriculum_does_not_advance_at_zero_success(self) -> None:
        curriculum = MasteryCurriculum(min_episodes=10)
        for _ in range(20):
            self.assertFalse(curriculum.observe("one_guided", False))
        self.assertEqual(curriculum.stage, "one_guided")

    def test_curriculum_can_wait_for_an_external_mastery_gate(self) -> None:
        curriculum = MasteryCurriculum(
            min_episodes=3, task_id="bridging"
        )
        for _ in range(3):
            self.assertFalse(
                curriculum.observe(
                    "one_gap_edge_aimed", True, advance=False
                )
            )
        self.assertTrue(curriculum.ready())
        self.assertEqual(curriculum.stage_index, 0)
        curriculum.advance()
        self.assertEqual(curriculum.stage_index, 1)
        self.assertEqual(len(curriculum.successes), 0)

    def test_bridging_final_stage_keeps_mastery_evidence(self) -> None:
        curriculum = MasteryCurriculum(
            stage_index=9, min_episodes=3, task_id="bridging"
        )
        for _ in range(3):
            self.assertFalse(
                curriculum.observe("variable_16_64_random", True, advance=False)
            )
        self.assertEqual(list(curriculum.successes), [1.0, 1.0, 1.0])
        self.assertFalse(curriculum.ready())

    def test_mixed_evaluation_seeds_are_reproducible_and_dispersed(self) -> None:
        first = [evaluation_seed(0, index) for index in range(100)]
        second = [evaluation_seed(0, index) for index in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 100)
        self.assertTrue(all(0 <= seed < 2**63 for seed in first))
        self.assertGreater(max(first) - min(first), 2**62)

    def test_bed_hint_dropout_counts_only_unguided_episodes(self) -> None:
        curriculum = MasteryCurriculum(
            stage_index=9, min_episodes=2, task_id="bed_defense_break"
        )
        self.assertFalse(
            curriculum.observe("layered_hint_75", True, eligible=False)
        )
        self.assertEqual(len(curriculum.successes), 0)
        self.assertFalse(curriculum.observe("layered_hint_75", True))
        self.assertTrue(curriculum.observe("layered_hint_75", True))
        self.assertEqual(curriculum.stage, "layered_hint_50")

    def test_bed_curriculum_teaches_balanced_tools_before_random_aim(self) -> None:
        curriculum = MasteryCurriculum(task_id="bed_defense_break")
        self.assertEqual(curriculum.stages[2], "single_mixed_aimed")
        self.assertEqual(curriculum.stages[3], "single_mixed")

    def test_bed_v2_requires_ninety_of_the_latest_hundred_successes(self) -> None:
        curriculum = MasteryCurriculum(task_id="bed_defense_break_v2")
        self.assertEqual(curriculum.success_thresholds[:5], (0.90,) * 5)
        self.assertEqual(curriculum.success_thresholds[5], 0.80)
        for _ in range(89):
            self.assertFalse(
                curriculum.observe("exposed_fixed_aimed", True)
            )
        for _ in range(10):
            self.assertFalse(
                curriculum.observe("exposed_fixed_aimed", False)
            )
        self.assertTrue(curriculum.observe("exposed_fixed_aimed", True))
        self.assertEqual(curriculum.stage, "exposed_random")

    def test_bed_v2_obsidian_stage_advances_at_eighty_percent(self) -> None:
        curriculum = MasteryCurriculum(
            stage_index=5,
            min_episodes=100,
            task_id="bed_defense_break_v2",
        )
        for _ in range(79):
            self.assertFalse(
                curriculum.observe("obsidian_inner_random", True)
            )
        for _ in range(20):
            self.assertFalse(
                curriculum.observe("obsidian_inner_random", False)
            )
        self.assertTrue(curriculum.observe("obsidian_inner_random", True))
        self.assertEqual(curriculum.stage, "full_random")

    def test_bed_periodic_checkpoint_default_is_fifty_thousand_steps(self) -> None:
        preset = get_task_preset("bed_defense_break")
        self.assertEqual(preset.default_checkpoint_freq, 50_000)

    def test_curriculum_variant_diagnostics_expire_after_one_hundred_episodes(self) -> None:
        callback = EpisodeConsoleCallback(log_every_episodes=10_000)
        callback.locals = {
            "dones": [True],
            "infos": [{"success": True, "curriculum_variant": "depth_3"}],
        }
        callback._on_step()
        for _ in range(100):
            callback.locals = {
                "dones": [True],
                "infos": [{"success": True, "curriculum_variant": ""}],
            }
            callback._on_step()

        self.assertEqual(len(callback.recent_variant_outcomes), 100)
        self.assertFalse(
            any(name for name, _ in callback.recent_variant_outcomes)
        )

    def test_bed_v2_console_diagnostics_report_actions_focus_and_stage_materials(self) -> None:
        callback = EpisodeConsoleCallback(log_every_episodes=1)
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "success": False,
                    "curriculum_stage": "one_layer_fixed_aimed",
                    "layer_materials": "wool",
                    "initial_selected_tool": "diamond_axe",
                    "tool_control_mode": "policy",
                    "curriculum_eligible": False,
                    "bed_focus_fraction": 0.4,
                    "breakable_focus_fraction": 0.8,
                    "attack_fraction": 0.5,
                    "air_attack_fraction": 0.2,
                    "bed_attack_fraction": 0.1,
                    "episode": {"r": -1.0, "l": 60},
                }
            ],
        }
        output = StringIO()
        with redirect_stdout(output):
            callback._on_step()

        rendered = output.getvalue()
        self.assertIn("[bed-v2] stage=one_layer_fixed_aimed n=1", rendered)
        self.assertIn("tool_control=policy", rendered)
        self.assertIn("rehearsal=100.0%", rendered)
        self.assertIn("attack=50.0%", rendered)
        self.assertIn("focus_bed=40.0%", rendered)
        self.assertIn("attack_air=20.0%", rendered)
        self.assertIn("material_success=wool:0%(n=1)", rendered)
        self.assertIn(
            "[bed-v2-material] stage=one_layer_fixed_aimed material=wool n=1",
            rendered,
        )
        self.assertIn("selected=(pickaxe:0%,axe:0%,shears:0%)", rendered)
        self.assertEqual(len(callback.recent_stage_outcomes), 0)

    def test_bed_v2_tool_head_is_uniform_until_stage_two_then_released(self) -> None:
        callback = TrainingProgressCallback(
            "bed_defense_break_v2", initial_stage=1
        )
        environment = Mock()
        action_space = spaces.MultiDiscrete([9, 9, 3, 2])
        action_net = torch.nn.Linear(4, int(np.sum(action_space.nvec)))
        optimizer = torch.optim.Adam(action_net.parameters(), lr=1e-3)
        model = Mock()
        model.action_space = action_space
        model.policy = SimpleNamespace(action_net=action_net, optimizer=optimizer)
        model.get_env.return_value = environment
        model.bed_v2_tool_head_released = False
        callback.model = model

        with torch.no_grad():
            action_net.weight[18:21].fill_(2.0)
            action_net.bias[18:21].fill_(2.0)
        callback._on_training_start()
        torch.testing.assert_close(
            action_net.weight[18:21], torch.zeros_like(action_net.weight[18:21])
        )
        self.assertFalse(model.bed_v2_tool_head_released)

        # Simulate another PPO update during exposed-bed training. The next
        # rollout must neutralize the tool rows again.
        with torch.no_grad():
            action_net.weight[18:21].fill_(3.0)
        callback._on_rollout_start()
        torch.testing.assert_close(
            action_net.weight[18:21], torch.zeros_like(action_net.weight[18:21])
        )

        callback.curriculum.successes.extend([1.0] * 99)
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "curriculum_stage": "exposed_random",
                    "curriculum_eligible": True,
                    "success": True,
                }
            ],
        }
        callback._on_step()

        self.assertEqual(callback.curriculum.stage_index, 2)
        self.assertTrue(model.bed_v2_tool_head_released)
        torch.testing.assert_close(
            action_net.weight[18:21], torch.zeros_like(action_net.weight[18:21])
        )

    def test_mastered_stage_is_saved_before_curriculum_switch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            callback = TrainingProgressCallback(
                "bed_defense_break",
                initial_stage=0,
                stage_checkpoint_dir=Path(temporary_directory),
                checkpoint_name_prefix="ppo_bed_defense_break",
            )
            environment = Mock()
            model = Mock()
            model.get_env.return_value = environment
            model.curriculum_stage_index = 0
            saved_stage_indices = []
            model.save.side_effect = lambda _: saved_stage_indices.append(
                model.curriculum_stage_index
            )
            callback.model = model
            callback.num_timesteps = 12_345
            callback.curriculum.successes.extend([1.0] * 99)
            callback.stage_eligible_episodes = 99
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "curriculum_stage": "exposed_near",
                        "curriculum_eligible": True,
                        "success": True,
                    }
                ],
            }

            self.assertTrue(callback._on_step())

            expected_path = (
                Path(temporary_directory)
                / "ppo_bed_defense_break_stage_00_exposed_near_mastered_12345_steps"
            )
            model.save.assert_called_once_with(expected_path)
            self.assertEqual(saved_stage_indices, [0])
            self.assertEqual(model.curriculum_stage_index, 1)
            self.assertEqual(callback.curriculum.stage, "exposed_random")

    def test_bridging_mastery_waits_for_deterministic_validation(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=0)
        model = SimpleNamespace(ent_coef=0.005)
        callback.model = model
        callback.num_timesteps = 50_000
        for _ in range(callback.BRIDGING_MASTERY_EPISODES):
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "curriculum_stage": "one_gap_edge_aimed",
                        "curriculum_eligible": True,
                        "success": True,
                    }
                ],
            }
            self.assertTrue(callback._on_step())
        self.assertEqual(callback.curriculum.stage_index, 0)
        self.assertTrue(callback.curriculum.ready())
        self.assertTrue(callback._bridging_validation_requested)
        self.assertEqual(callback._bridging_target_ent_coef(), 0.0)
        # The coefficient is committed once per completed PPO rollout.
        self.assertEqual(model.ent_coef, 0.005)

    def test_bridging_validation_is_deferred_if_window_drops_before_rollout_end(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=1)
        callback.model = SimpleNamespace(
            ent_coef=0.001, get_env=lambda: Mock()
        )
        callback.curriculum.successes.extend([1.0] * 450 + [0.0] * 50)
        callback.stage_eligible_episodes = 500
        callback._bridging_validation_requested = True
        # One more failure replaces a success: 450/500 becomes 449/500.
        callback.curriculum.successes.append(0.0)

        output = StringIO()
        with (
            redirect_stdout(output),
            patch.object(callback, "_run_bridging_validation") as validation,
        ):
            callback._on_rollout_end()

        validation.assert_not_called()
        self.assertEqual(callback.curriculum.stage_index, 1)
        self.assertFalse(callback._bridging_validation_requested)
        self.assertEqual(callback.model.ent_coef, 0.001)
        self.assertIn("deferred=stochastic_window_changed", output.getvalue())

    def test_bridging_validation_rechecks_every_learned_stage(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=2)
        self.assertEqual(
            callback._bridging_validation_plan(),
            [(0, 20, 0.90), (1, 100, 0.90), (2, 20, 0.85)],
        )

    def test_fifteen_gap_mastery_requirement_is_seventy_five_percent(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=6)
        self.assertEqual(callback.curriculum.stage, "fifteen_gap_random")
        self.assertEqual(callback.curriculum.success_thresholds[6], 0.75)
        callback.curriculum.successes.extend([1.0] * 375 + [0.0] * 125)
        self.assertTrue(callback._bridging_window_ready())

    def test_final_bridging_validation_rechecks_all_ten_stages(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=9)
        plan = callback._bridging_validation_plan()
        self.assertEqual([stage for stage, _, _ in plan], list(range(10)))
        self.assertEqual(plan[-1], (9, 100, 0.90))
        callback.curriculum.successes.extend([1.0] * 500)
        self.assertTrue(callback._bridging_window_ready())

    def test_final_bridging_certification_saves_without_advancing(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=9)
        callback.model = SimpleNamespace(
            ent_coef=0.0, get_env=lambda: Mock()
        )
        callback.num_timesteps = 2_000_000
        callback.stage_eligible_episodes = 500
        callback.curriculum.successes.extend([1.0] * 500)
        callback._bridging_validation_requested = True

        with (
            patch.object(callback, "_run_bridging_validation", return_value=True),
            patch.object(
                callback,
                "_save_mastered_stage",
                return_value=Path("certified"),
            ) as save_stage,
        ):
            callback._on_rollout_end()

        self.assertEqual(callback.curriculum.stage_index, 9)
        self.assertTrue(callback.model.bridging_final_certified)
        self.assertEqual(callback.model.bridging_final_certified_at, 2_000_000)
        save_stage.assert_called_once_with(9, "variable_16_64_random")

    def test_failed_bridging_validation_reopens_exploration(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=1)
        callback.curriculum.successes.extend([1.0] * 500)
        callback.stage_eligible_episodes = 500
        self.assertEqual(callback._bridging_target_ent_coef(), 0.0)
        callback._bridging_exploration_boost_until = 600
        self.assertEqual(callback._bridging_target_ent_coef(), 0.005)

    def test_rehearsal_is_balanced_by_transition_length_and_focuses_failure(self) -> None:
        callback = TrainingProgressCallback("bridging", initial_stage=4)
        callback._bridging_active_lengths.extend([166.0] * 100)
        callback._bridging_rehearsal_lengths.extend([20.0] * 100)

        percent, focus = callback._bridging_rehearsal_configuration()
        self.assertEqual((percent, focus), (80, -1))

        callback._bridging_retention_outcomes[3] = deque(
            [0.0] * callback.BRIDGING_RETENTION_MIN_EPISODES,
            maxlen=callback.BRIDGING_RETENTION_WINDOW,
        )
        percent, focus = callback._bridging_rehearsal_configuration()
        self.assertEqual((percent, focus), (90, 3))

    def test_bridging_validation_runs_all_stages_and_restores_training(self) -> None:
        class ValidationEnv(Env):
            observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            action_space = spaces.Discrete(1)

            def __init__(self) -> None:
                self.stage = 0
                self.validation = False
                self.seen_stages: set[int] = set()

            def set_curriculum_stage(self, stage_index: int) -> None:
                self.stage = stage_index

            def set_curriculum_validation_mode(self, enabled: bool) -> None:
                self.validation = enabled

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                self.seen_stages.add(self.stage)
                return (
                    np.zeros(1, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {"success": True},
                )

        raw_env = ValidationEnv()
        vector_env = DummyVecEnv([lambda: raw_env])
        callback = TrainingProgressCallback("bridging", initial_stage=2)
        model = Mock()
        model.get_env.return_value = vector_env
        model.predict.side_effect = lambda observation, deterministic: (
            np.zeros((vector_env.num_envs,), dtype=np.int64),
            None,
        )
        callback.model = model
        callback.num_timesteps = 123

        self.assertTrue(callback._run_bridging_validation())
        self.assertEqual(raw_env.seen_stages, {0, 1, 2})
        self.assertEqual(raw_env.stage, 2)
        self.assertFalse(raw_env.validation)
        self.assertEqual(model._last_episode_starts.tolist(), [True])
        vector_env.close()

    def test_failed_subprocess_shutdown_does_not_wait_for_pipe_replies(self) -> None:
        remote = Mock()
        process = Mock()
        process.exitcode = 1
        process.is_alive.side_effect = [True, False]
        subprocess_environment = Mock()
        subprocess_environment.remotes = [remote]
        subprocess_environment.processes = [process]
        wrapped_environment = Mock()

        with patch(
            "train._find_subproc_vec_env", return_value=subprocess_environment
        ):
            close_training_environment(wrapped_environment)

        remote.close.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with(timeout=5.0)
        wrapped_environment.close.assert_not_called()
        self.assertFalse(subprocess_environment.waiting)
        self.assertTrue(subprocess_environment.closed)

    def test_two_spawned_workers_bind_independent_servers(self) -> None:
        factories = [
            make_environment_factory(
                task_id="aim_break_tnt",
                max_episode_steps=10,
                host="127.0.0.1",
                port=0,
                instance_id=f"rl-{index}",
                visualized=False,
                target_tps=20.0,
                profile=True,
            )
            for index in range(2)
        ]
        env = SubprocVecEnv(factories, start_method="spawn")
        try:
            self.assertEqual(env.num_envs, 2)
            self.assertEqual(env.get_attr("instance_id"), ["rl-0", "rl-1"])
            self.assertEqual(env.get_attr("profile"), [True, True])
            self.assertEqual(env.get_attr("include_raw_info"), [False, False])
            addresses = env.get_attr("address")
            self.assertEqual(len({address[1] for address in addresses}), 2)
            env.env_method("set_training_progress", 12_345)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
