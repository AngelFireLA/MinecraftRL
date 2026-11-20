package com.minecraftrl.benchmark;

import com.minecraftrl.world.ClientBlockChangeTracker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class SynchronizedBenchmarkSelfTest {
    public static void main(String[] args) {
        Object world = new Object();
        int[] states = {1, 2, 3};
        SynchronizedBenchmark.BedDefenseVisibilityInputs baseline = inputs(
                world, 10, 20, 30, 1.5D, 22.62D, 3.5D, states);
        states[0] = 99;

        require(baseline.matches(inputs(
                        world, 10, 20, 30, 1.5D, 22.62D, 3.5D,
                        new int[]{1, 2, 3})),
                "identical visibility inputs must hit the cache");
        require(!baseline.matches(inputs(
                        new Object(), 10, 20, 30, 1.5D, 22.62D, 3.5D,
                        new int[]{1, 2, 3})),
                "a different world must invalidate visibility");
        require(!baseline.matches(inputs(
                        world, 11, 20, 30, 1.5D, 22.62D, 3.5D,
                        new int[]{1, 2, 3})),
                "a different grid origin must invalidate visibility");
        require(!baseline.matches(inputs(
                        world, 10, 20, 30, 1.5D, 22.63D, 3.5D,
                        new int[]{1, 2, 3})),
                "a moved eye must invalidate visibility");
        require(!baseline.matches(inputs(
                        world, 10, 20, 30, 1.5D, 22.62D, 3.5D,
                        new int[]{1, 4, 3})),
                "a changed block state must invalidate visibility");

        List<String> requested = new ArrayList<>(Arrays.asList(
                "player.x", "task_goal.distance", "bed_defense.block_grid"));
        SynchronizedBenchmark.ObservationPlan plan =
                SynchronizedBenchmark.ObservationPlan.compile(requested);
        requested.clear();
        require(plan.fields.size() == 3,
                "compiled observation fields must be stable snapshots");
        require(plan.requiresTaskGoal && plan.requiresBedDefense,
                "observation plan missed required captures");
        require(!plan.requiresRaycast && !plan.requiresNearestLiving
                        && !plan.requiresTaskEntity && !plan.requiresTaskBlocks,
                "observation plan selected an unused capture");
        require("\"player.x\"".equals(plan.quotedKeys[0]),
                "observation plan must cache JSON field keys");
        SynchronizedBenchmark.ObservationPlan v2Plan =
                SynchronizedBenchmark.ObservationPlan.compile(Arrays.asList(
                        "raycast.safe_yaw_correction",
                        "bed_defense_v2.block_grid"));
        require(v2Plan.requiresRaycast && v2Plan.requiresBedDefenseV2,
                "V2 observation plan missed raycast or grid capture");
        require(!v2Plan.requiresBedDefense,
                "V2 grid must not invoke the legacy bed-defense capture");
        SynchronizedBenchmark.ObservationPlan bridgingPlan =
                SynchronizedBenchmark.ObservationPlan.compile(Arrays.asList(
                        "bridging.ground_fraction", "bridging.block_grid"));
        require(bridgingPlan.requiresBridging,
                "bridging observation plan missed its snapshot");

        require(SynchronizedBenchmark.bridgingInventoryReady(0, 64, true),
                "bridging inventory gate rejected 64 wool in slot zero");
        require(!SynchronizedBenchmark.bridgingInventoryReady(1, 64, true),
                "bridging inventory gate accepted the wrong selected slot");
        require(!SynchronizedBenchmark.bridgingInventoryReady(
                        0, 63, true),
                "bridging inventory gate accepted the wrong stack size");
        require(!SynchronizedBenchmark.bridgingInventoryReady(
                        0, 64, false),
                "bridging inventory gate accepted the wrong block");
        double[] fixedStart = SynchronizedBenchmark.bridgingExpectedStart(
                0.5D, 64.5D, 6.5D, 2, 123L);
        require(fixedStart[0] == 0.5D && fixedStart[1] == 65.0D
                        && fixedStart[2] == 1.29D,
                "bridging fixed reset pose readiness coordinates are wrong");
        double[] randomStart = SynchronizedBenchmark.bridgingExpectedStart(
                0.5D, 64.5D, 2.5D, 1, 123L);
        require(randomStart[0] >= 0.31D && randomStart[0] <= 0.69D
                        && randomStart[1] == 65.0D
                        && randomStart[2] >= 0.31D && randomStart[2] <= 0.69D,
                "bridging random reset pose escaped the start block");

        long version = ClientBlockChangeTracker.version();
        require(SynchronizedBenchmark.canReuseBedDefenseVolume(
                        world, 7, 8, 9, version,
                        world, 7, 8, 9, version),
                "unchanged tracked volume must be reusable");
        ClientBlockChangeTracker.markChanged();
        require(!SynchronizedBenchmark.canReuseBedDefenseVolume(
                        world, 7, 8, 9, version,
                        world, 7, 8, 9, ClientBlockChangeTracker.version()),
                "client block changes must invalidate tracked volumes");
        require(SynchronizedBenchmark.sameBedDefenseEye(
                        1.0D, 2.0D, 3.0D, 1.0D, 2.0D, 3.0D),
                "unchanged eye position must reuse visibility");
        require(!SynchronizedBenchmark.sameBedDefenseEye(
                        1.0D, 2.0D, 3.0D, 1.0D, 2.01D, 3.0D),
                "moved eye position must recompute visibility");
        System.out.println("SynchronizedBenchmark self-test passed "
                + "(observation plan, reset gate, and visibility cache)");
    }

    private static SynchronizedBenchmark.BedDefenseVisibilityInputs inputs(
            Object world, int originX, int originY, int originZ,
            double eyeX, double eyeY, double eyeZ, int[] stateIds) {
        return new SynchronizedBenchmark.BedDefenseVisibilityInputs(
                world, originX, originY, originZ,
                eyeX, eyeY, eyeZ, stateIds);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
