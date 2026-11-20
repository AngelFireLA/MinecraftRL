package com.minecraftrl.environment;

import net.minecraft.util.BlockPos;

import java.util.Random;

public final class EnvironmentResetManagerSelfTest {
    private static final double SURVIVAL_REACH = 4.5D;
    private static final double EYE_HEIGHT = 1.62D;
    private static final double BED_HEIGHT = 0.5625D;

    public static void main(String[] args) {
        require(!EnvironmentResetManager.playerNeedsRespawn(false, true, 20.0F),
                "a healthy reset player must not be recreated");
        require(EnvironmentResetManager.playerNeedsRespawn(true, false, 0.0F),
                "a dead reset player must be recreated");
        require(EnvironmentResetManager.playerNeedsRespawn(false, false, 20.0F),
                "a non-living reset player must be recreated");
        require(EnvironmentResetManager.playerNeedsRespawn(false, true, 0.0F),
                "a zero-health reset player must be recreated");
        for (int layers = 1; layers <= 2; layers++) {
            double radius = EnvironmentResetManager.bedDefenseCornerRadius(layers);
            double distance = nearestBedHitboxDistance(radius + 0.15D);
            require(distance <= SURVIVAL_REACH,
                    "jittered corner spawn cannot reach exposed bed at layers=" + layers
                            + ": distance=" + distance);
        }

        double oldDistance = nearestBedHitboxDistance(2.0D + 1.75D);
        require(oldDistance > SURVIVAL_REACH,
                "regression fixture no longer demonstrates the old unreachable spawn");

        for (int tunnelDepth = 1; tunnelDepth <= 3; tunnelDepth++) {
            double radius = EnvironmentResetManager.bedDefenseAxisRadius(tunnelDepth, 0);
            double distance = nearestAxisBedHitboxDistance(
                    radius + 0.25D, 0.45D);
            require(distance <= SURVIVAL_REACH,
                    "jittered tunnel spawn cannot reach exposed bed at depth=" + tunnelDepth
                            + ": distance=" + distance);
        }
        for (int layers = 1; layers <= 2; layers++) {
            double radius = EnvironmentResetManager.bedDefenseAxisRadius(0, layers);
            double distance = nearestAxisBedHitboxDistance(
                    radius + 0.25D, 0.45D);
            require(distance <= SURVIVAL_REACH,
                    "jittered shell spawn cannot reach exposed bed at layers=" + layers
                            + ": distance=" + distance);
        }

        Random firstJitter = new Random(12345L);
        Random secondJitter = new Random(12345L);
        for (int index = 0; index < 32; index++) {
            double firstValue = EnvironmentResetManager.bedDefenseSymmetricJitter(
                    firstJitter, 0.45D);
            double secondValue = EnvironmentResetManager.bedDefenseSymmetricJitter(
                    secondJitter, 0.45D);
            require(firstValue == secondValue,
                    "equal seeds must produce identical bed-defense jitter");
            require(firstValue >= -0.45D && firstValue <= 0.45D,
                    "bed-defense jitter escaped its configured bounds");
        }

        int first = EnvironmentResetManager.bedDefenseSingleCoverMaterialIndex(100L);
        int second = EnvironmentResetManager.bedDefenseSingleCoverMaterialIndex(101L);
        int third = EnvironmentResetManager.bedDefenseSingleCoverMaterialIndex(102L);
        require(first != second && first != third && second != third,
                "three consecutive seeds must cover all single-cover materials");
        require(first == EnvironmentResetManager.bedDefenseSingleCoverMaterialIndex(103L),
                "single-cover material schedule must repeat every three seeds");

        require(EnvironmentResetManager.bedDefenseV2LayerCount(0, 0L) == 0,
                "V2 stage zero must be exposed");
        require(EnvironmentResetManager.bedDefenseV2LayerCount(4, 0L) == 2,
                "V2 stage four must always have two layers");
        require(EnvironmentResetManager.bedDefenseV2LayerCount(5, 0L) == 1
                        && EnvironmentResetManager.bedDefenseV2LayerCount(5, 1L) == 2,
                "V2 variable-depth stages must balance one and two layers");
        boolean[] singleMaterials = new boolean[3];
        for (long seed = 0; seed < 3; seed++) {
            singleMaterials[EnvironmentResetManager.bedDefenseV2MaterialIndices(
                    2, 1, seed)[0]] = true;
        }
        require(singleMaterials[0] && singleMaterials[1] && singleMaterials[2],
                "V2 one-layer stage must balance wool, wood, and end stone");
        int stageFiveRehearsals = 0;
        boolean[] rehearsalMaterials = new boolean[3];
        for (long seed = 0; seed < 30; seed++) {
            int layers = EnvironmentResetManager.bedDefenseV2LayerCount(5, seed);
            int[] materials = EnvironmentResetManager.bedDefenseV2MaterialIndices(
                    5, layers, seed);
            if (EnvironmentResetManager.bedDefenseV2StageFiveRehearsal(seed)) {
                stageFiveRehearsals++;
                for (int material : materials) {
                    require(material >= 0 && material < 3,
                            "V2 stage-five rehearsal cannot contain obsidian");
                    rehearsalMaterials[material] = true;
                }
            } else {
                require(materials[0] == 3,
                        "V2 stage five must normally put obsidian on the inner layer");
                require(layers == 1 || materials[1] != 3,
                        "V2 stage five cannot have two obsidian layers");
            }

            layers = EnvironmentResetManager.bedDefenseV2LayerCount(6, seed);
            materials = EnvironmentResetManager.bedDefenseV2MaterialIndices(
                    6, layers, seed);
            require(layers == 1 || materials[0] != 3 || materials[1] != 3,
                    "V2 final stage cannot have two obsidian layers");
        }
        require(stageFiveRehearsals == 6,
                "V2 stage five must schedule exactly one rehearsal in five seeds");
        require(rehearsalMaterials[0] && rehearsalMaterials[1]
                        && rehearsalMaterials[2],
                "V2 stage-five rehearsals must cover every non-obsidian material");
        for (int stage = 0; stage <= 6; stage++) {
            int partialRehearsals = 0;
            boolean sawOneLayerPartial = false;
            boolean sawTwoLayerPartial = false;
            for (long seed = 0; seed < 100; seed++) {
                boolean partial = EnvironmentResetManager
                        .bedDefenseV2PartialDefenseRehearsal(stage, seed);
                if (partial) {
                    partialRehearsals++;
                    int partialLayers = EnvironmentResetManager
                            .bedDefenseV2LayerCount(stage, seed);
                    sawOneLayerPartial |= partialLayers == 1;
                    sawTwoLayerPartial |= partialLayers == 2;
                }
                if (stage == 5) {
                    require(!(partial && EnvironmentResetManager
                                    .bedDefenseV2StageFiveRehearsal(seed)),
                            "partial and no-obsidian rehearsals must not overlap");
                }
            }
            require(partialRehearsals == (stage >= 3 ? 10 : 0),
                    "V2 partial-defense rehearsal cadence is wrong at stage="
                            + stage);
            if (stage >= 5) {
                require(sawOneLayerPartial && sawTwoLayerPartial,
                        "variable-depth partial rehearsals must include one and two layers");
            }
        }
        double v2Reach = EnvironmentResetManager.nearestBedDistance(
                0.0D, EYE_HEIGHT, 4.3D,
                new BlockPos(0, 0, 0), new BlockPos(1, 0, 0));
        require(v2Reach <= SURVIVAL_REACH,
                "V2 outer random spawn radius must keep an exposed bed reachable");

        int[] bridgingGaps = {1, 1, 5, 5, 8, 11, 15, 22, 30};
        for (int stage = 0; stage < bridgingGaps.length; stage++) {
            require(EnvironmentResetManager.bridgingGap(stage, 123L)
                            == bridgingGaps[stage],
                    "bridging fixed gap is wrong at stage=" + stage);
        }
        boolean[] sampledGaps = new boolean[49];
        for (long seed = 0; seed < 49; seed++) {
            int gap = EnvironmentResetManager.bridgingGap(9, seed);
            require(gap >= 16 && gap <= 64,
                    "bridging variable gap escaped 16..64");
            sampledGaps[gap - 16] = true;
        }
        for (boolean sampled : sampledGaps) {
            require(sampled, "bridging variable curriculum omitted a gap");
        }
        require(EnvironmentResetManager.BRIDGING_FIXED_LOCAL_Z == 1.29D
                        && EnvironmentResetManager.BRIDGING_FIXED_YAW == 180.0F
                        && EnvironmentResetManager.BRIDGING_FIXED_PITCH == 82.0F,
                "bridging fixed edge/aim pose changed unexpectedly");
        require(EnvironmentResetManager.bridgingClearMinZ(100, 1) == 35
                        && EnvironmentResetManager.bridgingClearMaxZ(100, 1) == 104,
                "one-gap bridge reset must cover its target-relative padded scan");
        require(EnvironmentResetManager.bridgingClearMinZ(100, 64) == 98
                        && EnvironmentResetManager.bridgingClearMaxZ(100, 64) == 167,
                "64-gap bridge reset must cover its target-relative padded scan");

        System.out.println(
                "EnvironmentResetManager self-test passed (bed V1/V2 and bridging)");
    }

    private static double nearestBedHitboxDistance(double radius) {
        double horizontalToNearFace = radius - 0.5D;
        double verticalToTop = EYE_HEIGHT - BED_HEIGHT;
        return Math.sqrt(2.0D * horizontalToNearFace * horizontalToNearFace
                + verticalToTop * verticalToTop);
    }

    private static double nearestAxisBedHitboxDistance(
            double radius, double lateralOffset) {
        double horizontalToNearFace = radius - 0.5D;
        double verticalToTop = EYE_HEIGHT - BED_HEIGHT;
        return Math.sqrt(horizontalToNearFace * horizontalToNearFace
                + lateralOffset * lateralOffset
                + verticalToTop * verticalToTop);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
