package com.minecraftrl.environment;

import net.minecraft.block.BlockBed;
import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.entity.monster.EntityZombie;
import net.minecraft.init.Blocks;
import net.minecraft.init.Items;
import net.minecraft.item.ItemStack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.BlockPos;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.MathHelper;
import net.minecraft.util.MovingObjectPosition;
import net.minecraft.util.Vec3;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.WorldServer;
import net.minecraft.world.WorldSettings;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

/** Applies bounded, task-specific environment resets on the integrated-server thread. */
public final class EnvironmentResetManager {
    public static final String HOLD_ATTACK_BLOCK_TASK = "hold_attack_block";
    public static final String AIM_BREAK_TNT_TASK = "aim_break_tnt";
    public static final String AIM_TOOL_BREAK_TASK = "aim_tool_break";
    public static final String MULTI_TOOL_BREAK_TASK = "multi_tool_break";
    public static final String BED_DEFENSE_BREAK_TASK = "bed_defense_break";
    public static final String BED_DEFENSE_BREAK_V2_TASK = "bed_defense_break_v2";
    public static final String BRIDGING_TASK = "bridging";
    public static final String PVP_BENCHMARK_TASK = "pvp_benchmark";
    // The player overhangs the target-side edge by 0.29 blocks while its
    // 0.6-wide hitbox retains a 0.01-block strip of support on the start block.
    public static final double BRIDGING_FIXED_LOCAL_Z = 1.29D;
    public static final float BRIDGING_FIXED_YAW = 180.0F;
    public static final float BRIDGING_FIXED_PITCH = 82.0F;

    private BreakBlockArena breakBlockArena;
    private AimBreakTntArena aimBreakTntArena;
    private AimToolBreakArena aimToolBreakArena;
    private MultiToolBreakArena multiToolBreakArena;
    private BedDefenseBreakArena bedDefenseBreakArena;
    private BridgingArena bridgingArena;
    private PvpArena pvpArena;

    public boolean supports(String taskId) {
        return HOLD_ATTACK_BLOCK_TASK.equals(taskId)
                || AIM_BREAK_TNT_TASK.equals(taskId)
                || AIM_TOOL_BREAK_TASK.equals(taskId)
                || MULTI_TOOL_BREAK_TASK.equals(taskId)
                || BED_DEFENSE_BREAK_TASK.equals(taskId)
                || BED_DEFENSE_BREAK_V2_TASK.equals(taskId)
                || BRIDGING_TASK.equals(taskId)
                || PVP_BENCHMARK_TASK.equals(taskId);
    }

    public boolean providesGoal(String taskId) {
        return AIM_BREAK_TNT_TASK.equals(taskId) || AIM_TOOL_BREAK_TASK.equals(taskId)
                || BED_DEFENSE_BREAK_TASK.equals(taskId)
                || BED_DEFENSE_BREAK_V2_TASK.equals(taskId)
                || BRIDGING_TASK.equals(taskId);
    }

    public ResetResult apply(MinecraftServer server, UUID playerUuid, String username,
                             String taskId, long seed, int requestedTargetCount,
                             int curriculumStage) {
        if (!supports(taskId)) {
            return ResetResult.notApplied();
        }
        Set<BlockPos> cleanupSyncPositions = new HashSet<>();
        if (!BED_DEFENSE_BREAK_TASK.equals(taskId)
                && !BED_DEFENSE_BREAK_V2_TASK.equals(taskId)
                && bedDefenseBreakArena != null) {
            // Remove the whole persistent BedWars arena, including its bedrock
            // floor. Clearing only dynamic bed/shell blocks left the old floor
            // surrounding later tasks.
            clearBedDefenseArena(bedDefenseBreakArena, cleanupSyncPositions);
            bedDefenseBreakArena = null;
        }
        if (!BRIDGING_TASK.equals(taskId) && bridgingArena != null) {
            clearBridgingArena(bridgingArena);
            bridgingArena = null;
        }
        if (!PVP_BENCHMARK_TASK.equals(taskId) && pvpArena != null) {
            clearPvpArena(pvpArena);
            pvpArena.world.getMinecraftServer().setDifficultyForAllWorlds(
                    pvpArena.previousDifficulty);
            pvpArena = null;
        }

        // The UUID is the stable identity. In offline HeadlessMC, Mojang's profile cache can
        // replace the requested session name when a mistakenly reused online UUID is present.
        // A name-only lookup then fails even though this exact local player is connected.
        EntityPlayerMP player = playerUuid == null
                ? null
                : server.getConfigurationManager().getPlayerByUUID(playerUuid);
        if (player == null && username != null) {
            player = server.getConfigurationManager().getPlayerByUsername(username);
        }
        if (player == null) {
            throw new IllegalStateException("integrated-server player not found: name="
                    + username + ", uuid=" + playerUuid);
        }
        player = respawnPlayerIfNeeded(server, player);
        if (HOLD_ATTACK_BLOCK_TASK.equals(taskId)) {
            resetBreakBlockArena(player);
            return ResetResult.appliedWithoutEntity();
        }
        if (AIM_BREAK_TNT_TASK.equals(taskId)) {
            return resetAimBreakTntArena(player, seed);
        }
        if (AIM_TOOL_BREAK_TASK.equals(taskId)) {
            return resetAimToolBreakArena(player, seed);
        }
        if (MULTI_TOOL_BREAK_TASK.equals(taskId)) {
            return resetMultiToolBreakArena(player, seed, requestedTargetCount);
        }
        if (BED_DEFENSE_BREAK_TASK.equals(taskId)) {
            return resetBedDefenseBreakArena(player, seed, curriculumStage);
        }
        if (BED_DEFENSE_BREAK_V2_TASK.equals(taskId)) {
            return resetBedDefenseBreakV2Arena(player, seed, curriculumStage);
        }
        if (BRIDGING_TASK.equals(taskId)) {
            return resetBridgingArena(
                    player, seed, curriculumStage, cleanupSyncPositions);
        }
        if (PVP_BENCHMARK_TASK.equals(taskId)) {
            return resetPvpArena(player, seed);
        }
        return ResetResult.notApplied();
    }

    private ResetResult resetPvpArena(EntityPlayerMP player, long seed) {
        WorldServer world = player.getServerForPlayer();
        if (pvpArena == null || pvpArena.world != world) {
            EnumDifficulty previousDifficulty = world.getWorldInfo().getDifficulty();
            player.getServerForPlayer().getMinecraftServer()
                    .setDifficultyForAllWorlds(EnumDifficulty.NORMAL);
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            pvpArena = new PvpArena(
                    world, x, feetY, z, previousDifficulty);
            // Build the immutable arena only once. Episode resets only restore the
            // two combatants, keeping the benchmark reset path intentionally small.
            for (int dz = -6; dz <= 6; dz++) {
                for (int dx = -6; dx <= 6; dx++) {
                    world.setBlockState(new BlockPos(x + dx, feetY - 1, z + dz),
                            Blocks.bedrock.getDefaultState(), 3);
                    for (int dy = 0; dy <= 3; dy++) {
                        world.setBlockState(new BlockPos(x + dx, feetY + dy, z + dz),
                                Blocks.air.getDefaultState(), 3);
                    }
                }
            }
        }

        PvpArena arena = pvpArena;
        clearArenaEntities(player, world, arena.x, arena.feetY, arena.z);
        resetPlayer(player, arena.x - 2.5D, arena.feetY, arena.z + 0.5D,
                -90.0F, 0.0F);
        player.inventory.setInventorySlotContents(0, new ItemStack(Items.wooden_sword));
        player.inventory.setInventorySlotContents(1, new ItemStack(Items.stone_sword));
        player.inventory.setInventorySlotContents(2, new ItemStack(Items.iron_sword));
        player.inventory.currentItem = (int) Math.floorMod(seed, 3L);
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);

        Random random = new Random(seed ^ 0x5056505F42454E43L);
        double lateral = (random.nextDouble() - 0.5D) * 3.0D;
        EntityZombie opponent = new EntityZombie(world);
        opponent.setPositionAndRotation(arena.x + 3.5D, arena.feetY,
                arena.z + 0.5D + lateral, 90.0F, 0.0F);
        opponent.enablePersistence();
        // A helmet prevents daylight combustion from turning the benchmark into
        // a time-of-day test. The held sword exercises ordinary melee AI/damage.
        opponent.setCurrentItemOrArmor(0, new ItemStack(Items.iron_sword));
        opponent.setCurrentItemOrArmor(4, new ItemStack(Items.iron_helmet));
        opponent.setHealth(opponent.getMaxHealth());
        if (!world.spawnEntityInWorld(opponent)) {
            throw new IllegalStateException("failed to spawn pvp_benchmark opponent");
        }
        return ResetResult.appliedWithEntity(opponent.getEntityId());
    }

    private static void clearPvpArena(PvpArena arena) {
        for (int dz = -6; dz <= 6; dz++) {
            for (int dx = -6; dx <= 6; dx++) {
                for (int dy = -1; dy <= 3; dy++) {
                    arena.world.setBlockState(
                            new BlockPos(arena.x + dx, arena.feetY + dy, arena.z + dz),
                            Blocks.air.getDefaultState(), 3);
                }
            }
        }
    }

    private static EntityPlayerMP respawnPlayerIfNeeded(MinecraftServer server,
                                                         EntityPlayerMP player) {
        if (!playerNeedsRespawn(
                player.isDead, player.isEntityAlive(), player.getHealth())) {
            return player;
        }
        EntityPlayerMP respawned = server.getConfigurationManager()
                .recreatePlayerEntity(player, player.dimension, false);
        if (respawned == null) {
            throw new IllegalStateException("integrated-server player respawn failed");
        }
        return respawned;
    }

    static boolean playerNeedsRespawn(boolean isDead, boolean isEntityAlive,
                                      float health) {
        return isDead || !isEntityAlive || health <= 0.0F;
    }

    private void resetBreakBlockArena(EntityPlayerMP player) {
        WorldServer world = player.getServerForPlayer();
        if (breakBlockArena == null || breakBlockArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            breakBlockArena = new BreakBlockArena(world, x, feetY, z);
        }

        BreakBlockArena arena = breakBlockArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);

        // Keep the reset deliberately tiny: one support block and one target block.
        // The surrounding air volume removes leftovers that could obstruct the player
        // or raycast after an interrupted episode.
        for (int y = arena.feetY - 1; y <= arena.feetY + 2; y++) {
            for (int z = arena.z - 1; z <= arena.z + 3; z++) {
                for (int x = arena.x - 1; x <= arena.x + 1; x++) {
                    world.setBlockState(new BlockPos(x, y, z), Blocks.air.getDefaultState(), 3);
                }
            }
        }
        BlockPos support = new BlockPos(arena.x, arena.feetY - 1, arena.z);
        BlockPos target = new BlockPos(arena.x, arena.feetY, arena.z + 2);
        world.setBlockState(support, Blocks.bedrock.getDefaultState(), 3);
        world.setBlockState(target, Blocks.stone.getDefaultState(), 3);

        double playerX = arena.x + 0.5D;
        double playerY = arena.feetY;
        double playerZ = arena.z + 0.5D;
        double targetX = target.getX() + 0.5D;
        double targetY = target.getY() + 0.5D;
        double targetZ = target.getZ() + 0.5D;
        double dx = targetX - playerX;
        double dy = targetY - (playerY + player.getEyeHeight());
        double dz = targetZ - playerZ;
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, horizontal));

        resetPlayer(player, playerX, playerY, playerZ, yaw, pitch);
    }

    private ResetResult resetAimBreakTntArena(EntityPlayerMP player, long seed) {
        WorldServer world = player.getServerForPlayer();
        if (aimBreakTntArena == null || aimBreakTntArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            aimBreakTntArena = new AimBreakTntArena(world, x, feetY, z);
        }

        AimBreakTntArena arena = aimBreakTntArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);
        // Clear the lower layer too, because aim_tool_break may have left a
        // support block there when switching tasks in the same test world.
        for (int y = arena.feetY - 1; y <= arena.feetY + 2; y++) {
            for (int z = arena.z - 4; z <= arena.z + 4; z++) {
                for (int x = arena.x - 4; x <= arena.x + 4; x++) {
                    world.setBlockState(new BlockPos(x, y, z), Blocks.air.getDefaultState(), 3);
                }
            }
        }

        BlockPos support = new BlockPos(arena.x, arena.feetY - 1, arena.z);
        world.setBlockState(support, Blocks.bedrock.getDefaultState(), 3);

        Random random = new Random(seed ^ 0x4D43524C5F544E54L);
        int targetX = arena.x + random.nextInt(5) - 2;
        int targetY = arena.feetY + random.nextInt(3);
        int targetZ = arena.z + 2 + random.nextInt(2);
        BlockPos target = new BlockPos(targetX, targetY, targetZ);
        world.setBlockState(target, Blocks.tnt.getDefaultState(), 3);

        resetPlayer(player, arena.x + 0.5D, arena.feetY, arena.z + 0.5D, 0.0F, 0.0F);
        return ResetResult.appliedWithGoal(
                target.getX() + 0.5D, target.getY() + 0.5D, target.getZ() + 0.5D);
    }

    private ResetResult resetAimToolBreakArena(EntityPlayerMP player, long seed) {
        WorldServer world = player.getServerForPlayer();
        if (aimToolBreakArena == null || aimToolBreakArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            aimToolBreakArena = new AimToolBreakArena(world, x, feetY, z);
        }

        AimToolBreakArena arena = aimToolBreakArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);
        for (int y = arena.feetY - 1; y <= arena.feetY + 2; y++) {
            for (int z = arena.z + 1; z <= arena.z + 4; z++) {
                for (int x = arena.x - 3; x <= arena.x + 3; x++) {
                    world.setBlockState(new BlockPos(x, y, z), Blocks.air.getDefaultState(), 3);
                }
            }
        }

        BlockPos playerSupport = new BlockPos(arena.x, arena.feetY - 1, arena.z);
        world.setBlockState(playerSupport, Blocks.bedrock.getDefaultState(), 3);

        Random random = new Random(seed ^ 0x4D43524C5F544F4FL);
        int targetX = arena.x + random.nextInt(5) - 2;
        int targetY = arena.feetY + random.nextInt(3);
        int targetZ = arena.z + 2 + random.nextInt(2);
        BlockPos target = new BlockPos(targetX, targetY, targetZ);
        // Sand must have support or it would fall before the first observation.
        world.setBlockState(target.down(), Blocks.bedrock.getDefaultState(), 3);
        // Consecutive episode seeds cycle evenly through all three materials,
        // while the target position remains seeded-random.
        switch ((int) Math.floorMod(seed, 3L)) {
            case 0:
                world.setBlockState(target, Blocks.stone.getDefaultState(), 3);
                break;
            case 1:
                world.setBlockState(target, Blocks.planks.getDefaultState(), 3);
                break;
            default:
                world.setBlockState(target, Blocks.sand.getDefaultState(), 3);
                break;
        }

        float startYaw = random.nextFloat() * 360.0F - 180.0F;
        float startPitch = random.nextFloat() * 90.0F - 45.0F;
        int startSlot = random.nextInt(3);
        resetPlayer(player, arena.x + 0.5D, arena.feetY, arena.z + 0.5D,
                startYaw, startPitch);
        player.inventory.setInventorySlotContents(0, new ItemStack(Items.diamond_pickaxe));
        player.inventory.setInventorySlotContents(1, new ItemStack(Items.diamond_axe));
        player.inventory.setInventorySlotContents(2, new ItemStack(Items.iron_shovel));
        player.inventory.currentItem = startSlot;
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);

        return ResetResult.appliedWithGoal(
                target.getX() + 0.5D, target.getY() + 0.5D, target.getZ() + 0.5D);
    }

    private ResetResult resetMultiToolBreakArena(EntityPlayerMP player, long seed,
                                                  int requestedTargetCount) {
        WorldServer world = player.getServerForPlayer();
        if (multiToolBreakArena == null || multiToolBreakArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            multiToolBreakArena = new MultiToolBreakArena(world, x, feetY, z);
        }

        MultiToolBreakArena arena = multiToolBreakArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);
        for (int y = arena.feetY - 1; y <= arena.feetY + 2; y++) {
            for (int z = arena.z - 4; z <= arena.z + 4; z++) {
                for (int x = arena.x - 4; x <= arena.x + 4; x++) {
                    world.setBlockState(new BlockPos(x, y, z), Blocks.air.getDefaultState(), 3);
                }
            }
        }
        world.setBlockState(new BlockPos(arena.x, arena.feetY - 1, arena.z),
                Blocks.bedrock.getDefaultState(), 3);

        Random random = new Random(seed ^ 0x4D43524C5F4D554CL);
        if (requestedTargetCount < 0 || requestedTargetCount > 5) {
            throw new IllegalArgumentException(
                    "multi_tool_break target_count must be 0 (random) or between 1 and 5");
        }
        int targetCount = requestedTargetCount == 0
                ? 3 + random.nextInt(3)
                : requestedTargetCount;
        int[][] candidateOffsets = {
                {0, 3}, {2, 2}, {3, 0}, {2, -2},
                {0, -3}, {-2, -2}, {-3, 0}, {-2, 2}
        };
        int[] candidateOrder = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = candidateOrder.length - 1; i > 0; i--) {
            int swap = random.nextInt(i + 1);
            int value = candidateOrder[i];
            candidateOrder[i] = candidateOrder[swap];
            candidateOrder[swap] = value;
        }

        double[] goalXs = new double[targetCount];
        double[] goalYs = new double[targetCount];
        double[] goalZs = new double[targetCount];
        int[] goalBlockIds = new int[targetCount];
        for (int i = 0; i < targetCount; i++) {
            int[] offset = candidateOffsets[candidateOrder[i]];
            BlockPos target = new BlockPos(
                    arena.x + offset[0],
                    arena.feetY + random.nextInt(3),
                    arena.z + offset[1]);
            world.setBlockState(target.down(), Blocks.bedrock.getDefaultState(), 3);
            switch ((int) Math.floorMod(seed + i, 3L)) {
                case 0: world.setBlockState(target, Blocks.stone.getDefaultState(), 3); break;
                case 1: world.setBlockState(target, Blocks.planks.getDefaultState(), 3); break;
                default: world.setBlockState(target, Blocks.sand.getDefaultState(), 3); break;
            }
            goalXs[i] = target.getX() + 0.5D;
            goalYs[i] = target.getY() + 0.5D;
            goalZs[i] = target.getZ() + 0.5D;
            goalBlockIds[i] = net.minecraft.block.Block.getStateId(
                    world.getBlockState(target));
        }

        float startYaw = random.nextFloat() * 360.0F - 180.0F;
        float startPitch = random.nextFloat() * 90.0F - 45.0F;
        int startSlot = random.nextInt(3);
        resetPlayer(player, arena.x + 0.5D, arena.feetY, arena.z + 0.5D,
                startYaw, startPitch);
        player.inventory.setInventorySlotContents(0, new ItemStack(Items.diamond_pickaxe));
        player.inventory.setInventorySlotContents(1, new ItemStack(Items.diamond_axe));
        player.inventory.setInventorySlotContents(2, new ItemStack(Items.iron_shovel));
        player.inventory.currentItem = startSlot;
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);
        return ResetResult.appliedWithGoals(goalXs, goalYs, goalZs, goalBlockIds);
    }

    private ResetResult resetBedDefenseBreakArena(EntityPlayerMP player, long seed,
                                                   int curriculumStage) {
        if (curriculumStage < 0 || curriculumStage > 12) {
            throw new IllegalArgumentException(
                    "bed_defense_break curriculum_stage must be between 0 and 12");
        }
        WorldServer world = player.getServerForPlayer();
        if (bedDefenseBreakArena == null || bedDefenseBreakArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            bedDefenseBreakArena = new BedDefenseBreakArena(world, x, feetY, z);
        }

        BedDefenseBreakArena arena = bedDefenseBreakArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);
        if (!arena.initialized) {
            for (int y = arena.feetY - 1; y <= arena.feetY + 5; y++) {
                for (int z = arena.z - 5; z <= arena.z + 5; z++) {
                    for (int x = arena.x - 5; x <= arena.x + 5; x++) {
                        world.setBlockState(new BlockPos(x, y, z),
                                Blocks.air.getDefaultState(), 3);
                    }
                }
            }
            for (int z = arena.z - 5; z <= arena.z + 5; z++) {
                for (int x = arena.x - 5; x <= arena.x + 5; x++) {
                    world.setBlockState(new BlockPos(x, arena.feetY - 1, z),
                            Blocks.bedrock.getDefaultState(), 3);
                }
            }
            arena.initialized = true;
        } else {
            for (BlockPos position : arena.dynamicBlocks) {
                world.setBlockState(position, Blocks.air.getDefaultState(), 3);
            }
            arena.dynamicBlocks.clear();
        }

        Random random = new Random(seed ^ 0x4D43524C5F424544L);
        EnumFacing[] horizontal = {
                EnumFacing.NORTH, EnumFacing.EAST, EnumFacing.SOUTH, EnumFacing.WEST
        };
        EnumFacing bedFacing = horizontal[random.nextInt(horizontal.length)];
        EnumFacing approach = perpendicular(bedFacing, random.nextBoolean());
        BlockPos foot = new BlockPos(arena.x, arena.feetY, arena.z);
        BlockPos head = foot.offset(bedFacing);
        IBlockState footState = Blocks.bed.getDefaultState()
                .withProperty(BlockBed.FACING, bedFacing)
                .withProperty(BlockBed.PART, BlockBed.EnumPartType.FOOT);
        IBlockState headState = Blocks.bed.getDefaultState()
                .withProperty(BlockBed.FACING, bedFacing)
                .withProperty(BlockBed.PART, BlockBed.EnumPartType.HEAD);
        setBedDefenseBlock(arena, foot, footState);
        setBedDefenseBlock(arena, head, headState);

        int layers = 0;
        int tunnelDepth = 0;
        if (curriculumStage == 2 || curriculumStage == 3) {
            tunnelDepth = 1;
        } else if (curriculumStage == 4) {
            // The preceding single-cover stages already teach depth one.
            // Spend this sequential-breaking stage on depth two and three;
            // Python requires depth-three mastery before advancing.
            tunnelDepth = 2 + random.nextInt(2);
        } else if (curriculumStage == 5) {
            layers = 1;
        } else if (curriculumStage == 6) {
            layers = 2;
        } else if (curriculumStage >= 7) {
            layers = 1 + random.nextInt(2);
        }

        BlockPos initialTarget = foot;
        if (tunnelDepth > 0) {
            for (int depth = 1; depth <= tunnelDepth; depth++) {
                IBlockState state;
                if (curriculumStage == 2 || curriculumStage == 3) {
                    state = singleCoverMaterial(seed);
                } else {
                    state = tunnelMaterial(depth, random);
                }
                BlockPos footCover = foot.offset(approach, depth);
                BlockPos headCover = head.offset(approach, depth);
                setBedDefenseBlock(arena, footCover, state);
                setBedDefenseBlock(arena, headCover, state);
                initialTarget = footCover;
            }
        } else if (layers > 0) {
            IBlockState[] layerMaterials = new IBlockState[layers + 1];
            for (int layer = 1; layer <= layers; layer++) {
                layerMaterials[layer] = layerMaterial(curriculumStage, layer, random);
            }
            for (int y = arena.feetY; y <= arena.feetY + layers; y++) {
                for (int z = arena.z - 3; z <= arena.z + 3; z++) {
                    for (int x = arena.x - 3; x <= arena.x + 3; x++) {
                        BlockPos pos = new BlockPos(x, y, z);
                        if (pos.equals(foot) || pos.equals(head)) continue;
                        int shell = Math.min(manhattan(pos, foot), manhattan(pos, head));
                        if (shell >= 1 && shell <= layers) {
                            setBedDefenseBlock(arena, pos, layerMaterials[shell]);
                        }
                    }
                }
            }
            initialTarget = foot.offset(approach, layers);
        }

        double playerX;
        double playerY;
        double playerZ;
        boolean topPose = false;
        boolean cornerPose = false;
        EnumFacing cornerSide = null;
        if (curriculumStage >= 7 && random.nextInt(4) == 0) {
            topPose = true;
            playerX = foot.getX() + 0.5D;
            playerY = arena.feetY + layers + 1.0D;
            playerZ = foot.getZ() + 0.5D;
            initialTarget = foot.up(layers);
        } else if (curriculumStage >= 7 && random.nextInt(3) == 0) {
            cornerPose = true;
            cornerSide = perpendicular(approach, random.nextBoolean());
            double radius = bedDefenseCornerRadius(layers);
            playerX = foot.getX() + 0.5D
                    + approach.getFrontOffsetX() * radius
                    + cornerSide.getFrontOffsetX() * radius;
            playerY = arena.feetY;
            playerZ = foot.getZ() + 0.5D
                    + approach.getFrontOffsetZ() * radius
                    + cornerSide.getFrontOffsetZ() * radius;
        } else {
            // Leave one block of standing room beyond a tunnel, while keeping
            // the bed's nearest hittable point within survival reach after the
            // final cover is removed. Shell layouts need the extra block
            // because the outer shell occupies space in every direction.
            double radius = bedDefenseAxisRadius(tunnelDepth, layers);
            playerX = foot.getX() + 0.5D + approach.getFrontOffsetX() * radius;
            playerY = arena.feetY;
            playerZ = foot.getZ() + 0.5D + approach.getFrontOffsetZ() * radius;
        }

        // Stage zero remains the canonical bootstrap.  Every later stage gets
        // deterministic continuous offsets, avoiding a policy that only ever
        // sees exact integer/half-block poses.  Offsets are bounded so the
        // player neither intersects the outer defense nor loses survival reach.
        if (curriculumStage >= 1) {
            if (topPose) {
                playerX += bedDefenseSymmetricJitter(random, 0.18D);
                playerZ += bedDefenseSymmetricJitter(random, 0.18D);
            } else if (cornerPose) {
                double approachJitter = bedDefenseSymmetricJitter(random, 0.15D);
                double sideJitter = bedDefenseSymmetricJitter(random, 0.15D);
                playerX += approach.getFrontOffsetX() * approachJitter
                        + cornerSide.getFrontOffsetX() * sideJitter;
                playerZ += approach.getFrontOffsetZ() * approachJitter
                        + cornerSide.getFrontOffsetZ() * sideJitter;
            } else {
                EnumFacing tangent = perpendicular(approach, true);
                double lateralLimit = curriculumStage <= 2 ? 0.25D : 0.45D;
                double outwardLimit = curriculumStage <= 2 ? 0.15D : 0.25D;
                double lateral = bedDefenseSymmetricJitter(random, lateralLimit);
                double outward = random.nextDouble() * outwardLimit;
                playerX += tangent.getFrontOffsetX() * lateral
                        + approach.getFrontOffsetX() * outward;
                playerZ += tangent.getFrontOffsetZ() * lateral
                        + approach.getFrontOffsetZ() * outward;
            }
        }

        float[] look = lookAt(playerX, playerY + player.getEyeHeight(), playerZ,
                initialTarget.getX() + 0.5D, initialTarget.getY() + 0.5D,
                initialTarget.getZ() + 0.5D);
        float yaw = look[0];
        float pitch = look[1];
        if (curriculumStage != 0 && curriculumStage != 2) {
            // Stage zero teaches exposed-bed breaking. Stage two then teaches
            // the balanced material/tool mapping while already aimed; the
            // following stage combines that mapping with randomized aim.
            yaw = random.nextFloat() * 360.0F - 180.0F;
            pitch = random.nextFloat() * 120.0F - 60.0F;
        }

        resetPlayer(player, playerX, playerY, playerZ, yaw, pitch, false);
        player.inventory.setInventorySlotContents(0, new ItemStack(Items.shears));
        player.inventory.setInventorySlotContents(1, new ItemStack(Items.diamond_axe));
        player.inventory.setInventorySlotContents(2, new ItemStack(Items.diamond_pickaxe));
        player.inventory.currentItem = random.nextInt(3);
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);

        double[] goalXs = {foot.getX() + 0.5D, head.getX() + 0.5D};
        double[] goalYs = {foot.getY() + 0.5D, head.getY() + 0.5D};
        double[] goalZs = {foot.getZ() + 0.5D, head.getZ() + 0.5D};
        int[] goalIds = {
                net.minecraft.block.Block.getStateId(world.getBlockState(foot)),
                net.minecraft.block.Block.getStateId(world.getBlockState(head))
        };

        int syncCount = 7 * 5 * 7;
        double[] syncXs = new double[syncCount];
        double[] syncYs = new double[syncCount];
        double[] syncZs = new double[syncCount];
        int[] syncIds = new int[syncCount];
        int syncIndex = 0;
        for (int dy = -1; dy <= 3; dy++) {
            for (int dz = -3; dz <= 3; dz++) {
                for (int dx = -3; dx <= 3; dx++) {
                    BlockPos pos = foot.add(dx, dy, dz);
                    syncXs[syncIndex] = pos.getX() + 0.5D;
                    syncYs[syncIndex] = pos.getY() + 0.5D;
                    syncZs[syncIndex] = pos.getZ() + 0.5D;
                    syncIds[syncIndex] = net.minecraft.block.Block.getStateId(
                            world.getBlockState(pos));
                    syncIndex++;
                }
            }
        }
        return ResetResult.appliedWithGoalGoalsAndSync(
                foot.getX() + 0.5D, foot.getY() + 0.5D, foot.getZ() + 0.5D,
                goalXs, goalYs, goalZs, goalIds,
                syncXs, syncYs, syncZs, syncIds);
    }

    /** Reset profile for the compact, unguided seven-stage BedWars curriculum. */
    private ResetResult resetBedDefenseBreakV2Arena(EntityPlayerMP player, long seed,
                                                     int curriculumStage) {
        if (curriculumStage < 0 || curriculumStage > 6) {
            throw new IllegalArgumentException(
                    "bed_defense_break_v2 curriculum_stage must be between 0 and 6");
        }
        WorldServer world = player.getServerForPlayer();
        if (bedDefenseBreakArena == null || bedDefenseBreakArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = safeArenaFeetY(player);
            int z = MathHelper.floor_double(player.posZ);
            bedDefenseBreakArena = new BedDefenseBreakArena(world, x, feetY, z);
        }

        BedDefenseBreakArena arena = bedDefenseBreakArena;
        clearArenaEntities(player, arena.world, arena.x, arena.feetY, arena.z);
        if (!arena.initialized) {
            for (int y = arena.feetY - 1; y <= arena.feetY + 5; y++) {
                for (int z = arena.z - 6; z <= arena.z + 6; z++) {
                    for (int x = arena.x - 6; x <= arena.x + 6; x++) {
                        world.setBlockState(new BlockPos(x, y, z),
                                Blocks.air.getDefaultState(), 3);
                    }
                }
            }
            for (int z = arena.z - 6; z <= arena.z + 6; z++) {
                for (int x = arena.x - 6; x <= arena.x + 6; x++) {
                    world.setBlockState(new BlockPos(x, arena.feetY - 1, z),
                            Blocks.bedrock.getDefaultState(), 3);
                }
            }
            arena.initialized = true;
        } else {
            for (BlockPos position : arena.dynamicBlocks) {
                world.setBlockState(position, Blocks.air.getDefaultState(), 3);
            }
            arena.dynamicBlocks.clear();
        }

        Random random = new Random(seed ^ 0x4D43524C5F423256L);
        EnumFacing[] horizontal = {
                EnumFacing.NORTH, EnumFacing.EAST, EnumFacing.SOUTH, EnumFacing.WEST
        };
        EnumFacing bedFacing = horizontal[random.nextInt(horizontal.length)];
        EnumFacing lateral = perpendicular(bedFacing, true);
        BlockPos foot = new BlockPos(arena.x, arena.feetY, arena.z);
        BlockPos head = foot.offset(bedFacing);
        setBedDefenseBlock(arena, foot, Blocks.bed.getDefaultState()
                .withProperty(BlockBed.FACING, bedFacing)
                .withProperty(BlockBed.PART, BlockBed.EnumPartType.FOOT));
        setBedDefenseBlock(arena, head, Blocks.bed.getDefaultState()
                .withProperty(BlockBed.FACING, bedFacing)
                .withProperty(BlockBed.PART, BlockBed.EnumPartType.HEAD));

        int layers = bedDefenseV2LayerCount(curriculumStage, seed);
        int[] materialIndices = bedDefenseV2MaterialIndices(
                curriculumStage, layers, seed);
        for (int y = arena.feetY; y <= arena.feetY + layers; y++) {
            for (int lateralOffset = -layers; lateralOffset <= layers; lateralOffset++) {
                for (int longitudinalOffset = -layers;
                     longitudinalOffset <= layers + 1; longitudinalOffset++) {
                    BlockPos position = foot.offset(bedFacing, longitudinalOffset)
                            .offset(lateral, lateralOffset).up(y - arena.feetY);
                    if (position.equals(foot) || position.equals(head)) continue;
                    int shell = Math.min(manhattan(position, foot),
                            manhattan(position, head));
                    if (shell >= 1 && shell <= layers) {
                        setBedDefenseBlock(arena, position,
                                bedDefenseV2Material(materialIndices[shell - 1]));
                    }
                }
            }
        }

        double centerX = (foot.getX() + head.getX() + 1.0D) * 0.5D;
        double centerZ = (foot.getZ() + head.getZ() + 1.0D) * 0.5D;
        double playerX;
        double playerY;
        double playerZ;
        boolean randomPose = curriculumStage != 0 && curriculumStage != 2;
        boolean partialDefenseRehearsal =
                bedDefenseV2PartialDefenseRehearsal(curriculumStage, seed);
        if (randomPose && layers > 0 && !partialDefenseRehearsal
                && random.nextInt(5) == 0) {
            // A supported top spawn can fall into a newly reachable pose after
            // the block underneath is broken. The sub-block jitter prevents a
            // curriculum made only of exact half-block coordinates.
            BlockPos support = (random.nextBoolean() ? foot : head).up(layers);
            playerX = support.getX() + 0.5D + bedDefenseSymmetricJitter(random, 0.18D);
            playerY = support.getY() + 1.0D;
            playerZ = support.getZ() + 0.5D + bedDefenseSymmetricJitter(random, 0.18D);
        } else if (randomPose) {
            double[] pose = randomReachableBedDefensePose(
                    player, foot, head, bedFacing, lateral, layers, random);
            playerX = pose[0];
            playerY = pose[1];
            playerZ = pose[2];
        } else {
            double radius = layers + 2.5D;
            playerX = centerX + lateral.getFrontOffsetX() * radius;
            playerY = arena.feetY;
            playerZ = centerZ + lateral.getFrontOffsetZ() * radius;
        }

        if (partialDefenseRehearsal) {
            openRandomBedSightlineFromPose(
                    arena, foot, head, playerX,
                    playerY + player.getEyeHeight(), playerZ, random);
        }

        float yaw;
        float pitch;
        if (curriculumStage == 0 || curriculumStage == 2) {
            BlockPos aimedHalf = random.nextBoolean() ? foot : head;
            double aimX = aimedHalf.getX() + 0.15D + random.nextDouble() * 0.70D;
            double aimY = aimedHalf.getY() + 0.08D + random.nextDouble() * 0.40D;
            double aimZ = aimedHalf.getZ() + 0.15D + random.nextDouble() * 0.70D;
            float[] look = lookAt(playerX, playerY + player.getEyeHeight(), playerZ,
                    aimX, aimY, aimZ);
            yaw = look[0];
            pitch = look[1];
        } else {
            yaw = random.nextFloat() * 360.0F - 180.0F;
            pitch = random.nextFloat() * 180.0F - 90.0F;
        }

        resetPlayer(player, playerX, playerY, playerZ, yaw, pitch, false);
        player.inventory.setInventorySlotContents(0, new ItemStack(Items.shears));
        player.inventory.setInventorySlotContents(1, new ItemStack(Items.diamond_axe));
        player.inventory.setInventorySlotContents(2, new ItemStack(Items.diamond_pickaxe));
        player.inventory.currentItem = random.nextInt(3);
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);

        double[] goalXs = {foot.getX() + 0.5D, head.getX() + 0.5D};
        double[] goalYs = {foot.getY() + 0.5D, head.getY() + 0.5D};
        double[] goalZs = {foot.getZ() + 0.5D, head.getZ() + 0.5D};
        int[] goalIds = {
                net.minecraft.block.Block.getStateId(world.getBlockState(foot)),
                net.minecraft.block.Block.getStateId(world.getBlockState(head))
        };

        int syncCount = 8 * 4 * 7;
        double[] syncXs = new double[syncCount];
        double[] syncYs = new double[syncCount];
        double[] syncZs = new double[syncCount];
        int[] syncIds = new int[syncCount];
        int syncIndex = 0;
        for (int vertical = 0; vertical < 4; vertical++) {
            for (int lateralIndex = -3; lateralIndex <= 3; lateralIndex++) {
                for (int longitudinal = -3; longitudinal <= 4; longitudinal++) {
                    BlockPos position = foot.offset(bedFacing, longitudinal)
                            .offset(lateral, lateralIndex).up(vertical);
                    syncXs[syncIndex] = position.getX() + 0.5D;
                    syncYs[syncIndex] = position.getY() + 0.5D;
                    syncZs[syncIndex] = position.getZ() + 0.5D;
                    syncIds[syncIndex] = net.minecraft.block.Block.getStateId(
                            world.getBlockState(position));
                    syncIndex++;
                }
            }
        }
        return ResetResult.appliedWithGoalGoalsAndSync(
                foot.getX() + 0.5D, foot.getY() + 0.5D, foot.getZ() + 0.5D,
                goalXs, goalYs, goalZs, goalIds,
                syncXs, syncYs, syncZs, syncIds);
    }

    private ResetResult resetBridgingArena(EntityPlayerMP player, long seed,
                                            int curriculumStage,
                                            Set<BlockPos> cleanupSyncPositions) {
        if (curriculumStage < 0 || curriculumStage > 9) {
            throw new IllegalArgumentException(
                    "bridging curriculum_stage must be between 0 and 9");
        }
        WorldServer world = player.getServerForPlayer();
        if (bridgingArena == null || bridgingArena.world != world) {
            int x = MathHelper.floor_double(player.posX);
            int feetY = MathHelper.clamp_int(
                    MathHelper.floor_double(player.posY + 0.001D), 1, 223);
            int z = MathHelper.floor_double(player.posZ);
            bridgingArena = new BridgingArena(world, x, feetY, z);
        }
        BridgingArena arena = bridgingArena;
        int gap = bridgingGap(curriculumStage, seed);
        Set<BlockPos> syncPositions = new HashSet<>(cleanupSyncPositions);
        BlockPos start = new BlockPos(arena.x, arena.feetY - 1, arena.z);
        BlockPos target = new BlockPos(arena.x, arena.feetY - 1, arena.z + gap + 1);
        int clearMinZ = bridgingClearMinZ(arena.z, gap);
        int clearMaxZ = bridgingClearMaxZ(arena.z, gap);

        AxisAlignedBB entityBounds = new AxisAlignedBB(
                arena.x - 9.0D, arena.feetY - 2.0D, clearMinZ - 1.0D,
                arena.x + 10.0D, arena.feetY + 33.0D, clearMaxZ + 2.0D);
        List<Entity> entities = new ArrayList<>(
                world.getEntitiesWithinAABB(Entity.class, entityBounds));
        for (Entity entity : entities) {
            if (entity != player && !(entity instanceof EntityPlayer)) {
                world.removePlayerEntityDangerously(entity);
            }
        }

        // Clear the exact target-relative volume exposed by BridgingSnapshot.
        // For short gaps this includes observation padding behind the start;
        // failing to clear it can expose stale or pre-existing blocks as illegal
        // above-level placements on the next reset.
        for (int y = arena.feetY - 1; y <= arena.feetY + 31; y++) {
            for (int z = clearMinZ; z <= clearMaxZ; z++) {
                for (int x = arena.x - 8; x <= arena.x + 8; x++) {
                    BlockPos position = new BlockPos(x, y, z);
                    if (!world.isAirBlock(position)) {
                        syncPositions.add(position);
                        world.setBlockState(position, Blocks.air.getDefaultState(), 3);
                    }
                }
            }
        }

        world.setBlockState(start, Blocks.bedrock.getDefaultState(), 3);
        world.setBlockState(target, Blocks.bedrock.getDefaultState(), 3);
        syncPositions.add(start);
        syncPositions.add(target);

        Random random = new Random(seed);
        boolean fixedPose = curriculumStage == 0 || curriculumStage == 2;
        double playerX;
        double playerZ;
        float yaw;
        float pitch;
        if (fixedPose) {
            playerX = start.getX() + 0.5D;
            playerZ = start.getZ() + BRIDGING_FIXED_LOCAL_Z;
            yaw = BRIDGING_FIXED_YAW;
            pitch = BRIDGING_FIXED_PITCH;
        } else {
            // Keep the full 0.6-wide hitbox above the one-block start surface.
            playerX = start.getX() + 0.31D + random.nextDouble() * 0.38D;
            playerZ = start.getZ() + 0.31D + random.nextDouble() * 0.38D;
            yaw = random.nextFloat() * 360.0F - 180.0F;
            pitch = random.nextFloat() * 180.0F - 90.0F;
        }

        resetPlayer(player, playerX, arena.feetY, playerZ, yaw, pitch, false);
        player.inventory.setInventorySlotContents(
                0, new ItemStack(Blocks.wool, 64));
        player.inventory.currentItem = 0;
        player.inventory.markDirty();
        player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                .syncPlayerInventory(player);

        double[] syncXs = new double[syncPositions.size()];
        double[] syncYs = new double[syncPositions.size()];
        double[] syncZs = new double[syncPositions.size()];
        int[] syncIds = new int[syncPositions.size()];
        int syncIndex = 0;
        for (BlockPos position : syncPositions) {
            syncXs[syncIndex] = position.getX() + 0.5D;
            syncYs[syncIndex] = position.getY() + 0.5D;
            syncZs[syncIndex] = position.getZ() + 0.5D;
            syncIds[syncIndex] = net.minecraft.block.Block.getStateId(
                    world.getBlockState(position));
            syncIndex++;
        }
        return ResetResult.appliedWithGoalGoalsAndSync(
                target.getX() + 0.5D, target.getY() + 0.5D, target.getZ() + 0.5D,
                new double[0], new double[0], new double[0], new int[0],
                syncXs, syncYs, syncZs, syncIds);
    }

    public static int bridgingGap(int curriculumStage, long seed) {
        switch (curriculumStage) {
            case 0:
            case 1: return 1;
            case 2:
            case 3: return 5;
            case 4: return 8;
            case 5: return 11;
            case 6: return 15;
            case 7: return 22;
            case 8: return 30;
            default: return 16 + (int) Math.floorMod(seed, 49L);
        }
    }

    static int bridgingClearMinZ(int arenaZ, int gap) {
        return arenaZ + gap - 66;
    }

    static int bridgingClearMaxZ(int arenaZ, int gap) {
        return arenaZ + gap + 3;
    }


    private static void clearBedDefenseArena(
            BedDefenseBreakArena arena, Set<BlockPos> syncPositions) {
        for (int y = arena.feetY - 1; y <= arena.feetY + 5; y++) {
            for (int z = arena.z - 6; z <= arena.z + 6; z++) {
                for (int x = arena.x - 6; x <= arena.x + 6; x++) {
                    BlockPos position = new BlockPos(x, y, z);
                    if (!arena.world.isAirBlock(position)) {
                        syncPositions.add(position);
                        arena.world.setBlockState(
                                position, Blocks.air.getDefaultState(), 3);
                    }
                }
            }
        }
        arena.dynamicBlocks.clear();
        arena.initialized = false;
    }

    private static void clearBridgingArena(BridgingArena arena) {
        for (int y = arena.feetY - 1; y <= arena.feetY + 31; y++) {
            // Union of target-relative scan ranges for gaps 1 through 64.
            for (int z = arena.z - 65; z <= arena.z + 67; z++) {
                for (int x = arena.x - 8; x <= arena.x + 8; x++) {
                    BlockPos position = new BlockPos(x, y, z);
                    if (!arena.world.isAirBlock(position)) {
                        arena.world.setBlockState(
                                position, Blocks.air.getDefaultState(), 3);
                    }
                }
            }
        }
    }

    static int bedDefenseV2LayerCount(int stage, long seed) {
        if (stage <= 1) return 0;
        if (stage <= 3) return 1;
        if (stage == 4) return 2;
        return 1 + (int) Math.floorMod(seed, 2L);
    }

    /** Material indices are wool=0, wood=1, end stone=2, obsidian=3. */
    static int[] bedDefenseV2MaterialIndices(int stage, int layers, long seed) {
        int[] result = new int[layers];
        if (layers == 0) return result;
        if (stage <= 3) {
            result[0] = (int) Math.floorMod(seed, 3L);
            return result;
        }
        if (stage == 4) {
            result[0] = (int) Math.floorMod(seed, 3L);
            result[1] = (int) Math.floorMod(Math.floorDiv(seed, 3L), 3L);
            return result;
        }
        if (stage == 5) {
            if (bedDefenseV2StageFiveRehearsal(seed)) {
                long rehearsalIndex = Math.floorDiv(seed, 5L);
                int divisor = 1;
                for (int layer = 0; layer < layers; layer++) {
                    result[layer] = (int) Math.floorMod(
                            Math.floorDiv(rehearsalIndex, divisor), 3L);
                    divisor *= 3;
                }
                return result;
            }
            result[0] = 3;
            if (layers == 2) {
                result[1] = (int) Math.floorMod(Math.floorDiv(seed, 2L), 3L);
            }
            return result;
        }
        if (layers == 1) {
            result[0] = (int) Math.floorMod(Math.floorDiv(seed, 2L), 4L);
            return result;
        }
        int pair = (int) Math.floorMod(Math.floorDiv(seed, 2L), 15L);
        int seen = 0;
        for (int inner = 0; inner < 4; inner++) {
            for (int outer = 0; outer < 4; outer++) {
                if (inner == 3 && outer == 3) continue;
                if (seen++ == pair) {
                    result[0] = inner;
                    result[1] = outer;
                    return result;
                }
            }
        }
        throw new IllegalStateException("unreachable V2 material pair");
    }

    /** One in five stage-five episodes rehearses pre-obsidian tool switching. */
    static boolean bedDefenseV2StageFiveRehearsal(long seed) {
        return Math.floorMod(seed, 5L) == 0L;
    }

    /** One in ten random-pose layered episodes starts with a reachable bed opening. */
    static boolean bedDefenseV2PartialDefenseRehearsal(int stage, long seed) {
        long position = Math.floorMod(seed, 20L);
        // Use one odd and one even seed per twenty so variable-depth stages
        // rehearse both two-layer and one-layer openings. Neither position is
        // divisible by five, keeping stage-five no-obsidian rehearsals disjoint.
        return stage >= 3 && (position == 1L || position == 12L);
    }

    private static void openRandomBedSightlineFromPose(
            BedDefenseBreakArena arena, BlockPos foot, BlockPos head,
            double eyeX, double eyeY, double eyeZ, Random random) {
        Vec3 eyes = new Vec3(eyeX, eyeY, eyeZ);
        List<BlockPos> removedPositions = new ArrayList<BlockPos>();
        List<IBlockState> removedStates = new ArrayList<IBlockState>();
        for (int attempt = 0; attempt < 64; attempt++) {
            restoreBedDefenseBlocks(arena.world, removedPositions, removedStates);
            removedPositions.clear();
            removedStates.clear();

            BlockPos targetBed = random.nextBoolean() ? foot : head;
            Vec3 target = randomBedVisibilitySample(
                    arena.world, targetBed, random);
            if (eyes.distanceTo(target) > 4.5D) continue;

            for (int obstruction = 0; obstruction < 12; obstruction++) {
                MovingObjectPosition hit = arena.world.rayTraceBlocks(
                        eyes, target, false, true, false);
                if (hit == null
                        || hit.typeOfHit
                        != MovingObjectPosition.MovingObjectType.BLOCK) {
                    break;
                }
                BlockPos hitPosition = hit.getBlockPos();
                if (targetBed.equals(hitPosition)) {
                    if (!removedPositions.isEmpty()) return;
                    break;
                }
                if (foot.equals(hitPosition) || head.equals(hitPosition)) break;
                IBlockState state = arena.world.getBlockState(hitPosition);
                if (!isBedDefenseV2Material(state)) break;
                removedPositions.add(hitPosition);
                removedStates.add(state);
                arena.world.setBlockState(
                        hitPosition, Blocks.air.getDefaultState(), 3);
            }
        }
        restoreBedDefenseBlocks(arena.world, removedPositions, removedStates);
        throw new IllegalStateException(
                "could not open a reachable partial-defense bed sightline");
    }

    private static Vec3 randomBedVisibilitySample(
            WorldServer world, BlockPos bed, Random random) {
        IBlockState state = world.getBlockState(bed);
        AxisAlignedBB bounds = state.getBlock().getSelectedBoundingBox(world, bed);
        if (bounds == null) {
            throw new IllegalStateException("bed has no selection hitbox");
        }
        double[] fractions = {0.15D, 0.5D, 0.85D};
        int face = random.nextInt(6);
        double first = fractions[random.nextInt(fractions.length)];
        double second = fractions[random.nextInt(fractions.length)];
        double x;
        double y;
        double z;
        switch (face) {
            case 0:
            case 1:
                x = mix(bounds.minX, bounds.maxX, first);
                y = face == 0 ? bounds.minY : bounds.maxY;
                z = mix(bounds.minZ, bounds.maxZ, second);
                break;
            case 2:
            case 3:
                x = mix(bounds.minX, bounds.maxX, first);
                y = mix(bounds.minY, bounds.maxY, second);
                z = face == 2 ? bounds.minZ : bounds.maxZ;
                break;
            default:
                x = face == 4 ? bounds.minX : bounds.maxX;
                y = mix(bounds.minY, bounds.maxY, first);
                z = mix(bounds.minZ, bounds.maxZ, second);
                break;
        }
        return new Vec3(x, y, z);
    }

    private static double mix(double minimum, double maximum, double fraction) {
        return minimum + (maximum - minimum) * fraction;
    }

    private static boolean isBedDefenseV2Material(IBlockState state) {
        return state.getBlock() == Blocks.wool
                || state.getBlock() == Blocks.planks
                || state.getBlock() == Blocks.end_stone
                || state.getBlock() == Blocks.obsidian;
    }

    private static void restoreBedDefenseBlocks(
            WorldServer world, List<BlockPos> positions,
            List<IBlockState> states) {
        for (int index = 0; index < positions.size(); index++) {
            world.setBlockState(positions.get(index), states.get(index), 3);
        }
    }

    private static IBlockState bedDefenseV2Material(int index) {
        switch (index) {
            case 0: return Blocks.wool.getDefaultState();
            case 1: return Blocks.planks.getDefaultState();
            case 2: return Blocks.end_stone.getDefaultState();
            case 3: return Blocks.obsidian.getDefaultState();
            default: throw new IllegalArgumentException("unknown V2 material index");
        }
    }

    private static double[] randomReachableBedDefensePose(
            EntityPlayerMP player, BlockPos foot, BlockPos head,
            EnumFacing longitudinal, EnumFacing lateral, int layers, Random random) {
        double centerX = (foot.getX() + head.getX() + 1.0D) * 0.5D;
        double centerZ = (foot.getZ() + head.getZ() + 1.0D) * 0.5D;
        double minimumRadius = layers + 2.1D;
        double maximumRadius = 4.3D;
        for (int attempt = 0; attempt < 64; attempt++) {
            double angle = random.nextDouble() * Math.PI * 2.0D;
            double radius = minimumRadius
                    + random.nextDouble() * (maximumRadius - minimumRadius);
            double along = Math.cos(angle) * radius;
            double across = Math.sin(angle) * radius;
            double x = centerX + longitudinal.getFrontOffsetX() * along
                    + lateral.getFrontOffsetX() * across;
            double z = centerZ + longitudinal.getFrontOffsetZ() * along
                    + lateral.getFrontOffsetZ() * across;
            double y = foot.getY();
            AxisAlignedBB playerBounds = new AxisAlignedBB(
                    x - 0.30D, y, z - 0.30D,
                    x + 0.30D, y + 1.80D, z + 0.30D);
            if (!player.getServerForPlayer().getCollidingBoundingBoxes(
                    player, playerBounds).isEmpty()) {
                continue;
            }
            if (nearestBedDistance(x, y + player.getEyeHeight(), z, foot, head)
                    <= 4.5D) {
                return new double[] {x, y, z};
            }
        }
        double radius = layers + 2.5D;
        return new double[] {
                centerX + lateral.getFrontOffsetX() * radius,
                foot.getY(),
                centerZ + lateral.getFrontOffsetZ() * radius
        };
    }

    static double nearestBedDistance(double eyeX, double eyeY, double eyeZ,
                                     BlockPos foot, BlockPos head) {
        double minX = Math.min(foot.getX(), head.getX());
        double maxX = Math.max(foot.getX(), head.getX()) + 1.0D;
        double minZ = Math.min(foot.getZ(), head.getZ());
        double maxZ = Math.max(foot.getZ(), head.getZ()) + 1.0D;
        double nearestX = Math.max(minX, Math.min(maxX, eyeX));
        double nearestY = Math.max(foot.getY(),
                Math.min(foot.getY() + 0.5625D, eyeY));
        double nearestZ = Math.max(minZ, Math.min(maxZ, eyeZ));
        double dx = eyeX - nearestX;
        double dy = eyeY - nearestY;
        double dz = eyeZ - nearestZ;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private static void setBedDefenseBlock(BedDefenseBreakArena arena,
                                           BlockPos position, IBlockState state) {
        arena.world.setBlockState(position, state, 3);
        arena.dynamicBlocks.add(position);
    }

    static double bedDefenseCornerRadius(int layers) {
        // Keep the diagonal pose outside the shell while reserving enough of
        // the 4.5-block reach budget for continuous corner jitter.
        return layers + 1.4D;
    }

    static double bedDefenseAxisRadius(int tunnelDepth, int layers) {
        return tunnelDepth > 0 ? tunnelDepth + 1.5D : layers + 2.5D;
    }

    static double bedDefenseSymmetricJitter(Random random, double limit) {
        if (limit < 0.0D) {
            throw new IllegalArgumentException("jitter limit cannot be negative");
        }
        return (random.nextDouble() * 2.0D - 1.0D) * limit;
    }

    static int bedDefenseSingleCoverMaterialIndex(long seed) {
        return (int) Math.floorMod(seed, 3L);
    }

    private static IBlockState singleCoverMaterial(long seed) {
        switch (bedDefenseSingleCoverMaterialIndex(seed)) {
            case 0: return Blocks.wool.getDefaultState();
            case 1: return Blocks.planks.getDefaultState();
            default: return Blocks.end_stone.getDefaultState();
        }
    }

    private static IBlockState tunnelMaterial(int depth, Random random) {
        int choices = 3;
        int choice = Math.floorMod(random.nextInt() + depth, choices);
        if (choice == 0) return Blocks.wool.getDefaultState();
        if (choice == 1) return Blocks.planks.getDefaultState();
        return Blocks.end_stone.getDefaultState();
    }

    private static IBlockState layerMaterial(int curriculumStage, int layer, Random random) {
        if (curriculumStage == 8 && layer == 1) {
            return Blocks.obsidian.getDefaultState();
        }
        if (curriculumStage >= 9 && layer == 1 && random.nextInt(5) == 0) {
            return Blocks.obsidian.getDefaultState();
        }
        int choice = random.nextInt(3);
        if (choice == 0) return Blocks.wool.getDefaultState();
        if (choice == 1) return Blocks.planks.getDefaultState();
        return Blocks.end_stone.getDefaultState();
    }

    private static int manhattan(BlockPos first, BlockPos second) {
        return Math.abs(first.getX() - second.getX())
                + Math.abs(first.getY() - second.getY())
                + Math.abs(first.getZ() - second.getZ());
    }

    private static EnumFacing perpendicular(EnumFacing facing, boolean clockwise) {
        switch (facing) {
            case NORTH: return clockwise ? EnumFacing.EAST : EnumFacing.WEST;
            case EAST: return clockwise ? EnumFacing.SOUTH : EnumFacing.NORTH;
            case SOUTH: return clockwise ? EnumFacing.WEST : EnumFacing.EAST;
            case WEST: return clockwise ? EnumFacing.NORTH : EnumFacing.SOUTH;
            default: throw new IllegalArgumentException("horizontal facing required");
        }
    }

    private static float[] lookAt(double playerX, double eyeY, double playerZ,
                                  double targetX, double targetY, double targetZ) {
        double dx = targetX - playerX;
        double dy = targetY - eyeY;
        double dz = targetZ - playerZ;
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        return new float[] {
                (float) Math.toDegrees(Math.atan2(-dx, dz)),
                (float) -Math.toDegrees(Math.atan2(dy, horizontal))
        };
    }

    private static int safeArenaFeetY(EntityPlayerMP player) {
        int currentFeetY = MathHelper.floor_double(player.posY + 0.001D);
        // A disconnected/crashed trainer restores vanilla ticking. In a void world the
        // player may therefore be below Y=0 by the next launch. Blocks outside 0..255 are
        // silently rejected, so anchoring an arena there produces an "air" reset result.
        // Keep room for the support below and the tallest target/clearance above.
        return MathHelper.clamp_int(currentFeetY, 1, 253);
    }

    private static void clearArenaEntities(EntityPlayerMP player, WorldServer world,
                                           int x, int feetY, int z) {
        AxisAlignedBB bounds = new AxisAlignedBB(
                x - 5.0D, feetY - 1.0D, z - 5.0D,
                x + 6.0D, feetY + 4.0D, z + 6.0D);
        List<Entity> entities = new ArrayList<>(world.getEntitiesWithinAABB(Entity.class, bounds));
        for (Entity entity : entities) {
            // Intentionally includes EntityItem drops (such as the broken TNT),
            // projectiles, XP orbs and task mobs. Nothing survives into the next episode.
            if (entity != player && !(entity instanceof EntityPlayer)) {
                // Despite its 1.8.9 MCP name, this accepts any Entity. It immediately
                // removes/untracks it and sends the destroy packet; setDead/removeEntity
                // alone can leave a ghost EntityItem rendered in the next episode.
                world.removePlayerEntityDangerously(entity);
            }
        }
    }

    private static void resetPlayer(EntityPlayerMP player, double x, double y, double z,
                                    float yaw, float pitch) {
        resetPlayer(player, x, y, z, yaw, pitch, true);
    }

    private static void resetPlayer(EntityPlayerMP player, double x, double y, double z,
                                    float yaw, float pitch, boolean syncInventory) {
        player.theItemInWorldManager.cancelDestroyingBlock();
        player.theItemInWorldManager.setGameType(WorldSettings.GameType.SURVIVAL);
        player.stopUsingItem();
        player.clearActivePotions();
        player.extinguish();
        player.setAir(300);
        player.setHealth(player.getMaxHealth());
        player.setAbsorptionAmount(0.0F);
        player.getFoodStats().setFoodLevel(20);
        player.getFoodStats().setFoodSaturationLevel(5.0F);
        player.inventory.clear();
        player.inventory.currentItem = 0;
        player.inventory.markDirty();
        player.setSprinting(false);
        player.setSneaking(false);
        player.motionX = 0.0D;
        player.motionY = 0.0D;
        player.motionZ = 0.0D;
        player.velocityChanged = true;
        player.fallDistance = 0.0F;
        player.onGround = true;
        player.hurtResistantTime = 0;
        player.isCollided = false;
        player.isCollidedHorizontally = false;
        player.isCollidedVertically = true;
        player.playerNetServerHandler.setPlayerLocation(x, y, z, yaw, pitch);
        player.prevPosX = player.lastTickPosX = x;
        player.prevPosY = player.lastTickPosY = y;
        player.prevPosZ = player.lastTickPosZ = z;
        player.prevRotationYaw = yaw;
        player.prevRotationPitch = pitch;
        if (syncInventory) {
            player.getServerForPlayer().getMinecraftServer().getConfigurationManager()
                    .syncPlayerInventory(player);
        }
    }

    public static final class ResetResult {
        public final boolean applied;
        public final int taskEntityId;
        public final double goalX;
        public final double goalY;
        public final double goalZ;
        public final double[] goalXs;
        public final double[] goalYs;
        public final double[] goalZs;
        public final int[] goalBlockIds;
        public final double[] syncXs;
        public final double[] syncYs;
        public final double[] syncZs;
        public final int[] syncBlockIds;

        private ResetResult(boolean applied, int taskEntityId,
                             double goalX, double goalY, double goalZ,
                             double[] goalXs, double[] goalYs, double[] goalZs,
                             int[] goalBlockIds,
                             double[] syncXs, double[] syncYs, double[] syncZs,
                             int[] syncBlockIds) {
            this.applied = applied;
            this.taskEntityId = taskEntityId;
            this.goalX = goalX;
            this.goalY = goalY;
            this.goalZ = goalZ;
            this.goalXs = goalXs;
            this.goalYs = goalYs;
            this.goalZs = goalZs;
            this.goalBlockIds = goalBlockIds;
            this.syncXs = syncXs;
            this.syncYs = syncYs;
            this.syncZs = syncZs;
            this.syncBlockIds = syncBlockIds;
        }

        public static ResetResult notApplied() {
            return new ResetResult(false, -1, Double.NaN, Double.NaN, Double.NaN,
                    null, null, null, null, null, null, null, null);
        }

        public static ResetResult appliedWithoutEntity() {
            return new ResetResult(true, -1, Double.NaN, Double.NaN, Double.NaN,
                    null, null, null, null, null, null, null, null);
        }

        public static ResetResult appliedWithEntity(int entityId) {
            return new ResetResult(true, entityId, Double.NaN, Double.NaN, Double.NaN,
                    null, null, null, null, null, null, null, null);
        }

        public static ResetResult appliedWithGoal(double x, double y, double z) {
            return new ResetResult(true, -1, x, y, z, null, null, null, null,
                    null, null, null, null);
        }

        public static ResetResult appliedWithGoals(
                double[] xs, double[] ys, double[] zs, int[] blockIds) {
            if (xs.length != ys.length || xs.length != zs.length
                    || xs.length != blockIds.length) {
                throw new IllegalArgumentException("multi-goal coordinate lengths differ");
            }
            return new ResetResult(true, -1, Double.NaN, Double.NaN, Double.NaN,
                    xs, ys, zs, blockIds, xs, ys, zs, blockIds);
        }

        public static ResetResult appliedWithGoalGoalsAndSync(
                double goalX, double goalY, double goalZ,
                double[] goalXs, double[] goalYs, double[] goalZs, int[] goalBlockIds,
                double[] syncXs, double[] syncYs, double[] syncZs, int[] syncBlockIds) {
            if (goalXs.length != goalYs.length || goalXs.length != goalZs.length
                    || goalXs.length != goalBlockIds.length) {
                throw new IllegalArgumentException("goal coordinate lengths differ");
            }
            if (syncXs.length != syncYs.length || syncXs.length != syncZs.length
                    || syncXs.length != syncBlockIds.length) {
                throw new IllegalArgumentException("sync coordinate lengths differ");
            }
            return new ResetResult(true, -1, goalX, goalY, goalZ,
                    goalXs, goalYs, goalZs, goalBlockIds,
                    syncXs, syncYs, syncZs, syncBlockIds);
        }
    }

    private static final class BreakBlockArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;

        private BreakBlockArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class AimBreakTntArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;

        private AimBreakTntArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class AimToolBreakArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;

        private AimToolBreakArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class MultiToolBreakArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;

        private MultiToolBreakArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class BedDefenseBreakArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;
        private final Set<BlockPos> dynamicBlocks = new HashSet<>();
        private boolean initialized;

        private BedDefenseBreakArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class BridgingArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;

        private BridgingArena(WorldServer world, int x, int feetY, int z) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
        }
    }

    private static final class PvpArena {
        private final WorldServer world;
        private final int x;
        private final int feetY;
        private final int z;
        private final EnumDifficulty previousDifficulty;

        private PvpArena(WorldServer world, int x, int feetY, int z,
                         EnumDifficulty previousDifficulty) {
            this.world = world;
            this.x = x;
            this.feetY = feetY;
            this.z = z;
            this.previousDifficulty = previousDifficulty;
        }
    }
}
