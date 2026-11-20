package com.minecraftrl.benchmark;

import com.minecraftrl.ClientEvents;
import com.minecraftrl.environment.EnvironmentResetManager;
import com.minecraftrl.network.TcpBridgeClient;
import com.minecraftrl.network.WireMessage;
import com.minecraftrl.tick.TickController;
import com.minecraftrl.world.ClientBlockChangeTracker;
import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.GameSettings;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLivingBase;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.init.Blocks;
import net.minecraft.init.Items;
import net.minecraft.item.Item;
import net.minecraft.item.ItemBlock;
import net.minecraft.item.ItemStack;
import net.minecraft.network.play.server.S23PacketBlockChange;
import net.minecraft.server.integrated.IntegratedServer;
import net.minecraft.util.BlockPos;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.MathHelper;
import net.minecraft.util.MovingObjectPosition;
import net.minecraft.util.Vec3;
import com.minecraftrl.mixin.AccessorPlayerControllerMP;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.locks.LockSupport;

/** Drives causal benchmark and persistent Gym-style environment sessions. */
public final class SynchronizedBenchmark {
    private enum SessionKind {
        NONE,
        BENCHMARK,
        ENVIRONMENT
    }

    private enum Phase {
        IDLE,
        ARMING,
        RESET_WAIT_SERVER,
        RESET_RESYNC_WAIT_SERVER,
        RESET_POST_SERVER_CLIENT_TICK,
        ACTION_CLIENT_TICK,
        WAIT_SERVER,
        POST_SERVER_CLIENT_TICK
    }

    private static final long PHASE_TIMEOUT_NANOS = 30_000_000_000L;
    private static final double MAX_MOUSE_DELTA = 10_000.0D;
    private static final int MAX_TASK_BLOCKS = 5;
    private static final int MAX_RESET_CLIENT_SETTLE_TICKS = 8;
    private static final int MAX_RESET_CLIENT_RECOVERY_TICKS = 250;
    private static final int MAX_RESET_CLIENT_RESYNC_ATTEMPTS = 1;
    private static final long RESET_CLIENT_RECOVERY_PARK_NANOS = 1_000_000L;
    private static final Set<String> SUPPORTED_ACTION_FIELDS = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList("mouse_dx", "mouse_dy", "forward", "back",
                    "left", "right", "jump", "attack", "sneak", "sprint", "use",
                    "attack_once", "use_once", "hotbar_slot")));
    private static final Set<String> SUPPORTED_OBSERVATION_FIELDS =
            createSupportedObservationFields();

    private static Set<String> createSupportedObservationFields() {
        Set<String> fields = new HashSet<>(Arrays.asList(
                    "player.x", "player.y", "player.z", "player.yaw",
                    "player.pitch", "player.health", "player.on_ground", "player.motion_x",
                    "player.motion_y", "player.motion_z", "player.sneaking", "player.sprinting",
                    "player.using_item", "player.item_use_ticks", "player.selected_slot",
                    "player.collided_horizontally", "player.collided_vertically",
                    "player.fall_distance", "player.block_break_progress",
                    "held.item_id", "held.item_name", "held.count", "held.damage",
                    "held.max_damage", "inventory.arrow_count", "inventory.hotbar",
                    "raycast.type", "raycast.name", "raycast.id", "raycast.x", "raycast.y",
                    "raycast.z", "raycast.side", "raycast.distance", "raycast.health",
                    "raycast.block_hardness", "raycast.can_harvest",
                    "raycast.category", "raycast.safe_yaw_correction",
                    "raycast.safe_pitch_correction",
                    "nearest_living.type", "nearest_living.id", "nearest_living.x",
                    "nearest_living.y", "nearest_living.z", "nearest_living.relative_x",
                    "nearest_living.relative_y", "nearest_living.relative_z",
                    "nearest_living.distance", "nearest_living.yaw_delta",
                    "nearest_living.pitch_delta", "nearest_living.motion_x",
                    "nearest_living.motion_y", "nearest_living.motion_z",
                    "nearest_living.health", "nearest_living.alive",
                    "task_entity.type", "task_entity.id", "task_entity.x", "task_entity.y",
                    "task_entity.z", "task_entity.relative_x", "task_entity.relative_y",
                    "task_entity.relative_z", "task_entity.distance", "task_entity.yaw_delta",
                    "task_entity.pitch_delta", "task_entity.motion_x", "task_entity.motion_y",
                    "task_entity.motion_z", "task_entity.health", "task_entity.alive",
                    "task_goal.x", "task_goal.y", "task_goal.z", "task_goal.relative_x",
                    "task_goal.relative_y", "task_goal.relative_z", "task_goal.distance",
                    "task_goal.horizontal_distance", "task_goal.yaw_delta",
                    "task_goal.pitch_delta", "task_goal.block_id",
                    "task_goal.tool_speed_0", "task_goal.tool_speed_1",
                    "task_goal.tool_speed_2",
                    "local.blocks_3x3x3", "task_blocks.count",
                    "bed_defense.origin_x", "bed_defense.origin_y",
                    "bed_defense.origin_z", "bed_defense.block_grid",
                    "bed_defense.visible_mask", "bed_defense_v2.block_grid",
                    "bed_defense_v2.visible_mask",
                    "bridging.ground_fraction", "bridging.distance_forward_edge",
                    "bridging.distance_left_edge", "bridging.distance_right_edge",
                    "bridging.distance_rear_edge", "bridging.can_place",
                    "bridging.block_grid", "bridging.above_level_count"));
        for (int index = 0; index < MAX_TASK_BLOCKS; index++) {
            String prefix = "task_block_" + index + ".";
            fields.add(prefix + "x");
            fields.add(prefix + "y");
            fields.add(prefix + "z");
            fields.add(prefix + "distance");
            fields.add(prefix + "yaw_delta");
            fields.add(prefix + "pitch_delta");
            fields.add(prefix + "block_id");
            fields.add(prefix + "tool_speed_0");
            fields.add(prefix + "tool_speed_1");
            fields.add(prefix + "tool_speed_2");
        }
        return Collections.unmodifiableSet(fields);
    }

    private final TcpBridgeClient bridge;
    private final TickController controller = TickController.INSTANCE;
    private final EnvironmentResetManager resetManager = new EnvironmentResetManager();
    private boolean active;
    private SessionKind sessionKind = SessionKind.NONE;
    private int expectedSteps;
    private int expectedSequence;
    private long startedNanos;
    private long lastProgressNanos;
    private long connectionEpoch;
    private long awaitedServerTick;
    private Set<String> environmentActionFields = Collections.emptySet();
    private List<String> environmentObservationFields = Collections.emptyList();
    private ObservationPlan environmentObservationPlan = ObservationPlan.EMPTY;
    private boolean attackHeld;
    private boolean useHeld;
    private int environmentTaskEntityId = -1;
    private double environmentGoalX = Double.NaN;
    private double environmentGoalY = Double.NaN;
    private double environmentGoalZ = Double.NaN;
    private double[] environmentBlockGoalXs = new double[0];
    private double[] environmentBlockGoalYs = new double[0];
    private double[] environmentBlockGoalZs = new double[0];
    private int[] environmentBlockGoalIds = new int[0];
    private double[] environmentSyncXs = new double[0];
    private double[] environmentSyncYs = new double[0];
    private double[] environmentSyncZs = new double[0];
    private int[] environmentSyncIds = new int[0];
    private BedDefenseSnapshot bedDefenseSnapshotCache;
    private BedDefenseV2Snapshot bedDefenseV2SnapshotCache;
    private BridgingSnapshot bridgingSnapshotCache;
    private int resetClientSettleTicks;
    private int resetClientResyncAttempts;
    private boolean environmentProfileJavaPhases;
    private long actionAcceptedNanos;
    private long actionClientCompletedNanos;
    private long serverRequestedNanos;
    private long serverCompletedNanos;
    private long postClientStartedNanos;
    private WireMessage pendingEnvironmentReset;
    private volatile EnvironmentResetManager.ResetResult environmentResetResult;
    private volatile String environmentResetError;
    private Phase phase = Phase.IDLE;

    public SynchronizedBenchmark(TcpBridgeClient bridge) {
        this.bridge = bridge;
    }

    public boolean isActive() {
        return active;
    }

    public void start(int steps) throws IOException {
        if (active) throw new IOException("A synchronized benchmark is already running");
        if (!bridge.isConnected()) throw new IOException("Python bridge is not connected");
        Minecraft minecraft = Minecraft.getMinecraft();
        if (!minecraft.isIntegratedServerRunning()
                || minecraft.thePlayer == null || minecraft.theWorld == null) {
            throw new IOException("Open a single-player world before starting the synchronized benchmark");
        }

        active = true;
        sessionKind = SessionKind.BENCHMARK;
        environmentActionFields = Collections.emptySet();
        environmentObservationFields = Collections.emptyList();
        environmentObservationPlan = ObservationPlan.EMPTY;
        environmentTaskEntityId = -1;
        clearPendingEnvironmentReset();
        clearEnvironmentGoal();
        expectedSteps = steps;
        expectedSequence = 0;
        phase = Phase.ARMING;
        startedNanos = System.nanoTime();
        lastProgressNanos = startedNanos;
        connectionEpoch = bridge.getConnectionEpoch();
        controller.setMode(TickController.Mode.SYNC);
        controller.setSynchronizedRenderingEnabled(false);
    }

    public boolean handleProtocol(WireMessage message) {
        if ("sync_action".equals(message.type)) {
            if (sessionKind != SessionKind.BENCHMARK) {
                sendEnvironmentError("sync_action received outside benchmark session");
                return true;
            }
            acceptAction(message);
            return true;
        }
        if ("sync_benchmark_complete".equals(message.type)) {
            acceptCompletion(message);
            return true;
        }
        if ("sync_benchmark_error".equals(message.type)) {
            abort("Python error: " + message.text, false);
            return true;
        }
        if ("env_reset".equals(message.type)) {
            acceptEnvironmentReset(message);
            return true;
        }
        if ("env_action".equals(message.type)) {
            if (sessionKind != SessionKind.ENVIRONMENT) {
                sendEnvironmentError("env_action received without an active environment");
                return true;
            }
            acceptAction(message);
            return true;
        }
        if ("env_close".equals(message.type)) {
            closeEnvironment();
            return true;
        }
        return false;
    }

    public void onClientTickStart() {
        // Controlled phase transitions are driven from RenderTickEvent because
        // SYNC intentionally produces no vanilla client ticks while waiting.
    }

    public boolean isAwaitingAction() { return active && phase == Phase.IDLE; }

    public void onSynchronizedMainLoop() {
        if (!active) return;
        if (!bridge.isConnected() || bridge.getConnectionEpoch() != connectionEpoch) {
            abort("TCP bridge disconnected or reconnected", false);
            return;
        }
        if (phase != Phase.IDLE && System.nanoTime() - lastProgressNanos > PHASE_TIMEOUT_NANOS) {
            abort("timed out in phase " + phase.name().toLowerCase(Locale.ROOT), true);
            return;
        }
        if (phase == Phase.ARMING && controller.isServerWaitingAtGate()) {
            if (sessionKind == SessionKind.BENCHMARK) {
                if (!bridge.sendProtocol(WireMessage.synchronizedBenchmarkStart(expectedSteps))) {
                    abort("TCP outbound queue is full while starting benchmark", false);
                    return;
                }
            } else if (sessionKind == SessionKind.ENVIRONMENT) {
                Minecraft minecraft = Minecraft.getMinecraft();
                if (pendingEnvironmentReset != null
                        && resetManager.supports(pendingEnvironmentReset.task)) {
                    scheduleEnvironmentReset(minecraft);
                } else {
                    lockNearestTaskEntityIfRequired(minecraft);
                    updateTargetingIfRequired(minecraft);
                    if (!bridge.sendRawJson(createEnvironmentObservation(
                            minecraft, "env_reset_result", 0, Boolean.FALSE))) {
                        abort("TCP outbound queue is full while resetting environment", false);
                        return;
                    }
                    pendingEnvironmentReset = null;
                    phase = Phase.IDLE;
                    markProgress();
                }
            }
            if (sessionKind == SessionKind.BENCHMARK) {
                phase = Phase.IDLE;
                markProgress();
            }
        } else if (phase == Phase.RESET_WAIT_SERVER
                && controller.getCompletedServerTicks() >= awaitedServerTick) {
            if (environmentResetError != null) {
                abort("environment reset failed: " + environmentResetError, true);
                return;
            }
            if (environmentResetResult == null) {
                abort("environment reset did not execute on the server thread", true);
                return;
            }
            environmentTaskEntityId = environmentResetResult.taskEntityId;
            if (Double.isFinite(environmentResetResult.goalX)
                    && Double.isFinite(environmentResetResult.goalY)
                    && Double.isFinite(environmentResetResult.goalZ)) {
                environmentGoalX = environmentResetResult.goalX;
                environmentGoalY = environmentResetResult.goalY;
                environmentGoalZ = environmentResetResult.goalZ;
            }
            if (environmentResetResult.goalXs != null) {
                environmentBlockGoalXs = environmentResetResult.goalXs;
                environmentBlockGoalYs = environmentResetResult.goalYs;
                environmentBlockGoalZs = environmentResetResult.goalZs;
                environmentBlockGoalIds = environmentResetResult.goalBlockIds;
            }
            if (environmentResetResult.syncXs != null) {
                environmentSyncXs = environmentResetResult.syncXs;
                environmentSyncYs = environmentResetResult.syncYs;
                environmentSyncZs = environmentResetResult.syncZs;
                environmentSyncIds = environmentResetResult.syncBlockIds;
            }
            phase = Phase.RESET_POST_SERVER_CLIENT_TICK;
            resetClientSettleTicks = 0;
            controller.setSuppressSynchronizedAttackHandling(true);
            setKey(Minecraft.getMinecraft().gameSettings.keyBindAttack, false);
            controller.requestSynchronizedClientTick();
            markProgress();
        } else if (phase == Phase.RESET_RESYNC_WAIT_SERVER
                && controller.getCompletedServerTicks() >= awaitedServerTick) {
            if (environmentResetError != null) {
                abort("environment reset block resync failed: " + environmentResetError, true);
                return;
            }
            phase = Phase.RESET_POST_SERVER_CLIENT_TICK;
            resetClientSettleTicks = 0;
            controller.requestSynchronizedClientTick();
            markProgress();
        } else if (phase == Phase.WAIT_SERVER
                && controller.getCompletedServerTicks() >= awaitedServerTick) {
            serverCompletedNanos = System.nanoTime();
            phase = Phase.POST_SERVER_CLIENT_TICK;
            // Integrate packets without adding a second block-damage update or
            // resetting an ongoing held attack. Keep the physical key state intact
            // so visualized rendering does not alternate between held/released.
            controller.setSuppressSynchronizedAttackHandling(true);
            postClientStartedNanos = System.nanoTime();
            controller.requestSynchronizedClientTick();
            markProgress();
        }
    }

    public void onClientTickEnd() {
        if (!active) return;
        if (phase == Phase.RESET_POST_SERVER_CLIENT_TICK) {
            Minecraft minecraft = Minecraft.getMinecraft();
            stabilizeClientPlayerAfterReset(minecraft);
            // The teleport/respawn packet is consumed during this client tick.
            // Refresh the crosshair before judging reset readiness so an aimed
            // bridging reset cannot be accepted with a stale pre-reset MISS.
            updateTargetingIfRequired(minecraft);
            if (!clientResetStateReady(minecraft)) {
                int settleLimit = resetClientResyncAttempts == 0
                        ? MAX_RESET_CLIENT_SETTLE_TICKS
                        : MAX_RESET_CLIENT_RECOVERY_TICKS;
                if (resetClientSettleTicks >= settleLimit) {
                    if (resetClientResyncAttempts < MAX_RESET_CLIENT_RESYNC_ATTEMPTS) {
                        resetClientResyncAttempts++;
                        scheduleEnvironmentBlockGoalResync(minecraft);
                        return;
                    }
                    abort("environment reset state did not reach the client: "
                            + describeUnsettledResetState(minecraft), true);
                    return;
                }
                // Client-only ticks are normally intentionally unpaced. During the
                // exceptional recovery path, briefly park the client thread so the
                // local Netty thread can deliver the explicit block-change packets.
                if (resetClientResyncAttempts > 0) {
                    LockSupport.parkNanos(RESET_CLIENT_RECOVERY_PARK_NANOS);
                }
                resetClientSettleTicks++;
                controller.requestSynchronizedClientTick();
                markProgress();
                return;
            }
            controller.setSuppressSynchronizedAttackHandling(false);
            lockNearestTaskEntityIfRequired(minecraft);
            updateTargetingIfRequired(minecraft);
            if (!bridge.sendRawJson(createEnvironmentObservation(
                    minecraft, "env_reset_result", 0,
                    Boolean.valueOf(environmentResetResult.applied)))) {
                abort("TCP outbound queue is full while resetting environment", false);
                return;
            }
            pendingEnvironmentReset = null;
            environmentResetResult = null;
            environmentResetError = null;
            phase = Phase.IDLE;
            markProgress();
        } else if (phase == Phase.ACTION_CLIENT_TICK) {
            // Movement is one server-tick action. Attack/use are held controls and
            // remain pressed across the integration tick and network wait.
            releaseTransientKeys(Minecraft.getMinecraft().gameSettings);
            actionClientCompletedNanos = System.nanoTime();
            awaitedServerTick = controller.getCompletedServerTicks() + 1L;
            serverRequestedNanos = System.nanoTime();
            controller.requestSynchronizedTick();
            phase = Phase.WAIT_SERVER;
            markProgress();
        } else if (phase == Phase.POST_SERVER_CLIENT_TICK) {
            controller.setSuppressSynchronizedAttackHandling(false);
            if (attackHeld) {
                setKey(Minecraft.getMinecraft().gameSettings.keyBindAttack, true);
            }
            Minecraft minecraft = Minecraft.getMinecraft();
            updateTargetingIfRequired(minecraft);
            String observation = sessionKind == SessionKind.BENCHMARK
                    ? createBenchmarkObservation(minecraft, expectedSequence)
                    : createEnvironmentObservation(minecraft, "env_step_result",
                            expectedSequence, null);
            if (!bridge.sendRawJson(observation)) {
                abort("TCP outbound queue is full while sending observation", true);
                return;
            }
            expectedSequence++;
            phase = Phase.IDLE;
            markProgress();
        }
    }

    public void abort(String reason, boolean notifyPython) {
        if (!active) return;
        SessionKind abortedKind = sessionKind;
        active = false;
        sessionKind = SessionKind.NONE;
        environmentActionFields = Collections.emptySet();
        environmentObservationFields = Collections.emptyList();
        environmentObservationPlan = ObservationPlan.EMPTY;
        environmentTaskEntityId = -1;
        clearPendingEnvironmentReset();
        clearEnvironmentGoal();
        phase = Phase.IDLE;
        releaseKeys();
        controller.setMode(TickController.Mode.VANILLA);
        if (notifyPython) {
            String messageType = abortedKind == SessionKind.BENCHMARK
                    ? "sync_benchmark_abort" : "env_error";
            bridge.sendRawJson("{\"type\":" + WireMessage.quote(messageType) + ",\"text\":"
                    + WireMessage.quote(reason) + "}");
        }
        ClientEvents.showMessage("[RL] Synchronized session aborted: " + reason);
    }

    private void acceptEnvironmentReset(WireMessage reset) {
        if (active && sessionKind != SessionKind.ENVIRONMENT) {
            sendEnvironmentError("cannot reset environment during synchronized benchmark");
            return;
        }
        if (active && phase != Phase.IDLE) {
            abort("environment reset received while a step is pending", true);
            return;
        }
        Minecraft minecraft = Minecraft.getMinecraft();
        if (!bridge.isConnected()) {
            sendEnvironmentError("Python bridge is not connected");
            return;
        }
        if (!minecraft.isIntegratedServerRunning()
                || minecraft.thePlayer == null || minecraft.theWorld == null) {
            sendEnvironmentError("open a single-player world before resetting the environment");
            return;
        }
        if (reset.task == null || reset.task.trim().isEmpty()) {
            sendEnvironmentError("environment reset requires a task id");
            return;
        }

        final Set<String> requestedActionFields;
        final List<String> requestedObservationFields;
        try {
            requestedActionFields = new HashSet<>(parseRequestedFields(
                    reset.actionFields, SUPPORTED_ACTION_FIELDS, "action"));
            requestedObservationFields = parseRequestedFields(
                    reset.observationFields, SUPPORTED_OBSERVATION_FIELDS, "observation");
        } catch (IllegalArgumentException invalidSchema) {
            sendEnvironmentError(invalidSchema.getMessage());
            return;
        }
        if (hasPrefix(requestedObservationFields, "task_goal.")
                && (!Double.isFinite(reset.goalX) || !Double.isFinite(reset.goalY)
                || !Double.isFinite(reset.goalZ))
                && !resetManager.providesGoal(reset.task)) {
            sendEnvironmentError("task_goal observations require finite goal_x, goal_y and goal_z");
            return;
        }

        releaseKeys();
        active = true;
        sessionKind = SessionKind.ENVIRONMENT;
        environmentActionFields = requestedActionFields;
        environmentObservationFields = requestedObservationFields;
        environmentObservationPlan = ObservationPlan.compile(requestedObservationFields);
        environmentProfileJavaPhases = "profile_java_phases".equals(reset.payload);
        environmentTaskEntityId = -1;
        environmentGoalX = reset.goalX;
        environmentGoalY = reset.goalY;
        environmentGoalZ = reset.goalZ;
        expectedSteps = Integer.MAX_VALUE;
        expectedSequence = 0;
        phase = Phase.ARMING;
        startedNanos = System.nanoTime();
        lastProgressNanos = startedNanos;
        connectionEpoch = bridge.getConnectionEpoch();
        pendingEnvironmentReset = reset;
        environmentResetResult = null;
        environmentResetError = null;
        bedDefenseSnapshotCache = null;
        bedDefenseV2SnapshotCache = null;
        bridgingSnapshotCache = null;
        resetClientResyncAttempts = 0;
        controller.setMode(TickController.Mode.SYNC);
        controller.setSynchronizedRenderingEnabled(reset.visualized);
    }

    private void closeEnvironment() {
        if (!active || sessionKind != SessionKind.ENVIRONMENT) {
            sendEnvironmentError("no active environment to close");
            return;
        }
        active = false;
        sessionKind = SessionKind.NONE;
        environmentActionFields = Collections.emptySet();
        environmentObservationFields = Collections.emptyList();
        environmentObservationPlan = ObservationPlan.EMPTY;
        environmentProfileJavaPhases = false;
        environmentTaskEntityId = -1;
        clearPendingEnvironmentReset();
        clearEnvironmentGoal();
        phase = Phase.IDLE;
        releaseKeys();
        controller.setMode(TickController.Mode.VANILLA);
        bridge.sendRawJson("{\"type\":\"env_closed\"}");
        ClientEvents.showMessage("[RL] Environment session closed; vanilla ticking restored");
    }

    private void sendEnvironmentError(String reason) {
        bridge.sendRawJson("{\"type\":\"env_error\",\"text\":"
                + WireMessage.quote(reason) + "}");
    }

    private void acceptAction(WireMessage action) {
        if (!active || phase != Phase.IDLE) {
            abort("action received while another step is pending", true);
            return;
        }
        if (action.sequence != expectedSequence || expectedSequence >= expectedSteps) {
            abort("unexpected action sequence " + action.sequence + ", expected " + expectedSequence, true);
            return;
        }
        if (!Double.isFinite(action.mouseDx) || !Double.isFinite(action.mouseDy)
                || Math.abs(action.mouseDx) > MAX_MOUSE_DELTA
                || Math.abs(action.mouseDy) > MAX_MOUSE_DELTA) {
            abort("invalid mouse delta for action " + action.sequence, true);
            return;
        }
        if (sessionKind == SessionKind.ENVIRONMENT
                && environmentActionFields.contains("hotbar_slot")
                && (action.hotbarSlot < 0 || action.hotbarSlot > 8)) {
            abort("hotbar_slot must be between 0 and 8", true);
            return;
        }

        Minecraft minecraft = Minecraft.getMinecraft();
        if (minecraft.thePlayer == null || minecraft.theWorld == null) {
            abort("no active client world", true);
            return;
        }

        actionAcceptedNanos = System.nanoTime();
        // Entity.setAngles is the same vanilla path used after mouse sensitivity is applied.
        applyAction(minecraft, action);
        updateTargetingIfRequired(minecraft);
        phase = Phase.ACTION_CLIENT_TICK;
        controller.requestSynchronizedClientTick();
        markProgress();
    }

    private void acceptCompletion(WireMessage completion) {
        if (!active) return;
        if (expectedSequence != expectedSteps || phase != Phase.IDLE) {
            abort("Python completed before all observations arrived", false);
            return;
        }

        double javaSeconds = Math.max(0.000001D, (System.nanoTime() - startedNanos) / 1_000_000_000.0D);
        double pythonSeconds = Math.max(0.000001D, completion.elapsedNanos / 1_000_000_000.0D);
        double averageObservationBytes = completion.observationBytes / (double) expectedSteps;
        active = false;
        sessionKind = SessionKind.NONE;
        environmentActionFields = Collections.emptySet();
        environmentObservationFields = Collections.emptyList();
        environmentObservationPlan = ObservationPlan.EMPTY;
        environmentTaskEntityId = -1;
        clearEnvironmentGoal();
        releaseKeys();
        controller.setMode(TickController.Mode.VANILLA);
        ClientEvents.showMessage(String.format(Locale.ROOT,
                "[RL] synchronized %d steps: Java wall=%.1f step/s, Python wall=%.1f step/s, avg observation=%.0f bytes",
                expectedSteps, expectedSteps / javaSeconds, expectedSteps / pythonSeconds, averageObservationBytes));
    }

    private void applyAction(Minecraft minecraft, WireMessage action) {
        boolean benchmark = sessionKind == SessionKind.BENCHMARK;
        boolean yawEnabled = benchmark || environmentActionFields.contains("mouse_dx");
        boolean pitchEnabled = benchmark || environmentActionFields.contains("mouse_dy");
        if (yawEnabled || pitchEnabled) {
            minecraft.thePlayer.setAngles(
                    yawEnabled ? (float) action.mouseDx : 0.0F,
                    pitchEnabled ? (float) action.mouseDy : 0.0F);
        }

        GameSettings settings = minecraft.gameSettings;
        if (environmentActionFields.contains("hotbar_slot")) {
            minecraft.thePlayer.inventory.currentItem = action.hotbarSlot;
        }
        setKey(settings.keyBindForward, action.forward
                && (benchmark || environmentActionFields.contains("forward")));
        setKey(settings.keyBindBack, action.back
                && (benchmark || environmentActionFields.contains("back")));
        setKey(settings.keyBindLeft, action.left
                && (benchmark || environmentActionFields.contains("left")));
        setKey(settings.keyBindRight, action.right
                && (benchmark || environmentActionFields.contains("right")));
        setKey(settings.keyBindJump, action.jump
                && (benchmark || environmentActionFields.contains("jump")));
        setKey(settings.keyBindSneak, action.sneak
                && (benchmark || environmentActionFields.contains("sneak")));
        setKey(settings.keyBindSprint, action.sprint
                && (benchmark || environmentActionFields.contains("sprint")));
        boolean attack = action.attack
                && (benchmark || environmentActionFields.contains("attack"));
        setKey(settings.keyBindAttack, attack);
        if (attack && !attackHeld) KeyBinding.onTick(settings.keyBindAttack.getKeyCode());
        if (action.attackOnce && environmentActionFields.contains("attack_once")) {
            KeyBinding.onTick(settings.keyBindAttack.getKeyCode());
        }
        attackHeld = attack;

        boolean use = action.use
                && (benchmark || environmentActionFields.contains("use"));
        setKey(settings.keyBindUseItem, use);
        if (use && !useHeld) KeyBinding.onTick(settings.keyBindUseItem.getKeyCode());
        if (action.useOnce && environmentActionFields.contains("use_once")) {
            KeyBinding.onTick(settings.keyBindUseItem.getKeyCode());
        }
        useHeld = use;
    }

    private static void setKey(KeyBinding key, boolean pressed) {
        KeyBinding.setKeyBindState(key.getKeyCode(), pressed);
    }

    private static void releaseTransientKeys(GameSettings settings) {
        setKey(settings.keyBindForward, false);
        setKey(settings.keyBindBack, false);
        setKey(settings.keyBindLeft, false);
        setKey(settings.keyBindRight, false);
        setKey(settings.keyBindJump, false);
        setKey(settings.keyBindSneak, false);
        setKey(settings.keyBindSprint, false);
    }

    private void releaseKeys() {
        KeyBinding.unPressAllKeys();
        attackHeld = false;
        useHeld = false;
        Minecraft minecraft = Minecraft.getMinecraft();
        if (minecraft.playerController != null) {
            minecraft.playerController.resetBlockRemoving();
            if (minecraft.thePlayer != null && minecraft.thePlayer.isUsingItem()) {
                minecraft.playerController.onStoppedUsingItem(minecraft.thePlayer);
            }
        }
    }

    private void markProgress() {
        lastProgressNanos = System.nanoTime();
    }

    private void updateTargetingIfRequired(Minecraft minecraft) {
        if (sessionKind == SessionKind.BENCHMARK || requiresRaycast()) {
            minecraft.entityRenderer.getMouseOver(1.0F);
        }
    }

    private boolean requiresRaycast() {
        if (environmentActionFields.contains("attack") || environmentActionFields.contains("use")
                || environmentActionFields.contains("attack_once")
                || environmentActionFields.contains("use_once")) {
            return true;
        }
        return environmentObservationPlan.requiresRaycast;
    }

    static final class ObservationPlan {
        static final ObservationPlan EMPTY = compile(Collections.<String>emptyList());

        final List<String> fields;
        final String[] quotedKeys;
        final boolean requiresRaycast;
        final boolean requiresNearestLiving;
        final boolean requiresTaskEntity;
        final boolean requiresTaskGoal;
        final boolean requiresTaskBlocks;
        final boolean requiresBedDefense;
        final boolean requiresBedDefenseV2;
        final boolean requiresBridging;
        final int estimatedJsonCapacity;

        private ObservationPlan(List<String> fields, String[] quotedKeys,
                                boolean requiresRaycast,
                                boolean requiresNearestLiving,
                                boolean requiresTaskEntity,
                                boolean requiresTaskGoal,
                                boolean requiresTaskBlocks,
                                boolean requiresBedDefense,
                                boolean requiresBedDefenseV2,
                                boolean requiresBridging,
                                int estimatedJsonCapacity) {
            this.fields = fields;
            this.quotedKeys = quotedKeys;
            this.requiresRaycast = requiresRaycast;
            this.requiresNearestLiving = requiresNearestLiving;
            this.requiresTaskEntity = requiresTaskEntity;
            this.requiresTaskGoal = requiresTaskGoal;
            this.requiresTaskBlocks = requiresTaskBlocks;
            this.requiresBedDefense = requiresBedDefense;
            this.requiresBedDefenseV2 = requiresBedDefenseV2;
            this.requiresBridging = requiresBridging;
            this.estimatedJsonCapacity = estimatedJsonCapacity;
        }

        static ObservationPlan compile(List<String> requestedFields) {
            List<String> fields = Collections.unmodifiableList(
                    new ArrayList<>(requestedFields));
            String[] quotedKeys = new String[fields.size()];
            boolean raycast = false;
            boolean nearestLiving = false;
            boolean taskEntity = false;
            boolean taskGoal = false;
            boolean taskBlocks = false;
            boolean bedDefense = false;
            boolean bedDefenseV2 = false;
            boolean bridging = false;
            int capacity = 96;
            for (int index = 0; index < fields.size(); index++) {
                String field = fields.get(index);
                quotedKeys[index] = WireMessage.quote(field);
                capacity += quotedKeys[index].length() + 20;
                raycast |= field.startsWith("raycast.");
                nearestLiving |= field.startsWith("nearest_living.");
                taskEntity |= field.startsWith("task_entity.");
                taskGoal |= field.startsWith("task_goal.");
                taskBlocks |= field.startsWith("task_block_")
                        || "task_blocks.count".equals(field);
                bedDefense |= field.startsWith("bed_defense.");
                bedDefenseV2 |= field.startsWith("bed_defense_v2.");
                bridging |= field.startsWith("bridging.");
            }
            if (bedDefense) capacity += 3_000;
            if (bedDefenseV2) capacity += 3_000;
            if (bridging) capacity += 5_000;
            return new ObservationPlan(fields, quotedKeys, raycast, nearestLiving,
                    taskEntity, taskGoal, taskBlocks, bedDefense, bedDefenseV2,
                    bridging, capacity);
        }
    }

    private static List<String> parseRequestedFields(
            String encoded, Set<String> supported, String fieldKind) {
        List<String> result = new ArrayList<>();
        if (encoded == null || encoded.trim().isEmpty()) return result;
        Set<String> seen = new HashSet<>();
        for (String part : encoded.split(",", -1)) {
            String field = part.trim();
            if (field.isEmpty()) {
                throw new IllegalArgumentException("empty " + fieldKind + " field in environment schema");
            }
            if (!supported.contains(field)) {
                throw new IllegalArgumentException("unsupported " + fieldKind + " field: " + field);
            }
            if (!seen.add(field)) {
                throw new IllegalArgumentException("duplicate " + fieldKind + " field: " + field);
            }
            result.add(field);
        }
        return result;
    }

    private String createEnvironmentObservation(Minecraft minecraft, String messageType,
                                                 int sequence, Boolean worldResetApplied) {
        long observationStartedNanos = System.nanoTime();
        ObservationPlan plan = environmentObservationPlan;
        List<String> fields = plan.fields;
        StringBuilder json = new StringBuilder(plan.estimatedJsonCapacity);
        json.append("{\"type\":").append(WireMessage.quote(messageType))
                .append(",\"sequence\":").append(sequence);
        RaycastSnapshot raycast = plan.requiresRaycast
                ? RaycastSnapshot.capture(minecraft, fields) : null;
        LivingSnapshot nearestLiving = plan.requiresNearestLiving
                ? LivingSnapshot.captureNearest(minecraft) : null;
        LivingSnapshot taskEntity = plan.requiresTaskEntity
                ? LivingSnapshot.captureById(minecraft, environmentTaskEntityId) : null;
        GoalSnapshot taskGoal = plan.requiresTaskGoal
                ? GoalSnapshot.capture(minecraft, environmentGoalX, environmentGoalY,
                        environmentGoalZ) : null;
        GoalSnapshot[] taskBlocks = plan.requiresTaskBlocks
                ? captureTaskBlocks(minecraft) : null;
        BedDefenseSnapshot bedDefense = null;
        if (plan.requiresBedDefense) {
            bedDefense = BedDefenseSnapshot.capture(
                    minecraft, environmentGoalX, environmentGoalY,
                    environmentGoalZ, bedDefenseSnapshotCache);
            bedDefenseSnapshotCache = bedDefense;
        }
        BedDefenseV2Snapshot bedDefenseV2 = null;
        if (plan.requiresBedDefenseV2) {
            bedDefenseV2 = BedDefenseV2Snapshot.capture(
                    minecraft, environmentBlockGoalXs, environmentBlockGoalYs,
                    environmentBlockGoalZs, bedDefenseV2SnapshotCache);
            bedDefenseV2SnapshotCache = bedDefenseV2;
        }
        BridgingSnapshot bridging = null;
        if (plan.requiresBridging) {
            bridging = BridgingSnapshot.capture(
                    minecraft, environmentGoalX, environmentGoalY,
                    environmentGoalZ, bridgingSnapshotCache);
            bridgingSnapshotCache = bridging;
        }
        ItemStack held = minecraft.thePlayer.getHeldItem();

        for (int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
            String field = fields.get(fieldIndex);
            json.append(',').append(plan.quotedKeys[fieldIndex]).append(':');
            if ("task_blocks.count".equals(field)) {
                json.append(environmentBlockGoalXs.length);
                continue;
            }
            if (field.startsWith("task_block_")) {
                appendTaskBlockField(json, field, taskBlocks);
                continue;
            }
            switch (field) {
                case "player.x": json.append(minecraft.thePlayer.posX); break;
                case "player.y": json.append(minecraft.thePlayer.posY); break;
                case "player.z": json.append(minecraft.thePlayer.posZ); break;
                case "player.yaw": json.append(minecraft.thePlayer.rotationYaw); break;
                case "player.pitch": json.append(minecraft.thePlayer.rotationPitch); break;
                case "player.health": json.append(minecraft.thePlayer.getHealth()); break;
                case "player.on_ground": json.append(minecraft.thePlayer.onGround); break;
                case "player.motion_x": json.append(minecraft.thePlayer.motionX); break;
                case "player.motion_y": json.append(minecraft.thePlayer.motionY); break;
                case "player.motion_z": json.append(minecraft.thePlayer.motionZ); break;
                case "player.sneaking": json.append(minecraft.thePlayer.isSneaking()); break;
                case "player.sprinting": json.append(minecraft.thePlayer.isSprinting()); break;
                case "player.using_item": json.append(minecraft.thePlayer.isUsingItem()); break;
                case "player.item_use_ticks": json.append(minecraft.thePlayer.getItemInUseDuration()); break;
                case "player.selected_slot": json.append(minecraft.thePlayer.inventory.currentItem); break;
                case "player.collided_horizontally": json.append(minecraft.thePlayer.isCollidedHorizontally); break;
                case "player.collided_vertically": json.append(minecraft.thePlayer.isCollidedVertically); break;
                case "player.fall_distance": json.append(minecraft.thePlayer.fallDistance); break;
                case "player.block_break_progress":
                    json.append(((AccessorPlayerControllerMP) minecraft.playerController)
                            .minecraftRl$getBlockBreakProgress());
                    break;
                case "held.item_id": json.append(held == null ? -1 : Item.getIdFromItem(held.getItem())); break;
                case "held.item_name": json.append(WireMessage.quote(itemName(held))); break;
                case "held.count": json.append(held == null ? 0 : held.stackSize); break;
                case "held.damage": json.append(held == null ? 0 : held.getItemDamage()); break;
                case "held.max_damage": json.append(held == null ? 0 : held.getMaxDamage()); break;
                case "inventory.arrow_count": json.append(countArrows(minecraft)); break;
                case "inventory.hotbar": appendHotbar(json, minecraft); break;
                case "raycast.type": json.append(WireMessage.quote(raycast.type)); break;
                case "raycast.name": json.append(WireMessage.quote(raycast.name)); break;
                case "raycast.id": json.append(raycast.id); break;
                case "raycast.x": json.append(raycast.x); break;
                case "raycast.y": json.append(raycast.y); break;
                case "raycast.z": json.append(raycast.z); break;
                case "raycast.side": json.append(WireMessage.quote(raycast.side)); break;
                case "raycast.distance": json.append(raycast.distance); break;
                case "raycast.health": json.append(raycast.health); break;
                case "raycast.block_hardness": json.append(raycast.blockHardness); break;
                case "raycast.can_harvest": json.append(raycast.canHarvest); break;
                case "raycast.category": json.append(raycast.category); break;
                case "raycast.safe_yaw_correction": json.append(raycast.safeYawCorrection); break;
                case "raycast.safe_pitch_correction": json.append(raycast.safePitchCorrection); break;
                case "nearest_living.type": json.append(WireMessage.quote(nearestLiving.type)); break;
                case "nearest_living.id": json.append(nearestLiving.id); break;
                case "nearest_living.x": json.append(nearestLiving.x); break;
                case "nearest_living.y": json.append(nearestLiving.y); break;
                case "nearest_living.z": json.append(nearestLiving.z); break;
                case "nearest_living.relative_x": json.append(nearestLiving.relativeX); break;
                case "nearest_living.relative_y": json.append(nearestLiving.relativeY); break;
                case "nearest_living.relative_z": json.append(nearestLiving.relativeZ); break;
                case "nearest_living.distance": json.append(nearestLiving.distance); break;
                case "nearest_living.yaw_delta": json.append(nearestLiving.yawDelta); break;
                case "nearest_living.pitch_delta": json.append(nearestLiving.pitchDelta); break;
                case "nearest_living.motion_x": json.append(nearestLiving.motionX); break;
                case "nearest_living.motion_y": json.append(nearestLiving.motionY); break;
                case "nearest_living.motion_z": json.append(nearestLiving.motionZ); break;
                case "nearest_living.health": json.append(nearestLiving.health); break;
                case "nearest_living.alive": json.append(nearestLiving.alive); break;
                case "task_entity.type": json.append(WireMessage.quote(taskEntity.type)); break;
                case "task_entity.id": json.append(taskEntity.id); break;
                case "task_entity.x": json.append(taskEntity.x); break;
                case "task_entity.y": json.append(taskEntity.y); break;
                case "task_entity.z": json.append(taskEntity.z); break;
                case "task_entity.relative_x": json.append(taskEntity.relativeX); break;
                case "task_entity.relative_y": json.append(taskEntity.relativeY); break;
                case "task_entity.relative_z": json.append(taskEntity.relativeZ); break;
                case "task_entity.distance": json.append(taskEntity.distance); break;
                case "task_entity.yaw_delta": json.append(taskEntity.yawDelta); break;
                case "task_entity.pitch_delta": json.append(taskEntity.pitchDelta); break;
                case "task_entity.motion_x": json.append(taskEntity.motionX); break;
                case "task_entity.motion_y": json.append(taskEntity.motionY); break;
                case "task_entity.motion_z": json.append(taskEntity.motionZ); break;
                case "task_entity.health": json.append(taskEntity.health); break;
                case "task_entity.alive": json.append(taskEntity.alive); break;
                case "task_goal.x": json.append(taskGoal.x); break;
                case "task_goal.y": json.append(taskGoal.y); break;
                case "task_goal.z": json.append(taskGoal.z); break;
                case "task_goal.relative_x": json.append(taskGoal.relativeX); break;
                case "task_goal.relative_y": json.append(taskGoal.relativeY); break;
                case "task_goal.relative_z": json.append(taskGoal.relativeZ); break;
                case "task_goal.distance": json.append(taskGoal.distance); break;
                case "task_goal.horizontal_distance": json.append(taskGoal.horizontalDistance); break;
                case "task_goal.yaw_delta": json.append(taskGoal.yawDelta); break;
                case "task_goal.pitch_delta": json.append(taskGoal.pitchDelta); break;
                case "task_goal.block_id": json.append(taskGoal.blockId); break;
                case "task_goal.tool_speed_0": json.append(taskGoal.toolSpeed0); break;
                case "task_goal.tool_speed_1": json.append(taskGoal.toolSpeed1); break;
                case "task_goal.tool_speed_2": json.append(taskGoal.toolSpeed2); break;
                case "local.blocks_3x3x3": appendLocalBlocks(json, minecraft); break;
                case "bed_defense.origin_x": json.append(bedDefense.originX); break;
                case "bed_defense.origin_y": json.append(bedDefense.originY); break;
                case "bed_defense.origin_z": json.append(bedDefense.originZ); break;
                case "bed_defense.block_grid": bedDefense.appendGrid(json); break;
                case "bed_defense.visible_mask": bedDefense.appendVisibleMask(json); break;
                case "bed_defense_v2.block_grid": bedDefenseV2.appendGrid(json); break;
                case "bed_defense_v2.visible_mask": bedDefenseV2.appendVisibleMask(json); break;
                case "bridging.ground_fraction": json.append(bridging.groundFraction); break;
                case "bridging.distance_forward_edge": json.append(bridging.forwardEdge); break;
                case "bridging.distance_left_edge": json.append(bridging.leftEdge); break;
                case "bridging.distance_right_edge": json.append(bridging.rightEdge); break;
                case "bridging.distance_rear_edge": json.append(bridging.rearEdge); break;
                case "bridging.can_place": json.append(bridging.canPlace); break;
                case "bridging.block_grid": bridging.appendGrid(json); break;
                case "bridging.above_level_count": json.append(bridging.aboveLevelCount); break;
                default: throw new IllegalStateException("Validated observation field disappeared: " + field);
            }
        }
        if (worldResetApplied != null) {
            json.append(",\"world_reset_applied\":").append(worldResetApplied.booleanValue());
        }
        if (environmentProfileJavaPhases && worldResetApplied == null) {
            long observationSerializedNanos = System.nanoTime();
            json.append(",\"java_profile.action_client_ms\":")
                    .append((actionClientCompletedNanos - actionAcceptedNanos) / 1_000_000.0D)
                    .append(",\"java_profile.server_tick_ms\":")
                    .append((serverCompletedNanos - serverRequestedNanos) / 1_000_000.0D)
                    .append(",\"java_profile.post_client_ms\":")
                    .append((observationStartedNanos - postClientStartedNanos) / 1_000_000.0D)
                    .append(",\"java_profile.observation_serialize_ms\":")
                    .append((observationSerializedNanos - observationStartedNanos) / 1_000_000.0D)
                    .append(",\"java_profile.java_total_ms\":")
                    .append((observationSerializedNanos - actionAcceptedNanos) / 1_000_000.0D);
        }
        return json.append('}').toString();
    }

    private void scheduleEnvironmentReset(Minecraft minecraft) {
        final IntegratedServer server = minecraft.getIntegratedServer();
        final WireMessage reset = pendingEnvironmentReset;
        final String username = minecraft.getSession().getUsername();
        final UUID playerUuid = minecraft.thePlayer.getUniqueID();
        environmentResetResult = null;
        environmentResetError = null;
        server.addScheduledTask(new Runnable() {
            @Override
            public void run() {
                try {
                    environmentResetResult = resetManager.apply(
                            server, playerUuid, username, reset.task, reset.seed,
                            reset.targetCount, reset.curriculumStage);
                } catch (Throwable failure) {
                    String message = failure.getMessage();
                    environmentResetError = failure.getClass().getSimpleName()
                            + (message == null ? "" : ": " + message);
                }
            }
        });
        awaitedServerTick = controller.getCompletedServerTicks() + 1L;
        controller.requestSynchronizedTick();
        phase = Phase.RESET_WAIT_SERVER;
        markProgress();
    }

    /**
     * Recovery for a rare integrated-server packet race after a multi-block reset.
     * The normal reset path never calls this method. The authoritative server state
     * is validated before targeted block-change packets are resent to this player.
     */
    private void scheduleEnvironmentBlockGoalResync(Minecraft minecraft) {
        final IntegratedServer server = minecraft.getIntegratedServer();
        final UUID playerUuid = minecraft.thePlayer.getUniqueID();
        final String username = minecraft.getSession().getUsername();
        final double[] goalXs = environmentSyncXs.clone();
        final double[] goalYs = environmentSyncYs.clone();
        final double[] goalZs = environmentSyncZs.clone();
        final int[] goalIds = environmentSyncIds.clone();
        final boolean resyncBridgingInventory = pendingEnvironmentReset != null
                && EnvironmentResetManager.BRIDGING_TASK.equals(
                pendingEnvironmentReset.task);
        environmentResetError = null;
        server.addScheduledTask(new Runnable() {
            @Override
            public void run() {
                try {
                    EntityPlayerMP player = server.getConfigurationManager()
                            .getPlayerByUUID(playerUuid);
                    if (player == null) {
                        player = server.getConfigurationManager().getPlayerByUsername(username);
                    }
                    if (player == null) {
                        throw new IllegalStateException("integrated-server player disappeared");
                    }
                    for (int index = 0; index < goalIds.length; index++) {
                        BlockPos pos = new BlockPos(goalXs[index], goalYs[index], goalZs[index]);
                        int serverId = Block.getStateId(player.getServerForPlayer()
                                .getBlockState(pos));
                        if (serverId != goalIds[index]) {
                            throw new IllegalStateException("server goal #" + index + " at "
                                    + pos + " expected=" + goalIds[index]
                                    + " actual=" + serverId);
                        }
                        player.playerNetServerHandler.sendPacket(
                                new S23PacketBlockChange(player.getServerForPlayer(), pos));
                    }
                    if (resyncBridgingInventory) {
                        // Resend the authoritative teleport as well. Block and
                        // inventory packets can arrive before the position packet,
                        // leaving the client briefly at its pre-reset void fall.
                        player.playerNetServerHandler.setPlayerLocation(
                                player.posX, player.posY, player.posZ,
                                player.rotationYaw, player.rotationPitch);
                        player.getServerForPlayer().getMinecraftServer()
                                .getConfigurationManager().syncPlayerInventory(player);
                    }
                } catch (Throwable failure) {
                    String message = failure.getMessage();
                    environmentResetError = failure.getClass().getSimpleName()
                            + (message == null ? "" : ": " + message);
                }
            }
        });
        awaitedServerTick = controller.getCompletedServerTicks() + 1L;
        controller.requestSynchronizedTick();
        phase = Phase.RESET_RESYNC_WAIT_SERVER;
        markProgress();
    }

    private void lockNearestTaskEntityIfRequired(Minecraft minecraft) {
        if (environmentTaskEntityId < 0
                && environmentObservationPlan.requiresTaskEntity) {
            environmentTaskEntityId = LivingSnapshot.findNearestLivingId(minecraft);
        }
    }

    private static void stabilizeClientPlayerAfterReset(Minecraft minecraft) {
        if (minecraft.thePlayer == null) return;
        minecraft.thePlayer.motionX = 0.0D;
        minecraft.thePlayer.motionY = 0.0D;
        minecraft.thePlayer.motionZ = 0.0D;
        minecraft.thePlayer.fallDistance = 0.0F;
        minecraft.thePlayer.prevPosX = minecraft.thePlayer.lastTickPosX = minecraft.thePlayer.posX;
        minecraft.thePlayer.prevPosY = minecraft.thePlayer.lastTickPosY = minecraft.thePlayer.posY;
        minecraft.thePlayer.prevPosZ = minecraft.thePlayer.lastTickPosZ = minecraft.thePlayer.posZ;
        minecraft.thePlayer.prevRotationYaw = minecraft.thePlayer.rotationYaw;
        minecraft.thePlayer.prevRotationPitch = minecraft.thePlayer.rotationPitch;
        if (minecraft.playerController != null) {
            minecraft.playerController.resetBlockRemoving();
            // Vanilla sets this to 5 after destroying a block; resetBlockRemoving()
            // does not clear it, which otherwise delays every later episode.
            ((AccessorPlayerControllerMP) minecraft.playerController)
                    .minecraftRl$setBlockHitDelay(0);
        }
    }

    private static boolean hasPrefix(List<String> fields, String prefix) {
        for (String field : fields) {
            if (field.startsWith(prefix)) return true;
        }
        return false;
    }

    private void clearEnvironmentGoal() {
        environmentGoalX = Double.NaN;
        environmentGoalY = Double.NaN;
        environmentGoalZ = Double.NaN;
        environmentBlockGoalXs = new double[0];
        environmentBlockGoalYs = new double[0];
        environmentBlockGoalZs = new double[0];
        environmentBlockGoalIds = new int[0];
        environmentSyncXs = new double[0];
        environmentSyncYs = new double[0];
        environmentSyncZs = new double[0];
        environmentSyncIds = new int[0];
        bedDefenseSnapshotCache = null;
        bedDefenseV2SnapshotCache = null;
        bridgingSnapshotCache = null;
        resetClientSettleTicks = 0;
        resetClientResyncAttempts = 0;
    }

    private boolean clientResetStateReady(Minecraft minecraft) {
        if (minecraft.thePlayer == null
                || minecraft.thePlayer.isDead
                || !minecraft.thePlayer.isEntityAlive()
                || minecraft.thePlayer.getHealth() <= 0.0F) {
            return false;
        }
        for (int index = 0; index < environmentSyncIds.length; index++) {
            int actual = Block.getStateId(minecraft.theWorld.getBlockState(new BlockPos(
                    environmentSyncXs[index], environmentSyncYs[index],
                    environmentSyncZs[index])));
            if (actual != environmentSyncIds[index]) return false;
        }
        if (environmentObservationPlan.requiresTaskEntity
                && environmentTaskEntityId >= 0) {
            LivingSnapshot taskEntity = LivingSnapshot.captureById(
                    minecraft, environmentTaskEntityId);
            if (taskEntity.id != environmentTaskEntityId || !taskEntity.alive) {
                return false;
            }
        }
        if (pendingEnvironmentReset != null
                && EnvironmentResetManager.BRIDGING_TASK.equals(
                pendingEnvironmentReset.task)) {
            return bridgingInventoryReady(
                    minecraft.thePlayer.inventory.currentItem,
                    minecraft.thePlayer.getHeldItem())
                    && bridgingPlayerReady(minecraft, pendingEnvironmentReset)
                    && bridgingAimedTargetReady(minecraft, pendingEnvironmentReset)
                    && BridgingSnapshot.countAboveLevel(
                    minecraft,
                    MathHelper.floor_double(environmentGoalX),
                    MathHelper.floor_double(environmentGoalY),
                    MathHelper.floor_double(environmentGoalZ)) == 0;
        }
        return true;
    }

    private boolean bridgingAimedTargetReady(Minecraft minecraft,
                                             WireMessage reset) {
        if (reset.curriculumStage != 0 && reset.curriculumStage != 2) {
            return true;
        }
        MovingObjectPosition hit = minecraft.objectMouseOver;
        if (hit == null
                || hit.typeOfHit != MovingObjectPosition.MovingObjectType.BLOCK
                || hit.sideHit != EnumFacing.SOUTH
                || !BridgingSnapshot.canPlaceHeldBlock(minecraft)) {
            return false;
        }
        int gap = EnvironmentResetManager.bridgingGap(
                reset.curriculumStage, reset.seed);
        BlockPos expectedStart = new BlockPos(
                MathHelper.floor_double(environmentGoalX),
                MathHelper.floor_double(environmentGoalY),
                MathHelper.floor_double(environmentGoalZ) - gap - 1);
        return expectedStart.equals(hit.getBlockPos());
    }

    static boolean bridgingInventoryReady(int currentSlot, ItemStack held) {
        return bridgingInventoryReady(
                currentSlot,
                held == null ? 0 : held.stackSize,
                held != null && Block.getBlockFromItem(held.getItem()) == Blocks.wool);
    }

    static boolean bridgingInventoryReady(int currentSlot, int stackSize,
                                           boolean wool) {
        return currentSlot == 0 && wool && stackSize == 64;
    }

    private boolean bridgingPlayerReady(Minecraft minecraft,
                                        WireMessage reset) {
        double[] expected = bridgingExpectedStart(
                environmentGoalX, environmentGoalY, environmentGoalZ,
                reset.curriculumStage, reset.seed);
        return Math.abs(minecraft.thePlayer.posX - expected[0]) <= 0.02D
                && Math.abs(minecraft.thePlayer.posY - expected[1]) <= 0.02D
                && Math.abs(minecraft.thePlayer.posZ - expected[2]) <= 0.02D;
    }

    static double[] bridgingExpectedStart(double goalX, double goalY,
                                          double goalZ, int stage, long seed) {
        int gap = EnvironmentResetManager.bridgingGap(stage, seed);
        // The bridge points south (+Z). Goal and start share X; the start is
        // gap+1 blocks north (-Z) of the target.
        double startZ = Math.floor(goalZ) - gap - 1.0D;
        double expectedX;
        double expectedZ;
        if (stage == 0 || stage == 2) {
            expectedX = Math.floor(goalX) + 0.5D;
            expectedZ = startZ
                    + EnvironmentResetManager.BRIDGING_FIXED_LOCAL_Z;
        } else {
            java.util.Random random = new java.util.Random(seed);
            expectedX = Math.floor(goalX)
                    + 0.31D + random.nextDouble() * 0.38D;
            expectedZ = startZ
                    + 0.31D + random.nextDouble() * 0.38D;
        }
        return new double[] {expectedX, goalY + 0.5D, expectedZ};
    }

    private String describeUnsettledResetState(Minecraft minecraft) {
        StringBuilder result = new StringBuilder();
        if (minecraft.thePlayer == null) {
            return "client player missing after reset";
        }
        if (minecraft.thePlayer.isDead
                || !minecraft.thePlayer.isEntityAlive()
                || minecraft.thePlayer.getHealth() <= 0.0F) {
            result.append("player not alive after reset");
        }
        for (int index = 0; index < environmentSyncIds.length; index++) {
            BlockPos pos = new BlockPos(environmentSyncXs[index],
                    environmentSyncYs[index], environmentSyncZs[index]);
            int actual = Block.getStateId(minecraft.theWorld.getBlockState(pos));
            if (actual == environmentSyncIds[index]) continue;
            if (result.length() > 0) result.append(", ");
            result.append('#').append(index).append('@').append(pos)
                    .append(" expected=").append(environmentSyncIds[index])
                    .append(" actual=").append(actual);
        }
        if (pendingEnvironmentReset != null
                && EnvironmentResetManager.BRIDGING_TASK.equals(
                pendingEnvironmentReset.task)
                && !bridgingInventoryReady(
                minecraft.thePlayer.inventory.currentItem,
                minecraft.thePlayer.getHeldItem())) {
            if (result.length() > 0) result.append(", ");
            ItemStack held = minecraft.thePlayer.getHeldItem();
            result.append("bridging inventory expected=slot0:64xwool actual=slot")
                    .append(minecraft.thePlayer.inventory.currentItem)
                    .append(':').append(held == null ? "empty"
                    : held.stackSize + "x" + itemName(held));
        }
        if (pendingEnvironmentReset != null
                && EnvironmentResetManager.BRIDGING_TASK.equals(
                pendingEnvironmentReset.task)
                && !bridgingPlayerReady(minecraft, pendingEnvironmentReset)) {
            if (result.length() > 0) result.append(", ");
            result.append("bridging player teleport stale at=")
                    .append(minecraft.thePlayer.posX).append(',')
                    .append(minecraft.thePlayer.posY).append(',')
                    .append(minecraft.thePlayer.posZ)
                    .append(" goal=").append(environmentGoalX).append(',')
                    .append(environmentGoalY).append(',')
                    .append(environmentGoalZ);
        }
        if (pendingEnvironmentReset != null
                && EnvironmentResetManager.BRIDGING_TASK.equals(
                pendingEnvironmentReset.task)) {
            int above = BridgingSnapshot.countAboveLevel(
                    minecraft,
                    MathHelper.floor_double(environmentGoalX),
                    MathHelper.floor_double(environmentGoalY),
                    MathHelper.floor_double(environmentGoalZ));
            if (above > 0) {
                if (result.length() > 0) result.append(", ");
                result.append("bridging above-level blocks still visible=")
                        .append(above);
            }
            if (!bridgingAimedTargetReady(minecraft, pendingEnvironmentReset)) {
                if (result.length() > 0) result.append(", ");
                MovingObjectPosition hit = minecraft.objectMouseOver;
                result.append("bridging aimed raycast=")
                        .append(hit == null ? "null" : hit.typeOfHit)
                        .append(" side=")
                        .append(hit == null ? "" : hit.sideHit);
            }
        }
        return result.toString();
    }

    private GoalSnapshot[] captureTaskBlocks(Minecraft minecraft) {
        GoalSnapshot[] snapshots = new GoalSnapshot[MAX_TASK_BLOCKS];
        for (int index = 0; index < MAX_TASK_BLOCKS; index++) {
            snapshots[index] = index < environmentBlockGoalXs.length
                    ? GoalSnapshot.capture(minecraft, environmentBlockGoalXs[index],
                            environmentBlockGoalYs[index], environmentBlockGoalZs[index])
                    : new GoalSnapshot();
        }
        return snapshots;
    }

    private static void appendTaskBlockField(
            StringBuilder json, String field, GoalSnapshot[] snapshots) {
        int separator = field.indexOf('.', "task_block_".length());
        int index = Integer.parseInt(field.substring("task_block_".length(), separator));
        GoalSnapshot goal = snapshots[index];
        String attribute = field.substring(separator + 1);
        switch (attribute) {
            case "x": json.append(goal.x); break;
            case "y": json.append(goal.y); break;
            case "z": json.append(goal.z); break;
            case "distance": json.append(goal.distance); break;
            case "yaw_delta": json.append(goal.yawDelta); break;
            case "pitch_delta": json.append(goal.pitchDelta); break;
            case "block_id": json.append(goal.blockId); break;
            case "tool_speed_0": json.append(goal.toolSpeed0); break;
            case "tool_speed_1": json.append(goal.toolSpeed1); break;
            case "tool_speed_2": json.append(goal.toolSpeed2); break;
            default: throw new IllegalStateException("Unsupported task block field: " + field);
        }
    }

    private void clearPendingEnvironmentReset() {
        pendingEnvironmentReset = null;
        environmentResetResult = null;
        environmentResetError = null;
    }

    private static String itemName(ItemStack stack) {
        if (stack == null) return "";
        Object name = Item.itemRegistry.getNameForObject(stack.getItem());
        return name == null ? "" : name.toString();
    }

    private static int countArrows(Minecraft minecraft) {
        int count = 0;
        for (ItemStack stack : minecraft.thePlayer.inventory.mainInventory) {
            if (stack != null && stack.getItem() == Items.arrow) count += stack.stackSize;
        }
        return count;
    }

    private static void appendHotbar(StringBuilder json, Minecraft minecraft) {
        json.append('[');
        for (int slot = 0; slot < 9; slot++) {
            if (slot > 0) json.append(',');
            ItemStack stack = minecraft.thePlayer.inventory.mainInventory[slot];
            if (stack == null) {
                json.append("null");
            } else {
                json.append("{\"item_id\":").append(Item.getIdFromItem(stack.getItem()))
                        .append(",\"count\":").append(stack.stackSize)
                        .append(",\"damage\":").append(stack.getItemDamage()).append('}');
            }
        }
        json.append(']');
    }

    private static void appendLocalBlocks(StringBuilder json, Minecraft minecraft) {
        BlockPos center = new BlockPos(minecraft.thePlayer.posX,
                minecraft.thePlayer.posY, minecraft.thePlayer.posZ);
        json.append('[');
        boolean first = true;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (!first) json.append(',');
                    first = false;
                    json.append(Block.getStateId(minecraft.theWorld.getBlockState(
                            center.add(dx, dy, dz))));
                }
            }
        }
        json.append(']');
    }

    private static String createBenchmarkObservation(Minecraft minecraft, int sequence) {
        MovingObjectPosition hit = minecraft.objectMouseOver;
        String targetType = hit == null ? "MISS" : hit.typeOfHit.name();
        String targetName = "";
        int targetId = -1;
        int targetX = 0;
        int targetY = 0;
        int targetZ = 0;
        double targetDistance = -1.0D;
        float targetHealth = -1.0F;

        if (hit != null && hit.hitVec != null) {
            targetDistance = minecraft.thePlayer.getPositionEyes(1.0F).distanceTo(hit.hitVec);
        }
        if (hit != null && hit.typeOfHit == MovingObjectPosition.MovingObjectType.BLOCK) {
            BlockPos pos = hit.getBlockPos();
            IBlockState state = minecraft.theWorld.getBlockState(pos);
            Block block = state.getBlock();
            targetId = Block.getStateId(state);
            targetName = block.getRegistryName();
            targetX = pos.getX();
            targetY = pos.getY();
            targetZ = pos.getZ();
        } else if (hit != null && hit.typeOfHit == MovingObjectPosition.MovingObjectType.ENTITY) {
            Entity entity = hit.entityHit;
            targetId = entity.getEntityId();
            targetName = entity.getClass().getSimpleName();
            targetX = MathHelper.floor_double(entity.posX);
            targetY = MathHelper.floor_double(entity.posY);
            targetZ = MathHelper.floor_double(entity.posZ);
            if (entity instanceof EntityLivingBase) targetHealth = ((EntityLivingBase) entity).getHealth();
        }

        String json = "{\"type\":\"sync_observation\",\"sequence\":" + sequence
                + ",\"x\":" + minecraft.thePlayer.posX
                + ",\"y\":" + minecraft.thePlayer.posY
                + ",\"z\":" + minecraft.thePlayer.posZ
                + ",\"yaw\":" + minecraft.thePlayer.rotationYaw
                + ",\"pitch\":" + minecraft.thePlayer.rotationPitch
                + ",\"health\":" + minecraft.thePlayer.getHealth()
                + ",\"on_ground\":" + minecraft.thePlayer.onGround
                + ",\"target_type\":" + WireMessage.quote(targetType)
                + ",\"target_name\":" + WireMessage.quote(targetName)
                + ",\"target_id\":" + targetId
                + ",\"target_x\":" + targetX
                + ",\"target_y\":" + targetY
                + ",\"target_z\":" + targetZ
                + ",\"target_distance\":" + targetDistance
                + ",\"target_health\":" + targetHealth
                + ",\"reward\":0.0,\"terminated\":false,\"truncated\":false}";
        return json;
    }

    private static final class BridgingSnapshot {
        private static final int GRID_LONGITUDINAL = 66;

        private double groundFraction;
        private double forwardEdge;
        private double leftEdge;
        private double rightEdge;
        private double rearEdge;
        private boolean canPlace;
        private int aboveLevelCount;
        private boolean[] grid;
        private Object world;
        private int targetX;
        private int targetY;
        private int targetZ;
        private long blockVersion;

        private static BridgingSnapshot capture(Minecraft minecraft,
                                                double goalX, double goalY,
                                                double goalZ,
                                                BridgingSnapshot previous) {
            if (!Double.isFinite(goalX) || !Double.isFinite(goalY)
                    || !Double.isFinite(goalZ)) {
                throw new IllegalStateException(
                        "bridging observations require a finite task goal");
            }
            BridgingSnapshot result = new BridgingSnapshot();
            AxisAlignedBB playerBox = minecraft.thePlayer.getEntityBoundingBox();
            result.groundFraction = supportFraction(minecraft, playerBox);
            int supportY = MathHelper.floor_double(playerBox.minY - 0.01D);
            result.forwardEdge = edgeDistance(
                    minecraft, playerBox, supportY, 0, 1);
            result.rearEdge = edgeDistance(
                    minecraft, playerBox, supportY, 0, -1);
            result.leftEdge = edgeDistance(
                    minecraft, playerBox, supportY, -1, 0);
            result.rightEdge = edgeDistance(
                    minecraft, playerBox, supportY, 1, 0);
            result.canPlace = canPlaceHeldBlock(minecraft);

            result.targetX = MathHelper.floor_double(goalX);
            result.targetY = MathHelper.floor_double(goalY);
            result.targetZ = MathHelper.floor_double(goalZ);
            result.world = minecraft.theWorld;
            result.blockVersion = ClientBlockChangeTracker.version();
            boolean reusable = previous != null
                    && previous.world == result.world
                    && previous.targetX == result.targetX
                    && previous.targetY == result.targetY
                    && previous.targetZ == result.targetZ
                    && previous.blockVersion == result.blockVersion;
            if (reusable) {
                result.grid = previous.grid;
                result.aboveLevelCount = previous.aboveLevelCount;
            } else {
                result.grid = new boolean[GRID_LONGITUDINAL];
                for (int longitudinal = 0;
                     longitudinal < GRID_LONGITUDINAL; longitudinal++) {
                    BlockPos position = new BlockPos(
                            result.targetX, result.targetY,
                            result.targetZ - GRID_LONGITUDINAL + 1 + longitudinal);
                    IBlockState state = minecraft.theWorld.getBlockState(position);
                    result.grid[longitudinal] = state.getBlock() != Blocks.air;
                }
                result.aboveLevelCount = countAboveLevel(
                        minecraft, result.targetX, result.targetY, result.targetZ);
            }
            return result;
        }

        private static int countAboveLevel(Minecraft minecraft,
                                           int targetX, int targetY,
                                           int targetZ) {
            int count = 0;
            // The first illegal placement must occupy the layer immediately
            // above bridge level; the task truncates before a higher layer can
            // be supported and placed.
            int illegalY = targetY + 1;
            for (int z = targetZ - 67; z <= targetZ + 2; z++) {
                for (int x = targetX - 8; x <= targetX + 8; x++) {
                    if (!minecraft.theWorld.isAirBlock(
                            new BlockPos(x, illegalY, z))) {
                        count++;
                    }
                }
            }
            return count;
        }

        private static boolean canPlaceHeldBlock(Minecraft minecraft) {
            MovingObjectPosition hit = minecraft.objectMouseOver;
            ItemStack held = minecraft.thePlayer.getHeldItem();
            if (hit == null
                    || hit.typeOfHit != MovingObjectPosition.MovingObjectType.BLOCK
                    || hit.sideHit == null
                    || held == null
                    || !(held.getItem() instanceof ItemBlock)) {
                return false;
            }
            ItemBlock item = (ItemBlock) held.getItem();
            return item.canPlaceBlockOnSide(
                    minecraft.theWorld, hit.getBlockPos(), hit.sideHit,
                    minecraft.thePlayer, held);
        }

        private static boolean supportingBlock(Minecraft minecraft, int x, int y, int z) {
            IBlockState state = minecraft.theWorld.getBlockState(new BlockPos(x, y, z));
            return state.getBlock() != Blocks.air
                    && state.getBlock().getMaterial().blocksMovement();
        }

        private static double edgeDistance(Minecraft minecraft, AxisAlignedBB playerBox,
                                           int supportY, int dx, int dz) {
            int minX = MathHelper.floor_double(playerBox.minX + 1.0E-7D);
            int maxX = MathHelper.floor_double(playerBox.maxX - 1.0E-7D);
            int minZ = MathHelper.floor_double(playerBox.minZ + 1.0E-7D);
            int maxZ = MathHelper.floor_double(playerBox.maxZ - 1.0E-7D);
            double best = 0.0D;
            for (int x = minX; x <= maxX; x++) {
                for (int z = minZ; z <= maxZ; z++) {
                    if (!supportingBlock(minecraft, x, supportY, z)) continue;
                    int scanX = x;
                    int scanZ = z;
                    int scanned = 0;
                    while (scanned++ < 66
                            && supportingBlock(minecraft, scanX + dx,
                            supportY, scanZ + dz)) {
                        scanX += dx;
                        scanZ += dz;
                    }
                    double boundaryX = dx > 0 ? scanX + 1.0D : scanX;
                    double boundaryZ = dz > 0 ? scanZ + 1.0D : scanZ;
                    double distance = dx != 0
                            ? (boundaryX - minecraft.thePlayer.posX) * dx
                            : (boundaryZ - minecraft.thePlayer.posZ) * dz;
                    best = Math.max(best, distance);
                }
            }
            return best;
        }

        private static double supportFraction(Minecraft minecraft,
                                              AxisAlignedBB playerBox) {
            AxisAlignedBB probe = new AxisAlignedBB(
                    playerBox.minX, playerBox.minY - 0.05D, playerBox.minZ,
                    playerBox.maxX, playerBox.minY + 0.001D, playerBox.maxZ);
            List<AxisAlignedBB> collisions = minecraft.theWorld
                    .getCollidingBoundingBoxes(minecraft.thePlayer, probe);
            List<AxisAlignedBB> supports = new ArrayList<>();
            List<Double> xs = new ArrayList<>();
            List<Double> zs = new ArrayList<>();
            xs.add(playerBox.minX);
            xs.add(playerBox.maxX);
            zs.add(playerBox.minZ);
            zs.add(playerBox.maxZ);
            for (AxisAlignedBB box : collisions) {
                if (box.maxY < playerBox.minY - 0.051D
                        || box.maxY > playerBox.minY + 0.01D) continue;
                double minX = Math.max(playerBox.minX, box.minX);
                double maxX = Math.min(playerBox.maxX, box.maxX);
                double minZ = Math.max(playerBox.minZ, box.minZ);
                double maxZ = Math.min(playerBox.maxZ, box.maxZ);
                if (minX >= maxX || minZ >= maxZ) continue;
                supports.add(new AxisAlignedBB(
                        minX, box.minY, minZ, maxX, box.maxY, maxZ));
                xs.add(minX);
                xs.add(maxX);
                zs.add(minZ);
                zs.add(maxZ);
            }
            if (supports.isEmpty()) return 0.0D;
            Collections.sort(xs);
            Collections.sort(zs);
            double covered = 0.0D;
            for (int xi = 0; xi + 1 < xs.size(); xi++) {
                double x0 = xs.get(xi);
                double x1 = xs.get(xi + 1);
                if (x1 <= x0) continue;
                for (int zi = 0; zi + 1 < zs.size(); zi++) {
                    double z0 = zs.get(zi);
                    double z1 = zs.get(zi + 1);
                    if (z1 <= z0) continue;
                    double centerX = (x0 + x1) * 0.5D;
                    double centerZ = (z0 + z1) * 0.5D;
                    boolean supported = false;
                    for (AxisAlignedBB box : supports) {
                        if (centerX >= box.minX && centerX <= box.maxX
                                && centerZ >= box.minZ && centerZ <= box.maxZ) {
                            supported = true;
                            break;
                        }
                    }
                    if (supported) covered += (x1 - x0) * (z1 - z0);
                }
            }
            double footprint = (playerBox.maxX - playerBox.minX)
                    * (playerBox.maxZ - playerBox.minZ);
            return footprint <= 0.0D ? 0.0D
                    : Math.max(0.0D, Math.min(1.0D, covered / footprint));
        }

        private void appendGrid(StringBuilder json) {
            json.append('[');
            for (int index = 0; index < grid.length; index++) {
                if (index > 0) json.append(',');
                json.append(grid[index] ? '1' : '0');
            }
            json.append(']');
        }
    }

    private static final class RaycastSnapshot {
        private String type = "MISS";
        private String name = "";
        private int id = -1;
        private int x;
        private int y;
        private int z;
        private double distance = -1.0D;
        private float health = -1.0F;
        private String side = "";
        private float blockHardness = -1.0F;
        private boolean canHarvest;
        private int category;
        private float safeYawCorrection;
        private float safePitchCorrection;

        private static RaycastSnapshot capture(Minecraft minecraft, List<String> requested) {
            RaycastSnapshot result = new RaycastSnapshot();
            MovingObjectPosition hit = minecraft.objectMouseOver;
            if (hit == null) return result;
            result.type = hit.typeOfHit.name();
            if (requested.contains("raycast.distance") && hit.hitVec != null) {
                result.distance = minecraft.thePlayer.getPositionEyes(1.0F).distanceTo(hit.hitVec);
            }

            boolean needsName = requested.contains("raycast.name");
            boolean needsId = requested.contains("raycast.id");
            boolean needsPosition = requested.contains("raycast.x")
                    || requested.contains("raycast.y") || requested.contains("raycast.z");
            if (hit.typeOfHit == MovingObjectPosition.MovingObjectType.BLOCK) {
                BlockPos pos = hit.getBlockPos();
                if (requested.contains("raycast.side") && hit.sideHit != null) {
                    result.side = hit.sideHit.name();
                }
                boolean needsCategory = requested.contains("raycast.category");
                boolean needsSafeCorrection = requested.contains("raycast.safe_yaw_correction")
                        || requested.contains("raycast.safe_pitch_correction");
                if (needsName || needsId || needsCategory || needsSafeCorrection
                        || requested.contains("raycast.block_hardness")
                        || requested.contains("raycast.can_harvest")) {
                    IBlockState state = minecraft.theWorld.getBlockState(pos);
                    if (needsName) result.name = state.getBlock().getRegistryName();
                    if (needsId) result.id = Block.getStateId(state);
                    if (needsCategory) result.category = bedDefenseCategory(state);
                    if (requested.contains("raycast.block_hardness")) {
                        result.blockHardness = state.getBlock().getBlockHardness(minecraft.theWorld, pos);
                    }
                    if (requested.contains("raycast.can_harvest")) {
                        result.canHarvest = minecraft.thePlayer.canHarvestBlock(state.getBlock());
                    }
                    if (needsSafeCorrection) {
                        float[] correction = safeFaceCorrection(
                                minecraft, hit, pos, state);
                        result.safeYawCorrection = correction[0];
                        result.safePitchCorrection = correction[1];
                    }
                }
                if (needsPosition) {
                    result.x = pos.getX();
                    result.y = pos.getY();
                    result.z = pos.getZ();
                }
            } else if (hit.typeOfHit == MovingObjectPosition.MovingObjectType.ENTITY) {
                Entity entity = hit.entityHit;
                if (needsName) result.name = entity.getClass().getSimpleName();
                if (needsId) result.id = entity.getEntityId();
                if (needsPosition) {
                    result.x = MathHelper.floor_double(entity.posX);
                    result.y = MathHelper.floor_double(entity.posY);
                    result.z = MathHelper.floor_double(entity.posZ);
                }
                if (requested.contains("raycast.health") && entity instanceof EntityLivingBase) {
                    result.health = ((EntityLivingBase) entity).getHealth();
                }
            }
            return result;
        }

        private static float[] safeFaceCorrection(
                Minecraft minecraft, MovingObjectPosition hit,
                BlockPos position, IBlockState state) {
            if (hit.hitVec == null || hit.sideHit == null) {
                return new float[] {0.0F, 0.0F};
            }
            AxisAlignedBB bounds = state.getBlock().getSelectedBoundingBox(
                    minecraft.theWorld, position);
            if (bounds == null) return new float[] {0.0F, 0.0F};

            double targetX = hit.hitVec.xCoord;
            double targetY = hit.hitVec.yCoord;
            double targetZ = hit.hitVec.zCoord;
            double safeMinX = bounds.minX + (bounds.maxX - bounds.minX) * 0.10D;
            double safeMaxX = bounds.maxX - (bounds.maxX - bounds.minX) * 0.10D;
            double safeMinY = bounds.minY + (bounds.maxY - bounds.minY) * 0.10D;
            double safeMaxY = bounds.maxY - (bounds.maxY - bounds.minY) * 0.10D;
            double safeMinZ = bounds.minZ + (bounds.maxZ - bounds.minZ) * 0.10D;
            double safeMaxZ = bounds.maxZ - (bounds.maxZ - bounds.minZ) * 0.10D;
            switch (hit.sideHit) {
                case DOWN:
                case UP:
                    targetX = clamp(targetX, safeMinX, safeMaxX);
                    targetZ = clamp(targetZ, safeMinZ, safeMaxZ);
                    break;
                case NORTH:
                case SOUTH:
                    targetX = clamp(targetX, safeMinX, safeMaxX);
                    targetY = clamp(targetY, safeMinY, safeMaxY);
                    break;
                case WEST:
                case EAST:
                    targetY = clamp(targetY, safeMinY, safeMaxY);
                    targetZ = clamp(targetZ, safeMinZ, safeMaxZ);
                    break;
                default:
                    return new float[] {0.0F, 0.0F};
            }
            if (targetX == hit.hitVec.xCoord && targetY == hit.hitVec.yCoord
                    && targetZ == hit.hitVec.zCoord) {
                return new float[] {0.0F, 0.0F};
            }
            Vec3 eyes = minecraft.thePlayer.getPositionEyes(1.0F);
            double dx = targetX - eyes.xCoord;
            double dy = targetY - eyes.yCoord;
            double dz = targetZ - eyes.zCoord;
            double horizontal = Math.sqrt(dx * dx + dz * dz);
            float desiredYaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
            float desiredPitch = (float) -Math.toDegrees(Math.atan2(dy, horizontal));
            return new float[] {
                    MathHelper.wrapAngleTo180_float(
                            desiredYaw - minecraft.thePlayer.rotationYaw),
                    desiredPitch - minecraft.thePlayer.rotationPitch
            };
        }

        private static double clamp(double value, double minimum, double maximum) {
            return Math.max(minimum, Math.min(maximum, value));
        }
    }

    static final class BedDefenseVisibilityInputs {
        private final Object world;
        private final int originX;
        private final int originY;
        private final int originZ;
        private final double eyeX;
        private final double eyeY;
        private final double eyeZ;
        private final int[] stateIds;

        BedDefenseVisibilityInputs(Object world, int originX, int originY, int originZ,
                                   double eyeX, double eyeY, double eyeZ,
                                   int[] stateIds) {
            this.world = world;
            this.originX = originX;
            this.originY = originY;
            this.originZ = originZ;
            this.eyeX = eyeX;
            this.eyeY = eyeY;
            this.eyeZ = eyeZ;
            this.stateIds = stateIds.clone();
        }

        boolean matches(BedDefenseVisibilityInputs other) {
            return other != null
                    && world == other.world
                    && originX == other.originX
                    && originY == other.originY
                    && originZ == other.originZ
                    && Double.doubleToLongBits(eyeX) == Double.doubleToLongBits(other.eyeX)
                    && Double.doubleToLongBits(eyeY) == Double.doubleToLongBits(other.eyeY)
                    && Double.doubleToLongBits(eyeZ) == Double.doubleToLongBits(other.eyeZ)
                    && Arrays.equals(stateIds, other.stateIds);
        }
    }

    static boolean canReuseBedDefenseVolume(
            Object previousWorld, int previousOriginX, int previousOriginY,
            int previousOriginZ, long previousVersion,
            Object currentWorld, int currentOriginX, int currentOriginY,
            int currentOriginZ, long currentVersion) {
        return previousWorld == currentWorld
                && previousOriginX == currentOriginX
                && previousOriginY == currentOriginY
                && previousOriginZ == currentOriginZ
                && previousVersion == currentVersion;
    }

    static boolean sameBedDefenseEye(double previousX, double previousY, double previousZ,
                                     double currentX, double currentY, double currentZ) {
        return Double.doubleToLongBits(previousX) == Double.doubleToLongBits(currentX)
                && Double.doubleToLongBits(previousY) == Double.doubleToLongBits(currentY)
                && Double.doubleToLongBits(previousZ) == Double.doubleToLongBits(currentZ);
    }

    private static final class BedDefenseSnapshot {
        private static final int SIZE_X = 7;
        private static final int SIZE_Y = 5;
        private static final int SIZE_Z = 7;
        private static final int CELL_COUNT = SIZE_X * SIZE_Y * SIZE_Z;
        private int originX;
        private int originY;
        private int originZ;
        private Object world;
        private long blockChangeVersion;
        private final int[] categories = new int[CELL_COUNT];
        private final boolean[] visible = new boolean[CELL_COUNT];
        private BedDefenseVisibilityInputs visibilityInputs;

        private static BedDefenseSnapshot capture(Minecraft minecraft,
                                                  double goalX, double goalY,
                                                  double goalZ,
                                                  BedDefenseSnapshot previous) {
            if (!Double.isFinite(goalX) || !Double.isFinite(goalY)
                    || !Double.isFinite(goalZ)) {
                throw new IllegalStateException(
                        "bed_defense observations require a finite reset goal");
            }
            BlockPos foot = new BlockPos(goalX, goalY, goalZ);
            int originX = foot.getX() - 3;
            int originY = foot.getY() - 1;
            int originZ = foot.getZ() - 3;
            Vec3 eyes = minecraft.thePlayer.getPositionEyes(1.0F);
            long blockChangeVersion = ClientBlockChangeTracker.version();
            boolean unchangedVolume = previous != null
                    && canReuseBedDefenseVolume(
                            previous.world, previous.originX, previous.originY,
                            previous.originZ, previous.blockChangeVersion,
                            minecraft.theWorld, originX, originY, originZ,
                            blockChangeVersion);
            if (unchangedVolume
                    && sameBedDefenseEye(
                            previous.visibilityInputs.eyeX,
                            previous.visibilityInputs.eyeY,
                            previous.visibilityInputs.eyeZ,
                            eyes.xCoord, eyes.yCoord, eyes.zCoord)) {
                return previous;
            }

            BedDefenseSnapshot result = new BedDefenseSnapshot();
            result.world = minecraft.theWorld;
            result.blockChangeVersion = blockChangeVersion;
            result.originX = originX;
            result.originY = originY;
            result.originZ = originZ;
            int[] stateIds = new int[CELL_COUNT];

            if (unchangedVolume) {
                System.arraycopy(previous.visibilityInputs.stateIds, 0,
                        stateIds, 0, CELL_COUNT);
                System.arraycopy(previous.categories, 0,
                        result.categories, 0, CELL_COUNT);
            } else {
                for (int y = 0; y < SIZE_Y; y++) {
                    for (int z = 0; z < SIZE_Z; z++) {
                        for (int x = 0; x < SIZE_X; x++) {
                            int index = index(x, y, z);
                            IBlockState state = minecraft.theWorld.getBlockState(new BlockPos(
                                    result.originX + x, result.originY + y,
                                    result.originZ + z));
                            stateIds[index] = Block.getStateId(state);
                            result.categories[index] = bedDefenseCategory(state);
                        }
                    }
                }
            }

            result.visibilityInputs = new BedDefenseVisibilityInputs(
                    minecraft.theWorld, result.originX, result.originY, result.originZ,
                    eyes.xCoord, eyes.yCoord, eyes.zCoord, stateIds);
            if (previous != null
                    && previous.visibilityInputs.matches(result.visibilityInputs)) {
                System.arraycopy(previous.visible, 0, result.visible, 0, CELL_COUNT);
                return result;
            }
            for (int y = 0; y < SIZE_Y; y++) {
                for (int z = 0; z < SIZE_Z; z++) {
                    for (int x = 0; x < SIZE_X; x++) {
                        int current = index(x, y, z);
                        BlockPos position = new BlockPos(
                                result.originX + x, result.originY + y,
                                result.originZ + z);
                        MovingObjectPosition hit = minecraft.theWorld.rayTraceBlocks(
                                eyes,
                                new Vec3(position.getX() + 0.5D,
                                        position.getY() + 0.5D,
                                        position.getZ() + 0.5D),
                                false, true, false);
                        if (result.categories[current] == 1) {
                            // Air is known only when no nearer block occludes its
                            // center. This distinguishes visible empty space from
                            // unknown space without revealing anything behind a wall.
                            result.visible[current] = hit == null;
                        } else {
                            result.visible[current] = hit != null
                                    && hit.typeOfHit == MovingObjectPosition.MovingObjectType.BLOCK
                                    && position.equals(hit.getBlockPos());
                        }
                    }
                }
            }
            return result;
        }

        private static int index(int x, int y, int z) {
            return (y * SIZE_Z + z) * SIZE_X + x;
        }

        private void appendGrid(StringBuilder json) {
            json.append('[');
            for (int index = 0; index < categories.length; index++) {
                if (index > 0) json.append(',');
                json.append(categories[index]);
            }
            json.append(']');
        }

        private void appendVisibleMask(StringBuilder json) {
            json.append('[');
            for (int index = 0; index < visible.length; index++) {
                if (index > 0) json.append(',');
                json.append(visible[index] ? 1 : 0);
            }
            json.append(']');
        }
    }

    private static int bedDefenseCategory(IBlockState state) {
        Block block = state.getBlock();
        if (block == Blocks.air) return 1;
        if (block == Blocks.bed) return 2;
        if (block == Blocks.wool) return 3;
        if (block == Blocks.planks || block == Blocks.log || block == Blocks.log2) return 4;
        if (block == Blocks.end_stone) return 5;
        if (block == Blocks.obsidian) return 6;
        return 7;
    }

    /** Bed-oriented 8 longitudinal x 4 vertical x 7 lateral V2 volume. */
    private static final class BedDefenseV2Snapshot {
        private static final int SIZE_LONGITUDINAL = 8;
        private static final int SIZE_VERTICAL = 4;
        private static final int SIZE_LATERAL = 7;
        private static final int CELL_COUNT =
                SIZE_LONGITUDINAL * SIZE_VERTICAL * SIZE_LATERAL;
        private Object world;
        private long blockChangeVersion;
        private BlockPos foot;
        private int longitudinalX;
        private int longitudinalZ;
        private double eyeX;
        private double eyeY;
        private double eyeZ;
        private final int[] categories = new int[CELL_COUNT];
        private final boolean[] visible = new boolean[CELL_COUNT];

        private static BedDefenseV2Snapshot capture(
                Minecraft minecraft, double[] goalXs, double[] goalYs,
                double[] goalZs, BedDefenseV2Snapshot previous) {
            if (goalXs.length != 2 || goalYs.length != 2 || goalZs.length != 2) {
                throw new IllegalStateException(
                        "bed_defense_v2 observations require exactly two bed halves");
            }
            BlockPos foot = new BlockPos(goalXs[0], goalYs[0], goalZs[0]);
            BlockPos head = new BlockPos(goalXs[1], goalYs[1], goalZs[1]);
            int longitudinalX = head.getX() - foot.getX();
            int longitudinalZ = head.getZ() - foot.getZ();
            if (head.getY() != foot.getY()
                    || Math.abs(longitudinalX) + Math.abs(longitudinalZ) != 1) {
                throw new IllegalStateException(
                        "bed_defense_v2 bed halves are not horizontally adjacent");
            }
            Vec3 eyes = minecraft.thePlayer.getPositionEyes(1.0F);
            long version = ClientBlockChangeTracker.version();
            boolean unchangedVolume = previous != null
                    && previous.world == minecraft.theWorld
                    && previous.foot.equals(foot)
                    && previous.longitudinalX == longitudinalX
                    && previous.longitudinalZ == longitudinalZ
                    && previous.blockChangeVersion == version;
            if (unchangedVolume
                    && sameBedDefenseEye(previous.eyeX, previous.eyeY, previous.eyeZ,
                            eyes.xCoord, eyes.yCoord, eyes.zCoord)) {
                return previous;
            }

            BedDefenseV2Snapshot result = new BedDefenseV2Snapshot();
            result.world = minecraft.theWorld;
            result.blockChangeVersion = version;
            result.foot = foot;
            result.longitudinalX = longitudinalX;
            result.longitudinalZ = longitudinalZ;
            result.eyeX = eyes.xCoord;
            result.eyeY = eyes.yCoord;
            result.eyeZ = eyes.zCoord;
            if (unchangedVolume) {
                System.arraycopy(previous.categories, 0, result.categories, 0,
                        CELL_COUNT);
            } else {
                for (int vertical = 0; vertical < SIZE_VERTICAL; vertical++) {
                    for (int lateral = 0; lateral < SIZE_LATERAL; lateral++) {
                        for (int longitudinal = 0;
                             longitudinal < SIZE_LONGITUDINAL; longitudinal++) {
                            int index = index(longitudinal, vertical, lateral);
                            BlockPos position = result.position(
                                    longitudinal, vertical, lateral);
                            result.categories[index] = bedDefenseCategory(
                                    minecraft.theWorld.getBlockState(position));
                        }
                    }
                }
            }

            for (int vertical = 0; vertical < SIZE_VERTICAL; vertical++) {
                for (int lateral = 0; lateral < SIZE_LATERAL; lateral++) {
                    for (int longitudinal = 0;
                         longitudinal < SIZE_LONGITUDINAL; longitudinal++) {
                        int index = index(longitudinal, vertical, lateral);
                        BlockPos position = result.position(
                                longitudinal, vertical, lateral);
                        if (result.categories[index] == 1) {
                            MovingObjectPosition hit = minecraft.theWorld.rayTraceBlocks(
                                    eyes,
                                    new Vec3(position.getX() + 0.5D,
                                            position.getY() + 0.5D,
                                            position.getZ() + 0.5D),
                                    false, true, false);
                            result.visible[index] = hit == null;
                        } else {
                            result.visible[index] = canSeeBlock(
                                    minecraft, eyes, position);
                        }
                    }
                }
            }
            return result;
        }

        private BlockPos position(int longitudinal, int vertical, int lateral) {
            int longitudinalOffset = longitudinal - 3;
            int lateralOffset = lateral - 3;
            int lateralX = -longitudinalZ;
            int lateralZ = longitudinalX;
            return foot.add(
                    longitudinalX * longitudinalOffset + lateralX * lateralOffset,
                    vertical,
                    longitudinalZ * longitudinalOffset + lateralZ * lateralOffset);
        }

        private static boolean canSeeBlock(
                Minecraft minecraft, Vec3 eyes, BlockPos position) {
            IBlockState state = minecraft.theWorld.getBlockState(position);
            AxisAlignedBB bounds = state.getBlock().getSelectedBoundingBox(
                    minecraft.theWorld, position);
            if (bounds == null) return false;
            double[] fractions = {0.15D, 0.5D, 0.85D};
            for (int face = 0; face < 6; face++) {
                for (double first : fractions) {
                    for (double second : fractions) {
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
                        MovingObjectPosition hit = minecraft.theWorld.rayTraceBlocks(
                                eyes, new Vec3(x, y, z), false, true, false);
                        if (hit != null
                                && hit.typeOfHit == MovingObjectPosition.MovingObjectType.BLOCK
                                && position.equals(hit.getBlockPos())) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        private static double mix(double minimum, double maximum, double fraction) {
            return minimum + (maximum - minimum) * fraction;
        }

        private static int index(int longitudinal, int vertical, int lateral) {
            return (vertical * SIZE_LATERAL + lateral) * SIZE_LONGITUDINAL
                    + longitudinal;
        }

        private void appendGrid(StringBuilder json) {
            json.append('[');
            for (int index = 0; index < categories.length; index++) {
                if (index > 0) json.append(',');
                json.append(categories[index]);
            }
            json.append(']');
        }

        private void appendVisibleMask(StringBuilder json) {
            json.append('[');
            for (int index = 0; index < visible.length; index++) {
                if (index > 0) json.append(',');
                json.append(visible[index] ? 1 : 0);
            }
            json.append(']');
        }
    }

    private static final class LivingSnapshot {
        private String type = "";
        private int id = -1;
        private double x;
        private double y;
        private double z;
        private double relativeX;
        private double relativeY;
        private double relativeZ;
        private double distance = -1.0D;
        private float yawDelta;
        private float pitchDelta;
        private double motionX;
        private double motionY;
        private double motionZ;
        private float health = -1.0F;
        private boolean alive;

        private static int findNearestLivingId(Minecraft minecraft) {
            EntityLivingBase nearest = findNearest(minecraft);
            return nearest == null ? -1 : nearest.getEntityId();
        }

        private static LivingSnapshot captureNearest(Minecraft minecraft) {
            return capture(minecraft, findNearest(minecraft));
        }

        private static LivingSnapshot captureById(Minecraft minecraft, int entityId) {
            Entity entity = entityId < 0 ? null : minecraft.theWorld.getEntityByID(entityId);
            LivingSnapshot result = capture(minecraft, entity instanceof EntityLivingBase
                    ? (EntityLivingBase) entity : null);
            if (entityId >= 0 && entity == null) result.id = entityId;
            return result;
        }

        private static EntityLivingBase findNearest(Minecraft minecraft) {
            EntityLivingBase nearest = null;
            double nearestDistanceSq = Double.POSITIVE_INFINITY;
            for (Object candidate : minecraft.theWorld.loadedEntityList) {
                if (!(candidate instanceof EntityLivingBase) || candidate == minecraft.thePlayer) continue;
                EntityLivingBase living = (EntityLivingBase) candidate;
                double distanceSq = living.getDistanceSqToEntity(minecraft.thePlayer);
                if (living.isEntityAlive() && distanceSq < nearestDistanceSq) {
                    nearest = living;
                    nearestDistanceSq = distanceSq;
                }
            }
            return nearest;
        }

        private static LivingSnapshot capture(Minecraft minecraft, EntityLivingBase nearest) {
            LivingSnapshot result = new LivingSnapshot();
            if (nearest == null) return result;

            result.type = nearest.getClass().getSimpleName();
            result.id = nearest.getEntityId();
            result.x = nearest.posX;
            result.y = nearest.posY;
            result.z = nearest.posZ;
            result.relativeX = nearest.posX - minecraft.thePlayer.posX;
            result.relativeY = nearest.posY - minecraft.thePlayer.posY;
            result.relativeZ = nearest.posZ - minecraft.thePlayer.posZ;
            result.distance = Math.sqrt(nearest.getDistanceSqToEntity(minecraft.thePlayer));
            double horizontal = Math.sqrt(result.relativeX * result.relativeX
                    + result.relativeZ * result.relativeZ);
            float desiredYaw = (float) (Math.atan2(result.relativeZ, result.relativeX)
                    * 180.0D / Math.PI) - 90.0F;
            double eyeDeltaY = nearest.posY + nearest.getEyeHeight()
                    - (minecraft.thePlayer.posY + minecraft.thePlayer.getEyeHeight());
            float desiredPitch = (float) -(Math.atan2(eyeDeltaY, horizontal)
                    * 180.0D / Math.PI);
            result.yawDelta = MathHelper.wrapAngleTo180_float(
                    desiredYaw - minecraft.thePlayer.rotationYaw);
            result.pitchDelta = desiredPitch - minecraft.thePlayer.rotationPitch;
            result.motionX = nearest.motionX;
            result.motionY = nearest.motionY;
            result.motionZ = nearest.motionZ;
            result.health = nearest.getHealth();
            result.alive = nearest.isEntityAlive();
            return result;
        }
    }

    private static final class GoalSnapshot {
        private double x;
        private double y;
        private double z;
        private double relativeX;
        private double relativeY;
        private double relativeZ;
        private double distance;
        private double horizontalDistance;
        private float yawDelta;
        private float pitchDelta;
        private int blockId;
        private float toolSpeed0;
        private float toolSpeed1;
        private float toolSpeed2;

        private static GoalSnapshot capture(Minecraft minecraft, double x, double y, double z) {
            GoalSnapshot result = new GoalSnapshot();
            result.x = x;
            result.y = y;
            result.z = z;
            result.relativeX = x - minecraft.thePlayer.posX;
            result.relativeY = y - minecraft.thePlayer.posY;
            result.relativeZ = z - minecraft.thePlayer.posZ;
            result.horizontalDistance = Math.sqrt(result.relativeX * result.relativeX
                    + result.relativeZ * result.relativeZ);
            result.distance = Math.sqrt(result.horizontalDistance * result.horizontalDistance
                    + result.relativeY * result.relativeY);
            BlockPos blockPos = new BlockPos(x, y, z);
            IBlockState blockState = minecraft.theWorld.getBlockState(blockPos);
            result.blockId = Block.getStateId(blockState);
            result.toolSpeed0 = toolBreakSpeed(minecraft, blockState, blockPos, 0);
            result.toolSpeed1 = toolBreakSpeed(minecraft, blockState, blockPos, 1);
            result.toolSpeed2 = toolBreakSpeed(minecraft, blockState, blockPos, 2);
            float desiredYaw = (float) (Math.atan2(result.relativeZ, result.relativeX)
                    * 180.0D / Math.PI) - 90.0F;
            double eyeRelativeY = y - (minecraft.thePlayer.posY
                    + minecraft.thePlayer.getEyeHeight());
            float desiredPitch = (float) -(Math.atan2(eyeRelativeY, result.horizontalDistance)
                    * 180.0D / Math.PI);
            result.yawDelta = MathHelper.wrapAngleTo180_float(
                    desiredYaw - minecraft.thePlayer.rotationYaw);
            result.pitchDelta = desiredPitch - minecraft.thePlayer.rotationPitch;
            return result;
        }

        private static float toolBreakSpeed(Minecraft minecraft, IBlockState state,
                                            BlockPos pos, int hotbarSlot) {
            Block block = state.getBlock();
            if (block == net.minecraft.init.Blocks.air) return 0.0F;
            float hardness = block.getBlockHardness(minecraft.theWorld, pos);
            if (hardness <= 0.0F) return 0.0F;
            ItemStack stack = minecraft.thePlayer.inventory.getStackInSlot(hotbarSlot);
            float strength = stack == null ? 1.0F : stack.getStrVsBlock(block);
            boolean canHarvest = block.getMaterial().isToolNotRequired()
                    || (stack != null && stack.canHarvestBlock(block));
            return strength / hardness / (canHarvest ? 30.0F : 100.0F);
        }
    }
}
