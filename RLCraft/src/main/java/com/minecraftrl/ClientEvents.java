package com.minecraftrl;

import com.minecraftrl.benchmark.SynchronizedBenchmark;
import com.minecraftrl.network.TcpBridgeClient;
import com.minecraftrl.network.WireMessage;
import com.minecraftrl.tick.TickController;
import com.minecraftrl.world.AutomaticWorldLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.util.ChatComponentText;
import net.minecraftforge.event.world.WorldEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;

import java.awt.GraphicsEnvironment;

public final class ClientEvents {
    private final TcpBridgeClient bridge;
    private final SynchronizedBenchmark synchronizedBenchmark;
    private final RuntimeConfig config;
    private final AutomaticWorldLoader automaticWorldLoader;
    private long readyConnectionEpoch = -1L;
    private boolean readyAnnounced;

    public ClientEvents(TcpBridgeClient bridge, SynchronizedBenchmark synchronizedBenchmark,
                        RuntimeConfig config) {
        this.bridge = bridge;
        this.synchronizedBenchmark = synchronizedBenchmark;
        this.config = config;
        this.automaticWorldLoader = new AutomaticWorldLoader(config, bridge);
    }

    private void pumpIncomingMessages() {
        WireMessage message;
        int handled = 0;
        while (handled++ < 100 && (message = bridge.pollIncoming()) != null) {
            if (synchronizedBenchmark.handleProtocol(message)) {
                continue;
            } else if ("control".equals(message.type)) {
                applyControl(message);
            } else {
                showMessage("[RL/Python] " + message.text);
            }
        }
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase == TickEvent.Phase.START) {
            automaticWorldLoader.onClientTick();
            announceClientReady();
            if (TickController.INSTANCE.getMode() != TickController.Mode.SYNC) {
                pumpIncomingMessages();
            }
            synchronizedBenchmark.onClientTickStart();
        } else {
            synchronizedBenchmark.onClientTickEnd();
            TickController.INSTANCE.recordClientTick();
        }
    }

    @SubscribeEvent
    public void onRenderTick(TickEvent.RenderTickEvent event) {
        if (event.phase == TickEvent.Phase.START
                && TickController.INSTANCE.getMode() == TickController.Mode.SYNC) {
            // SYNC deliberately runs zero vanilla client ticks while waiting for
            // Python. Render-loop events still pump TCP and release controlled ticks.
            long generation = TickController.INSTANCE.workGeneration();
            pumpIncomingMessages();
            synchronizedBenchmark.onSynchronizedMainLoop();
            if (synchronizedBenchmark.isAwaitingAction() && !bridge.hasIncoming()) {
                TickController.INSTANCE.awaitIdleWork(generation);
            }
        }
    }

    @SubscribeEvent
    public void onWorldUnload(WorldEvent.Unload event) {
        if (event.world.isRemote) {
            readyAnnounced = false;
            synchronizedBenchmark.abort("world unloaded", true);
            TickController.INSTANCE.setMode(TickController.Mode.VANILLA);
        }
    }

    private void announceClientReady() {
        Minecraft minecraft = Minecraft.getMinecraft();
        boolean worldReady = minecraft.isIntegratedServerRunning()
                && minecraft.getIntegratedServer() != null
                && minecraft.thePlayer != null
                && minecraft.theWorld != null;
        if (!worldReady) {
            readyAnnounced = false;
            return;
        }
        if (!bridge.isConnected()) return;

        long connectionEpoch = bridge.getConnectionEpoch();
        if (readyAnnounced && readyConnectionEpoch == connectionEpoch) return;
        if (bridge.sendProtocol(WireMessage.clientReady(
                config.getInstanceId(), config.getWorld()))) {
            readyAnnounced = true;
            readyConnectionEpoch = connectionEpoch;
            MinecraftRLMod.LOGGER.info("Instance {} is ready for RL commands in world {}",
                    config.getInstanceId(), config.getWorld());
        }
    }

    private void applyControl(WireMessage message) {
        if ("step".equals(message.action)) {
            if (synchronizedBenchmark.isActive()) {
                showMessage("[RL] Ignoring manual step during synchronized benchmark");
                return;
            }
            int count = Math.max(1, Math.min(message.ticks, 100000));
            TickController.INSTANCE.requestSteps(count);
            showMessage("[RL] Python requested " + count + " integrated-server tick(s)");
            return;
        }

        if ("mode".equals(message.action)) {
            try {
                TickController.Mode mode = TickController.Mode.valueOf(message.mode.toUpperCase());
                if (mode == TickController.Mode.SYNC) throw new IllegalArgumentException();
                if (synchronizedBenchmark.isActive()) synchronizedBenchmark.abort("mode changed from Python", true);
                TickController.INSTANCE.setMode(mode);
                showMessage("[RL] Tick mode: " + mode.name().toLowerCase());
            } catch (IllegalArgumentException ignored) {
                showMessage("[RL] Unknown tick mode from Python: " + message.mode);
            }
        }
    }

    public static void showMessage(String text) {
        Minecraft minecraft = Minecraft.getMinecraft();
        if (minecraft.thePlayer != null && !GraphicsEnvironment.isHeadless()) {
            minecraft.thePlayer.addChatMessage(new ChatComponentText(text));
        } else if (MinecraftRLMod.LOGGER != null) {
            MinecraftRLMod.LOGGER.info(text);
        }
    }
}
