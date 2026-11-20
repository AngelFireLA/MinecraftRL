package com.minecraftrl.world;

import com.minecraftrl.MinecraftRLMod;
import com.minecraftrl.RuntimeConfig;
import com.minecraftrl.network.TcpBridgeClient;
import com.minecraftrl.network.WireMessage;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiMainMenu;
import net.minecraft.world.storage.WorldInfo;

/** Opens one configured existing single-player save when the main menu becomes ready. */
public final class AutomaticWorldLoader {
    private enum State {
        DISABLED,
        WAITING_FOR_MENU,
        STARTED,
        FAILED
    }

    private final RuntimeConfig config;
    private final TcpBridgeClient bridge;
    private State state;

    public AutomaticWorldLoader(RuntimeConfig config, TcpBridgeClient bridge) {
        this.config = config;
        this.bridge = bridge;
        this.state = config.shouldAutoLoadWorld() ? State.WAITING_FOR_MENU : State.DISABLED;
    }

    public void onClientTick() {
        if (state != State.WAITING_FOR_MENU) return;

        Minecraft minecraft = Minecraft.getMinecraft();
        if (minecraft.theWorld != null || minecraft.isIntegratedServerRunning()) {
            state = State.STARTED;
            return;
        }
        if (!(minecraft.currentScreen instanceof GuiMainMenu)) return;

        String saveDirectory = config.getWorld();
        try {
            WorldInfo worldInfo = minecraft.getSaveLoader().getWorldInfo(saveDirectory);
            if (worldInfo == null || !minecraft.getSaveLoader().canLoadWorld(saveDirectory)) {
                fail("configured world save does not exist or cannot be loaded: " + saveDirectory);
                return;
            }

            state = State.STARTED;
            MinecraftRLMod.LOGGER.info("Instance {} auto-loading world {}",
                    config.getInstanceId(), saveDirectory);
            minecraft.launchIntegratedServer(saveDirectory, worldInfo.getWorldName(), null);
        } catch (RuntimeException error) {
            fail("failed to auto-load world " + saveDirectory + ": " + error.getMessage());
        }
    }

    private void fail(String message) {
        state = State.FAILED;
        MinecraftRLMod.LOGGER.error("Instance {}: {}", config.getInstanceId(), message);
        bridge.sendProtocol(WireMessage.clientError(message));
    }
}
