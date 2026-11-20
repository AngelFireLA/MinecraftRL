package com.minecraftrl;

import com.minecraftrl.benchmark.SynchronizedBenchmark;
import com.minecraftrl.command.RlCommand;
import com.minecraftrl.network.TcpBridgeClient;
import com.minecraftrl.tick.TickController;
import net.minecraftforge.client.ClientCommandHandler;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStoppingEvent;
import org.apache.logging.log4j.Logger;

@Mod(modid = MinecraftRLMod.MOD_ID, name = "MinecraftRL", version = "1.0.0", clientSideOnly = true)
public final class MinecraftRLMod {
    public static final String MOD_ID = "minecraftrl";
    public static Logger LOGGER;

    private RuntimeConfig config;
    private TcpBridgeClient bridge;

    @Mod.EventHandler
    public void preInit(FMLPreInitializationEvent event) {
        LOGGER = event.getModLog();
        config = RuntimeConfig.fromSystemProperties();
        bridge = new TcpBridgeClient(config.getHost(), config.getPort(),
                config.getInstanceId(), config.getWorld());
    }

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        SynchronizedBenchmark synchronizedBenchmark = new SynchronizedBenchmark(bridge);
        ClientEvents events = new ClientEvents(bridge, synchronizedBenchmark, config);
        MinecraftForge.EVENT_BUS.register(events);
        net.minecraftforge.fml.common.FMLCommonHandler.instance().bus().register(events);
        ClientCommandHandler.instance.registerCommand(new RlCommand(bridge, synchronizedBenchmark));
        bridge.start();
        LOGGER.info("MinecraftRL ready; instance={}, TCP={}:{}, auto-world={}",
                config.getInstanceId(), bridge.getHost(), bridge.getPort(),
                config.shouldAutoLoadWorld() ? config.getWorld() : "disabled");
    }

    @Mod.EventHandler
    public void serverStopping(FMLServerStoppingEvent event) {
        TickController.INSTANCE.setMode(TickController.Mode.VANILLA);
    }
}
