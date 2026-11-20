package com.minecraftrl.command;

import com.minecraftrl.ClientEvents;
import com.minecraftrl.benchmark.SynchronizedBenchmark;
import com.minecraftrl.network.TcpBridgeClient;
import com.minecraftrl.tick.TickController;
import net.minecraft.client.Minecraft;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.util.BlockPos;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public final class RlCommand extends CommandBase {
    private final TcpBridgeClient bridge;
    private final SynchronizedBenchmark synchronizedBenchmark;
    private final AtomicBoolean benchmarkRunning = new AtomicBoolean();

    public RlCommand(TcpBridgeClient bridge, SynchronizedBenchmark synchronizedBenchmark) {
        this.bridge = bridge;
        this.synchronizedBenchmark = synchronizedBenchmark;
    }

    @Override
    public String getCommandName() {
        return "rl";
    }

    @Override
    public String getCommandUsage(ICommandSender sender) {
        return "/rl <send|status|mode|step|benchmark|syncbench|stats|resetstats>";
    }

    @Override
    public int getRequiredPermissionLevel() {
        return 0;
    }

    @Override
    public void processCommand(ICommandSender sender, String[] args) throws CommandException {
        if (args.length == 0) {
            feedback(getCommandUsage(sender));
            return;
        }

        switch (args[0].toLowerCase()) {
            case "send":
                if (args.length < 2) throw new CommandException("Usage: /rl send <message>");
                boolean queued = bridge.sendMessage(buildString(args, 1));
                feedback(queued ? "Message queued for Python" : "Outbound queue is full");
                break;
            case "status":
                feedback("TCP " + (bridge.isConnected() ? "connected" : "disconnected") + " -> "
                        + bridge.getHost() + ":" + bridge.getPort());
                break;
            case "mode":
                if (args.length != 2) throw new CommandException("Usage: /rl mode <vanilla|fast|step>");
                try {
                    TickController.Mode mode = TickController.Mode.valueOf(args[1].toUpperCase());
                    if (mode == TickController.Mode.SYNC) throw new IllegalArgumentException();
                    if (synchronizedBenchmark.isActive()) synchronizedBenchmark.abort("mode changed", true);
                    TickController.INSTANCE.setMode(mode);
                    feedback("Tick mode: " + args[1].toLowerCase());
                } catch (IllegalArgumentException error) {
                    throw new CommandException("Unknown mode: " + args[1]);
                }
                break;
            case "step":
                if (synchronizedBenchmark.isActive()) {
                    throw new CommandException("Cannot manually step during a synchronized benchmark");
                }
                int count = args.length >= 2 ? parseInt(args[1], 1, 100000) : 1;
                TickController.INSTANCE.requestSteps(count);
                feedback("Released " + count + " integrated-server tick(s); client UI remains active");
                break;
            case "benchmark":
                int seconds = args.length >= 2 ? parseInt(args[1], 1, 60) : 5;
                startBenchmark(seconds);
                break;
            case "syncbench":
                if (benchmarkRunning.get()) {
                    throw new CommandException("The throughput benchmark is already running");
                }
                int steps = args.length >= 2 ? parseInt(args[1], 10, 100000) : 1000;
                try {
                    synchronizedBenchmark.start(steps);
                    feedback("Synchronized RL benchmark arming for " + steps
                            + " steps; waiting for the server tick barrier");
                } catch (Exception error) {
                    throw new CommandException(error.getMessage());
                }
                break;
            case "stats":
                feedback(TickController.INSTANCE.formatStats());
                break;
            case "resetstats":
                TickController.INSTANCE.resetMeasurements();
                feedback("Tick measurements reset");
                break;
            default:
                throw new CommandException(getCommandUsage(sender));
        }
    }

    @Override
    public List<String> addTabCompletionOptions(ICommandSender sender, String[] args, BlockPos pos) {
        if (args.length == 1) {
            return getListOfStringsMatchingLastWord(args, "send", "status", "mode", "step", "benchmark", "syncbench", "stats", "resetstats");
        }
        if (args.length == 2 && "mode".equalsIgnoreCase(args[0])) {
            return getListOfStringsMatchingLastWord(args, "vanilla", "fast", "step");
        }
        return null;
    }

    private static void feedback(String text) {
        ClientEvents.showMessage("[RL] " + text);
    }

    private void startBenchmark(int seconds) throws CommandException {
        if (synchronizedBenchmark.isActive()) {
            throw new CommandException("The synchronized benchmark is already running");
        }
        if (!benchmarkRunning.compareAndSet(false, true)) {
            throw new CommandException("A benchmark is already running");
        }

        TickController controller = TickController.INSTANCE;
        controller.setMode(TickController.Mode.FAST);
        controller.setRenderSuppressed(false);
        controller.resetMeasurements();
        feedback("Benchmark started: " + seconds + "s rendered + " + seconds
                + "s renderless + TCP 100x64KiB (keep VSync off)");
        Thread benchmark = new Thread(() -> {
            TickController.Measurement rendered = null;
            TickController.Measurement renderless = null;
            String networkResult = "TCP benchmark not run";
            String failure = null;
            try {
                Thread.sleep(seconds * 1000L);
                rendered = controller.measurement();

                controller.setRenderSuppressed(true);
                controller.resetMeasurements();
                Thread.sleep(seconds * 1000L);
                renderless = controller.measurement();

                controller.setRenderSuppressed(false);
                controller.setMode(TickController.Mode.VANILLA);
                networkResult = bridge.benchmarkEcho(64 * 1024, 100, 5000L).format();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                failure = "benchmark interrupted";
            } catch (Exception error) {
                failure = error.getMessage();
            } finally {
                controller.setRenderSuppressed(false);
                controller.setMode(TickController.Mode.VANILLA);
                benchmarkRunning.set(false);
            }

            final TickController.Measurement finalRendered = rendered;
            final TickController.Measurement finalRenderless = renderless;
            final String finalNetworkResult = networkResult;
            final String finalFailure = failure;
            Minecraft.getMinecraft().addScheduledTask(() -> {
                feedback("Benchmark complete; mode restored to vanilla");
                if (finalRendered != null) feedback(formatMeasurement("rendered", finalRendered));
                if (finalRenderless != null) feedback(formatMeasurement("renderless", finalRenderless));
                feedback(finalNetworkResult);
                if (finalFailure != null) feedback("Benchmark warning: " + finalFailure);
            });
        }, "MinecraftRL-Benchmark");
        benchmark.setDaemon(true);
        benchmark.start();
    }

    private static String formatMeasurement(String label, TickController.Measurement measurement) {
        return String.format(Locale.ROOT,
                "%s %.2fs: client=%.1f tick/s server=%.1f tick/s",
                label, measurement.seconds,
                measurement.clientTicksPerSecond, measurement.serverTicksPerSecond);
    }
}
