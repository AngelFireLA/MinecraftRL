package com.minecraftrl.tick;

public final class TickControllerSelfTest {
    public static void main(String[] args) throws Exception {
        System.setProperty("minecraftrl.idleWait", "event");
        TickController controller = TickController.INSTANCE;
        controller.setMode(TickController.Mode.SYNC);
        Thread server = new Thread(controller::awaitServerStep, "tick-controller-self-test");
        try {
            require(controller.isRenderSuppressed(), "SYNC rendering was enabled by default");
            require(controller.shouldSuppressSynchronizedAudio(),
                    "renderless SYNC audio was enabled by default");
            controller.setSynchronizedRenderingEnabled(true);
            require(!controller.isRenderSuppressed(), "visualized SYNC still suppressed rendering");
            require(!controller.shouldSuppressSynchronizedAudio(),
                    "visualized SYNC still suppressed audio");
            controller.setSynchronizedRenderingEnabled(false);
            require(controller.isRenderSuppressed(), "renderless SYNC did not suppress rendering");
            require(controller.shouldSuppressSynchronizedAudio(),
                    "renderless SYNC did not suppress audio");
            require(!controller.consumeSynchronizedClientTick(),
                    "client tick ran without a synchronized permit");
            controller.requestSynchronizedClientTick();
            require(controller.consumeSynchronizedClientTick(),
                    "client tick permit did not release one tick");
            require(!controller.consumeSynchronizedClientTick(),
                    "one client permit released more than one tick");

            long generation = controller.workGeneration();
            controller.signalWork();
            require(controller.workGeneration() != generation, "work signal was lost");
            controller.awaitIdleWork(generation);
            Thread idleClient = new Thread(() -> controller.awaitIdleWork(controller.workGeneration()));
            idleClient.start();
            controller.signalWork();
            idleClient.join(1000L);
            require(!idleClient.isAlive(), "idle client did not wake within its maintenance bound");

            server.start();
            long deadline = System.nanoTime() + 1_000_000_000L;
            while (!controller.isServerWaitingAtGate() && System.nanoTime() < deadline) {
                Thread.yield();
            }
            require(controller.isServerWaitingAtGate(), "server never reported reaching the gate");
            require(server.isAlive(), "server passed a gate without a permit");

            controller.requestSynchronizedTick();
            server.join(1_000L);
            require(!server.isAlive(), "one synchronized permit did not release one server tick");
            require(!controller.isServerWaitingAtGate(), "gate remained marked as waiting after release");
            System.out.println("TickController self-test passed (park, permit, release)");
        } finally {
            controller.setMode(TickController.Mode.VANILLA);
            server.join(1_000L);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
