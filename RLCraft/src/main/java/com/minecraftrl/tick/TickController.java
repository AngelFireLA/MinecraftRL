package com.minecraftrl.tick;

import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

/** Coordinates the experimental client and integrated-server tick policies. */
public final class TickController {
    public enum Mode {
        VANILLA,
        FAST,
        STEP,
        SYNC
    }

    public static final TickController INSTANCE = new TickController();

    private final AtomicLong serverPermits = new AtomicLong();
    private final AtomicLong synchronizedClientPermits = new AtomicLong();
    private final AtomicLong clientTicks = new AtomicLong();
    private final AtomicLong serverTicks = new AtomicLong();
    private final AtomicLong completedServerTicks = new AtomicLong();
    private final Object permitMonitor = new Object();
    private final Object workMonitor = new Object();
    private long workGeneration;
    private final boolean eventIdleWait = "event".equals(System.getProperty("minecraftrl.idleWait", "off"));

    public long workGeneration() {
        synchronized (workMonitor) { return workGeneration; }
    }

    public void signalWork() {
        synchronized (workMonitor) {
            workGeneration++;
            workMonitor.notifyAll();
        }
    }

    /** Signals are remembered, including a command arriving just before wait(). */
    public void awaitIdleWork(long observedGeneration) {
        if (!eventIdleWait || mode != Mode.SYNC || !isRenderSuppressed()) return;
        synchronized (workMonitor) {
            if (observedGeneration != workGeneration || mode != Mode.SYNC) return;
            try { workMonitor.wait(10L); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        }
    }

    private volatile Mode mode = Mode.VANILLA;
    private volatile boolean renderSuppressed;
    private volatile boolean serverWaitingAtGate;
    private volatile boolean suppressSynchronizedAttackHandling;
    private volatile boolean synchronizedRenderingEnabled;
    private volatile long synchronizedClientTickNanos = System.nanoTime();
    private volatile long measurementStartedNanos = System.nanoTime();
    private long virtualServerTime = System.currentTimeMillis();
    private long lastWallTime = virtualServerTime;

    private TickController() {
    }

    public Mode getMode() {
        return mode;
    }

    public void setMode(Mode newMode) {
        if (newMode == null || newMode == mode) {
            return;
        }
        mode = newMode;
        signalWork();
        if (newMode != Mode.FAST) {
            renderSuppressed = false;
        }
        if (newMode != Mode.SYNC) {
            synchronizedRenderingEnabled = false;
        }
        serverPermits.set(0L);
        synchronizedClientPermits.set(0L);
        suppressSynchronizedAttackHandling = false;
        resetMeasurements();
        synchronized (permitMonitor) {
            permitMonitor.notifyAll();
        }
    }

    public void requestSteps(int count) {
        if (count <= 0) {
            return;
        }
        if (mode != Mode.STEP) {
            setMode(Mode.STEP);
        }
        serverPermits.addAndGet(count);
        synchronized (permitMonitor) {
            permitMonitor.notifyAll();
        }
    }

    public void requestSynchronizedTick() {
        if (mode != Mode.SYNC) {
            throw new IllegalStateException("Synchronized tick requested outside SYNC mode");
        }
        serverPermits.incrementAndGet();
        synchronized (permitMonitor) {
            permitMonitor.notifyAll();
        }
    }

    public void requestSynchronizedClientTick() {
        if (mode != Mode.SYNC) {
            throw new IllegalStateException("Synchronized client tick requested outside SYNC mode");
        }
        synchronizedClientPermits.incrementAndGet();
    }

    public boolean consumeSynchronizedClientTick() {
        return mode == Mode.SYNC && consume(synchronizedClientPermits);
    }

    public void setSuppressSynchronizedAttackHandling(boolean suppress) {
        suppressSynchronizedAttackHandling = suppress && mode == Mode.SYNC;
    }

    public boolean shouldSuppressSynchronizedAttackHandling() {
        return mode == Mode.SYNC && suppressSynchronizedAttackHandling;
    }

    /** Allows frames during a synchronized evaluation without ungating simulation ticks. */
    public void setSynchronizedRenderingEnabled(boolean enabled) {
        synchronizedRenderingEnabled = enabled && mode == Mode.SYNC;
        synchronizedClientTickNanos = System.nanoTime();
    }

    /** Wall-clock interpolation only; it never grants an extra simulation tick. */
    public float synchronizedRenderPartialTicks(boolean clientTickStarted) {
        long now = System.nanoTime();
        if (clientTickStarted) {
            synchronizedClientTickNanos = now;
        }
        if (mode != Mode.SYNC || !synchronizedRenderingEnabled) {
            return 0.0F;
        }
        double partial = (now - synchronizedClientTickNanos) / 50_000_000.0D;
        return (float) Math.max(0.0D, Math.min(1.0D, partial));
    }

    public void awaitServerStep() {
        try {
            while (isServerGated() && !consume(serverPermits)) {
                synchronized (permitMonitor) {
                    if (isServerGated() && serverPermits.get() == 0L) {
                        serverWaitingAtGate = true;
                        permitMonitor.notifyAll();
                        try {
                            permitMonitor.wait(100L);
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                }
            }
        } finally {
            serverWaitingAtGate = false;
        }
    }

    /** True only after the integrated-server thread has reached its gated tick head. */
    public boolean isServerWaitingAtGate() {
        return serverWaitingAtGate;
    }

    public boolean isServerGated() {
        return mode == Mode.STEP || mode == Mode.SYNC;
    }

    private boolean consume(AtomicLong permits) {
        long available;
        do {
            available = permits.get();
            if (available <= 0L) {
                return false;
            }
        } while (!permits.compareAndSet(available, available - 1L));
        return true;
    }

    /**
     * Supplies a monotonic clock to the vanilla server loop. FAST/STEP advance one
     * logical 50 ms interval per loop; VANILLA advances by real elapsed time.
     */
    public synchronized long serverTimeMillis() {
        long wallNow = System.currentTimeMillis();
        if (mode == Mode.VANILLA) {
            virtualServerTime += Math.max(0L, wallNow - lastWallTime);
        } else {
            virtualServerTime += 50L;
        }
        lastWallTime = wallNow;
        return virtualServerTime;
    }

    public void serverSleep(long vanillaDelayMillis) throws InterruptedException {
        if (mode == Mode.VANILLA) {
            Thread.sleep(vanillaDelayMillis);
        }
        // FAST intentionally runs flat-out. STEP and SYNC do not need a courtesy
        // yield here either: the next server tick blocks in awaitServerStep() when
        // no permit is available. Thread.yield() is especially expensive on Windows,
        // where four instances can each lose a full scheduler quantum per RL step.
    }

    public void recordClientTick() {
        clientTicks.incrementAndGet();
    }

    public void recordServerTick() {
        serverTicks.incrementAndGet();
    }

    public void markIntegratedServerTickComplete() {
        completedServerTicks.incrementAndGet();
    }

    public long getCompletedServerTicks() {
        return completedServerTicks.get();
    }

    public void setRenderSuppressed(boolean suppressed) {
        renderSuppressed = suppressed;
    }

    public boolean isRenderSuppressed() {
        return (mode == Mode.SYNC && !synchronizedRenderingEnabled)
                || (renderSuppressed && mode == Mode.FAST);
    }

    public boolean shouldSuppressSynchronizedAudio() {
        return mode == Mode.SYNC && !synchronizedRenderingEnabled;
    }

    public void resetMeasurements() {
        clientTicks.set(0L);
        serverTicks.set(0L);
        measurementStartedNanos = System.nanoTime();
    }

    public String formatStats() {
        Measurement measurement = measurement();
        return String.format(Locale.ROOT,
                "mode=%s elapsed=%.2fs client=%d (%.1f tick/s) server=%d (%.1f tick/s) serverPermits=%d",
                mode.name().toLowerCase(Locale.ROOT), measurement.seconds,
                measurement.clientTicks, measurement.clientTicksPerSecond,
                measurement.serverTicks, measurement.serverTicksPerSecond,
                serverPermits.get());
    }

    public Measurement measurement() {
        double seconds = Math.max(0.001D, (System.nanoTime() - measurementStartedNanos) / 1_000_000_000.0D);
        long clients = clientTicks.get();
        long servers = serverTicks.get();
        return new Measurement(seconds, clients, servers);
    }

    public static final class Measurement {
        public final double seconds;
        public final long clientTicks;
        public final long serverTicks;
        public final double clientTicksPerSecond;
        public final double serverTicksPerSecond;

        private Measurement(double seconds, long clientTicks, long serverTicks) {
            this.seconds = seconds;
            this.clientTicks = clientTicks;
            this.serverTicks = serverTicks;
            this.clientTicksPerSecond = clientTicks / seconds;
            this.serverTicksPerSecond = serverTicks / seconds;
        }
    }
}
