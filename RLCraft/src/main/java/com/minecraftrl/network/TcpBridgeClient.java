package com.minecraftrl.network;

import com.minecraftrl.MinecraftRLMod;
import com.minecraftrl.tick.TickController;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

/** A reconnecting TCP client whose blocking I/O always runs outside Minecraft's threads. */
public final class TcpBridgeClient {
    public static final int PROTOCOL_VERSION = 1;

    private final String host;
    private final int port;
    private final String instanceId;
    private final String world;
    private final BlockingQueue<WireMessage> incoming = new ArrayBlockingQueue<>(1024);
    private final BlockingQueue<WireMessage> outgoing = new ArrayBlockingQueue<>(1024);
    private final ConcurrentHashMap<String, CompletableFuture<Long>> echoReplies = new ConcurrentHashMap<>();
    private final AtomicLong echoIds = new AtomicLong();
    private final AtomicLong connectionEpoch = new AtomicLong();
    private volatile boolean running;
    private volatile boolean connected;
    private volatile Socket socket;

    public TcpBridgeClient(String host, int port) {
        this(host, port, "rl-0", "");
    }

    public TcpBridgeClient(String host, int port, String instanceId, String world) {
        this.host = host;
        this.port = port;
        this.instanceId = instanceId;
        this.world = world;
    }

    public void start() {
        if (running) return;
        running = true;
        Thread worker = new Thread(this::runWorker, "MinecraftRL-TCP");
        worker.setDaemon(true);
        worker.start();
    }

    public void stop() {
        running = false;
        closeSocket();
    }

    public boolean sendMessage(String text) {
        return outgoing.offer(WireMessage.message(text));
    }

    public boolean sendProtocol(WireMessage message) {
        return outgoing.offer(message);
    }

    public boolean sendRawJson(String json) {
        return outgoing.offer(WireMessage.raw(json));
    }

    public boolean hasIncoming() { return !incoming.isEmpty(); }

    public WireMessage pollIncoming() {
        return incoming.poll();
    }

    public boolean isConnected() {
        return connected;
    }

    /** Changes for every successful TCP connection, including reconnects. */
    public long getConnectionEpoch() {
        return connectionEpoch.get();
    }

    public String getHost() {
        return host;
    }

    public int getPort() {
        return port;
    }

    public String getInstanceId() {
        return instanceId;
    }

    /** Runs sequential request/reply exchanges without blocking a Minecraft thread. */
    public NetworkBenchmark benchmarkEcho(int payloadBytes, int rounds, long timeoutMillis)
            throws IOException, InterruptedException {
        if (!connected) {
            throw new IOException("Python bridge is not connected");
        }
        char[] payloadChars = new char[payloadBytes];
        Arrays.fill(payloadChars, 'x');
        String payload = new String(payloadChars);
        long[] durations = new long[rounds];
        long benchmarkStarted = System.nanoTime();

        for (int round = 0; round < rounds; round++) {
            String id = Long.toString(echoIds.incrementAndGet());
            CompletableFuture<Long> reply = new CompletableFuture<>();
            echoReplies.put(id, reply);
            long started = System.nanoTime();
            if (!outgoing.offer(WireMessage.benchmarkEcho(id, payload))) {
                echoReplies.remove(id);
                throw new IOException("TCP outbound queue is full");
            }
            try {
                durations[round] = reply.get(timeoutMillis, TimeUnit.MILLISECONDS) - started;
            } catch (ExecutionException error) {
                throw new IOException("TCP connection failed during benchmark", error.getCause());
            } catch (TimeoutException timeout) {
                throw new IOException("Timed out waiting for Python echo reply");
            } finally {
                echoReplies.remove(id);
            }
        }
        return new NetworkBenchmark(payloadBytes, rounds, durations, System.nanoTime() - benchmarkStarted);
    }

    private void runWorker() {
        while (running) {
            try {
                connectAndPump();
            } catch (IOException error) {
                connected = false;
                if (MinecraftRLMod.LOGGER != null) {
                    MinecraftRLMod.LOGGER.debug("TCP bridge unavailable: {}", error.getMessage());
                }
            } finally {
                connected = false;
                closeSocket();
            }

            if (running) {
                try {
                    TimeUnit.SECONDS.sleep(1L);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private void connectAndPump() throws IOException {
        Socket newSocket = new Socket();
        socket = newSocket;
        newSocket.setTcpNoDelay(true);
        newSocket.connect(new InetSocketAddress(host, port), 1000);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(newSocket.getInputStream(), StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(newSocket.getOutputStream(), StandardCharsets.UTF_8))) {
            // Identity must be the first line so Python never binds the wrong game
            // process to an environment merely because it connected first.
            writer.write(WireMessage.hello(PROTOCOL_VERSION, instanceId, world).toJson());
            writer.newLine();
            writer.flush();
            connectionEpoch.incrementAndGet();
            connected = true;
            MinecraftRLMod.LOGGER.info("TCP bridge connected: instance={} target={}:{}",
                    instanceId, host, port);

            Thread writerThread = new Thread(() -> writeLoop(newSocket, writer), "MinecraftRL-TCP-Writer");
            writerThread.setDaemon(true);
            writerThread.start();
            while (running && !newSocket.isClosed()) {
                try {
                    String line = reader.readLine();
                    if (line == null) return;
                    if (line.length() <= 1_048_576) {
                        WireMessage message = WireMessage.parse(line);
                        if ("benchmark_echo_reply".equals(message.type)) {
                            CompletableFuture<Long> reply = echoReplies.get(message.id);
                            if (reply != null && message.payload.length() <= 1_048_576) {
                                reply.complete(System.nanoTime());
                            }
                        } else {
                            incoming.offer(message);
                            TickController.INSTANCE.signalWork();
                        }
                    }
                } catch (IllegalArgumentException malformed) {
                    MinecraftRLMod.LOGGER.warn("Ignoring malformed TCP message: {}", malformed.getMessage());
                }
            }
        }
    }

    private void writeLoop(Socket targetSocket, BufferedWriter writer) {
        try {
            while (running && connected && socket == targetSocket && !targetSocket.isClosed()) {
                WireMessage message = outgoing.poll(100L, TimeUnit.MILLISECONDS);
                if (message != null) {
                    writer.write(message.toJson());
                    writer.newLine();
                    writer.flush();
                }
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        } catch (IOException error) {
            try {
                targetSocket.close();
            } catch (IOException ignored) {
            }
        }
    }

    private void closeSocket() {
        Socket current = socket;
        socket = null;
        if (current != null) {
            try {
                current.close();
            } catch (IOException ignored) {
            }
        }
        IOException disconnected = new IOException("TCP bridge disconnected");
        echoReplies.forEach((id, reply) -> reply.completeExceptionally(disconnected));
        echoReplies.clear();
        // Protocol messages from a dead connection belong to the dead RL session.
        // Replaying an action or observation after reconnect would break causality.
        TickController.INSTANCE.signalWork();
        incoming.clear();
        outgoing.clear();
    }

    public static final class NetworkBenchmark {
        public final int payloadBytes;
        public final int rounds;
        public final double averageRoundTripMillis;
        public final double p95RoundTripMillis;
        public final double payloadMebibytesPerSecond;

        private NetworkBenchmark(int payloadBytes, int rounds, long[] durations, long totalNanos) {
            this.payloadBytes = payloadBytes;
            this.rounds = rounds;
            long sum = 0L;
            for (long duration : durations) sum += duration;
            long[] sorted = durations.clone();
            Arrays.sort(sorted);
            this.averageRoundTripMillis = sum / (double) rounds / 1_000_000.0D;
            this.p95RoundTripMillis = sorted[Math.min(rounds - 1, (int) Math.ceil(rounds * 0.95D) - 1)] / 1_000_000.0D;
            double transferredBytes = (double) payloadBytes * 2.0D * rounds;
            this.payloadMebibytesPerSecond = transferredBytes / (1024.0D * 1024.0D)
                    / (totalNanos / 1_000_000_000.0D);
        }

        public String format() {
            return String.format(Locale.ROOT,
                    "TCP %d x %dKiB round trips: avg=%.3fms p95=%.3fms payload=%.1fMiB/s",
                    rounds, payloadBytes / 1024, averageRoundTripMillis,
                    p95RoundTripMillis, payloadMebibytesPerSecond);
        }
    }
}
