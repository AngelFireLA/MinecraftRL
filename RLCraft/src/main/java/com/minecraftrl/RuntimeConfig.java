package com.minecraftrl;

import java.util.Properties;
import java.util.regex.Pattern;

/** Validated per-process settings supplied by the HeadlessMC instance launcher. */
public final class RuntimeConfig {
    public static final String HOST_PROPERTY = "minecraftrl.host";
    public static final String PORT_PROPERTY = "minecraftrl.port";
    public static final String INSTANCE_ID_PROPERTY = "minecraftrl.instanceId";
    public static final String WORLD_PROPERTY = "minecraftrl.world";

    private static final Pattern INSTANCE_ID_PATTERN =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,63}");

    private final String host;
    private final int port;
    private final String instanceId;
    private final String world;

    private RuntimeConfig(String host, int port, String instanceId, String world) {
        this.host = host;
        this.port = port;
        this.instanceId = instanceId;
        this.world = world;
    }

    public static RuntimeConfig fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    public static RuntimeConfig fromProperties(Properties properties) {
        String host = value(properties, HOST_PROPERTY, "127.0.0.1");
        if (host.isEmpty()) {
            throw new IllegalArgumentException(HOST_PROPERTY + " must not be empty");
        }

        String rawPort = value(properties, PORT_PROPERTY, "8765");
        final int port;
        try {
            port = Integer.parseInt(rawPort);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(PORT_PROPERTY + " must be an integer: " + rawPort);
        }
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException(PORT_PROPERTY + " must be between 1 and 65535");
        }

        String instanceId = value(properties, INSTANCE_ID_PROPERTY, "rl-0");
        if (!INSTANCE_ID_PATTERN.matcher(instanceId).matches()) {
            throw new IllegalArgumentException(INSTANCE_ID_PROPERTY
                    + " must match " + INSTANCE_ID_PATTERN.pattern());
        }

        String world = value(properties, WORLD_PROPERTY, "");
        if (world.length() > 128 || world.equals(".") || world.equals("..")
                || world.indexOf('/') >= 0 || world.indexOf('\\') >= 0) {
            throw new IllegalArgumentException(WORLD_PROPERTY
                    + " must be a single save-directory name of at most 128 characters");
        }
        return new RuntimeConfig(host, port, instanceId, world);
    }

    private static String value(Properties properties, String name, String defaultValue) {
        String value = properties.getProperty(name);
        return value == null ? defaultValue : value.trim();
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

    public String getWorld() {
        return world;
    }

    public boolean shouldAutoLoadWorld() {
        return !world.isEmpty();
    }
}
