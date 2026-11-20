package com.minecraftrl;

import java.util.Properties;

public final class RuntimeConfigSelfTest {
    public static void main(String[] args) {
        RuntimeConfig defaults = RuntimeConfig.fromProperties(new Properties());
        require("127.0.0.1".equals(defaults.getHost()), "wrong default host");
        require(defaults.getPort() == 8765, "wrong default port");
        require("rl-0".equals(defaults.getInstanceId()), "wrong default instance id");
        require(!defaults.shouldAutoLoadWorld(), "default unexpectedly auto-loads a world");

        Properties configured = new Properties();
        configured.setProperty(RuntimeConfig.HOST_PROPERTY, "localhost");
        configured.setProperty(RuntimeConfig.PORT_PROPERTY, "8768");
        configured.setProperty(RuntimeConfig.INSTANCE_ID_PROPERTY, "rl-3");
        configured.setProperty(RuntimeConfig.WORLD_PROPERTY, "void");
        RuntimeConfig parsed = RuntimeConfig.fromProperties(configured);
        require(parsed.getPort() == 8768, "wrong configured port");
        require("rl-3".equals(parsed.getInstanceId()), "wrong configured instance id");
        require("void".equals(parsed.getWorld()) && parsed.shouldAutoLoadWorld(),
                "wrong configured world");

        expectInvalid(RuntimeConfig.PORT_PROPERTY, "0");
        expectInvalid(RuntimeConfig.PORT_PROPERTY, "not-a-port");
        expectInvalid(RuntimeConfig.INSTANCE_ID_PROPERTY, "rl/3");
        expectInvalid(RuntimeConfig.WORLD_PROPERTY, "../void");
        System.out.println("RuntimeConfig self-test passed");
    }

    private static void expectInvalid(String property, String value) {
        Properties properties = new Properties();
        properties.setProperty(property, value);
        try {
            RuntimeConfig.fromProperties(properties);
            throw new AssertionError("accepted invalid " + property + "=" + value);
        } catch (IllegalArgumentException expected) {
            // Expected.
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
