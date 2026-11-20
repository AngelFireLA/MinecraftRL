package com.minecraftrl.world;

import java.util.concurrent.atomic.AtomicLong;

/** Conservative client-world generation used by reusable observation caches. */
public final class ClientBlockChangeTracker {
    private static final AtomicLong VERSION = new AtomicLong();

    private ClientBlockChangeTracker() {}

    public static void markChanged() {
        VERSION.incrementAndGet();
    }

    public static long version() {
        return VERSION.get();
    }
}
