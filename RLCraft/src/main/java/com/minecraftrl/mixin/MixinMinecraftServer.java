package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftServer.class)
public abstract class MixinMinecraftServer {
    @Redirect(method = "run", at = @At(value = "INVOKE", target = "Lnet/minecraft/server/MinecraftServer;getCurrentTimeMillis()J"))
    private long minecraftRl$useControllableClock() {
        return TickController.INSTANCE.serverTimeMillis();
    }

    @Redirect(method = "run", at = @At(value = "INVOKE", target = "Ljava/lang/Thread;sleep(J)V"))
    private void minecraftRl$controlServerSleep(long delayMillis) throws InterruptedException {
        TickController.INSTANCE.serverSleep(delayMillis);
    }

    @Inject(method = "tick", at = @At("HEAD"))
    private void minecraftRl$beforeServerTick(CallbackInfo callback) {
        if (TickController.INSTANCE.isServerGated()) {
            TickController.INSTANCE.awaitServerStep();
        }
    }

    @Inject(method = "tick", at = @At("RETURN"))
    private void minecraftRl$afterServerTick(CallbackInfo callback) {
        TickController.INSTANCE.recordServerTick();
    }
}
