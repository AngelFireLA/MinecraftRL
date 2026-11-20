package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.server.integrated.IntegratedServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(IntegratedServer.class)
public abstract class MixinIntegratedServer {
    @Inject(method = "tick", at = @At("RETURN"))
    private void minecraftRl$afterCompleteIntegratedTick(CallbackInfo callback) {
        TickController.INSTANCE.markIntegratedServerTickComplete();
    }
}
