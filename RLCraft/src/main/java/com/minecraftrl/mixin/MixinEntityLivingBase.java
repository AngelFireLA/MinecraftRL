package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.entity.EntityLivingBase;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Prevents the packet-integration client tick from advancing arm swings twice. */
@Mixin(EntityLivingBase.class)
public abstract class MixinEntityLivingBase {
    @Shadow public float prevSwingProgress;
    @Unique private float minecraftRl$savedPrevSwingProgress;

    @Inject(method = "onEntityUpdate", at = @At("HEAD"))
    private void minecraftRl$saveArmSwingBeforeIntegrationTick(CallbackInfo callback) {
        if (TickController.INSTANCE.shouldSuppressSynchronizedAttackHandling()) {
            minecraftRl$savedPrevSwingProgress = prevSwingProgress;
        }
    }

    @Inject(method = "updateArmSwingProgress", at = @At("HEAD"), cancellable = true)
    private void minecraftRl$freezeArmSwingDuringIntegrationTick(CallbackInfo callback) {
        if (TickController.INSTANCE.shouldSuppressSynchronizedAttackHandling()) {
            prevSwingProgress = minecraftRl$savedPrevSwingProgress;
            callback.cancel();
        }
    }
}
