package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.client.audio.ISound;
import net.minecraft.client.audio.SoundManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Silences accelerated training while retaining sound during visualized evaluation. */
@Mixin(SoundManager.class)
public abstract class MixinSoundManager {
    @Inject(method = "playSound", at = @At("HEAD"), cancellable = true)
    private void minecraftRl$suppressRenderlessTrainingSound(
            ISound sound, CallbackInfo callback) {
        if (TickController.INSTANCE.shouldSuppressSynchronizedAudio()) {
            callback.cancel();
        }
    }
}
