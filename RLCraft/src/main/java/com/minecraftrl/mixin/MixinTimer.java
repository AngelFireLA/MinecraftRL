package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.util.Timer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Timer.class)
public abstract class MixinTimer {
    @Shadow public int elapsedTicks;
    @Shadow public float renderPartialTicks;
    @Shadow public float elapsedPartialTicks;

    @Inject(method = "updateTimer", at = @At("HEAD"), cancellable = true)
    private void minecraftRl$controlClientTimer(CallbackInfo callback) {
        TickController.Mode mode = TickController.INSTANCE.getMode();
        // STEP must not cancel Minecraft.runTick(): that method also pumps keyboard,
        // mouse, GUI and network input. Only the integrated server is gated in STEP.
        if (mode != TickController.Mode.FAST && mode != TickController.Mode.SYNC) {
            return;
        }

        boolean synchronizedClientTick = mode == TickController.Mode.SYNC
                && TickController.INSTANCE.consumeSynchronizedClientTick();
        elapsedTicks = mode == TickController.Mode.FAST || synchronizedClientTick ? 1 : 0;
        renderPartialTicks = mode == TickController.Mode.SYNC
                ? TickController.INSTANCE.synchronizedRenderPartialTicks(synchronizedClientTick)
                : 0.0F;
        elapsedPartialTicks = renderPartialTicks;
        callback.cancel();
    }
}
