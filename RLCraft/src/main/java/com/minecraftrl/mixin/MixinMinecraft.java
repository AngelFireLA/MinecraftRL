package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.client.Minecraft;
import org.lwjgl.opengl.Display;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public abstract class MixinMinecraft {
    @Redirect(
            method = "updateDisplay",
            at = @At(value = "INVOKE", target = "Lorg/lwjgl/opengl/Display;update()V"))
    private void minecraftRl$skipHeadlessDisplayYieldDuringTraining() {
        TickController controller = TickController.INSTANCE;
        if (controller.getMode() != TickController.Mode.SYNC
                || !controller.isRenderSuppressed()) {
            Display.update();
        }
    }

    @Inject(method = "sendClickBlockToController", at = @At("HEAD"), cancellable = true)
    private void minecraftRl$suppressAttackDuringIntegrationTick(
            boolean leftClick, CallbackInfo callback) {
        if (TickController.INSTANCE.shouldSuppressSynchronizedAttackHandling()) {
            callback.cancel();
        }
    }
}
