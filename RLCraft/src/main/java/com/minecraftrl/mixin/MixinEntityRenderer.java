package com.minecraftrl.mixin;

import com.minecraftrl.tick.TickController;
import net.minecraft.client.renderer.EntityRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Skips the complete world/GUI render pass during the renderless benchmark phase. */
@Mixin(EntityRenderer.class)
public abstract class MixinEntityRenderer {
    @Inject(method = "updateCameraAndRender", at = @At("HEAD"), cancellable = true)
    private void minecraftRl$skipRendering(float partialTicks, long nanoTime, CallbackInfo callback) {
        if (TickController.INSTANCE.isRenderSuppressed()) {
            // Keep vanilla targeting current even though no OpenGL frame is drawn.
            // SYNC performs this at its precise pre-action/post-server barriers.
            if (TickController.INSTANCE.getMode() != TickController.Mode.SYNC) {
                ((EntityRenderer) (Object) this).getMouseOver(partialTicks);
            }
            callback.cancel();
        }
    }
}
