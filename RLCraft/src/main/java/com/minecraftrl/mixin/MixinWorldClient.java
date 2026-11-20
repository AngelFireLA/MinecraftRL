package com.minecraftrl.mixin;

import com.minecraftrl.world.ClientBlockChangeTracker;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.util.BlockPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Invalidates cached block volumes whenever the client applies an authoritative change. */
@Mixin(WorldClient.class)
public abstract class MixinWorldClient {
    @Inject(method = "invalidateRegionAndSetBlock", at = @At("HEAD"))
    private void minecraftRl$markBlockChange(BlockPos pos, IBlockState state,
                                              CallbackInfoReturnable<Boolean> callback) {
        ClientBlockChangeTracker.markChanged();
    }
}
