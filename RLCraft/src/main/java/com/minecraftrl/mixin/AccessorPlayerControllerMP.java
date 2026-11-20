package com.minecraftrl.mixin;

import net.minecraft.client.multiplayer.PlayerControllerMP;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(PlayerControllerMP.class)
public interface AccessorPlayerControllerMP {
    @Accessor("curBlockDamageMP")
    float minecraftRl$getBlockBreakProgress();

    @Accessor("blockHitDelay")
    void minecraftRl$setBlockHitDelay(int ticks);
}
