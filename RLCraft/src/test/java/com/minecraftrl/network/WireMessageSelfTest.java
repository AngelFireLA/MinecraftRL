package com.minecraftrl.network;

import java.util.Arrays;

public final class WireMessageSelfTest {
    public static void main(String[] args) {
        char[] payloadChars = new char[64 * 1024];
        Arrays.fill(payloadChars, 'x');
        String payload = new String(payloadChars);

        WireMessage encoded = WireMessage.benchmarkEcho("round-1", payload);
        WireMessage decoded = WireMessage.parse(encoded.toJson());
        require("benchmark_echo".equals(decoded.type), "wrong type");
        require("round-1".equals(decoded.id), "wrong id");
        require(payload.equals(decoded.payload), "64 KiB payload changed");

        WireMessage escaped = WireMessage.message("quotes: \" slash: \\ newline:\n unicode: é");
        WireMessage escapedDecoded = WireMessage.parse(escaped.toJson());
        require(escaped.text.equals(escapedDecoded.text), "escaped text changed");

        WireMessage action = WireMessage.parse("{\"type\":\"sync_action\",\"sequence\":7,"
                + "\"mouse_dx\":1.25,\"mouse_dy\":-0.5,\"forward\":true,\"attack\":false}");
        require(action.sequence == 7, "wrong action sequence");
        require(action.mouseDx == 1.25D && action.mouseDy == -0.5D, "wrong mouse action");
        require(action.forward && !action.attack, "wrong key action");

        WireMessage environmentAction = WireMessage.parse("{\"type\":\"env_action\",\"sequence\":3,"
                + "\"mouse_dx\":-2.0,\"jump\":true,\"attack\":true}");
        require("env_action".equals(environmentAction.type), "wrong environment action type");
        require(environmentAction.sequence == 3 && environmentAction.mouseDx == -2.0D,
                "wrong environment action fields");
        require(environmentAction.jump && environmentAction.attack,
                "wrong environment action buttons");

        WireMessage extendedAction = WireMessage.parse("{\"type\":\"env_action\","
                + "\"sneak\":true,\"sprint\":true,\"use\":true,\"hotbar_slot\":4}");
        require(extendedAction.sneak && extendedAction.sprint && extendedAction.use,
                "wrong held action fields");
        require(extendedAction.hotbarSlot == 4, "wrong hotbar slot");

        WireMessage environmentReset = WireMessage.parse("{\"type\":\"env_reset\","
                + "\"task\":\"hold_attack_block\",\"seed\":1234567890123,"
                + "\"visualized\":true,\"target_count\":3,\"curriculum_stage\":7,"
                + "\"action_fields\":\"mouse_dx,attack\","
                + "\"observation_fields\":\"player.yaw,raycast.type\","
                + "\"goal_x\":12.5,\"goal_y\":64.0,\"goal_z\":-3.0}");
        require("mouse_dx,attack".equals(environmentReset.actionFields),
                "wrong requested action fields");
        require("hold_attack_block".equals(environmentReset.task)
                && environmentReset.seed == 1234567890123L, "wrong task reset identity");
        require(environmentReset.visualized, "wrong visualization flag");
        require(environmentReset.targetCount == 3, "wrong target count");
        require(environmentReset.curriculumStage == 7, "wrong curriculum stage");
        require("player.yaw,raycast.type".equals(environmentReset.observationFields),
                "wrong requested observation fields");
        require(environmentReset.goalX == 12.5D && environmentReset.goalY == 64.0D
                && environmentReset.goalZ == -3.0D, "wrong task goal");

        WireMessage hello = WireMessage.parse(WireMessage.hello(1, "rl-3", "void").toJson());
        require("hello".equals(hello.type) && hello.protocolVersion == 1,
                "wrong hello protocol version");
        require("rl-3".equals(hello.instanceId) && "void".equals(hello.world),
                "wrong hello identity");

        WireMessage ready = WireMessage.parse(WireMessage.clientReady("rl-3", "void").toJson());
        require("client_ready".equals(ready.type) && "rl-3".equals(ready.instanceId),
                "wrong ready identity");
        System.out.println("WireMessage self-test passed (64 KiB payload)");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
