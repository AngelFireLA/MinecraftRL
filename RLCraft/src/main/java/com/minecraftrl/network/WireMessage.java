package com.minecraftrl.network;

import java.util.HashMap;
import java.util.Map;

/** Minimal newline-delimited JSON message used by the Java/Python prototype. */
public final class WireMessage {
    public final String type;
    public final String text;
    public final String action;
    public final String mode;
    public final int ticks;
    public final String id;
    public final String payload;
    public final int protocolVersion;
    public final String instanceId;
    public final String world;
    public final String task;
    public final long seed;
    public final boolean visualized;
    public final int targetCount;
    public final int curriculumStage;
    public final int steps;
    public final int sequence;
    public final double mouseDx;
    public final double mouseDy;
    public final boolean forward;
    public final boolean back;
    public final boolean left;
    public final boolean right;
    public final boolean jump;
    public final boolean attack;
    public final boolean sneak;
    public final boolean sprint;
    public final boolean use;
    public final boolean attackOnce;
    public final boolean useOnce;
    public final int hotbarSlot;
    public final long elapsedNanos;
    public final long observationBytes;
    public final String actionFields;
    public final String observationFields;
    public final double goalX;
    public final double goalY;
    public final double goalZ;

    private WireMessage(String type, String text, String action, String mode, int ticks, String id, String payload,
                        int protocolVersion, String instanceId, String world,
                        String task, long seed, boolean visualized, int targetCount,
                        int curriculumStage,
                        int steps, int sequence, double mouseDx, double mouseDy,
                        boolean forward, boolean back, boolean left, boolean right, boolean jump, boolean attack,
                        boolean sneak, boolean sprint, boolean use, boolean attackOnce,
                        boolean useOnce, int hotbarSlot,
                        long elapsedNanos, long observationBytes,
                        String actionFields, String observationFields,
                        double goalX, double goalY, double goalZ) {
        this.type = type;
        this.text = text;
        this.action = action;
        this.mode = mode;
        this.ticks = ticks;
        this.id = id;
        this.payload = payload;
        this.protocolVersion = protocolVersion;
        this.instanceId = instanceId;
        this.world = world;
        this.task = task;
        this.seed = seed;
        this.visualized = visualized;
        this.targetCount = targetCount;
        this.curriculumStage = curriculumStage;
        this.steps = steps;
        this.sequence = sequence;
        this.mouseDx = mouseDx;
        this.mouseDy = mouseDy;
        this.forward = forward;
        this.back = back;
        this.left = left;
        this.right = right;
        this.jump = jump;
        this.attack = attack;
        this.sneak = sneak;
        this.sprint = sprint;
        this.use = use;
        this.attackOnce = attackOnce;
        this.useOnce = useOnce;
        this.hotbarSlot = hotbarSlot;
        this.elapsedNanos = elapsedNanos;
        this.observationBytes = observationBytes;
        this.actionFields = actionFields;
        this.observationFields = observationFields;
        this.goalX = goalX;
        this.goalY = goalY;
        this.goalZ = goalZ;
    }

    public static WireMessage message(String text) {
        return basic("message", text, "", "", 0, "", "");
    }

    public static WireMessage benchmarkEcho(String id, String payload) {
        return basic("benchmark_echo", "", "", "", 0, id, payload);
    }

    public static WireMessage raw(String json) {
        return basic("_raw", "", "", "", 0, "", json);
    }

    public static WireMessage hello(int protocolVersion, String instanceId, String world) {
        return new WireMessage("hello", "", "", "", 0, "", "",
                protocolVersion, instanceId, world, "", 0L, false, 0, 0,
                0, 0, 0.0D, 0.0D,
                false, false, false, false, false, false,
                false, false, false, false, false, -1, 0L, 0L, "", "",
                Double.NaN, Double.NaN, Double.NaN);
    }

    public static WireMessage clientReady(String instanceId, String world) {
        return new WireMessage("client_ready", "", "", "", 0, "", "",
                0, instanceId, world, "", 0L, false, 0, 0,
                0, 0, 0.0D, 0.0D,
                false, false, false, false, false, false,
                false, false, false, false, false, -1, 0L, 0L, "", "",
                Double.NaN, Double.NaN, Double.NaN);
    }

    public static WireMessage clientError(String text) {
        return basic("client_error", text, "", "", 0, "", "");
    }

    public static WireMessage synchronizedBenchmarkStart(int steps) {
        return new WireMessage("sync_benchmark_start", "", "", "", 0, "", "",
                0, "", "", "", 0L, false, 0, 0,
                steps, 0, 0.0D, 0.0D,
                false, false, false, false, false, false,
                false, false, false, false, false, -1, 0L, 0L, "", "",
                Double.NaN, Double.NaN, Double.NaN);
    }

    private static WireMessage basic(String type, String text, String action, String mode,
                                     int ticks, String id, String payload) {
        return new WireMessage(type, text, action, mode, ticks, id, payload,
                0, "", "", "", 0L, false, 0, 0,
                0, 0, 0.0D, 0.0D,
                false, false, false, false, false, false,
                false, false, false, false, false, -1, 0L, 0L, "", "",
                Double.NaN, Double.NaN, Double.NaN);
    }

    public static WireMessage parse(String json) {
        Map<String, String> fields = parseFlatObject(json);
        String type = value(fields, "type");
        if (type.isEmpty()) {
            throw new IllegalArgumentException("Missing message type");
        }
        return new WireMessage(type, value(fields, "text"), value(fields, "action"),
                value(fields, "mode"), intField(fields, "ticks", 1),
                value(fields, "id"), value(fields, "payload"),
                intField(fields, "protocol_version", 0), value(fields, "instance_id"),
                value(fields, "world"), value(fields, "task"),
                longField(fields, "seed", 0L), booleanField(fields, "visualized"),
                intField(fields, "target_count", 0),
                intField(fields, "curriculum_stage", 0),
                intField(fields, "steps", 0), intField(fields, "sequence", 0),
                doubleField(fields, "mouse_dx", 0.0D), doubleField(fields, "mouse_dy", 0.0D),
                booleanField(fields, "forward"), booleanField(fields, "back"),
                booleanField(fields, "left"), booleanField(fields, "right"),
                booleanField(fields, "jump"), booleanField(fields, "attack"),
                booleanField(fields, "sneak"), booleanField(fields, "sprint"),
                booleanField(fields, "use"), booleanField(fields, "attack_once"),
                booleanField(fields, "use_once"), intField(fields, "hotbar_slot", -1),
                longField(fields, "elapsed_nanos", 0L), longField(fields, "observation_bytes", 0L),
                value(fields, "action_fields"), value(fields, "observation_fields"),
                doubleField(fields, "goal_x", Double.NaN),
                doubleField(fields, "goal_y", Double.NaN),
                doubleField(fields, "goal_z", Double.NaN));
    }

    public String toJson() {
        if ("hello".equals(type)) {
            return "{\"type\":\"hello\",\"protocol_version\":" + protocolVersion
                    + ",\"instance_id\":\"" + escape(instanceId)
                    + "\",\"world\":\"" + escape(world) + "\"}";
        }
        if ("client_ready".equals(type)) {
            return "{\"type\":\"client_ready\",\"instance_id\":\""
                    + escape(instanceId) + "\",\"world\":\"" + escape(world) + "\"}";
        }
        if ("benchmark_echo".equals(type)) {
            return "{\"type\":\"benchmark_echo\",\"id\":\"" + escape(id)
                    + "\",\"payload\":\"" + escape(payload) + "\"}";
        }
        if ("sync_benchmark_start".equals(type)) {
            return "{\"type\":\"sync_benchmark_start\",\"steps\":" + steps + "}";
        }
        if ("_raw".equals(type)) {
            return payload;
        }
        return "{\"type\":\"" + escape(type) + "\",\"text\":\"" + escape(text) + "\"}";
    }

    private static String value(Map<String, String> fields, String name) {
        String value = fields.get(name);
        return value == null ? "" : value;
    }

    private static int intField(Map<String, String> fields, String name, int defaultValue) {
        String value = fields.get(name);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static long longField(Map<String, String> fields, String name, long defaultValue) {
        String value = fields.get(name);
        if (value == null) return defaultValue;
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static double doubleField(Map<String, String> fields, String name, double defaultValue) {
        String value = fields.get(name);
        if (value == null) return defaultValue;
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static boolean booleanField(Map<String, String> fields, String name) {
        return Boolean.parseBoolean(value(fields, name));
    }

    /** Parses the flat JSON objects used by this protocol without regex recursion. */
    private static Map<String, String> parseFlatObject(String json) {
        Map<String, String> fields = new HashMap<>();
        int index = skipWhitespace(json, 0);
        if (index >= json.length() || json.charAt(index++) != '{') {
            throw new IllegalArgumentException("Expected JSON object");
        }

        while (true) {
            index = skipWhitespace(json, index);
            if (index >= json.length()) throw new IllegalArgumentException("Unterminated JSON object");
            if (json.charAt(index) == '}') return fields;

            ParsedString key = parseString(json, index);
            index = skipWhitespace(json, key.nextIndex);
            if (index >= json.length() || json.charAt(index++) != ':') {
                throw new IllegalArgumentException("Expected ':' after JSON field");
            }
            index = skipWhitespace(json, index);

            String fieldValue;
            if (index < json.length() && json.charAt(index) == '"') {
                ParsedString parsedValue = parseString(json, index);
                fieldValue = parsedValue.value;
                index = parsedValue.nextIndex;
            } else {
                int valueStart = index;
                while (index < json.length() && json.charAt(index) != ',' && json.charAt(index) != '}') index++;
                fieldValue = json.substring(valueStart, index).trim();
            }
            fields.put(key.value, fieldValue);

            index = skipWhitespace(json, index);
            if (index >= json.length()) throw new IllegalArgumentException("Unterminated JSON object");
            char separator = json.charAt(index++);
            if (separator == '}') return fields;
            if (separator != ',') throw new IllegalArgumentException("Expected ',' between JSON fields");
        }
    }

    private static ParsedString parseString(String json, int start) {
        if (start >= json.length() || json.charAt(start) != '"') {
            throw new IllegalArgumentException("Expected JSON string");
        }
        StringBuilder result = new StringBuilder();
        for (int index = start + 1; index < json.length(); index++) {
            char c = json.charAt(index);
            if (c == '"') return new ParsedString(result.toString(), index + 1);
            if (c != '\\') {
                result.append(c);
                continue;
            }
            if (++index >= json.length()) throw new IllegalArgumentException("Incomplete JSON escape");
            char escaped = json.charAt(index);
            switch (escaped) {
                case '"': result.append('"'); break;
                case '\\': result.append('\\'); break;
                case '/': result.append('/'); break;
                case 'b': result.append('\b'); break;
                case 'f': result.append('\f'); break;
                case 'n': result.append('\n'); break;
                case 'r': result.append('\r'); break;
                case 't': result.append('\t'); break;
                case 'u':
                    if (index + 4 >= json.length()) throw new IllegalArgumentException("Incomplete unicode escape");
                    try {
                        result.append((char) Integer.parseInt(json.substring(index + 1, index + 5), 16));
                    } catch (NumberFormatException error) {
                        throw new IllegalArgumentException("Invalid unicode escape");
                    }
                    index += 4;
                    break;
                default: throw new IllegalArgumentException("Invalid JSON escape");
            }
        }
        throw new IllegalArgumentException("Unterminated JSON string");
    }

    private static int skipWhitespace(String json, int index) {
        while (index < json.length() && Character.isWhitespace(json.charAt(index))) index++;
        return index;
    }

    private static final class ParsedString {
        private final String value;
        private final int nextIndex;

        private ParsedString(String value, int nextIndex) {
            this.value = value;
            this.nextIndex = nextIndex;
        }
    }

    private static String escape(String value) {
        StringBuilder result = new StringBuilder(value.length() + 16);
        for (int index = 0; index < value.length(); index++) {
            char c = value.charAt(index);
            switch (c) {
                case '\\': result.append("\\\\"); break;
                case '"': result.append("\\\""); break;
                case '\n': result.append("\\n"); break;
                case '\r': result.append("\\r"); break;
                case '\t': result.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        result.append(String.format("\\u%04x", (int) c));
                    } else {
                        result.append(c);
                    }
            }
        }
        return result.toString();
    }

    public static String quote(String value) {
        return "\"" + escape(value == null ? "" : value) + "\"";
    }

}
