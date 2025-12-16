// db/schema-sqlite.ts
import {
    sqliteTable,
    integer,
    text,
    real
} from "drizzle-orm/sqlite-core";

export const evaluationData = sqliteTable("evaluation_data", {
    id: integer("id").primaryKey({ autoIncrement: true }),
    dataset: text("dataset").default("default"),
    dataset_index: integer("dataset_index"),
    trace_id: text("trace_id").default("default"),
    trace_url: text("trace_url").default("default"),
    exp_id: text("exp_id").default("default"),
    source: text("source").default(""),
    raw_question: text("raw_question").default(""),
    level: integer("level").default(0),
    augmented_question: text("augmented_question"),
    correct_answer: text("correct_answer"),
    file_name: text("file_name"),
    stage: text("stage").default("init"),
    response: text("response"),
    time_cost: real("time_cost"),
    trajectory: text("trajectory"),
    trajectories: text("trajectories"),
    extracted_final_answer: text("extracted_final_answer"),
    judged_response: text("judged_response"),
    reasoning: text("reasoning"),
    correct: integer("correct", { mode: "boolean" }),
    confidence: integer("confidence"),
});

export const tracingGenerationData = sqliteTable("tracing_generation", {
    id: integer("id").primaryKey({ autoIncrement: true }),
    trace_id: text("trace_id").default(""),
    span_id: text("span_id").default(""),
    input: text("input", { mode: "json" }),
    output: text("output", { mode: "json" }),
    model: text("model").default(""),
    model_configs: text("model_configs", { mode: "json" }),
    usage: text("usage", { mode: "json" }),
});

export const tracingToolData = sqliteTable("tracing_tool", {
    id: integer("id").primaryKey({ autoIncrement: true }),
    trace_id: text("trace_id").default(""),
    span_id: text("span_id").default(""),
    name: text("name").default(""),
    input: text("input", { mode: "json" }),
    output: text("output", { mode: "json" }),
    mcp_data: text("mcp_data", { mode: "json" }),
});

export const trajectory = sqliteTable("trajectory", {
    id: integer("id").primaryKey({ autoIncrement: true }),
    trace_id: text("trace_id").notNull(),
    trace_url: text("trace_url"),
    d_input: text("d_input"),
    d_output: text("d_output"),
    trajectories: text("trajectories"),
    time_cost: real("time_cost"),
});

// 导出类型
export type EvaluationData = typeof evaluationData.$inferSelect;
export type TracingGenerationData = typeof tracingGenerationData.$inferSelect;
export type TracingToolData = typeof tracingToolData.$inferSelect;
export type Trajectory = typeof trajectory.$inferSelect;