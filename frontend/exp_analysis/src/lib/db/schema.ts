import {
    pgTable,
    serial,
    varchar,
    text,
    integer,
    boolean,
    real,
    json
} from "drizzle-orm/pg-core";


export const evaluationData = pgTable("evaluation_data", {
  id: serial("id").primaryKey(),
  dataset: varchar("dataset", { length: 256 }).default("default"),
  dataset_index: integer("dataset_index"),
  trace_id: varchar("trace_id", { length: 256 }).default("default"),
  trace_url: varchar("trace_url", { length: 256 }).default("default"),
  exp_id: varchar("exp_id", { length: 256 }).default("default"),
  source: varchar("source", { length: 256 }).default(""),
  raw_question: text("raw_question").default(""),
  level: integer("level").default(0),
  augmented_question: text("augmented_question"),
  correct_answer: text("correct_answer"),
  file_name: varchar("file_name", { length: 256 }),
  stage: varchar("stage", { length: 50 }).default("init"),
  response: text("response"),
  time_cost: real("time_cost"),
  trajectory: text("trajectory"),
  trajectories: text("trajectories"),
  extracted_final_answer: text("extracted_final_answer"),
  judged_response: text("judged_response"),
  reasoning: text("reasoning"),
  correct: boolean("correct"),
  confidence: integer("confidence"),
});


export const tracingGenerationData = pgTable("tracing_generation", {
  id: serial("id").primaryKey(),
  trace_id: varchar("trace_id", { length: 256 }).default(""),
  span_id: varchar("span_id", { length: 256 }).default(""),
  input: json("input"),
  output: json("output"),
  model: varchar("model", { length: 256 }).default(""),
  model_configs: json("model_configs"),
  usage: json("usage"),
});


export const tracingToolData = pgTable("tracing_tool", {
  id: serial("id").primaryKey(),
  trace_id: varchar("trace_id", { length: 256 }).default(""),
  span_id: varchar("span_id", { length: 256 }).default(""),
  name: varchar("name", { length: 256 }).default(""),
  input: json("input"),
  output: json("output"),
  mcp_data: json("mcp_data"),
});

export const trajectory = pgTable("trajectory", {
  id: serial("id").primaryKey(),
  trace_id: varchar("trace_id", { length: 256 }).notNull(),
  trace_url: varchar("trace_url", { length: 256 }),
  d_input: text("d_input"),
  d_output: text("d_output"),
  trajectories: text("trajectories"),
  time_cost: real("time_cost"),
});