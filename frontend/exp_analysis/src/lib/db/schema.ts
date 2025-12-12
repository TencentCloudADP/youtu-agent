// db/schema.ts
import * as dotenv from "dotenv";

dotenv.config({ path: "../../.env" });

// 根据环境变量返回对应的 schema
function getSchema() {
  if (process.env.UTU_DB_URL?.startsWith("sqlite:///")) {
    return require("./schema-sqlite");
  } else {
    return require("./schema-postgresql");
  }
}

// 导出表结构
const schema = getSchema();
export const evaluationData = schema.evaluationData;
export const tracingGenerationData = schema.tracingGenerationData;
export const tracingToolData = schema.tracingToolData;
export const trajectory = schema.trajectory;

// 导出类型
export type EvaluationData = typeof evaluationData.$inferSelect;
export type TracingGenerationData = typeof tracingGenerationData.$inferSelect;
export type TracingToolData = typeof tracingToolData.$inferSelect;
export type Trajectory = typeof trajectory.$inferSelect;