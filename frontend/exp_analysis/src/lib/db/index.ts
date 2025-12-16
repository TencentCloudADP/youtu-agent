// db/index.ts
import * as schema from "./schema";
import * as dotenv from "dotenv";

dotenv.config({ path: "../../.env" });

if (!process.env.UTU_DB_URL) {
  throw new Error("UTU_DB_URL is not set");
}

const dbUrl = process.env.UTU_DB_URL;

export const db = (() => {
  if (dbUrl.startsWith("sqlite:///")) {
    // 导入 SQLite
    const { drizzle: sqliteDrizzle } = require("drizzle-orm/better-sqlite3");
    const Database = require("better-sqlite3");

    const dbPath = dbUrl.replace("sqlite:///", "");
    const sqlite = new Database(dbPath);

    return sqliteDrizzle(sqlite, { schema });
  } else {
    // 导入 PostgreSQL
    const { drizzle: pgDrizzle } = require("drizzle-orm/postgres-js");
    const postgres = require("postgres");

    const client = postgres(dbUrl);
    return pgDrizzle(client, { schema });
  }
})();
