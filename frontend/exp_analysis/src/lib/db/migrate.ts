// db/migrate.ts
import * as dotenv from "dotenv";

dotenv.config({ path: "../../.env" });

if (!process.env.UTU_DB_URL) {
  throw new Error("UTU_DB_URL is not set");
}

const main = async () => {
  const dbUrl = process.env.UTU_DB_URL!;

  if (dbUrl.startsWith("sqlite:///")) {
    // SQLite
    const { drizzle: sqliteDrizzle } = require("drizzle-orm/better-sqlite3");
    const { migrate: sqliteMigrate } = require("drizzle-orm/better-sqlite3/migrator");
    const Database = require("better-sqlite3");

    const dbPath = dbUrl.replace("sqlite:///", "");
    const sqlite = new Database(dbPath);
    const db = sqliteDrizzle(sqlite);

    await sqliteMigrate(db, { migrationsFolder: "drizzle" });
    console.log("SQLite migrations applied successfully");
  } else {
    // PostgreSQL
    const { drizzle: pgDrizzle } = require("drizzle-orm/postgres-js");
    const { migrate: pgMigrate } = require("drizzle-orm/postgres-js/migrator");
    const postgres = require("postgres");

    const connection = postgres(dbUrl, { max: 1 });
    const db = pgDrizzle(connection);
    await pgMigrate(db, { migrationsFolder: "drizzle" });
    console.log("PostgreSQL migrations applied successfully");
  }

  process.exit(0);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});