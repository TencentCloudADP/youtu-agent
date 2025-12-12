import { db } from "@/lib/db";
import { evaluationData } from "@/lib/db/schema";
import { NextResponse } from "next/server";
import { eq, count } from "drizzle-orm";

export async function GET(
    req: Request
) {
  const url = new URL(req.url);
  const pathSegments = url.pathname.split('/');
  const exp_id = pathSegments[pathSegments.length - 2]; // exp_id is before /stats

  const formattedStats: { [key: string]: number } = {
    init: 0,
    rollout: 0,
    judged: 0,
  };

  const stats = await db
      .select({
        stage: evaluationData.stage,
        count: count(evaluationData.stage),
      })
      .from(evaluationData)
      .where(eq(evaluationData.exp_id, exp_id))
      .groupBy(evaluationData.stage);

  stats.forEach((s: { stage: string | null; count: number }) => {
    if (s.stage !== null) { // Ensure stage is not null
      formattedStats[s.stage] = s.count;
    }
  });

  return NextResponse.json(formattedStats);
}