import { db } from "@/lib/db";
import { evaluationData } from "@/lib/db/schema";
import { NextResponse } from "next/server";
import { desc } from "drizzle-orm";

export async function GET() {
    const data = await db
        .select({
            exp_id: evaluationData.exp_id,
        })
        .from(evaluationData)
        .groupBy(evaluationData.exp_id)
        .orderBy(desc(evaluationData.exp_id));

    const expIds = data.map((d: { exp_id: string }) => d.exp_id);
    return NextResponse.json(expIds);
}