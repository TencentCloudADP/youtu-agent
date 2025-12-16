import { db } from "@/lib/db";
import { trajectory } from "@/lib/db/schema";
import { NextResponse } from "next/server";
import { ilike, sql, eq } from "drizzle-orm";

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get('page') || '1');
    const pageSize = parseInt(searchParams.get('pageSize') || '20');
    const traceId = searchParams.get('trace_id');
    const keyword = searchParams.get('keyword');

    // 构建查询条件
    const whereConditions = [];
    if (traceId) {
        // 使用 eq 进行完全匹配
        whereConditions.push(eq(trajectory.trace_id, traceId));
    }
    if (keyword) {
        // 轨迹内容保持模糊匹配
        whereConditions.push(ilike(trajectory.trajectories, `%${keyword}%`));
    }

    const data = await db
        .select({
            id: trajectory.id,
            trace_id: trajectory.trace_id,
            trace_url: trajectory.trace_url,
            d_input: trajectory.d_input,
            d_output: trajectory.d_output,
            trajectories: trajectory.trajectories,
            time_cost: trajectory.time_cost,
        })
        .from(trajectory)
        .where(whereConditions.length ? sql`${whereConditions.reduce((a, b) => sql`${a} and ${b}`)}` : undefined)
        .limit(pageSize)
        .offset((page - 1) * pageSize);

    const totalCountResult = await db
        .select({ count: sql<number>`count(*)` })
        .from(trajectory)
        .where(whereConditions.length ? sql`${whereConditions.reduce((a, b) => sql`${a} and ${b}`)}` : undefined);
    const totalCount = totalCountResult[0]?.count || 0;

    return NextResponse.json({
        data,
        totalCount
    });
}