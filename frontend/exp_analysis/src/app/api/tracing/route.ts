import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { eq } from "drizzle-orm";
import {
    tracingGenerationData,
    tracingToolData,
    type TracingGenerationData,
    type TracingToolData
} from "@/lib/db/schema";

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const trace_id = searchParams.get("trace_id");

    if (!trace_id) {
        return NextResponse.json({ error: "Missing trace_id" }, { status: 400 });
    }

    try {
        // 查询两张表
        const generationData: TracingGenerationData[] = await db
            .select()
            .from(tracingGenerationData)
            .where(eq(tracingGenerationData.trace_id, trace_id))
            .execute();

        const toolData: TracingToolData[] = await db
            .select()
            .from(tracingToolData)
            .where(eq(tracingToolData.trace_id, trace_id))
            .execute();

        // 格式化数据并分别排序
        const generations = generationData
            .map(item => ({
                source: "tracing_generation" as const,
                id: item.id,
                trace_id: item.trace_id,
                span_id: item.span_id,
                model: item.model,
                input: item.input,
                output: item.output,
                model_configs: item.model_configs,
                usage: item.usage,
            }))
            .sort((a, b) => {
                const numA = Number(a.id);
                const numB = Number(b.id);
                if (!isNaN(numA) && !isNaN(numB)) {
                    return numA - numB;
                }
                return String(a.id).localeCompare(String(b.id));
            });

        const tools = toolData
            .map(item => ({
                source: "tracing_tool" as const,
                id: item.id,
                span_id: item.span_id,
                name: item.name,
                input: item.input,
                output: item.output,
                mcp_data: item.mcp_data,
            }))
            .sort((a, b) => {
                const numA = Number(a.id);
                const numB = Number(b.id);
                if (!isNaN(numA) && !isNaN(numB)) {
                    return numA - numB;
                }
                return String(a.id).localeCompare(String(b.id));
            });

        return NextResponse.json({ generations, tools });
    } catch (error) {
        console.error("Failed to fetch tracing data:", error);
        return NextResponse.json({ error: "Database error" }, { status: 500 });
    }
}