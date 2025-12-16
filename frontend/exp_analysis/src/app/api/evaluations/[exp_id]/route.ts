import { db } from "@/lib/db";
import { evaluationData } from "@/lib/db/schema";
import { NextResponse } from "next/server";
import { eq, count, and, sql, asc, desc, Column } from "drizzle-orm";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const pathSegments = url.pathname.split('/');
  const exp_id_from_path = pathSegments[pathSegments.length - 1];

  const exp_id = exp_id_from_path === 'all' ? null : exp_id_from_path;

  const searchParams = url.searchParams;

  const page = searchParams.get('page') ? parseInt(searchParams.get('page')!, 10) : 1;
  const pageSize = searchParams.get('pageSize') ? parseInt(searchParams.get('pageSize')!, 10) : 20;
  const offset = (page - 1) * pageSize;

  const orderBy = searchParams.get('order') || 'dataset_index';
  const orderDirection = searchParams.get('orderDirection') || 'asc';

  const keyword = searchParams.get('keyword') || '';
  const tools = searchParams.get('tools') || '';
  const trace_id = searchParams.get('trace_id') || '';
  const correct = searchParams.get('correct') || '';

  if (exp_id === null && !trace_id) {
    return NextResponse.json({
      data: [],
      totalCount: 0
    });
  }

  const buildConditions = () => {
    const conditions = [];

    if (exp_id) {
      conditions.push(eq(evaluationData.exp_id, exp_id));
    }

    if (keyword) {
      conditions.push(sql`
        (LOWER(${evaluationData.raw_question}) LIKE ${'%' + keyword.toLowerCase() + '%'} OR
         LOWER(${evaluationData.response}) LIKE ${'%' + keyword.toLowerCase() + '%'} OR
         LOWER(${evaluationData.correct_answer}) LIKE ${'%' + keyword.toLowerCase() + '%'})
      `);
    }

    if (tools) {
      conditions.push(sql`
        LOWER(${evaluationData.trajectories}) LIKE ${'%' + tools.toLowerCase() + '%'}
      `);
    }

    if (trace_id) {
      conditions.push(eq(evaluationData.trace_id, trace_id));
    }

    if (correct && correct !== 'all') {
      const isCorrect = correct === 'true';
      conditions.push(eq(evaluationData.correct, isCorrect));
    }

    return conditions;
  };

  const validOrderFields: Record<string, Column> = {
    id: evaluationData.id,
    trace_id: evaluationData.trace_id,
    exp_id: evaluationData.exp_id,
    source: evaluationData.source,
    raw_question: evaluationData.raw_question,
    level: evaluationData.level,
    augmented_question: evaluationData.augmented_question,
    correct_answer: evaluationData.correct_answer,
    file_name: evaluationData.file_name,
    stage: evaluationData.stage,
    response: evaluationData.response,
    time_cost: evaluationData.time_cost,
    trajectory: evaluationData.trajectory,
    extracted_final_answer: evaluationData.extracted_final_answer,
    judged_response: evaluationData.judged_response,
    reasoning: evaluationData.reasoning,
    correct: evaluationData.correct,
    confidence: evaluationData.confidence,
    dataset_index: evaluationData.dataset_index
  };

  // 默认排序字段
  const defaultOrderField = evaluationData.dataset_index;

  // 获取排序字段
  const orderColumn = validOrderFields[orderBy] || defaultOrderField;

  // 设置排序方向
  const orderFn = orderDirection === 'asc' ? asc : desc;

  // 构建基础查询的条件
  const conditions = buildConditions();

  // 构建基础查询
  const baseQuery = db
      .select()
      .from(evaluationData)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(orderFn(orderColumn));

  // 添加分页
  const data = await baseQuery.limit(pageSize).offset(offset);

  // 获取总数 - 使用相同的条件
  const totalCountResult = await db
      .select({ count: count() })
      .from(evaluationData)
      .where(conditions.length > 0 ? and(...conditions) : undefined);

  const totalCount = totalCountResult[0]?.count || 0;

  return NextResponse.json({
    data,
    totalCount
  });
}