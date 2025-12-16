export interface TextDeltaContent {
  type: 'reason' | 'tool_call' | 'tool_call_output' | 'text';
  delta: string;
  callid?: string;
  argument?: string;
  inprogress?: boolean;
}

export interface ExampleContent {
  query: string;
}

export interface PlanItem {
  analysis: string;
  todo: string[];
}

export interface WorkerItem {
  task: string;
  output: string;
}

export interface ReportItem {
  output: string;
}

export interface NewAgentContent {
  name: string;
}

export interface InitContent {
  type: 'init';
  default_agent: string;
  agent_type: "simple" | "orchestra" | "other";
  sub_agents: string[] | null;
}

export interface PlanItemOrchestrator {
  type: 'plan';
  analysis: string;
  tasks: string[];
}

export interface TaskItemOrchestrator {
  type: 'task';
  agent_name: string;
  task: string;
  is_reporter: boolean;
  report: string | null;
}

export type OrchestraContent =
  | { type: 'plan'; item: PlanItem }
  | { type: 'worker'; item: WorkerItem }
  | { type: 'report'; item: ReportItem }
  | { type: 'plan_start'; item: null }
  | { type: 'report_start'; item: null };

export type OrchestratorContent =
  | { type: 'orchestrator'; sub_type: 'plan.start'; item: null }
  | { type: 'orchestrator'; sub_type: 'plan.done'; item: PlanItemOrchestrator }
  | { type: 'orchestrator'; sub_type: 'task.start'; item: TaskItemOrchestrator | null }
  | { type: 'orchestrator'; sub_type: 'task.done'; item: TaskItemOrchestrator | null}

export interface ListAgentsContent {
  type: 'list_agents';
  agents: string[];
}

export interface SwitchAgentContent {
  type: 'switch_agent';
  ok: boolean;
  name: string;
  agent_type: "simple" | "orchestra" | "other";
  sub_agents: string[] | null;
}

export interface AskContent {
  type: 'ask';
  question: string;
  ask_id: string;
}

export interface GeneratedAgentContent {
  type: 'generated_agent_config';
  filename: string;
  config_content: string;
}

export interface ErrorContent {
  type: 'error';
  message: string;
}

export interface Event {
  type: 'raw' | 'orchestra' | 'orchestrator' | 'finish' | 'example' | 'new' | 'init' | 'list_agents' | 'switch_agent' | 'gen_agent' | 'ask' | 'generated_agent_config' | 'error';
  data: TextDeltaContent | OrchestraContent | OrchestratorContent | ExampleContent | NewAgentContent | InitContent | ListAgentsContent | SwitchAgentContent | AskContent | GeneratedAgentContent | ErrorContent | null;
  requireConfirm?: boolean;
}
