"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import {
    Card,
    Button,
    Tabs,
    Switch,
    Space,
    ConfigProvider,
    Collapse
} from "antd";
import { BulbOutlined, BulbFilled } from "@ant-design/icons";
import dynamic from 'next/dynamic';
import ReactMarkdown from "react-markdown";

// 动态导入 ReactJson 以避免 SSR 问题
const ReactJson = dynamic(() => import('react-json-view'), { ssr: false });

// 检查是否在浏览器环境中
const isBrowser = typeof window !== 'undefined';

type Evaluation = {
    id: number;
    dataset: string;
    dataset_index: number;
    trace_id: string;
    trace_url: string;
    exp_id: string;
    source: string;
    raw_question: string;
    level: number | null;
    augmented_question: string | null;
    correct_answer: string | null;
    file_name: string | null;
    stage: string;
    response: string | null;
    time_cost: number | null;
    trajectory: string | null;
    trajectories: string | null;
    extracted_final_answer: string | null;
    judged_response: string | null;
    reasoning: string | null;
    correct: boolean | null;
    confidence: number | null;
};

type TrajectoryEntry = {
    role: string;
    content?: string;
    tool_calls?: ToolCallDetail[];
    tool_call_id?: string;
    usage?: unknown;
};

type ToolCallDetail = {
    id: string;
    type?: string;
    function?: {
        name: string;
        arguments?: string;
    };
};

type RoleTrajectory = {
    agent: string;
    trajectory: TrajectoryEntry[];
};

type TracingData = {
    id: number;
    span_id: string;
    trace_id: string;
    trace_url: string;
    source: "tracing_generation" | "tracing_tool";
    model?: string;
    name?: string;
    input: Record<string, unknown> | null;
    output?: Record<string, unknown> | null;
    model_configs?: Record<string, unknown> | null;
    usage?: Record<string, unknown> | null;
    mcp_data?: Record<string, unknown> | null;
};

function EvaluationDetailPage() {
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const { id } = params;
    const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [parsedTrajectory, setParsedTrajectory] = useState<RoleTrajectory[] | null>(null);
    const [viewMode, setViewMode] = useState<"formatted" | "json">("formatted");
    const [isDarkMode, setIsDarkMode] = useState(false);

    const [generations, setGenerations] = useState<TracingData[]>([]);
    const [tools, setTools] = useState<TracingData[]>([]);

    const [tracingError, setTracingError] = useState<string | null>(null);

    const [selectedSpan, setSelectedSpan] = useState<TracingData | null>(null);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [tracingLoading, setTracingLoading] = useState(false);

    const [expandedChildKeys, setExpandedChildKeys] = useState<Record<string, boolean>>({});
    const [isJsonExpanded, setIsJsonExpanded] = useState(false);
    const [activeTab, setActiveTab] = useState<string>("general");

    useEffect(() => {
        if (isBrowser) {
            const savedTheme = localStorage.getItem("theme");
            const initialDarkMode = savedTheme === "dark";
            setIsDarkMode(initialDarkMode);
            document.body.className = initialDarkMode ? "dark-mode" : "light-mode";
        }
    }, []);

    useEffect(() => {
        if (isBrowser) {
            localStorage.setItem("theme", isDarkMode ? "dark" : "light");
            document.body.className = isDarkMode ? "dark-mode" : "light-mode";
        }
    }, [isDarkMode]);

    useEffect(() => {
        if (id) {
            fetch(`/api/evaluation/${id}`)
                .then((res) => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then(async (data) => {
                    setEvaluation(data);
                    setLoading(false);

                    if (data.trajectories) {
                        try {
                            const trajectory: RoleTrajectory[] = JSON.parse(data.trajectories);
                            setParsedTrajectory(trajectory);
                        } catch (err) {
                            console.error("Failed to parse trajectories JSON:", err);
                            if (data.trajectory) {
                                try {
                                    const oldTrajectory: TrajectoryEntry[] = JSON.parse(data.trajectory);
                                    setParsedTrajectory([
                                        {
                                            agent: "Null",
                                            trajectory: oldTrajectory,
                                        },
                                    ]);
                                } catch (err) {
                                    console.error("Failed to parse trajectory JSON:", err);
                                    setParsedTrajectory(null);
                                }
                            } else {
                                setParsedTrajectory(null);
                            }
                        }
                    } else if (data.trajectory) {
                        try {
                            const oldTrajectory: TrajectoryEntry[] = JSON.parse(data.trajectory);
                            setParsedTrajectory([
                                {
                                    agent: "Null",
                                    trajectory: oldTrajectory,
                                },
                            ]);
                        } catch (err) {
                            console.error("Failed to parse trajectory JSON:", err);
                            setParsedTrajectory(null);
                        }
                    } else {
                        setParsedTrajectory(null);
                    }

                    if (data.trace_id) {
                        setTracingLoading(true);
                        setTracingError(null); // 重置错误状态

                        fetch(`/api/tracing?trace_id=${data.trace_id}`)
                            .then(async (tracingRes) => {
                                if (!tracingRes.ok) {
                                    throw new Error(`Tracing API error: ${tracingRes.status}`);
                                }
                                const tracingData = await tracingRes.json();
                                setGenerations(tracingData.generations || []);
                                setTools(tracingData.tools || []);
                            })
                            .catch((err) => {
                                console.error("Failed to fetch tracing data:", err);
                                setTracingError(err.message || "Failed to load tracing data");
                            })
                            .finally(() => {
                                setTracingLoading(false);
                            });
                        }
                    }
                )
                .catch((e) => {
                    setError(e.message);
                    setLoading(false);
                });
        }
    }, [id]);

    if (loading) {
        return <div className="container mx-auto p-4">Loading...</div>;
    }

    if (error) {
        return (
            <div className="container mx-auto p-4 text-red-500">Error: {error}</div>
        );
    }

    if (!evaluation) {
        return <div className="container mx-auto p-4">Evaluation not found.</div>;
    }

    const renderFormattedTrajectory = () => {
        if (!parsedTrajectory) {
            return (
                <p className="text-gray-500 dark:text-gray-400">
                    No trajectory data available.
                </p>
            );
        }

        return (
            <Collapse
                accordion={false}
                defaultActiveKey={parsedTrajectory.map((_, index) => index)}
            >
                {parsedTrajectory.map((roleTraj, roleIndex) => {
                    const roleKey = roleIndex.toString();
                    const childKeys = roleTraj.trajectory.map((_, idx) => `${roleKey}-${idx}`);
                    const isAllExpanded = childKeys.every(key => expandedChildKeys[key]);

                    return (
                        <Collapse.Panel
                            key={roleIndex}
                            header={
                                <div className="flex justify-between items-center">
                                    <span>{`${roleIndex + 1}. ${roleTraj.agent.toUpperCase()}`}</span>
                                    <Button
                                        size="small"
                                        type="link"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            const newChildKeys = {...expandedChildKeys};
                                            childKeys.forEach(key => {
                                                newChildKeys[key] = !isAllExpanded;
                                            });
                                            setExpandedChildKeys(newChildKeys);
                                        }}
                                    >
                                        {isAllExpanded ? "全部折叠" : "全部展开"}
                                    </Button>
                                </div>
                            }
                            className={`mb-2 ${
                                isDarkMode
                                    ? "bg-[#3c3d32] text-[#f8f8f2]"
                                    : "bg-white text-gray-800"
                            }`}
                        >
                            <Collapse
                                accordion={false}
                                activeKey={childKeys.filter(key => expandedChildKeys[key])}
                                onChange={(keys) => {
                                    const newKeys = Array.isArray(keys) ? keys : [keys];
                                    const newState = {...expandedChildKeys};

                                    childKeys.forEach(key => {
                                        newState[key] = newKeys.includes(key);
                                    });

                                    setExpandedChildKeys(newState);
                                }}
                            >
                                {roleTraj.trajectory.map((entry, entryIndex) => {
                                    const childKey = `${roleKey}-${entryIndex}`;

                                    return (
                                        <Collapse.Panel
                                            key={childKey}
                                            header={`${roleIndex + 1}.${entryIndex + 1}. ${entry.role.toUpperCase()}`}
                                            className={`mb-2 ${
                                                isDarkMode
                                                    ? "bg-[#272822] text-[#f8f8f2]"
                                                    : "bg-gray-100 text-gray-800"
                                            }`}
                                            forceRender={true}
                                        >
                                            {/* 子项内容保持不变 */}
                                            {entry.content && (
                                                <div className="mb-3">
                                                    <div className="text-sm font-semibold text-gray-500 dark:text-gray-400">
                                                        Content:
                                                    </div>
                                                    <pre
                                                        className={`whitespace-pre-wrap p-2 rounded mt-1 text-sm ${
                                                            isDarkMode
                                                                ? "bg-[#1e1f1c] text-[#f8f8f2]"
                                                                : "bg-gray-50 text-gray-800"
                                                        }`}
                                                    >
                                                      {entry.content}
                                                    </pre>
                                                </div>
                                            )}

                                            {entry.role === "assistant" && entry.tool_calls && (
                                                <div className="mb-3">
                                                    <div className="text-sm font-semibold text-gray-500 dark:text-gray-400">
                                                        Tool Calls:
                                                    </div>
                                                    <div className="space-y-2 mt-1">
                                                        {entry.tool_calls.map((tool, toolIndex) => {
                                                            const name = tool.function?.name || "Unknown Tool";
                                                            const args = tool.function?.arguments || "";
                                                            return (
                                                                <div
                                                                    key={`${roleIndex}-${entryIndex}-${toolIndex}`}
                                                                    className={`p-2 rounded ${
                                                                        isDarkMode
                                                                            ? "bg-[#1e1f1c] text-[#f8f8f2]"
                                                                            : "bg-gray-50 text-gray-800"
                                                                    }`}
                                                                >
                                                                    <div className="font-medium">ID: {tool.id}</div>
                                                                    <div>Name: {name}</div>
                                                                    {args && <div>Arguments: {args}</div>}
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}

                                            {entry.role === "tool" && entry.tool_call_id && (
                                                <div>
                                                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                        Call ID: {entry.tool_call_id}
                                                    </div>
                                                </div>
                                            )}
                                        </Collapse.Panel>
                                    );
                                })}
                            </Collapse>
                        </Collapse.Panel>
                    );
                })}
            </Collapse>
        );
    };


    const renderJsonTrajectory = () => {
        if (!parsedTrajectory) {
            return (
                <p className="text-gray-500 dark:text-gray-400">
                    No trajectory data available.
                </p>
            );
        }

        return (
            <div>
                {/* 添加展开/收起控制按钮 */}
                <div className="flex justify-end mb-2">
                    <Button
                        size="small"
                        type="primary"
                        onClick={() => setIsJsonExpanded(!isJsonExpanded)}
                    >
                        {isJsonExpanded ? "一键收起" : "一键展开"}
                    </Button>
                </div>

                <div className="react-json-view">
                    <ReactJson
                        src={parsedTrajectory}
                        theme={isDarkMode ? "monokai" : "rjv-default"}
                        name="trajectories"
                        collapsed={isJsonExpanded ? false : 3} // 根据状态控制展开/收起
                        enableClipboard={true}
                        displayObjectSize={true}
                        displayDataTypes={false}
                        style={{
                            padding: "1rem",
                            borderRadius: "0.5rem",
                            backgroundColor: isDarkMode ? "#272822" : "#ffffff",
                            height: "100%",
                            overflow: "auto",
                        }}
                        onEdit={false}
                        onAdd={false}
                        onDelete={false}
                    />
                </div>
            </div>
        );
    };

    const renderTracingContent = () => (
        <div className="h-full space-y-6">
            {/* Generations 部分 */}
            <div>
                <h3 className={`text-lg font-bold mb-2 ${isDarkMode ? 'text-[#66d9ef]' : 'text-blue-600'}`}>
                    Generations
                </h3>

                {tracingLoading ? (
                    <p className="text-gray-500 dark:text-gray-400">
                        Loading generation data...
                    </p>
                ) : tracingError ? (
                    <p className="text-red-500 dark:text-red-400">
                        Error: {tracingError}
                    </p>
                ) : generations.length === 0 ? (
                    <p className="text-gray-500 dark:text-gray-400">
                        No generation data found
                    </p>
                ) : (
                    <div className="space-y-2">
                        {generations.map((item, index) => (
                            <div
                                key={`gen-${item.id}`}
                                className={`flex items-center py-1 hover:bg-opacity-30 rounded px-2 cursor-pointer 
                                ${
                                    isDarkMode
                                        ? "hover:bg-gray-700 text-[#f8f8f2]"
                                        : "hover:bg-gray-200 text-gray-800"
                                }`}
                                onClick={() => {
                                    setSelectedSpan(item);
                                    setIsDialogOpen(true);
                                }}
                            >
                                <div className="text-sm font-medium min-w-[30px]">
                                    {index + 1}.
                                </div>
                                <div className="flex-grow">
                                    <div className="text-sm">
                                        {item.source === "tracing_generation" ? item.model : "Unknown"}
                                        :{" "}
                                        <span
                                            className={`font-mono ml-1 ${
                                                isDarkMode ? "text-blue-300" : "text-blue-500"
                                            } underline`}
                                        >
                                        {item.span_id}
                                    </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Tools 部分 */}
            <div>
                <h3 className={`text-lg font-bold mb-2 ${isDarkMode ? 'text-[#66d9ef]' : 'text-blue-600'}`}>
                    Tools
                </h3>

                {tracingLoading ? (
                    <p className="text-gray-500 dark:text-gray-400">
                        Loading tool data...
                    </p>
                ) : tracingError ? (
                    <p className="text-red-500 dark:text-red-400">
                        Error: {tracingError}
                    </p>
                ) : tools.length === 0 ? (
                    <p className="text-gray-500 dark:text-gray-400">
                        No tool data found
                    </p>
                ) : (
                    <div className="space-y-2">
                        {tools.map((item, index) => (
                            <div
                                key={`tool-${item.id}`}
                                className={`flex items-center py-1 hover:bg-opacity-30 rounded px-2 cursor-pointer 
                                ${
                                    isDarkMode
                                        ? "hover:bg-gray-700 text-[#f8f8f2]"
                                        : "hover:bg-gray-200 text-gray-800"
                                }`}
                                onClick={() => {
                                    setSelectedSpan(item);
                                    setIsDialogOpen(true);
                                }}
                            >
                                <div className="text-sm font-medium min-w-[30px]">
                                    {index + 1}.
                                </div>
                                <div className="flex-grow">
                                    <div className="text-sm">
                                        {item.source === "tracing_tool" ? item.name : "Unknown"}
                                        :{" "}
                                        <span
                                            className={`font-mono ml-1 ${
                                                isDarkMode ? "text-blue-300" : "text-blue-500"
                                            } underline`}
                                        >
                                        {item.span_id}
                                    </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );



    const renderGeneralInfo = () => {
        const correctTag = (
            <div className="mb-6">
        <span
            className={`inline-block px-4 py-2 rounded-full text-base font-bold ${
                evaluation.correct === true
                    ? "bg-green-500 text-white"
                    : evaluation.correct === false
                        ? "bg-red-500 text-white"
                        : "bg-gray-300 text-gray-700"
            }`}
        >
          {evaluation.correct === true
              ? "✓ Correct"
              : evaluation.correct === false
                  ? "✗ Incorrect"
                  : "Unknown Status"}
        </span>
            </div>
        );

        const InfoBlock: React.FC<{
            title: string;
            items: Array<{
                label: string;
                value: string | number | boolean | null;
                render?: (value: string | number | boolean | null) => React.ReactNode;
            }>;
            darkMode?: boolean;
        }> = ({title, items, darkMode = false}) => (
            <div
                className={`rounded-lg border mb-6 ${
                    darkMode ? "border-gray-600" : "border-gray-200"
                }`}
            >
                <div
                    className={`px-4 py-2 font-bold border-b ${
                        darkMode
                            ? "border-gray-600 bg-gray-700 text-gray-200"
                            : "border-gray-200 bg-gray-100 text-gray-800"
                    }`}
                >
                    {title}
                </div>
                <div className="p-3">
                    {items.map((item, index) => (
                        <div key={index} className="grid grid-cols-4 gap-4 mb-3 last:mb-0">
                            <div
                                className={`text-sm font-medium self-start ${
                                    darkMode ? "text-gray-400" : "text-gray-500"
                                }`}
                            >
                                {item.label}:
                            </div>
                            <div className="col-span-3 text-sm break-words">
                                {item.render
                                    ? item.render(item.value)
                                    : item.value != null
                                        ? item.value.toString()
                                        : "-"}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );

        const basicInfoData = [
            {
                label: "Experiment ID",
                value: evaluation.exp_id,
            },
            {
                label: "Trace ID",
                value: evaluation.trace_id,
            },
            {
                label: "Trace URL",
                value: evaluation.trace_url,
                render: (value: string | number | boolean | null) =>
                    value ? (
                        <a
                            href={value.toString()}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={`max-w-full break-all ${
                                isDarkMode
                                    ? "text-blue-300 hover:text-blue-400"
                                    : "text-blue-600 hover:text-blue-800"
                            }`}
                        >
                            {value}
                        </a>
                    ) : (
                        "-"
                    ),
            },
        ];

        const dataInfoData = [
            {
                label: "Dataset",
                value: evaluation.dataset,
            },
            {
                label: "Dataset Index",
                value: evaluation.dataset_index,
            },
            {
                label: "Source",
                value: evaluation.source,
            },
            {
                label: "Raw Question",
                value: evaluation.raw_question,
            },
            {
                label: "Correct Answer",
                value: evaluation.correct_answer,
            },
            {
                label: "File Name",
                value: evaluation.file_name,
            },
        ];

        const rolloutData = [
            {
                label: "time_cost",
                value: evaluation.time_cost?.toString() ?? "-",
            },
            {
                label: "Response",
                value: evaluation.response,
                render: (value: string | number | boolean | null) =>
                    value ? (
                        <div
                            className={`markdown-container p-2 rounded ${
                                isDarkMode ? "bg-[#272822]" : "bg-gray-100"
                            }`}
                        >
                            <ReactMarkdown>{value.toString()}</ReactMarkdown>
                        </div>
                    ) : (
                        "-"
                    ),
            },
        ];

        const judgeInfoData = [
            {
                label: "Correct",
                value: evaluation.correct?.toString() ?? "-",
            },
            {
                label: "Judged Response",
                value: evaluation.judged_response,
                render: (value: string | number | boolean | null) =>
                    value ? (
                        <div
                            className={`markdown-container p-2 rounded ${
                                isDarkMode ? "bg-[#272822]" : "bg-gray-100"
                            }`}
                        >
                            <ReactMarkdown>{value.toString()}</ReactMarkdown>
                        </div>
                    ) : (
                        "-"
                    ),
            },
        ];

        return (
            <div>
                {correctTag}
                <InfoBlock
                    title="Basic Information"
                    items={basicInfoData}
                    darkMode={isDarkMode}
                />

                <InfoBlock title="Data" items={dataInfoData} darkMode={isDarkMode}/>

                <InfoBlock
                    title="Rollout"
                    items={rolloutData}
                    darkMode={isDarkMode}
                />
                <InfoBlock
                    title="Judge"
                    items={judgeInfoData}
                    darkMode={isDarkMode}
                />
            </div>
        );
    };

    const renderJsonDialog = () => (
        <div
            className={`fixed inset-0 flex items-center justify-center z-50 ${
                isDarkMode ? "bg-[#272822] bg-opacity-70" : "bg-gray-200 bg-opacity-70"
            }`}
        >
            <div
                className={`rounded-lg p-6 w-4/5 max-h-[90vh] overflow-auto shadow-2xl ${
                    isDarkMode
                        ? "bg-[#272822] text-[#f8f8f2] border border-[#49483e]"
                        : "bg-white text-gray-800 border border-gray-300"
                }`}
            >
                <div
                    className={`flex justify-between items-center mb-4 p-3 rounded ${
                        isDarkMode
                            ? "bg-[#3c3d32] border-b border-[#49483e]"
                            : "bg-gray-100 border-b border-gray-200"
                    }`}
                >
                    <h2 className="text-xl font-bold">
                        {selectedSpan?.source === "tracing_generation" ? (
                            <>
                                <span className="text-blue-300">Model:</span> {selectedSpan?.model}
                                <br/>
                                <span className="text-blue-300">Span ID:</span>{" "}
                                {selectedSpan?.span_id}
                            </>
                        ) : (
                            <>
                                <span className="text-blue-300">Tool:</span> {selectedSpan?.name}
                                <br/>
                                <span className="text-blue-300">Span ID:</span>{" "}
                                {selectedSpan?.span_id}
                            </>
                        )}
                    </h2>
                    <Button
                        type={isDarkMode ? "default" : "primary"}
                        onClick={() => setIsDialogOpen(false)}
                    >
                        Close
                    </Button>
                </div>

                {selectedSpan && (
                    <div className="space-y-6">
                        <div>
                            <h3
                                className={`text-md font-bold mb-2 p-2 rounded ${
                                    isDarkMode
                                        ? "bg-[#3c3d32] text-[#66d9ef]"
                                        : "bg-blue-50 text-black"
                                }`}
                            >
                                Input:
                            </h3>
                            <div
                                className={`rounded p-1 ${
                                    isDarkMode
                                        ? "bg-[#1e1f1c] border border-[#49483e]"
                                        : "bg-gray-50 border border-gray-200"
                                }`}
                            >
                                <JsonOrTextView
                                    data={selectedSpan.input}
                                    name="input"
                                    themeStyle={{borderRadius: "0.25rem"}}
                                    isDarkMode={isDarkMode}
                                />
                            </div>
                        </div>

                        {selectedSpan.source === "tracing_generation" && (
                            <>
                                {selectedSpan.output && (
                                    <div>
                                        <h3
                                            className={`text-md font-bold mb-2 p-2 rounded ${
                                                isDarkMode
                                                    ? "bg-[#3c3d32] text-[#66d9ef]"
                                                    : "bg-blue-50 text-black"
                                            }`}
                                        >
                                            Output:
                                        </h3>
                                        <div
                                            className={`rounded p-1 ${
                                                isDarkMode
                                                    ? "bg-[#1e1f1c] border border-[#49483e]"
                                                    : "bg-gray-50 border border-gray-200"
                                            }`}
                                        >
                                            <JsonOrTextView
                                                data={selectedSpan.output}
                                                name="output"
                                                themeStyle={{borderRadius: "0.25rem"}}
                                                isDarkMode={isDarkMode}
                                            />
                                        </div>
                                    </div>
                                )}

                                {selectedSpan.model_configs && (
                                    <div>
                                        <h3
                                            className={`text-md font-bold mb-2 p-2 rounded ${
                                                isDarkMode
                                                    ? "bg-[#3c3d32] text-[#66d9ef]"
                                                    : "bg-blue-50 text-black"
                                            }`}
                                        >
                                            Model Configs:
                                        </h3>
                                        <div
                                            className={`rounded p-1 ${
                                                isDarkMode
                                                    ? "bg-[#1e1f1c] border border-[#49483e]"
                                                    : "bg-gray-50 border border-gray-200"
                                            }`}
                                        >
                                            <JsonOrTextView
                                                data={selectedSpan.model_configs}
                                                name="model_configs"
                                                themeStyle={{borderRadius: "0.25rem"}}
                                                isDarkMode={isDarkMode}
                                            />
                                        </div>
                                    </div>
                                )}

                                {selectedSpan.usage && (
                                    <div>
                                        <h3
                                            className={`text-md font-bold mb-2 p-2 rounded ${
                                                isDarkMode
                                                    ? "bg-[#3c3d32] text-[#66d9ef]"
                                                    : "bg-blue-50 text-black"
                                            }`}
                                        >
                                            Usage:
                                        </h3>
                                        <div
                                            className={`rounded p-1 ${
                                                isDarkMode
                                                    ? "bg-[#1e1f1c] border border-[#49483e]"
                                                    : "bg-gray-50 border border-gray-200"
                                            }`}
                                        >
                                            <JsonOrTextView
                                                data={selectedSpan.usage}
                                                name="usage"
                                                themeStyle={{borderRadius: "0.25rem"}}
                                                isDarkMode={isDarkMode}
                                            />
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        {selectedSpan.source === "tracing_tool" && (
                            <>
                                {selectedSpan.output && (
                                    <div>
                                        <h3
                                            className={`text-md font-bold mb-2 p-2 rounded ${
                                                isDarkMode
                                                    ? "bg-[#3c3d32] text-[#66d9ef]"
                                                    : "bg-blue-50 text-black"
                                            }`}
                                        >
                                            Output:
                                        </h3>
                                        <div
                                            className={`rounded p-1 ${
                                                isDarkMode
                                                    ? "bg-[#1e1f1c] border border-[#49483e]"
                                                    : "bg-gray-50 border border-gray-200"
                                            }`}
                                        >
                                            <JsonOrTextView
                                                data={selectedSpan.output}
                                                name="output"
                                                themeStyle={{borderRadius: "0.25rem"}}
                                                isDarkMode={isDarkMode}
                                            />
                                        </div>
                                    </div>
                                )}

                                {selectedSpan.mcp_data && (
                                    <div>
                                        <h3
                                            className={`text-md font-bold mb-2 p-2 rounded ${
                                                isDarkMode
                                                    ? "bg-[#3c3d32] text-[#66d9ef]"
                                                    : "bg-blue-50 text-black"
                                            }`}
                                        >
                                            MCP Data:
                                        </h3>
                                        <div
                                            className={`rounded p-1 ${
                                                isDarkMode
                                                    ? "bg-[#1e1f1c] border border-[#49483e]"
                                                    : "bg-gray-50 border border-gray-200"
                                            }`}
                                        >
                                            <JsonOrTextView
                                                data={selectedSpan.mcp_data}
                                                name="mcp_data"
                                                themeStyle={{borderRadius: "0.25rem"}}
                                                isDarkMode={isDarkMode}
                                            />
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );

    type JsonOrTextViewProps = {
        data: unknown;
        name: string;
        themeStyle?: React.CSSProperties;
        isDarkMode?: boolean;
    };

    const JsonOrTextView: React.FC<JsonOrTextViewProps> = ({
                                                               data,
                                                               name,
                                                               themeStyle,
                                                               isDarkMode = false,
                                                           }) => {
        const parseData = (): unknown => {
            if (typeof data === "string") {
                try {
                    return JSON.parse(data);
                } catch {
                    return data; // 返回原始字符串
                }
            }
            return data;
        };

        const parsedData = parseData();

        if (parsedData && typeof parsedData === "object") {
            return (
                <div>
                    <ReactJson
                        src={parsedData as Record<string, unknown>}
                        name={name}
                        theme={isDarkMode ? "monokai" : "rjv-default"} // 根据主题选择主题
                        collapsed={false}
                        enableClipboard={true}
                        displayObjectSize={true}
                        displayDataTypes={false}
                        style={{
                            padding: "1rem",
                            borderRadius: "0.5rem",
                            backgroundColor: isDarkMode ? "#272822" : "#ffffff",
                            ...themeStyle,
                        }}
                    />
                </div>
            );
        }

        const text = String(parsedData);
        const hasNewlines = text.includes("\n");

        // 纯文本显示模式
        return (
            <div
                style={{
                    padding: "1rem",
                    borderRadius: "0.5rem",
                    backgroundColor: isDarkMode ? "#272822" : "#ffffff",
                    color: isDarkMode ? "#f8f8f2" : "#333",
                    fontFamily: "monospace",
                    ...themeStyle,
                }}
            >
                {hasNewlines ? (
                    <pre
                        style={{
                            whiteSpace: "pre-wrap",
                            overflow: "auto",
                            margin: 0,
                        }}
                    >
                        {text}
                    </pre>
                ) : (
                    <div style={{whiteSpace: "pre-wrap"}}>{text}</div>
                )}
            </div>
        );
    };

    const handleBackToList = () => {
        const queryParams = new URLSearchParams(searchParams.toString());

        queryParams.delete("id");

        if (!queryParams.has("exp_id")) {
            queryParams.set("exp_id", evaluation.exp_id);
        }

        router.push(`/?${queryParams.toString()}`);
    };

    const darkModeClasses = "bg-[#272822] text-[#f8f8f2]";
    const lightModeClasses = "bg-white text-gray-800";

    return (
        <ConfigProvider
            theme={{
                token: {
                    colorBgContainer: isDarkMode ? "#3c3d32" : "#ffffff",
                    colorText: isDarkMode ? "#f8f8f2" : "#333",
                },
                components: {
                    Card: {
                        headerBg: isDarkMode ? "#3c3d32" : "#fafafa",
                        colorTextHeading: isDarkMode ? "#f8f8f2" : "#333",
                    },
                    Table: {
                        headerBg: isDarkMode ? "#3c3d32" : "#fafafa",
                        headerColor: isDarkMode ? "#f8f8f2" : "#333",
                    },
                },
            }}
        >
            <div
                className={`max-w-screen-2xl mx-auto p-4 min-h-screen ${
                    isDarkMode ? darkModeClasses : lightModeClasses
                }`}
            >
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-2xl font-bold">
                        Evaluation Details (ID: {evaluation.id})
                    </h1>
                    <div className="flex items-center space-x-4">
                        <Button onClick={handleBackToList}>Back to List</Button>
                        <Space size={8} className="ml-4">
                            <Switch
                                checkedChildren={<BulbFilled/>}
                                unCheckedChildren={<BulbOutlined/>}
                                checked={isDarkMode}
                                onChange={setIsDarkMode}
                            />
                            <span>{isDarkMode ? "Dark" : "Light"}</span>
                        </Space>
                    </div>
                </div>

                <div
                    className="grid grid-cols-1 md:grid-cols-2 gap-4"
                    style={{height: "calc(100vh - 150px)"}}
                >
                    {/* 左列：包含使用Tab切换的General Information和Tracing */}
                    <Card className="h-full flex flex-col">
                        <Tabs
                            activeKey={activeTab}
                            onChange={setActiveTab}
                            items={[
                                {
                                    key: "general",
                                    label: "General Information",
                                    children: renderGeneralInfo(),
                                },
                                {
                                    key: "tracing",
                                    label: "Tracing",
                                    children: renderTracingContent(),
                                },
                            ]}
                        />
                    </Card>

                    <Card className="h-full flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-bold">Trajectory</h2>
                            <Tabs
                                activeKey={viewMode}
                                onChange={(value) => setViewMode(value as "formatted" | "json")}
                                items={[
                                    {key: "formatted", label: "卡片视图"},
                                    {key: "json", label: "JSON视图"},
                                ]}
                            />
                        </div>
                        <div className="flex-grow overflow-auto">
                            {viewMode === "formatted"
                                ? renderFormattedTrajectory()
                                : renderJsonTrajectory()}
                        </div>
                    </Card>
                </div>

                {/* 渲染 JSON 对话框 */}
                {isDialogOpen && renderJsonDialog()}
            </div>
        </ConfigProvider>
    );
}

// 添加全局样式
const GlobalStyles = () => (
    <style jsx global>{`
        body.light-mode {
            background-color: #ffffff;
            color: #333333;
            transition: background-color 0.3s ease;
        }

        body.dark-mode {
            background-color: #272822;
            color: #f8f8f2;
            transition: background-color 0.3s ease;
        }
    `}</style>
);

export default function EvaluationDetailPageWrapper() {
    return (
        <>
            <GlobalStyles/>
            <EvaluationDetailPage/>
        </>
    );
}