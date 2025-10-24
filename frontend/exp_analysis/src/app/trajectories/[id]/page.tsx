"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import {
    Card,
    Button,
    Tabs,
    ConfigProvider,
    Collapse,
} from "antd";
import {
    ArrowLeftOutlined
} from "@ant-design/icons";
import ReactJson from "react-json-view";
import InfoBlock from "@/components/InfoBlock";
import ThemeToggle from "@/components/ThemeToggle";

type Trajectory = {
    id: number;
    trace_id: string;
    trace_url: string | null;
    d_input: string | null;
    d_output: string | null;
    trajectories: string | null;
    time_cost: number | null;
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

export default function TrajectoryDetailPage() {
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const {id} = params;
    const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [parsedTrajectory, setParsedTrajectory] = useState<RoleTrajectory[] | null>(null);
    const [viewMode, setViewMode] = useState<"formatted" | "json">("formatted");
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [expandedChildKeys, setExpandedChildKeys] = useState<Record<string, boolean>>({});
    const [isJsonExpanded, setIsJsonExpanded] = useState(false);

    useEffect(() => {
        const savedTheme = localStorage.getItem("theme");
        const initialDarkMode = savedTheme === "dark";
        setIsDarkMode(initialDarkMode);
        document.body.className = initialDarkMode ? "dark-mode" : "light-mode";
    }, []);

    useEffect(() => {
        localStorage.setItem("theme", isDarkMode ? "dark" : "light");
        document.body.className = isDarkMode ? "dark-mode" : "light-mode";
    }, [isDarkMode]);

    useEffect(() => {
        if (id) {
            fetch(`/api/trajectories/${id}`)
                .then((res) => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then(async (data) => {
                    setTrajectory(data);
                    setLoading(false);

                    if (data.trajectories) {
                        try {
                            const trajectoryData: RoleTrajectory[] = JSON.parse(data.trajectories);
                            setParsedTrajectory(trajectoryData);
                        } catch (err) {
                            console.error("Failed to parse trajectories JSON:", err);
                            setParsedTrajectory(null);
                        }
                    } else {
                        setParsedTrajectory(null);
                    }
                })
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

    if (!trajectory) {
        return <div className="container mx-auto p-4">Trajectory not found.</div>;
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
                        collapsed={isJsonExpanded ? false : 3}
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

    const renderTrajectoryInfo = () => {
        const basicInfoData = [
            {
                label: "Trace ID",
                value: trajectory.trace_id,
            },
            {
                label: "Trace URL",
                value: trajectory.trace_url,
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
            {
                label: "Time Cost",
                value: trajectory.time_cost ? `${trajectory.time_cost.toFixed(2)}s` : "-",
            },
        ];

        const inputOutputData = [
            {
                label: "Input",
                value: trajectory.d_input,
                render: (value: string | number | boolean | null) =>
                    value ? (
                        <div
                            className={`p-2 rounded ${
                                isDarkMode ? "bg-[#272822]" : "bg-gray-100"
                            }`}
                        >
                            <pre className="whitespace-pre-wrap">{value}</pre>
                        </div>
                    ) : (
                        "-"
                    ),
            },
            {
                label: "Output",
                value: trajectory.d_output,
                render: (value: string | number | boolean | null) =>
                    value ? (
                        <div
                            className={`p-2 rounded ${
                                isDarkMode ? "bg-[#272822]" : "bg-gray-100"
                            }`}
                        >
                            <pre className="whitespace-pre-wrap">{value}</pre>
                        </div>
                    ) : (
                        "-"
                    ),
            },
        ];

        return (
            <div>
                <InfoBlock
                    title="Basic Information"
                    items={basicInfoData}
                    isDarkMode={isDarkMode}
                />

                <InfoBlock
                    title="Input & Output"
                    items={inputOutputData}
                    isDarkMode={isDarkMode}
                />
            </div>
        );
    };

    const handleBackToList = () => {
        const queryParams = new URLSearchParams();

        // 保留原始查询参数 - 使用正确的参数名称
        if (searchParams.has("traj_trace_id")) queryParams.set("traj_trace_id", searchParams.get("traj_trace_id")!);
        if (searchParams.has("traj_keyword")) queryParams.set("traj_keyword", searchParams.get("traj_keyword")!);
        if (searchParams.has("trajectoryPage")) queryParams.set("trajectoryPage", searchParams.get("trajectoryPage")!);
        if (searchParams.has("trajectoryPageSize")) queryParams.set("trajectoryPageSize", searchParams.get("trajectoryPageSize")!);

        // 添加tab参数，确保回到Trajectories标签页
        queryParams.set("tab", "trajectories");

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
                    <Button
                        type="text"
                        icon={<ArrowLeftOutlined />}
                        onClick={handleBackToList}
                    >
                        Back to List
                    </Button>
                    <h1 className="text-2xl font-bold">
                        Trajectory Details (ID: {trajectory.id})
                    </h1>
                    <ThemeToggle isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />
                </div>

                <div
                    className="grid grid-cols-1 md:grid-cols-2 gap-4"
                    style={{height: "calc(100vh - 150px)"}}
                >
                    <Card className="h-full flex flex-col">
                        <Tabs
                            defaultActiveKey="info"
                            items={[
                                {
                                    key: "info",
                                    label: "Trajectory Information",
                                    children: renderTrajectoryInfo(),
                                },
                            ]}
                        />
                    </Card>

                    <Card className="h-full flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-bold">Trajectory Details</h2>
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
            </div>
        </ConfigProvider>
    );
}