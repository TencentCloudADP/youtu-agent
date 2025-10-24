"use client";

import { useEffect, useState, Suspense, useRef } from "react";
import {
  Card,
  Table,
  Button,
  Input,
  Select,
  Statistic,
  Row,
  Col,
  Pagination,
  Dropdown,
  Menu,
  ConfigProvider,
  theme,
  Switch,
  Space,
  Tabs,
  Tooltip
} from "antd";
import { useRouter, useSearchParams } from "next/navigation";
import {
  FilterOutlined,
  FilterFilled,
  BulbOutlined,
  BulbFilled,
  ExperimentOutlined,
  RadarChartOutlined
} from "@ant-design/icons";

type Evaluation = {
  id: number;
  trace_id: string;
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
  extracted_final_answer: string | null;
  judged_response: string | null;
  reasoning: string | null;
  correct: boolean | null;
  confidence: number | null;
  dataset_index: number;
};

type Trajectory = {
  id: number;
  trace_id: string;
  trace_url: string | null;
  d_input: string | null;
  d_output: string | null;
  trajectories: string | null;
  time_cost: number | null;
};

type FilterState = {
  keyword: string;
  tools: string;
  traceId: string;
};

type TrajectoryFilterState = {
  traceId: string;
  trajectoryContent: string;
};

function HomePageContentWrapper() {
  return (
      <Suspense fallback={<div className="text-center py-8">Loading...</div>}>
        <HomePageContent />
      </Suspense>
  );
}

function HomePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [isDarkMode, setIsDarkMode] = useState(false);
  const urlTab = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<string>(urlTab || "evaluations");

  const [expIds, setExpIds] = useState<string[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [allEvaluations, setAllEvaluations] = useState<Evaluation[]>([]);
  const [allTrajectories, setAllTrajectories] = useState<Trajectory[]>([]);
  const [stats, setStats] = useState<{ [key: string]: number }>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isTrajectoriesLoading, setIsTrajectoriesLoading] = useState(false);

  const [filter, setFilter] = useState<FilterState>({
    keyword: "",
    tools: "",
    traceId: ""
  });

  const [appliedFilter, setAppliedFilter] = useState<FilterState>({
    keyword: "",
    tools: "",
    traceId: ""
  });

  const [trajectoryFilter, setTrajectoryFilter] = useState<TrajectoryFilterState>({
    traceId: "",
    trajectoryContent: ""
  });

  const [appliedTrajectoryFilter, setAppliedTrajectoryFilter] = useState<TrajectoryFilterState>({
    traceId: "",
    trajectoryContent: ""
  });

  const [correctFilter, setCorrectFilter] = useState<string>("all");
  const [appliedCorrectFilter, setAppliedCorrectFilter] = useState<string>("all");

  const urlPage = searchParams.get('page');
  const urlPageSize = searchParams.get('pageSize');
  const urlTrajectoryPage = searchParams.get('trajectoryPage');
  const urlTrajectoryPageSize = searchParams.get('trajectoryPageSize');

  const [currentPage, setCurrentPage] = useState(
      urlPage ? Number(urlPage) : 1
  );
  const [pageSize, setPageSize] = useState(
      urlPageSize ? Number(urlPageSize) : 20
  );

  const [trajectoryCurrentPage, setTrajectoryCurrentPage] = useState(
      urlTrajectoryPage ? Number(urlTrajectoryPage) : 1
  );
  const [trajectoryPageSize, setTrajectoryPageSize] = useState(
      urlTrajectoryPageSize ? Number(urlTrajectoryPageSize) : 20
  );

  const [totalItems, setTotalItems] = useState(0);
  const [totalTrajectories, setTotalTrajectories] = useState(0);

  const initializedRef = useRef(false);

  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    // 安全地检查 localStorage 和 document
    const savedTheme = typeof localStorage !== 'undefined' ? localStorage.getItem("theme") : null;
    const initialDarkMode = savedTheme === "dark";
    setIsDarkMode(initialDarkMode);

    // 安全地设置 body className
    if (typeof document !== 'undefined') {
      document.body.className = initialDarkMode ? "dark-mode" : "light-mode";
    }

    // 首先读取所有URL参数
    const urlExpId = searchParams.get("exp_id");
    const urlCorrect = searchParams.get("correct") || "all";
    const urlKeyword = searchParams.get("eval_keyword") || "";
    const urlTools = searchParams.get("eval_tools") || "";
    const urlTraceId = searchParams.get("eval_trace_id") || "";

    // 轨迹过滤参数
    const urlTrajTraceId = searchParams.get("traj_trace_id") || "";
    const urlTrajKeyword = searchParams.get("traj_keyword") || "";

    // 设置evaluations过滤状态
    setFilter({
      keyword: urlKeyword,
      tools: urlTools,
      traceId: urlTraceId
    });

    setAppliedFilter({
      keyword: urlKeyword,
      tools: urlTools,
      traceId: urlTraceId
    });

    setCorrectFilter(urlCorrect);
    setAppliedCorrectFilter(urlCorrect);

    // 设置trajectories过滤状态
    setTrajectoryFilter({
      traceId: urlTrajTraceId,
      trajectoryContent: urlTrajKeyword
    });

    setAppliedTrajectoryFilter({
      traceId: urlTrajTraceId,
      trajectoryContent: urlTrajKeyword
    });

    // 然后获取实验ID列表
    fetch("/api/exp_ids")
        .then((res) => res.json())
        .then((data) => {
          setExpIds(data);
          if (urlExpId && (data.includes(urlExpId) || urlExpId === "all")) {
            setSelectedExpId(urlExpId);
          } else if (data.length > 0) {
            setSelectedExpId(data[0]);
          } else {
            setSelectedExpId("all");
          }
        });
  }, [searchParams]);

  useEffect(() => {
    // 安全地操作 localStorage 和 document
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem("theme", isDarkMode ? "dark" : "light");
    }
    if (typeof document !== 'undefined') {
      document.body.className = isDarkMode ? "dark-mode" : "light-mode";
    }
  }, [isDarkMode]);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);

    // 清除其他标签页的参数
    const newSearchParams = new URLSearchParams();
    newSearchParams.set('tab', tab);

    // 保留当前标签页的参数
    if (tab === "evaluations") {
      if (selectedExpId) newSearchParams.set("exp_id", selectedExpId);
      newSearchParams.set("page", currentPage.toString());
      newSearchParams.set("pageSize", pageSize.toString());
      if (appliedFilter.keyword) newSearchParams.set("eval_keyword", appliedFilter.keyword);
      if (appliedFilter.tools) newSearchParams.set("eval_tools", appliedFilter.tools);
      if (appliedFilter.traceId) newSearchParams.set("eval_trace_id", appliedFilter.traceId);
      if (appliedCorrectFilter !== "all") newSearchParams.set("correct", appliedCorrectFilter);
    } else if (tab === "trajectories") {
      newSearchParams.set("trajectoryPage", trajectoryCurrentPage.toString());
      newSearchParams.set("trajectoryPageSize", trajectoryPageSize.toString());
      if (appliedTrajectoryFilter.traceId) newSearchParams.set("traj_trace_id", appliedTrajectoryFilter.traceId);
      if (appliedTrajectoryFilter.trajectoryContent) newSearchParams.set("traj_keyword", appliedTrajectoryFilter.trajectoryContent);
    }

    router.replace(`/?${newSearchParams.toString()}`, { scroll: false });
  };

  useEffect(() => {
    if (activeTab === "evaluations" && (selectedExpId || selectedExpId === "all")) {
      setIsLoading(true);

      const newSearchParams = new URLSearchParams();
      if (selectedExpId !== "all") {
        newSearchParams.set("exp_id", selectedExpId);
      }
      newSearchParams.set("page", currentPage.toString());
      newSearchParams.set("pageSize", pageSize.toString());
      newSearchParams.set("order", "dataset_index");

      if (appliedFilter.keyword) newSearchParams.set("keyword", appliedFilter.keyword);
      if (appliedFilter.tools) newSearchParams.set("tools", appliedFilter.tools);
      if (appliedFilter.traceId) newSearchParams.set("trace_id", appliedFilter.traceId);
      if (appliedCorrectFilter !== "all") newSearchParams.set("correct", appliedCorrectFilter);

      const urlParams = new URLSearchParams(newSearchParams);
      if (urlParams.toString() !== searchParams.toString()) {
        router.replace(`/?${urlParams.toString()}`, { scroll: false });
      }

      if (selectedExpId === "all" && !appliedFilter.traceId) {
        setAllEvaluations([]);
        setTotalItems(0);
        setIsLoading(false);
        return;
      }

      const apiPath = selectedExpId === "all"
          ? `/api/evaluations/all?${newSearchParams.toString()}`
          : `/api/evaluations/${selectedExpId}?${newSearchParams.toString()}`;

      fetch(apiPath)
          .then((res) => res.json())
          .then((data) => {
            setAllEvaluations(data.data || []);
            setTotalItems(data.totalCount || 0);
            setIsLoading(false);
          })
          .catch((error) => {
            console.error("Error fetching evaluations:", error);
            setIsLoading(false);
            setAllEvaluations([]);
            setTotalItems(0);
          });

      const statsParams = new URLSearchParams();
      if (selectedExpId !== "all") statsParams.set("exp_id", selectedExpId);
      if (appliedFilter.keyword) statsParams.set("keyword", appliedFilter.keyword);
      if (appliedFilter.tools) statsParams.set("tools", appliedFilter.tools);
      if (appliedFilter.traceId) statsParams.set("trace_id", appliedFilter.traceId);
      if (appliedCorrectFilter !== "all") statsParams.set("correct", appliedCorrectFilter);

      const statsPath = selectedExpId === "all"
          ? `/api/evaluations/all/stats?${statsParams.toString()}`
          : `/api/evaluations/${selectedExpId}/stats?${statsParams.toString()}`;

      fetch(statsPath)
          .then((res) => res.json())
          .then(setStats);
    }
  }, [selectedExpId, appliedFilter, appliedCorrectFilter, searchParams, router, currentPage, pageSize, activeTab]);

  useEffect(() => {
    if (activeTab === "trajectories") {
      setIsTrajectoriesLoading(true);

      const newSearchParams = new URLSearchParams();
      newSearchParams.set("page", trajectoryCurrentPage.toString());
      newSearchParams.set("pageSize", trajectoryPageSize.toString());

      // 使用后端API期望的参数名
      if (appliedTrajectoryFilter.traceId) {
        newSearchParams.set("trace_id", appliedTrajectoryFilter.traceId);
      }
      if (appliedTrajectoryFilter.trajectoryContent) {
        newSearchParams.set("keyword", appliedTrajectoryFilter.trajectoryContent);
      }

      // 更新URL参数
      const urlParams = new URLSearchParams();
      urlParams.set('tab', 'trajectories');
      urlParams.set("trajectoryPage", trajectoryCurrentPage.toString());
      urlParams.set("trajectoryPageSize", trajectoryPageSize.toString());
      if (appliedTrajectoryFilter.traceId) urlParams.set("traj_trace_id", appliedTrajectoryFilter.traceId);
      if (appliedTrajectoryFilter.trajectoryContent) urlParams.set("traj_keyword", appliedTrajectoryFilter.trajectoryContent);

      if (urlParams.toString() !== searchParams.toString()) {
        router.replace(`/?${urlParams.toString()}`, { scroll: false });
      }

      fetch(`/api/trajectories?${newSearchParams.toString()}`)
          .then((res) => res.json())
          .then((data) => {
            setAllTrajectories(data.data || []);
            setTotalTrajectories(data.totalCount || 0);
            setIsTrajectoriesLoading(false);
          })
          .catch((error) => {
            console.error("Error fetching trajectories:", error);
            setIsTrajectoriesLoading(false);
            setAllTrajectories([]);
            setTotalTrajectories(0);
          });
    }
  }, [appliedTrajectoryFilter, trajectoryCurrentPage, trajectoryPageSize, activeTab, searchParams, router]);

  const handleFilterChange = (field: keyof FilterState, value: string) => {
    setFilter(prev => ({ ...prev, [field]: value }));
  };

  const handleTrajectoryFilterChange = (field: keyof TrajectoryFilterState, value: string) => {
    setTrajectoryFilter(prev => ({ ...prev, [field]: value }));
  };

  const handleQuery = () => {
    if (activeTab === "evaluations") {
      setAppliedFilter(filter);
      setAppliedCorrectFilter(correctFilter);
      setCurrentPage(1);
    } else {
      setAppliedTrajectoryFilter(trajectoryFilter);
      setTrajectoryCurrentPage(1);
    }
  };

  const handleClearFilters = () => {
    if (activeTab === "evaluations") {
      setFilter({ keyword: "", tools: "", traceId: "" });
      setCorrectFilter("all");
      setAppliedFilter({ keyword: "", tools: "", traceId: "" });
      setAppliedCorrectFilter("all");
      setCurrentPage(1);
    } else {
      setTrajectoryFilter({ traceId: "", trajectoryContent: "" });
      setAppliedTrajectoryFilter({ traceId: "", trajectoryContent: "" });
      setTrajectoryCurrentPage(1);
    }
  };

  const handleRowClick = (evaluationId: number) => {
    const currentSearchParams = new URLSearchParams();
    if (selectedExpId && selectedExpId !== "all") currentSearchParams.set("exp_id", selectedExpId);
    currentSearchParams.set("correct", appliedCorrectFilter);
    if (appliedFilter.keyword) currentSearchParams.set("eval_keyword", appliedFilter.keyword);
    if (appliedFilter.tools) currentSearchParams.set("eval_tools", appliedFilter.tools);
    if (appliedFilter.traceId) currentSearchParams.set("eval_trace_id", appliedFilter.traceId);

    currentSearchParams.set("page", currentPage.toString());
    currentSearchParams.set("pageSize", pageSize.toString());

    router.push(`/evaluations/${evaluationId}?${currentSearchParams.toString()}`);
  };

  const handleTrajectoryClick = (trajectoryId: number) => {
    const currentSearchParams = new URLSearchParams();
    // 使用不同的参数名称避免冲突
    if (appliedTrajectoryFilter.traceId) currentSearchParams.set("traj_trace_id", appliedTrajectoryFilter.traceId);
    if (appliedTrajectoryFilter.trajectoryContent) currentSearchParams.set("traj_keyword", appliedTrajectoryFilter.trajectoryContent);
    currentSearchParams.set("trajectoryPage", trajectoryCurrentPage.toString());
    currentSearchParams.set("trajectoryPageSize", trajectoryPageSize.toString());
    currentSearchParams.set("tab", "trajectories");

    router.push(`/trajectories/${trajectoryId}?${currentSearchParams.toString()}`);
  };

  const handlePageChange = (page: number) => {
    if (activeTab === "evaluations") {
      setCurrentPage(page);
    } else {
      setTrajectoryCurrentPage(page);
    }
  };

  const handlePageSizeChange = (size: number) => {
    if (activeTab === "evaluations") {
      setPageSize(size);
      setCurrentPage(1);
    } else {
      setTrajectoryPageSize(size);
      setTrajectoryCurrentPage(1);
    }
  };

  const correctFilterMenu = (
      <Menu
          selectedKeys={[appliedCorrectFilter]}
          onClick={({ key }) => {
            setCorrectFilter(key);
            setAppliedCorrectFilter(key);
            setCurrentPage(1);
          }}
      >
        <Menu.Item key="all">All</Menu.Item>
        <Menu.Item key="true">True</Menu.Item>
        <Menu.Item key="false">False</Menu.Item>
      </Menu>
  );

  const FilterIcon = appliedCorrectFilter !== "all" ? FilterFilled : FilterOutlined;

  const evaluationColumns = [
    {
      title: "Trace ID",
      dataIndex: "trace_id",
      key: "trace_id",
      width: 120,
      render: (text: string) => (
          <span className={isDarkMode ? "text-blue-300" : "text-blue-500"}>
          {text}
        </span>
      )
    },
    {
      title: "Dataset Index",
      dataIndex: "dataset_index",
      key: "dataset_index",
      width: 100,
      sorter: true
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      width: 100
    },
    {
      title: "Question",
      dataIndex: "raw_question",
      key: "raw_question",
      width: 200,
      ellipsis: true
    },
    {
      title: "Level",
      dataIndex: "level",
      key: "level",
      width: 80
    },
    {
      title: "Correct Answer",
      dataIndex: "correct_answer",
      key: "correct_answer",
      width: 150,
      ellipsis: true
    },
    {
      title: "Stage",
      dataIndex: "stage",
      key: "stage",
      width: 80
    },
    {
      title: "Response",
      dataIndex: "response",
      key: "response",
      width: 200,
      ellipsis: true
    },
    {
      title: (
          <div className="flex items-center">
            <span>Correct</span>
            <Dropdown
                overlay={correctFilterMenu}
                trigger={['click']}
                open={undefined}
            >
              <FilterIcon
                  className={`ml-2 ${appliedCorrectFilter !== "all" ? (isDarkMode ? "text-blue-300" : "text-blue-500") : "text-gray-400"}`}
              />
            </Dropdown>
          </div>
      ),
      dataIndex: "correct",
      key: "correct",
      width: 100,
      render: (correct: boolean | null) => (
          <span className={`font-medium ${
              correct === true ? (isDarkMode ? "text-green-400" : "text-green-600") :
                  correct === false ? (isDarkMode ? "text-red-400" : "text-red-500") :
                      (isDarkMode ? "text-gray-400" : "text-gray-500")
          }`}>
      {correct === true ? "True" :
          correct === false ? "False" : "N/A"}
    </span>
      )
    },
    {
      title: "Confidence",
      dataIndex: "confidence",
      key: "confidence",
      width: 120,
      render: (value: number | null) => (
          <span className={value ? (isDarkMode ? "text-purple-400" : "text-purple-600") : (isDarkMode ? "text-gray-400" : "text-gray-500")}>
          {value ?? "N/A"}
        </span>
      )
    }
  ];

  const trajectoryColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 80,
      render: (id: number) => (
          <span className={isDarkMode ? "text-blue-300" : "text-blue-500"}>
          #{id}
        </span>
      )
    },
    {
      title: "Trace ID",
      dataIndex: "trace_id",
      key: "trace_id",
      width: 120,
      render: (text: string) => (
          <span className={isDarkMode ? "text-blue-300" : "text-blue-500"}>
          {text}
        </span>
      )
    },
    {
      title: "Trace URL",
      dataIndex: "trace_url",
      key: "trace_url",
      width: 200, // 增加宽度以显示更多URL内容
      ellipsis: true,
      render: (url: string | null) => (
          url ? (
              <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`block truncate ${isDarkMode ? "text-blue-300 hover:text-blue-200" : "text-blue-500 hover:text-blue-700"}`}
                  style={{
                    maxWidth: "100%",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap"
                  }}
              >
                {url}
              </a>
          ) : (
              <span className={isDarkMode ? "text-gray-400" : "text-gray-500"}>N/A</span>
          )
      )
    },
    {
      title: "Time Cost",
      dataIndex: "time_cost",
      key: "time_cost",
      width: 100,
      render: (value: number | null) => (
          <span className={value ? (isDarkMode ? "text-yellow-400" : "text-yellow-600") : (isDarkMode ? "text-gray-400" : "text-gray-500")}>
          {value ? `${value.toFixed(2)}s` : "N/A"}
        </span>
      )
    },
    {
      title: "Input",
      dataIndex: "d_input",
      key: "d_input",
      width: 200,
      ellipsis: true,
      render: (text: string | null) => (
          <Tooltip title={text || ""}>
          <span className={isDarkMode ? "text-gray-300" : "text-gray-600"}>
            {text ? `${text.substring(0, 50)}${text.length > 50 ? "..." : ""}` : "N/A"}
          </span>
          </Tooltip>
      )
    },
    {
      title: "Output",
      dataIndex: "d_output",
      key: "d_output",
      width: 200,
      ellipsis: true,
      render: (text: string | null) => (
          <Tooltip title={text || ""}>
          <span className={isDarkMode ? "text-gray-300" : "text-gray-600"}>
            {text ? `${text.substring(0, 50)}${text.length > 50 ? "..." : ""}` : "N/A"}
          </span>
          </Tooltip>
      )
    },
    {
      title: "Trajectories",
      dataIndex: "trajectories",
      key: "trajectories",
      width: 200,
      ellipsis: true,
      render: (text: string | null) => (
          <Tooltip title={text || ""}>
          <span className={isDarkMode ? "text-gray-300" : "text-gray-600"}>
            {text ? `${text.substring(0, 50)}${text.length > 50 ? "..." : ""}` : "N/A"}
          </span>
          </Tooltip>
      )
    }
  ];

  return (
      <ConfigProvider
          theme={{
            algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
            token: {
              colorBgContainer: isDarkMode ? "#3c3d32" : "#ffffff",
              colorText: isDarkMode ? "#f8f8f2" : "#333333",
              colorBorder: isDarkMode ? "#49483e" : "#d9d9d9",
            },
            components: {
              Card: {
                headerBg: isDarkMode ? "#3c3d32" : "#fafafa",
                colorTextHeading: isDarkMode ? "#f8f8f2" : "#333333",
              },
              Table: {
                headerBg: isDarkMode ? "#3c3d32" : "#fafafa",
                headerColor: isDarkMode ? "#f8f8f2" : "#333333",
                rowHoverBg: isDarkMode ? "rgba(60, 61, 50, 0.6)" : "#f5f5f5",
              },
              Input: {
                colorBgContainer: isDarkMode ? "#272822" : "#ffffff",
                colorText: isDarkMode ? "#f8f8f2" : "#333333",
                colorBorder: isDarkMode ? "#49483e" : "#d9d9d9",
              },
              Select: {
                colorBgContainer: isDarkMode ? "#272822" : "#ffffff",
                colorText: isDarkMode ? "#f8f8f2" : "#333333",
                colorBorder: isDarkMode ? "#49483e" : "#d9d9d9",
              },
              Button: {
                colorBgContainer: isDarkMode ? "#3c3d32" : "#f0f0f0",
                colorText: isDarkMode ? "#f8f8f2" : "#333333",
              },
              Tabs: {
                itemColor: isDarkMode ? "#f8f8f2" : "#333333",
                itemHoverColor: isDarkMode ? "#a6e22e" : "#52c41a",
                itemSelectedColor: isDarkMode ? "#a6e22e" : "#52c41a",
                inkBarColor: isDarkMode ? "#a6e22e" : "#52c41a",
              }
            }
          }}
      >
        <div className={`max-w-screen-2xl mx-auto p-4 min-h-screen ${isDarkMode ? "bg-[#272822] text-[#f8f8f2]" : "bg-white text-gray-800"}`}>
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold">LLM Agent Experiments</h1>
            <div className="flex items-center space-x-4">
              <Space size={8} className="ml-4">
                <Switch
                    checkedChildren={<BulbFilled />}
                    unCheckedChildren={<BulbOutlined />}
                    checked={isDarkMode}
                    onChange={setIsDarkMode}
                />
                <span>{isDarkMode ? "Dark" : "Light"}</span>
              </Space>
            </div>
          </div>

          <Tabs
              activeKey={activeTab}
              onChange={handleTabChange}
              items={[
                {
                  label: (
                      <span className="flex items-center">
                    <ExperimentOutlined className="mr-2" />
                    Evaluations
                  </span>
                  ),
                  key: "evaluations",
                },
                {
                  label: (
                      <span className="flex items-center">
                    <RadarChartOutlined className="mr-2" />
                    Trajectories
                  </span>
                  ),
                  key: "trajectories",
                }
              ]}
              className={`${isDarkMode ? "text-gray-200" : "text-gray-700"}`}
          />

          {activeTab === "evaluations" && (
              <>
                <div className="flex justify-between items-center mb-6">
                  <Select
                      style={{ width: 280 }}
                      onChange={(value) => {
                        setSelectedExpId(value);
                        setCurrentPage(1);
                      }}
                      value={selectedExpId}
                      placeholder="Select an experiment"
                      options={[
                        { label: "All Experiments", value: "all" },
                        ...expIds.map(id => ({ label: id, value: id }))
                      ]}
                      showSearch
                      allowClear
                  />
                </div>

                {selectedExpId === "all" && !appliedFilter.traceId && allEvaluations.length === 0 && (
                    <div className={`p-6 mb-6 rounded-md ${isDarkMode ? "bg-[#49483e] text-red-300" : "bg-red-50 text-red-700"}`}>
                      <p className="font-medium">⚠️提示：</p>
                      <p className="mt-2">
                        当选择 All Experiments 且未输入Trace ID时，系统不会查询全部数据（避免性能问题）。
                        请输入Trace ID或者选择特定实验ID进行查询。
                      </p>
                    </div>
                )}

                {selectedExpId && (
                    <>
                      <Row gutter={16} className="mb-6">
                        <Col span={8}>
                          <Card
                              bordered={false}
                              className={`rounded-lg shadow-sm ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-gray-50"}`}
                          >
                            <Statistic
                                title="Init"
                                value={stats.init || 0}
                                valueStyle={{ color: isDarkMode ? '#a6e22e' : '#3f8600' }}
                            />
                          </Card>
                        </Col>
                        <Col span={8}>
                          <Card
                              bordered={false}
                              className={`rounded-lg shadow-sm ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-gray极狐-50"}`}
                          >
                            <Statistic
                                title="Rollout"
                                value={stats.rollout || 0}
                                valueStyle={{ color: isDarkMode ? '#66d9ef' : '#1890ff' }}
                            />
                          </Card>
                        </Col>
                        <Col span={8}>
                          <Card
                              bordered={false}
                              className={`rounded-lg shadow-sm ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-gray-50"}`}
                          >
                            <Statistic
                                title="Judged"
                                value={stats.judged || 0}
                                valueStyle={{ color: isDarkMode ? '#ae81ff' : '#722ed1' }}
                            />
                          </Card>
                        </Col>
                      </Row>

                      <Card
                          bodyStyle={{ padding: 16 }}
                          className={`rounded-lg shadow-sm mb-6 ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-white"}`}
                      >
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div>
                            <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>
                              Filter by Trace ID
                            </label>
                            <Input
                                value={filter.traceId}
                                onChange={e => handleFilterChange('traceId', e.target.value)}
                                placeholder="Search Trace ID..."
                                allowClear
                            />
                          </div>

                          <div>
                            <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>
                              Filter by Keyword
                            </label>
                            <Input
                                value={filter.keyword}
                                onChange={e => handleFilterChange('keyword', e.target.value)}
                                placeholder="Search keyword..."
                                allowClear
                            />
                          </div>

                          <div>
                            <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>
                              Filter by Tools
                            </label>
                            <Input
                                value={filter.tools}
                                onChange={e => handleFilterChange('tools', e.target.value)}
                                placeholder="Search tools..."
                                allowClear
                            />
                          </div>

                          <div className="flex items-end space-x-2 md:col-span-3 justify-end">
                            <Button
                                type="primary"
                                onClick={handleQuery}
                                size="middle"
                                className="w-32"
                            >
                              Apply
                            </Button>
                            <Button
                                onClick={handleClearFilters}
                                size="middle"
                                className="w-32"
                            >
                              Reset
                            </Button>
                          </div>
                        </div>
                      </Card>

                      <Card
                          bodyStyle={{ padding: 0 }}
                          className={`rounded-lg shadow-sm ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-white"}`}
                      >
                        <Table
                            columns={evaluationColumns}
                            dataSource={allEvaluations.map(e => ({ ...e, key: e.id }))}
                            pagination={false}
                            scroll={{ x: 1500 }}
                            loading={isLoading}
                            onRow={(record) => ({
                              onClick: () => handleRowClick(record.id)
                            })}
                            rowClassName={`cursor-pointer ${isDarkMode ? "hover:bg-[#49483e]" : "hover:bg-blue-50"}`}
                        />

                        <div className={`mt-6 flex justify-between items-center p-4 ${isDarkMode ? "bg-[#3c3d32]" : "bg-gray-50"}`}>
                          <div className="flex items-center">
                            <span className={`mr-2 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>Items per page:</span>
                            <Select
                                value={pageSize}
                                style={{ width: 120 }}
                                onChange={handlePageSizeChange}
                                options={[
                                  { value: 20, label: "20 / page" },
                                  { value: 50, label: "50 / page" },
                                  { value: 100, label: "100 / page" }
                                ]}
                            />
                          </div>

                          <Pagination
                              current={currentPage}
                              pageSize={pageSize}
                              total={totalItems}
                              onChange={handlePageChange}
                              showTotal={(total) => (
                                  <span className={isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}>
                        Total {total} items
                      </span>
                              )}
                              showSizeChanger={false}
                              className={isDarkMode ? "ant-pagination-dark" : ""}
                          />
                        </div>
                      </Card>
                    </>
                )}
              </>
          )}

          {activeTab === "trajectories" && (
              <>
                <Card
                    bodyStyle={{ padding: 16 }}
                    className={`rounded-lg shadow-sm mb-6 ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-white"}`}
                >
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>
                        Filter by Trace ID
                      </label>
                      <Input
                          value={trajectoryFilter.traceId}
                          onChange={e => handleTrajectoryFilterChange('traceId', e.target.value)}
                          placeholder="Search Trace ID..."
                          allowClear
                      />
                    </div>

                    <div>
                      <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>
                        Filter by Trajectory Content
                      </label>
                      <Input
                          value={trajectoryFilter.trajectoryContent}
                          onChange={e => handleTrajectoryFilterChange('trajectoryContent', e.target.value)}
                          placeholder="Search trajectory content..."
                          allowClear
                      />
                    </div>

                    <div className="flex items-end space-x-2 md:col-span-3 justify-end">
                      <Button
                          type="primary"
                          onClick={handleQuery}
                          size="middle"
                          className="w-32"
                      >
                        Apply
                      </Button>
                      <Button
                          onClick={handleClearFilters}
                          size="middle"
                          className="w-32"
                      >
                        Reset
                      </Button>
                    </div>
                  </div>
                </Card>

                <Card
                    bodyStyle={{ padding: 0 }}
                    className={`rounded-lg shadow-sm ${isDarkMode ? "bg-[#3c3d32] border border-[#49483e]" : "bg-white"}`}
                >
                  <Table
                      columns={trajectoryColumns}
                      dataSource={allTrajectories.map(t => ({ ...t, key: t.id }))}
                      pagination={false}
                      scroll={{ x: 1200 }}
                      loading={isTrajectoriesLoading}
                      onRow={(record) => ({
                        onClick: () => handleTrajectoryClick(record.id)
                      })}
                      rowClassName={`cursor-pointer ${isDarkMode ? "hover:bg-[#49483e]" : "hover:bg-blue-50"}`}
                  />

                  <div className={`mt-6 flex justify-between items-center p-4 ${isDarkMode ? "bg-[#3c3d32]" : "bg-gray-50"}`}>
                    <div className="flex items-center">
                      <span className={`mr-2 ${isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}`}>Items per page:</span>
                      <Select
                          value={trajectoryPageSize}
                          style={{ width: 120 }}
                          onChange={(value) => handlePageSizeChange(value)}
                          options={[
                            { value: 20, label: "20 / page" },
                            { value: 50, label: "50 / page" },
                            { value: 100, label: "100 / page" }
                          ]}
                      />
                    </div>

                    <Pagination
                        current={trajectoryCurrentPage}
                        pageSize={trajectoryPageSize}
                        total={totalTrajectories}
                        onChange={handlePageChange}
                        showTotal={(total) => (
                            <span className={isDarkMode ? "text-[#f8f8f2]" : "text-gray-700"}>
                        Total {total} items
                      </span>
                        )}
                        showSizeChanger={false}
                        className={isDarkMode ? "ant-pagination-dark" : ""}
                    />
                  </div>
                </Card>
              </>
          )}
        </div>
      </ConfigProvider>
  );
}

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

      .ant-pagination-dark .ant-pagination-item a {
        color: #f8f8f2 !important;
      }

      .ant-pagination-dark .ant-pagination-item-active a {
        color: #272822 !important;
      }
    `}</style>
);

export default function HomePage() {
  return (
      <>
        <GlobalStyles />
        <HomePageContentWrapper />
      </>
  );
}