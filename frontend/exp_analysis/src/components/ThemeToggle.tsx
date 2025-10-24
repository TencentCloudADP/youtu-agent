import React from "react";
import { Switch, Space } from "antd";
import { BulbOutlined, BulbFilled } from "@ant-design/icons";

type ThemeToggleProps = {
    isDarkMode: boolean;
    setIsDarkMode: (value: boolean) => void;
};

const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDarkMode, setIsDarkMode }) => (
    <Space size={8} className="ml-4">
        <Switch
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
            checked={isDarkMode}
            onChange={setIsDarkMode}
        />
        <span>{isDarkMode ? "Dark" : "Light"}</span>
    </Space>
);

export default ThemeToggle;