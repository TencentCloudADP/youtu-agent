import React from "react";

type InfoItem = {
    label: string;
    value: string | number | boolean | null;
    render?: (value: string | number | boolean | null) => React.ReactNode;
};

type InfoBlockProps = {
    title: string;
    items: InfoItem[];
    isDarkMode?: boolean;
};

const InfoBlock: React.FC<InfoBlockProps> = ({ title, items, isDarkMode = false }) => (
    <div
        className={`rounded-lg border mb-6 ${
            isDarkMode ? "border-gray-600" : "border-gray-200"
        }`}
    >
        <div
            className={`px-4 py-2 font-bold border-b ${
                isDarkMode
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
                            isDarkMode ? "text-gray-400" : "text-gray-500"
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

export default InfoBlock;