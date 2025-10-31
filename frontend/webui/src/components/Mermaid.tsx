import React, { useEffect, useState } from 'react';
import mermaid from 'mermaid';
import BrokenImagePlaceholder from './BrokenImagePlaceholder';

interface MermaidProps {
  chart: string;
  mermaidId: string;
}

// Initialize Mermaid once with default config
mermaid.initialize({
  startOnLoad: false, // We'll handle rendering manually
  theme: 'default',
  securityLevel: 'loose',
  fontFamily: 'Arial, sans-serif',
  // themeCSS: `
  //   .node rect, .node circle, .node polygon {
  //     stroke: var(--mermaid-node, #4a6cf7) !important;
  //     fill: var(--mermaid-bg, #ffffff) !important;
  //   }
  //   .node text {
  //     fill: var(--mermaid-text, #213547) !important;
  //   }
  //   .edgePath path {
  //     stroke: var(--mermaid-node, #4a6cf7) !important;
  //   }
  // `,
});

const Mermaid: React.FC<MermaidProps> = ({ chart, mermaidId }) => {
  const [wasOK, setWasOK] = useState(false);
  const [svg, setSvg] = useState("");

  let parseOptions = {
    suppressErrors: true,
  };

  const renderChart = async () => {
    const { svg } = await mermaid.render("mermaidDiv" + mermaidId, chart);
    return svg;
  };

  const validateChart = async () => {
    const isOk = await mermaid.parse(chart, parseOptions);
    return !(!isOk);
  };

  const renderChartAsync = async () => {
    const svg = await renderChart();
    setSvg(svg);
    setWasOK(true);
  };

  useEffect(() => {
    const try_render = async () => {
      const isOk = await validateChart();
      if (isOk && !wasOK) {
        setWasOK(true);
      }
      if (chart && isOk) {
        renderChartAsync();
      }
    };
    try_render();
  }, [chart]);

  return (
    <div className="mermaid-container">
      {wasOK ? <div className="mermaid" dangerouslySetInnerHTML={{ __html: svg }} /> : <BrokenImagePlaceholder
        src={chart}
        alt="Mermaid Chart"
      />}
    </div>
  );
};

export default Mermaid;
