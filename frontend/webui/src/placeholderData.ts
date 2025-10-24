import type { Event } from "./types/events";
import type { TextDeltaContent } from "./types/events";

export const placeholderResponses: string[] = [
  "Hello!\n",
  "I am your AI assistant. How can I help you today?\n",
  "Here's a markdown table example:\n| Feature | Support | Version |\n|---------|---------|---------|\n| Tables  | ✓       | 1.0     |\n| Mermaid | ✓       | 9.4.3   |\n| Emoji   | ✓       | 2.0     |\nAnd here's a simple mermaid diagram:\n```mermaid\ngraph TD\n    A[Start] --> B{Is it?}\n    B -->|Yes| C[OK]\n    C --> D[Rethink]\n    D --> B\n    B -->|No| E[End]\n```",
  "Here's a more complex mermaid example showing a Gantt chart:\n\n```mermaid\ngantt\n    title Project Timeline\n    dateFormat  YYYY-MM-DD\n    section Section\n    Task 1           :a1, 2023-01-01, 30d\n    Task 2           :after a1, 20d\n    section Another\n    Task 3           :2023-01-12, 12d\n    Task 4           : 24d\n```",
  "Here's a markdown table with alignment and formatting:\n\n| Syntax      | Description | Test Text     |\n| :---        |    :----:   |          ---: |\n| Header      | Title       | Here's this   |\n| Paragraph   | Text        | And more      |\n| **Bold**    | *Italic*    | ~~Strikethrough~~ |\n\nAnd a sequence diagram in mermaid:\n\n```mermaid\nsequenceDiagram\n    participant Alice\n    participant Bob\n    Alice->>John: Hello John, how are you?\n    loop Healthcheck\n        John->>John: Fight against hypochondria\n    end\n    Note right of John: Rational thoughts <br/>prevail!\n    John-->>Alice: Great!\n    John->>Bob: How about you?\n    Bob-->>John: Jolly good!\n```",
  "Here are some image examples in markdown:\n\n1. Basic image with alt text:\n   ![A beautiful landscape](https://example.com/images/landscape.jpg)\n\n2. Image with title and size:\n   <img src=\"https://example.com/images/portrait.jpg\" alt=\"Portrait\" title=\"Portrait\" width=\"200\"/>\n\n3. Image with link:\n   [![Clickable Image](https://example.com/images/button.png)](https://example.com)",
  "Here's an example with multiple images in a grid layout using HTML (since markdown doesn't support grids natively):\n\n<div style=\"display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;\">\n  <img src=\"https://example.com/images/1.jpg\" alt=\"Image 1\" style=\"width: 100%;\"/>\n  <img src=\"https://example.com/images/2.jpg\" alt=\"Image 2\" style=\"width: 100%;\"/>\n  <img src=\"https://example.com/images/3.jpg\" alt=\"Image 3\" style=\"width: 100%;\"/>\n  <img src=\"https://example.com/images/4.jpg\" alt=\"Image 4\" style=\"width: 100%;\"/>\n</div>",
  "The answer is 42.",
];

export const placeholderEvents: Event[] = [
  // Initial greeting - no confirmation needed
  {
    type: 'raw',
    data: {
      type: 'text',
      delta: 'Hello! I\'m your AI assistant. I have some actions that require your confirmation before proceeding.',
      inprogress: false,
    }
  },
  
  // New agent creation event
  {
    type: 'new',
    data: {
      name: 'Research Agent'
    }
  },
  
  // System update notification
  {
    type: 'raw',
    data: {
      type: 'text',
      delta: 'I need to perform a system update. This will take about 5 minutes.',
      inprogress: false,
    }
  },
  
  // New data analysis agent
  {
    type: 'new',
    data: {
      name: 'Data Analysis Agent',
    }
  },
  
  // Plan that requires confirmation
  {
    type: 'orchestra',
    data: {
      type: 'plan',
      item: {
        analysis: 'System optimization plan',
        todo: [
          'Update system dependencies',
          'Optimize database indexes',
          'Clear cache',
          'Restart services'
        ]
      }
    },
    requireConfirm: true
  },
  
  // Worker task
  {
    type: 'orchestra',
    data: {
      type: 'worker',
      item: {
        task: 'Database backup',
        output: 'Creating a backup of the current database state.'
      }
    }
  },
  
  // Tool call
  {
    type: 'raw',
    data: {
      type: 'tool_call',
      delta: 'execute_payment',
      argument: '{"amount": 100, "currency": "USD", "recipient": "merchant123"}',
      callid: "pay_001"
    }
  },
  
  // Information message
  {
    type: 'raw',
    data: {
      type: 'text',
      delta: 'Sending notification to all users about the upcoming maintenance.',
      inprogress: false,
    }
  },
  
  // Report generation
  {
    type: 'orchestra',
    data: {
      type: 'report',
      item: {
        output: 'Generating a detailed report of all changes made.'
      }
    }
  },
  
  // New customer support agent
  {
    type: 'new',
    data: {
      name: 'Customer Support Agent'
    }
  },
  
  // Worker completion with output
  {
    type: 'orchestra',
    data: {
      type: 'worker',
      item: {
        task: 'Process user data',
        output: 'Data processing completed successfully. Generated 5 reports.'
      }
    }
  },
  
  // New marketing agent
  {
    type: 'new',
    data: {
      name: 'Marketing Automation Agent',
    }
  },
  
  // Final report
  {
    type: 'orchestra',
    data: {
      type: 'report',
      item: {
        output: 'All tasks completed successfully. System is running optimally.'
      }
    }
  },
  // HTML Report Example
  {
    type: 'orchestra',
    data: {
      type: 'report',
      item: {
        output: `<!DOCTYPE html>
<html>
<head>
<title>HTML Report</title>
<style>
  body { font-family: sans-serif; margin: 20px; background-color: #f0f0f0; }
  h1 { color: #333; }
  p { color: #666; }
  .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
</head>
<body>
  <div class="container">
    <h1>Sample HTML Report</h1>
    <p>This is a sample report rendered as HTML inside an iframe.</p>
    <p>It demonstrates the new functionality of displaying HTML content directly.</p>
    <ul>
      <li>Item 1</li>
      <li>Item 2</li>
      <li>Item 3</li>
    </ul>
  </div>
</body>
</html>`
      }
    }
  },
  // SVG report example
  {
    type: 'orchestra',
    data: {
      type: 'report',
      item: {
        output: ` 如下
\`\`\`html
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="blue" />
</svg>
\`\`\`
`
      }
    }
  },
  
  // Text delta with tool call
  {
    type: 'raw',
    data: {
      type: 'tool_call',
      delta: 'Calling weather API',
      callid: 'weather-123',
      argument: '{"location": "San Francisco"}',
      inprogress: true
    }
  },
  
  // Tool call output
  {
    type: 'raw',
    data: {
      type: 'tool_call_output',
      delta: 'Weather data retrieved',
      callid: 'weather-123',
      inprogress: false
    }
  },
  
  // New agent for analytics
  {
    type: 'new',
    data: {
      name: 'Analytics Dashboard Agent'
    }
  },
  
  // Markdown example with code blocks
  {
    type: 'raw',
    data: {
      type: 'text',
      delta: 'Here\'s how to create a table in markdown:\n\n```markdown\n| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| Data 1   | Data 2   | Data 3   |\n| Data 4   | Data 5   | Data 6   |\n```',
      inprogress: false,
    }
  },
  
  // Mermaid example - pie chart
  {
    type: 'raw',
    data: {
      type: 'text',
      delta: 'Here\'s a pie chart example using mermaid:\n\n```mermaid\npie\n    title Pets adopted by volunteers\n    "Dogs" : 386\n    "Cats" : 85\n    "Rats" : 15\n```',
      inprogress: false,
    }
  },
  
  // Finish event - no confirmation needed
  {
    type: 'finish',
    data: null,
  }
];

// function returns random events
export const getPlaceholderEvents = () => {
    const events = JSON.parse(JSON.stringify(placeholderEvents));
    events.forEach((event: Event) => {
        if (event.type === 'raw' && (event.data as TextDeltaContent).type === 'text') {
            (event.data as TextDeltaContent).delta = placeholderResponses[Math.floor(Math.random() * placeholderResponses.length)];
        }
    });
    return events;
}

export const simulateEvents = (handleEvent: (event: Event) => void) => {
    const events = getPlaceholderEvents();
    events.forEach((event: Event, index: number) => {
        setTimeout(() => {
            handleEvent(event);
        }, 500 * index);
    })
}