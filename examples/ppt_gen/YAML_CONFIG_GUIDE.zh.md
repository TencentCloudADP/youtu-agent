# PPT YAML 配置指南

本文档说明如何使用 YAML 描述 PPT 模板。

## YAML 文件结构

YAML配置文件（见 [`yaml_example.yaml`](./yaml_example.yaml)）包括两部分：

1. **`type_map`**：按顺序将每种幻灯片的 `type` 映射到参考 PPT 模板中的幻灯片索引（从 0 开始）。渲染器根据该索引复制相应母版。
2. **页面定义块**：每个 `<name>_page` 条目描述某类幻灯片的 schema，必须包含：
   - `description`：对页面的文字说明，会用于文档和 JSON Schema。
   - `type`：幻灯片类型，必须出现在 `type_map` 中。
   - 至少一个 **字段定义**，用于描述每个占位符希望接收的数据。

### 字段定义可用的键

| 键名         | 说明 |
|--------------|------|
| `type`       | 字段类型（见下方支持列表）。 |
| `description`| 可选提示，会被写入 JSON Schema 供 LLM 参考。 |
| `min_len`    | 对字符串或列表的最小长度约束（转换为 `minLength` / `minItems`）。 |
| `max_len`    | 字符串或列表的最大长度（转换为 `maxLength` / `maxItems`）。 |
| `optional`   | 若设为 `true` 则字段为可选；默认必填。 |

### 支持的字段类型

| 字段类型         | 含义 / 渲染行为 |
|------------------|----------------|
| `str` / `int`    | 直接写入同名形状，`int` 会被转成字符串。 |
| `content`        | 交由 `handle_content` 处理的富媒体字段，使用下文列出的 `TextContent`/`ImageContent`/`TableContent` 结构。 |
| `content_list`   | `BaseContent` 数组，与 `content` 相同的负载，每个条目映射到 `<字段名>1`, `<字段名>2` 等。 |
| `item_list`      | `Item` 对象数组（标题 + 文本），使用 `handle_item` 渲染。 |
| `str_list`       | 短标签列表（如 SWOT 字母），映射到 `label1`, `label2`... 等形状。 |
| `image`          | `BasicImage`（仅包含 `image_url`），插入图像占位符。 |

> ⚠️ **形状命名**：PPT 模板中的形状名称需与 YAML 字段一致，`fill_template_with_yaml_config` 才能定位并替换。

### 负载结构参考

这些类型会直接序列化为 [`ppt_template_model.py`](./ppt_template_model.py) 中的 Pydantic 模型：

1. **`content` / `content_list`** → `BaseContent` 联合类型（由 `content_type` 决定具体模型）：
   - `TextContent`：`{"content_type": "text", "paragraph": [Paragraph] | str}`；`Paragraph` 包含 `{text, bullet, level}`。
   - `ImageContent`：`{"content_type": "image", "image_url": "https://...", "caption": "可选"}`。
   - `TableContent`：`{"content_type": "table", "header": [str], "rows": [[str]], "n_rows": int, "n_cols": int, "caption": "可选"}`。
2. **`item_list`** → `Item` 列表：`{"title": "2-4 个词", "content": "≤10 个词"}`，渲染时映射到 `item_title{n}` / `item_content{n}`。
3. **`image`** → `BasicImage`：`{"image_url": "https://..."}`，渲染器会下载图片并替换占位符。
4. **`str_list`** → 字符串数组，`min_len`/`max_len` 会写入 JSON Schema，渲染器按顺序填充 `label1`、`label2` 等形状。
5. **`content_list`** 补充：每个条目与单独的 `content` 字段一致，请在 PPT 中使用 `<字段名>1`、`<字段名>2` 命名。

## YAML 如何转成 JSON Schema

运行 [`main.py`](./main.py) 时，程序会读取 `--yaml_path` 指向的 YAML，并调用 `build_schema` 将生成的 JSON Schema 注入 Agent 指令，无需手动处理 @examples/ppt_gen/main.py#39-84。

`gen_schema.build_schema` 在内部完成以下步骤：

1. 依据 `type_map` 得到允许的幻灯片类型集合。
2. 将每个页面定义转成 `oneOf` 项，其 `properties` 与字段说明一致。
3. 把 `min_len` / `max_len` 映射到 JSON Schema 的 `minLength` / `maxLength` / `minItems` / `maxItems`。
4. 对 `content`、`item_list`、`image` 等复杂类型引用 `$defs`，确保验证逻辑与渲染器保持一致。

> 如需调试，可手动运行 `gen_schema.py` 导出 schema，但正常流程会自动执行。

## 自定义页面类型示例

假设需要新增“insight_grid_page”幻灯片展示三个核心指标：

1. **准备 PPT 模板**：复制现有幻灯片或重新设计，并在 Selection Pane 中将形状命名为 `title`、`item_title1`、`item_content1` 等。
2. **扩展 `type_map`**：加入新类型对应的模板索引。

```yaml
type_map:
  - content: 0
  - title: 1
  - insight_grid: 13   # 新 slide 位于模板索引 13
```

3. **定义页面块**：

```yaml
insight_grid_page:
  description: 并排展示三个关键指标
  type: insight_grid
  title:
    type: str
    description: 6 个词以内的标题
    max_len: 6
  items:
    type: item_list
    description: 每个元素包含指标名称和简短说明
    min_len: 3
    max_len: 3
  image:
    type: image
    optional: true
```

4. **运行 main.py**：携带 `--yaml_path <your_yaml>`，Agent 会输出带 `type: insight_grid` 的 slide，`fill_template_with_yaml_config` 会复制索引 13 的幻灯片并填充字段。

注意，需要始终保持 YAML、JSON Schema 与 PPT 中的形状命名一致。
