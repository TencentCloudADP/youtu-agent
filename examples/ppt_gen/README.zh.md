# PowerPoint 模板示例

## 环境依赖

**安装方式：**

```bash
# 项目根目录
cd youtu-agent

# 安装 PPT 生成所需依赖
uv sync --extra ppt-gen
```

## 快速开始

1. 下载[模板文件](https://cdn.jsdelivr.net/gh/TencentCloudADP-DevRel/picgo-images@main/assets/templates.zip)到 `examples/ppt_gen/templates` 目录。
2. 准备需要转成 PPT 的参考资料（纯文本、Markdown 或网页）。例如下载 [诺贝尔奖介绍网页](https://www.nobelprize.org/prizes/physics/2025/popular-information/)。

```bash
# 进入示例目录
cd examples/ppt_gen

# 下载示例网页
wget https://www.nobelprize.org/prizes/physics/2025/popular-information/ -O webpage.html
```

运行 PPT 生成脚本：

```python
python main.py \
  --file webpage.html \
  --template_path template/0.pptx \
  --yaml_path yaml_example.yaml \
  --pages 10 \
  --disable_tooluse \
  --extra_prompt "确保PPT语言是中文"
```

脚本会生成同名的 `json` 与 `pptx` 文件。通过 `--yaml_path` 可以切换不同的模板配置。

## YAML 驱动的模板配置

> 详见 [YAML 配置指南](YAML_CONFIG_GUIDE.zh.md)。

套模板流程由 YAML 配置驱动（参见 `yaml_example.yaml`），配置文件主要有两部分：

1. **type_map**：将每类幻灯片的 `type`（如 `title`, `items_page_4`）映射到模板 PPT 中的幻灯片索引，渲染器据此复制对应母板。
2. **页面定义块**：每个 `<name>_page` 描述该类幻灯片的全部字段，字段会声明 `type`、长度限制以及会被注入到 JSON Schema 的提示。

支持的字段类型包括 `str`、`int`、`content`（富文本/图片/表格容器）、`content_list`、`item_list`、`str_list` 与 `image`。当 Agent 输出内容后，`fill_template_with_yaml_config` 会读取 YAML、根据 `type_map` 找到目标幻灯片、复制模板，并按照字段类型渲染。自定义模板步骤：

1. 复制 `yaml_example.yaml`，调整 `type_map` 使其与 PPT 模板的幻灯片顺序一致。
2. 更新或新增页面块，确保所有期望生成的 slide 类型都拥有正确的字段定义。
3. 运行脚本时使用 `--yaml_path <your_config.yaml>`，即可加载新的 schema 并与模板贴合。

### 如何标注新的 PPT 模板

- 在 PowerPoint 的「选择窗格」（Selection Pane）中重命名形状，确保名称与 YAML 字段完全一致（例如：`title`、`subtitle`、`item_title1`、`item_content1`、`label1`、`content1`）。
- 同一页不可出现重复的名称。每个形状名称必须唯一，以便渲染器能准确定位并替换。
- 可参考 `examples/ppt_gen/templates` 中的现有模板，以及 `yaml_example.yaml` 的字段定义进行命名。