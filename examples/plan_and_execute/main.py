import asyncio

from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils


async def main():
    config = ConfigLoader.load_agent_config("orchestrator/adp_plan_and_execute")
    runner = OrchestratorAgent(config)

    # question = input("Input your question: ")
    # question = "深入分析上海2025年上半年GDP的主要增长动力，并预测下半年相应的增长情况，给出一份图文并茂的研究报告"
    # question = "制作一份上海热点旅游景点推荐网页报告，需要带有地图位置标记"

    # 测试反馈 case
    # question = "查询清华教职工因违规报备入校被拘热点事件的相关信息，并给出具体的参考文献；然后提取出涉及到的法律法规并标注生效时间；接着搜索和本热点事件类似的案件信息进行分析（但注意输出的文档中要按照类案原文返回，不要修改其中的内容），最后根据检索到的事件信息、法律法规信息和类案信息输出采访提纲。如果事件涉及多个流程或者多个人物，最好针对不同的主要人物梳理采访提纲。（注意，最后输出的阶段和标题格式需要包括：一、事件概述类案信息，二、相关法律法规分析，三、类案信息对比分析，其中需要标注出类案原文，四、采访提纲生成）"
    # question = "围绕“十五五”这个专题，生成几个比较详细的选题。我希望每个选题能够从近期的一些热点事件出发，然后能够自然地过渡到十五五的相关话题，这些相关话题是观众或者日常百姓比较关注比较贴合的角度。并且给出一些怎么丝滑过渡切换的内容和语句。"
    # question = "查询长春航展“南天门计划”盛况空前：白帝概念战机引围观，玄女舰载太空模拟仓科幻元素拉满的详细背景和信息，给出一些选题方向，选题内容需要凸显中国的强大实力，每一条选题内容需要搜索中国和其他不同国家的相关历史结果和综合数据作为数据和理论支撑。给出3-4个选题方向，每个方向给出3-4个选题内容。"
    # question = "提取出来这个议程文档中的人员名单，【议程】自然资源部中央广播电视总台战略合作框架协议签约暨大型系列节目《自然中国》启播活动议程.pdf然后根据这些人员所在的单位、人员的职务和姓名，查询并推断出他们对应的职级（查询推断的逻辑是这样的：1. 查询单位级别（正部级>副部级>正厅级>副厅级>正处级>副处级>正科级>副科级）2. 单位级别的最高职务的人和单位的职级相同，以此类推可以推断出不同职务人的职级）（你推断出的职级需要给出相应的理论依据、推断方式和查询到的职务的参考链接，不要胡编乱造），最后将所有人按照职级大小排序，以下是议程文档：https://database-1301722672.cos.ap-beijing.myqcloud.com/%E5%8C%97%E4%BA%AC%E6%97%A5%E6%8A%A5/"

    # 测试 adp case
    # question = "请写作一份计算机行业未来发展分析报告，要求全面解析计算机行业未来趋势、技术发展与就业前景，并提出面向学生的就业选择建议"
    # question = "我们团队搞活动或开会时需要随机抽人，手动抽不公平。我希望能有个工具，把“张三、李四、王五、赵六、孙七、周八、吴九、郑十、陈晨、刘洋”这个名单预先填好在一个文本框里，我可以随时修改。每点一下“抽取”按钮，它就随机弹出一个名字，还能勾选一个“不允许重复”的选项，勾选后抽过的人就不会再被抽到。"
    # question = "北京学区房评测需求聚焦东、西、海等核心城区及潜力区域，涵盖多类房产。从教育资源（学校质量、升学政策）、房价（均价、走势）、居住环境（房屋、配套）、市场交易（活跃度、周期）等维度，结合官方数据、实地调研等方式分析，图表丰富，提供综合评测与个性化建议。"

    # 测试 writing 分流
    # question = "写一段2000字关于人工智能的小说，直接写不要搜索"
    # question = "制作一张体检打分表，直接写不要搜索"

    # 测试 attachments 读取
    # question = "这里提到的人的信息帮我总结下 https://www.nobelprize.org/prizes/physics/2025/popular-information"
    # question = "北京学区房综合评测报告.docx 为我总结一下价格趋势"

    # 测试ppt生成
    question = "制作一个腾讯游戏版图介绍ppt"

    res = runner.run_streamed(question)
    await AgentsUtils.print_stream_events(res.stream_events())
    with open(f"{question[:min(20, len(question))]}_output", "w") as fout:
        fout.write(res.final_output)


if __name__ == "__main__":
    asyncio.run(main())
