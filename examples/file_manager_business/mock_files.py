import os
import random
from datetime import datetime, timedelta

# 创建模拟文件目录
output_dir = "/tmp/mock_files"
os.makedirs(output_dir, exist_ok=True)

# 常见后缀变体
version_suffixes = [
    "v1", "v2", "v3", "v1.0", "v2.1", "v3.5",
    "最新版", "最终版", "修订版", "修改版", "审核版",
    "正式版", "草案", "初稿", "终稿",
    "latest", "final", "draft", "review", "approved",
    "最新最终版", "最终修改版", "最终确认版", "最终审核版",
    "FINAL", "FINAL2", "FINAL_FINAL", "URGENT",
    "（密）", "【内部】", "_confidential", "_internal"
]

# 合同文件模板
contract_templates = [
    "采购合同-{date}-{company}{suffix}",
    "Contract_{code}_{date}_signed{suffix}", 
    "{person}-零部件采购-修订{version}版{suffix}",
    "临时协议-{date}-改{suffix}",
    "备份_合同扫描件_{date}{suffix}",
    "Contract_{code}_{date}_unsigned{suffix}",
    "{company}_采购协议_{date}{suffix}",
    "Agreement_{person}_{date}{suffix}"
]

# 发票模板  
invoice_templates = [
    "Invoice_{date}_{company}{suffix}",
    "{year}年{month}月报销单_{person}{suffix}",
    "Receipt_{date}_CNY{amount}{suffix}",
    "发票丢失说明-{date}{suffix}",
    "{province}增值税专用发票_No{number}{suffix}",
    "Payment_Request_Form_{year}_{month}{suffix}",
    "财务凭证_{date}_{person}{suffix}",
    "Voucher_{date}_{company}{suffix}"
]

# 报告模板
report_templates = [
    "季度报告-Q{quarter}-{year}-终版{suffix}",
    "{certificate}证书-{year}更新{suffix}", 
    "尽调报告_保密版本_{date}{suffix}",
    "Audit_Report_FY{year}_Final{suffix}",
    "投标文件-{project}-正本{suffix}",
    "项目报告_{project}_{date}{suffix}",
    "Analysis_Report_{quarter}Q{year}{suffix}",
    "{person}_工作总结_{date}{suffix}"
]

# 无关文件模板
junk_templates = [
    "临时文档_{random}{suffix}",
    "备份_{random}{suffix}",
    "old_version_{random}{suffix}",
    "draft_{random}{suffix}",
    "test_file_{random}{suffix}",
    "~$临时文件_{random}{suffix}",
    "未命名文档_{random}{suffix}",
    "新建文档_{random}{suffix}"
]

# 模拟数据
companies = ["XX科技", "YY零件", "ZZ材料", "ABC公司", "XYZ集团", "环球贸易", "创新科技"]
persons = ["张三", "李四", "王五", "赵六", "James", "Alice", "Robert", "Lisa"]
projects = ["智慧城市", "新能源", "智能制造", "数字化转型", "云计算", "物联网"]
certificates = ["ISO9001", "ISO14001", "高新技术企业", "AAA信用", "CMMI", "ITSS"]
provinces = ["北京", "上海", "广东", "江苏", "浙江", "福建", "四川", "湖北"]

def generate_date(offset_days):
    date = datetime.now() - timedelta(days=offset_days)
    return date.strftime("%Y%m%d"), date.strftime("%Y-%m-%d")

def get_random_suffix():
    # 60%的文件有后缀，40%没有
    if random.random() < 0.6:
        suffix = random.choice(version_suffixes)
        # 随机添加括号或下划线
        if random.random() < 0.3:
            if any(char in suffix for char in "（）【】"):
                return suffix
            else:
                return f"({suffix})" if random.random() < 0.5 else f"_{suffix}"
        return suffix
    return ""

def create_mock_file(filename, content):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"创建文件: {filename}")

def generate_contract_files(count=15):
    for i in range(count):
        date_short, date_long = generate_date(random.randint(1, 90))
        version = random.randint(1, 5)
        company = random.choice(companies)
        person = random.choice(persons)
        code = f"AA-{random.randint(10, 99)}B"
        suffix = get_random_suffix()
        
        template = random.choice(contract_templates)
        filename = template.format(
            date=date_short, company=company, person=person, 
            version=version, code=code, suffix=suffix
        ) + random.choice([".docx", ".pdf", ".pptx", ".jpg", ".doc"])
        
        content = f"""合同文件: {filename}
签约方: {company}
签署日期: {date_long}
合同金额: ￥{random.randint(10000, 1000000)}
合同类型: 采购合同
版本: {version}
备注: {suffix if suffix else '无特殊标记'}
"""
        create_mock_file(filename, content)

def generate_invoice_files(count=10):
    for i in range(count):
        date_short, date_long = generate_date(random.randint(1, 60))
        year = datetime.now().year
        month = random.choice(["March", "April", "May", "3", "4", "5"])
        person = random.choice(persons)
        company = random.choice(companies)
        amount = random.randint(100, 50000)
        province = random.choice(provinces)
        number = random.randint(1000000, 9999999)
        suffix = get_random_suffix()
        
        template = random.choice(invoice_templates)
        filename = template.format(
            date=date_short, company=company, person=person,
            year=year, month=month, amount=amount, 
            province=province, number=number, suffix=suffix
        ) + random.choice([".pdf", ".xlsx", ".png", ".jpg", ".doc"])
        
        content = f"""发票文件: {filename}
开票日期: {date_long}
金额: ￥{amount}
开票方: {company}
收款方: {person if '报销单' in filename else '本公司'}
状态: {suffix if suffix else '正式'}
"""
        create_mock_file(filename, content)

def generate_report_files(count=8):
    for i in range(count):
        date_short, date_long = generate_date(random.randint(1, 120))
        year = datetime.now().year
        quarter = random.randint(1, 4)
        certificate = random.choice(certificates)
        project = random.choice(projects)
        suffix = get_random_suffix()
        
        template = random.choice(report_templates)
        filename = template.format(
            date=date_short, year=year, quarter=quarter,
            certificate=certificate, project=project, suffix=suffix,
            person=random.choice(persons),
            month=random.randint(1, 12),  # For invoice templates that might use {month}
            code=f"{random.randint(1000, 9999)}",  # For templates using {code}
            amount=random.randint(1000, 50000),  # For invoice amount placeholders
            number=random.randint(10000000, 99999999)  # For invoice numbers
        ) + random.choice([".pptx", ".pdf", ".docx", ".ppt", ".zip"])
        
        content = f"""报告文件: {filename}
生成日期: {date_long}
报告类型: {'资质证书' if '证书' in filename else '业务报告'}
项目: {project if '投标' in filename else '通用'}
版本状态: {suffix if suffix else '标准版'}
"""
        create_mock_file(filename, content)

def generate_junk_files(count=12):
    for i in range(count):
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        suffix = get_random_suffix()
        
        template = random.choice(junk_templates)
        filename = template.format(random=random_str, suffix=suffix) + random.choice([".tmp", ".bak", ".old", ".docx", ".pdf", ".xlsx"])
        
        content = f"""无关文件: {filename}
创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
备注: 测试文件或临时备份
文件状态: {suffix if suffix else '临时文件'}
"""
        create_mock_file(filename, content)

if __name__ == "__main__":
    print("开始生成模拟文件...")
    generate_contract_files(15)
    generate_invoice_files(10) 
    generate_report_files(8)
    generate_junk_files(12)
    print(f"\n文件生成完成，共生成 {len(os.listdir(output_dir))} 个文件在 {output_dir} 目录")
