import argparse
import os
import random


def prepare_messy_files(work_dir="/tmp/mock_files/"):
    """
    Prepare messy files for testing.
    """
    os.makedirs(work_dir, exist_ok=True)

    student_names = [
        "张三",
        "李四",
        "王五",
        "赵六",
        "孙七",
        "周八",
        "吴九",
        "郑十",
        "冯十一",
        "陈十二",
        "褚十三",
        "卫十四",
        "蒋十五",
        "沈十六",
        "韩十七",
        "杨十八",
        "朱十九",
        "秦二十",
        "尤二一",
        "许二二",
    ]
    student_numbers = [
        "2021001",
        "2021002",
        "2021003",
        "2021004",
        "2021005",
        "2021006",
        "2021007",
        "2021008",
        "2021009",
        "2021010",
        "2021011",
        "2021012",
        "2021013",
        "2021014",
        "2021015",
        "2021016",
        "2021017",
        "2021018",
        "2021019",
        "2021020",
    ]
    titles = [
        "实验报告",
        "课程报告",
        "数据结构大作业",
        "数据结构report",
        "数据结构实验报告",
        "数据结构课程报告",
        "数据结构实验",
        "数据结构大作业报告",
        "数据结构课程设计",
        "数据结构课程设计报告",
        "report",
    ]
    extensions = ["pdf", "docx"]
    formats = [
        "{name}{delimiter}{number}{delimiter}{title}.{extension}",
        "{number}{delimiter}{name}{delimiter}{title}.{extension}",
        "{title}{delimiter}{name}{delimiter}{number}.{extension}",
    ]

    for ind, student_name in enumerate(student_names):
        # create a report file for each student
        num_repeats = random.choices([1, 2], weights=[0.7, 0.3])[0]
        for _ in range(num_repeats):
            file_name_format = random.choice(formats)
            file_name = file_name_format.format(
                name=student_name,
                number=student_numbers[ind],
                title=random.choice(titles),
                delimiter=random.choice(["_", "-", "", " ", "  ", "__"]),
                extension=random.choices(extensions, weights=[0.7, 0.3])[0],
            )
            file_path = os.path.join(work_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"这是{student_name}的{file_name}。\n")
    print(f"Simulated messy files in {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare messy files for testing")
    parser.add_argument("--work_dir", default="/tmp/mock_files", help="Work directory")
    args = parser.parse_args()

    # warn user if work_dir is not empty
    if os.path.exists(args.work_dir) and os.listdir(args.work_dir):
        print(f"Warning: {args.work_dir} is not empty. Files may be overwritten.")
        input("Press Enter to continue or Ctrl+C to abort.")
        os.system(f"rm -rf {args.work_dir}")

    prepare_messy_files(args.work_dir)
