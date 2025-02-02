import pandas as pd


def xlsx_to_csv_pd(src_file, dst_file):
    data_xls = pd.read_excel(src_file, index_col=0)
    data_xls.to_csv(dst_file, encoding='utf-8')


# if __name__ == '__main__':
#     src_file = r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.xlsx"
#     dst_file = r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.csv"
#     xlsx_to_csv_pd(src_file, dst_file)


# 读取CSV文件
df = pd.read_csv(r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.csv")

prefix_path = r"C:\Users\y8549\Desktop\confuse\外包数据集"

# 假设第一列是名字，第二列是正常，接下来的7列是疾病数据
# 提取正常列和疾病列
normal_and_disease_columns = df.columns[1:9]  # 第二列到第九列

# 检查是否所有相关列全为 0
invalid_data = df[df[normal_and_disease_columns].sum(axis=1) == 0]
if not invalid_data.empty:
    invalid_data.to_csv(rf"{prefix_path}\Left_invalid_data.csv", index=False)
    print(f"已导出：invalid_data.csv, 记录数:{len(invalid_data)}")

# 提取疾病列（从第三列到第九列）
disease_columns = df.columns[2:]

# 创建一个新的列，计算每行的疾病数量
df['disease_count'] = df[disease_columns].sum(axis=1)

# 筛选“无疾病或只有一个疾病”的记录
group_0_or_1 = df[(df['disease_count'] <= 1) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
if not group_0_or_1.empty:
    group_0_or_1.to_csv(fr"{prefix_path}\Left_group_0_or_1_diseases.csv", index=False)
    print(f"已导出：group_0_or_1_diseases.csv, 记录数:{len(group_0_or_1)}")

# 筛选“有两个及以上疾病”的记录，并分别导出
for count in range(2, 9):  # 疾病数量从2到8
    group_df = df[(df['disease_count'] == count) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
    if not group_df.empty:  # 仅在有记录时导出
        filename = fr"{prefix_path}\Left_group_{count}_diseases.csv"
        group_df.to_csv(filename, index=False)
        print(f"已导出：{filename}, 记录数:{len(group_df)}")
