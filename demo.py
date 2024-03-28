import pandas as pd

# 创建一个数据框
data = {
    '姓名': ['张三', '李四'],
    '年龄': [25, 30]
}
df = pd.DataFrame(data)

# 将数据写入 Excel 文件
df.to_excel('sample.xlsx', index=False)
