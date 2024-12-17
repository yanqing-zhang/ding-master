'''
@Project ：transformer-master 
@File    ：excel_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/21 17:06 
'''
import pandas as pd

def read_excel():
    df = pd.read_excel("./tests.xlsx")
    specific_columns = df.iloc[:, 3]
    print(f"specific_columns:{specific_columns}")
    strx = ""
    for column in specific_columns:
        print(f"column:{column}")
        strx += str(column) + ","
    strx = strx.rstrip(',')
    print(f"strx：{strx}")
if __name__ == '__main__':
    read_excel()
