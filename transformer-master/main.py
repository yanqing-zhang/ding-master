'''
@Project ：transformer-master 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/21 20:09 
'''
import pandas as pd
from utils import subsequent_mask
import altair as alt

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking":x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.x("Window:0"),
            alt.Y("Masking:0"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )

example_mask()