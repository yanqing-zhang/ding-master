'''
@Project ：transformer-master 
@File    ：example_position_testcase.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/27 19:54 
'''
import altair as alt
from positional_encoding import PositionalEncoding
import torch
import pandas as pd

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


example_positional()