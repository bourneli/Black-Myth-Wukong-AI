from collections import deque
from itertools import islice

def get_latest_elements(dq, num_elements):
    """
    使用迭代器获取 dq 中最近的 num_elements 个元素。
    
    :param dq: 包含元素的 deque 实例
    :param num_elements: 要获取的最新元素数量
    :return: 包含最新元素的迭代器
    """
    # 确保 num_elements 不大于 dq 的长度
    num_elements = min(len(dq), num_elements)
    
    # 通过 islice 获取队列尾部的迭代器
    return list(islice(dq, len(dq) - num_elements, len(dq)))

# 示例用法
dq = deque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], maxlen=7)

# 获取 dq 中最近 3 个元素的迭代器
latest_elements_iter = get_latest_elements(dq, 3)

# 将迭代器转化为列表进行打印
print(list(dq))
print(latest_elements_iter)  # 输出: [8, 9, 10]
print("first %s" % latest_elements_iter[0] )
print("last %s" % latest_elements_iter[-1] )