#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: test.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/12
"""
import subprocess


def execute_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline().decode("utf-8")
        line = line.strip()
        if line:
            print('Subprogram output: [{}]'.format(line))
    if p.returncode == 0:
        return True
    else:
        return False



if __name__ == "__main__":
    print("vision/test.py")
    # main()