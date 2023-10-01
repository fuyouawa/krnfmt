# krnl-fmt
驱动中可用的C++20 format库
改写自[fmt](https://github.com/fmtlib/fmt)https://github.com/fmtlib/fmt
## 修改内容:
1. 去除了浮点数的运算(驱动中无法直接使用浮点数)
2. Header Only
3. 限制静态本地化
## 注意事项
目前暂时仅测试了基础功能, 格式化数字什么的
