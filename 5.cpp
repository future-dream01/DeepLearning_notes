#include <iostream>

int main() {
    int number;

    std::cout << "请输入一个整数: ";
    std::cin >> number;

    if (number > 0) {
        std::cout << "这个数字是正数。" << std::endl;
    } else if (number < 0) {
        std::cout << "这个数字是负数。" << std::endl;
    } else {
        std::cout << "这个数字是零。" << std::endl;
    }

    return 0;
}