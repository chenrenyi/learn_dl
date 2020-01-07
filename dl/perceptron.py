# coding=utf-8
from random import choice


class Perceptron:
    """
    感知器
    """

    def __init__(self, activator, vector_num):
        """
        初始化
        :param activator: 激活函数
        :param vector_num: 向量维数
        """
        self.activator = activator
        self.weight = [1] * vector_num
        self.bias = 0

    def __str__(self):
        return 'weight: {0}, bias: {1}'.format(self.weight, self.bias)

    def predict(self, input_x):
        """
        输入向量，返回模型的计算结果
        :return:
        """
        zip_data = list(zip(input_x, self.weight))
        sum_result = sum(map(lambda x_w: x_w[0] * x_w[1], zip_data))
        return self.activator(sum_result + self.bias)

    def train(self, input_args, labels, train_count, rate):
        """
        输入训练数据和训练轮数，完成训练，并返回训练后的模型
        :return:
        """
        for i in range(train_count):
            self._single_train(input_args, labels, rate)

    def _single_train(self, input_args, labels, rate):
        """
        单次训练，使用输入，计算模型输出
        得到模型输出与给定输出后，使用优化规则，更新权重
        :param input_args:
        :return:
        """
        cp = zip(input_args, labels)
        # input_arg, label = choice(cp)
        print '-----------'
        for input_arg, label in cp:
            y = self.predict(input_arg)
            self.__update_weight(input_arg, y, label, rate)

    def __update_weight(self, input_arg, y, label, rate):
        """
        用优化规则，更新权重
        :param input_arg:
        :param y:
        :param label:
        :param rate:
        :return:
        """
        x_w_cp = zip(input_arg, self.weight)
        basic = rate * (label - y)
        self.weight = [basic * x_w[0] + x_w[1] for x_w in x_w_cp]
        self.bias += basic
        print self


def logic(x):
    """
    感知器激活函数
    :param x:
    :return:
    """
    return 1 if x > 0 else 0


def linear_unit(x):
    """
    线性单元函数
    :param x:
    :return:
    """
    return x


def get_train_data():
    """
    返回训练数据
    :return:
    """
    train_x_set = [[3, 4], [123, 22], [9, 1], [0, 0]]
    train_y_set = [34, 1252, 91, 0]
    return train_x_set, train_y_set


if __name__ == '__main__':
    input_args, labels = get_train_data()
    p = Perceptron(linear_unit, 2)
    p.train(input_args, labels, 10, 0.2)
    print p.predict([3, 5])
    print p.predict([-2, 3])
    print p.predict([10, 0])
    print p
