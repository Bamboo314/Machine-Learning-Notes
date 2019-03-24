class KNN:
    def __init__(self, X_train, y_train, k=3, p=2):
        """
        parammeter:
        - k：选取邻近点个数
        - p：距离度量 Lp
        """
        self.k = k
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def single_predict(self, x):
        """
        单个实例类别预测函数：
        输入 -- 单个实例
        输出 -- 单个输入实例的类别
        """
        # 初始化，用于后续更新。计算出 X 与前三个训练实例点的距离
        knn_list = []  # 存放邻近点的距离和类别
        for i in range(self.k):
            dist = np.linalg.norm(x - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        # linear scan. 找到更小的值就更新
        for i in range(self.k, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(x - self.X_train[i], ord=self.p)
            if dist < knn_list[max_index][0]:
                knn_list[max_index] = (dist, y_train[i])

        # 分类决策规则：多数表决
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)   # 根据列表频数生成字典，key为类别值，values为类别值频数
        # 按values升序排序后取出出现最多的类别
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]  
        return max_count
    
    def predict(self, X):
        """
        类别预测函数：
        输入 -- 实例
        输出 -- 通过 KNN 算法预测的实例类别
        """
        y_pred = []  # 初始化一个列表，存放每个输入实例的预测类
        if X.ndim == 1:    # 单个实例
            y_pred = self.single_predict(X)
        else:
            for i in range(len(X)):   # 依次判断每个输入实例的类别
                y_pred.append(self.single_predict(X[i]))
        return y_pred
            
    def score(self, X_test, y_test):
        """
        计算分类正确的实例数占总数的比例
        """
        y_pred = self.predict(X_test)
        right_count = (y_pred == y_test).sum()
        rate = right_count / len(y_test)
        return rate


# clf = KNN(X_train, y_train)
# clf.score(X_test, y_test)
# clf.predict(X)
