from sklearn import linear_model
import os, numpy


class SalesPredictor:

    def __init__(self):
        self.train_file = os.path.join("data", 'train.csv')
        self.features_file = os.path.join("data", 'features.csv')
        self.x_data_date = []
        self.y_data_sales = []

    def run_main(self):
        self.load_data()
        self.train()
        #self.test()

    def load_data(self):
        for line in open(self.train_file, 'r').readlines():
            if not line or len(line.split(',')) != 5:continue
            train_data = line.split(',')
            self.y_data_sales.append([train_data[3]])
        self.y_data_sales.pop(0)
        self.y_data_sales = [[float(x[0])] for x in self.y_data_sales]
        self.x_data_date = [ [x] for x in range(len(self.y_data_sales))]

    def train(self):
        regr = linear_model.LinearRegression()
        X = numpy.array(self.x_data_date)
        Y = numpy.array(self.y_data_sales)
        regr.fit(X, Y)
        print regr.coef_      

if __name__ == "__main__":
    sp_obj = SalesPredictor()
    sp_obj.run_main()
