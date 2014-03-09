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
        self.test()

    def load_data(self):
        self.load_features_data()
        self.load_train_data()
        self.merge_data()

    def load_features_data(self):
        self.features_dict = {}
        #key = (Store, Date)
        #value = (Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday)
        #line -  #Store,Date,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday - 12 fields
        with open(self.features_file, 'r') as feat_file:
            for line in feat_file.readlines():
                if line.startswith("Store"):continue
                feat_data = line.split(',')
                if not line or len(feat_data) != 12:continue
                key = (feat_data[0], feat_data[1]) 
                self.features_dict[key] = feat_data[2:]
        feat_file.close()        


    def load_train_data(self):
        self.train_dict = {}
        #key = (Store, Dept, Date)
        #value = (Weekly_Sales)
        #line - Store,Dept,Date,Weekly_Sales,IsHoliday - 5 fields
        with open(self.train_file, 'r') as train_file:
            for line in train_file.readlines():
                if line.startswith("Store"):continue
                train_data = line.split(',')
                if not line or len(train_data) != 5:continue
                key = (train_data[0], train_data[1], train_data[2])
                self.train_dict[key] = train_data[3]
        train_file.close()


    def merge_data(self):
        self.training_data = []
        self.target_values = []
        for key, value in self.train_dict.iteritems():
            store, dept ,date = key
            data = self.make_numeric(self.features_dict.get((store, date)))
            if not data:continue                
            self.training_data.append(data)
            self.target_values.append([float(value)])        
   
 
    def make_numeric(self, data):
        if not data:return None
        #value = (Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,IsHoliday)
        temp, fuel, m1, m2, m3, m4, m5, cpi, ue, hol = data
        keys = ["temp", "fuel", "m1", "m2", "m3", "m4", "m5", "cpi", "ue", "hol"]
        values = [temp, fuel, m1, m2, m3, m4, m5, cpi, ue, hol]
        temp_dict = dict(zip(keys, values))
        
        for key, value in temp_dict.iteritems():
            
            if value == "NA":
                temp_dict[key] = 0

            elif key  == "hol":
                value = value.strip()
                if value == "FALSE":
                    temp_dict[key] = 0 
                elif value == "TRUE":
                    temp_dict[key] = 1

            else:
                temp_dict[key] = float(value)
 
        results = []
        for key in keys:
            results.append(temp_dict.get(key))

        return results

    def load_sample_data(self):
        for line in open(self.train_file, 'r').readlines():
            if not line or len(line.split(',')) != 5:continue
            train_data = line.split(',')
            self.y_data_sales.append([train_data[3]])
        self.y_data_sales.pop(0)
        self.y_data_sales = [[float(x[0])] for x in self.y_data_sales]
        self.x_data_date = [[x] for x in range(len(self.y_data_sales))]

    def train(self):
        self.regr = linear_model.LinearRegression()
        self.cutoff = int(len(self.training_data) * 0.75)
        X = numpy.array(self.training_data[:self.cutoff])
        Y = numpy.array(self.target_values[:self.cutoff])
        self.regr.fit(X, Y)
        print self.regr.coef_      

    def test(self):
        test_X = numpy.array(self.training_data[self.cutoff:])
        target_X = numpy.array(self.target_values[self.cutoff:])
        predicted_X  = self.regr.predict(test_X)
        print predicted_X     
        print target_X
        score = self.regr.score(test_X, target_X)
        print score
        

if __name__ == "__main__":
    sp_obj = SalesPredictor()
    sp_obj.run_main()
