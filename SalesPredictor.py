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

        #Least squares
        print "Linear Squares Method"
        self.regr = linear_model.LinearRegression()
        self.train()
        self.test()

        #SGD
        print "SGD"
        self.regr = linear_model.SGDRegressor()
        self.patch_target_data()
        self.train()
        self.test() 

        #Bayesian Ridge Regression
        print "Bayesian Ridge Regression"
        self.regr = linear_model.BayesianRidge()
        self.train()
        self.test() 
        
        
        
    def load_data(self):
        self.load_features_data()
        self.load_train_data()
        self.merge_data()
        self.normalize_data()

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
   
    def normalize_data(self):
        self.get_max_min()
        self.get_normalized_data()

    def get_max_min(self):
        #result - temp, fuel, cpi, ue, hol
        self.temp_max = self.temp_min = 0
        self.fuel_max = self.fuel_min = 0
        self.cpi_max = self.cpi_min = 0
        self.ue_max = self.ue_min = 0
        self.hol_max = self.hol_min = 0

        for result in self.training_data:
            temp, fuel, cpi, ue, hol = result
            
            if temp > self.temp_max:self.temp_max = temp
            elif temp <= self.temp_min:self.temp_min =temp

            if fuel > self.fuel_max:self.fuel_max = fuel
            elif fuel <= self.fuel_min:self.fuel_min = fuel
        
            if cpi > self.cpi_max:self.cpi_max = cpi
            elif cpi <= self.cpi_min:self.cpi_min = cpi

            if ue > self.ue_max:self.ue_max = ue
            elif ue <= self.ue_min:self.ue_min = ue

            if hol > self.hol_max:self.hol_max = hol
            elif hol <= self.hol_min:self.hol_min = hol


    def get_normalized_data(self):
        norm_training_data = []           

        for result in self.training_data:
            temp, fuel, cpi, ue, hol = result

            temp = self.get_norm_temp(temp)
            fuel = self.get_norm_fuel(fuel)
            cpi = self.get_norm_cpi(cpi)
            ue = self.get_norm_ue(ue)
            hol = self.get_norm_hol(hol)

            norm_training_data.append([temp, fuel, cpi, ue, hol])
        
        self.training_data = norm_training_data        

    def get_norm_temp(self, temp):
        return (float)(temp - self.temp_min)/(self.temp_max - self.temp_min)

    def get_norm_fuel(self, fuel):
        return (float)(fuel - self.fuel_min)/(self.fuel_max - self.fuel_min)

    def get_norm_cpi(self, cpi):
        return (float)(cpi - self.cpi_min)/(self.cpi_max - self.cpi_min)

    def get_norm_ue(self, ue):
        return (float)(ue - self.ue_min)/(self.ue_max - self.ue_min)

    def get_norm_hol(self, hol):
        return (float)(hol - self.hol_min)/(self.hol_max - self.hol_min)

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
            if key in {"m1", "m2", "m3", "m4", "m5"}:continue
            results.append(temp_dict.get(key))
        #results - temp, fuel, cpi, ue, hol
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
        self.cutoff = int(len(self.training_data) * 0.85)
        X = numpy.array(self.training_data[:self.cutoff])
        Y = numpy.array(self.target_values[:self.cutoff])
        self.regr.fit(X, Y)

    def test(self):
        self.test_X = numpy.array(self.training_data[self.cutoff:])
        self.target_X = numpy.array(self.target_values[self.cutoff:])
        self.predicted_X  = self.regr.predict(self.test_X)
        score = self.regr.score(self.test_X, self.target_X)
        print score

    def patch_target_data(self):
        new_target_values = [x[0] for x in self.target_values]
        self.target_values = new_target_values

if __name__ == "__main__":
    sp_obj = SalesPredictor()
    sp_obj.run_main()
