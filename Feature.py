from collections import defaultdict
import pickle
import json

class Feature:
    def __init__(self):

        self.weight={}
        self.avg_weight={}
        self.r={}

    def add(self,local_feature,name,*args):

        feature_name=' '.join((name,)+tuple(args))

        local_feature[feature_name]=1
        



    def get_features(self,item):

        w1=item[-1]
        w2=None
        if len(item)>1:
            w2=item[-2]
        local_feature={}
        self.add(local_feature,"w",w1)
        if w2 !=None:
            self.add(local_feature,"w1w2",w1,w2)
        if len(w1)==1:
            self.add(local_feature,"single-c_w",w1)
            if w2!=None :
                self.add(local_feature,"separated c1c2",w2[-1],w1)
        else:
            self.add(local_feature,"bigram c1c2",w1[-2],w1[-1])
        self.add(local_feature,"start(c)len(w)",w1[0],str(len(w1)))
        self.add(local_feature,"end(c)len(w)",w1[-1],str(len(w1)))
        self.add(local_feature,"first(c1)lst(c2)",w1[0],w1[-1])
        if w2!=None:
            self.add(local_feature,"w2-c",w2,w1[-1])
            self.add(local_feature,"c-w1",w2[-1],w1)
            self.add(local_feature,"start(c1)start(c2)",w2[0],w1[0])
            self.add(local_feature,"end(c1)end(c2)",w2[-1],w1[-1])
            self.add(local_feature,"len(w1)w2",str(len(w1)),w2)
            self.add(local_feature,"len(w2)w1",str(len(w2)),w1)
        return local_feature


    def get_score(self,item):
        score=0
        local_feature=self.get_features(item)
        for feature_name in local_feature.keys():
            score+=local_feature[feature_name]*(self.weight[feature_name] if feature_name in self.weight.keys() else 0)
        return score

    def get_global_feature(self,item):
        global_feature={}
        for i in range(len(item)):
            local_feature=self.get_features(item[0:i+1])
            for feature_name in local_feature.keys():
                global_feature[feature_name]=(global_feature[feature_name] if feature_name in global_feature.keys() else 0)+local_feature[feature_name]
        return global_feature

    def update_weight(self,y,z):
        y_feature=self.get_global_feature(y)
        z_feature=self.get_global_feature(z)
        for feature_name in y_feature.keys():
            self.weight[feature_name]=(self.weight[feature_name] if feature_name in self.weight.keys() else 0)+y_feature[feature_name]
        for feature_name in z_feature.keys():
            self.weight[feature_name]=(self.weight[feature_name] if feature_name in self.weight.keys() else 0)-z_feature[feature_name]


    def update_avgWeight(self,y,z,n,t,data_size):

        z_feature = self.get_global_feature(z)
        for feature_name in z_feature.keys():
            self.weight[feature_name] = self.weight[feature_name] if feature_name in self.weight.keys() else 0
            self.r[feature_name]=self.r[feature_name] if feature_name in self.r.keys() else (0,0)
            self.avg_weight[feature_name]=(self.avg_weight[feature_name] if feature_name in self.avg_weight.keys() else 0)\
                                          +self.weight[feature_name]*(t*data_size+n-self.r[feature_name][1]*data_size-self.r[feature_name][0])
            self.weight[feature_name]-=z_feature[feature_name]
            self.r[feature_name]=(n,t)


        y_feature = self.get_global_feature(y)
        for feature_name in y_feature.keys():
            self.weight[feature_name] = self.weight[feature_name] if feature_name in self.weight.keys() else 0
            self.r[feature_name] = self.r[feature_name] if feature_name in self.r.keys() else (0, 0)
            self.avg_weight[feature_name]=(self.avg_weight[feature_name] if feature_name in self.avg_weight.keys() else 0)\
                                          +self.weight[feature_name]*(t*data_size+n-self.r[feature_name][1]*data_size-self.r[feature_name][0])
            self.weight[feature_name]+=y_feature[feature_name]
            self.r[feature_name]=(n,t)

    def last_update(self,iterations,data_size):
        for feature_name in self.weight.keys():
            if feature_name not in self.avg_weight.keys():
                self.avg_weight[feature_name]=self.weight[feature_name]*iterations*data_size
            else:
                self.avg_weight[feature_name]+=self.weight[feature_name]*(iterations*data_size-self.r[feature_name][1]*data_size-self.r[feature_name][0]+1)

    def cal_avg_weight(self,iterations,data_size):
        for feature_name in self.avg_weight.keys():
            self.weight[feature_name]=self.avg_weight[feature_name]/(iterations*data_size)

    def save_model(self,model_file):
        pickle.dump(self.weight,model_file)

    def load_model(self,model_file):

        self.weight=pickle.load(model_file)


if __name__ == "__main__":
    feature=Feature()
    feature_file='./data/feature.json'
    res=feature.load_feature(feature_file)
    print(len(res))
