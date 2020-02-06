import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.manifold import TSNE

class Zodiac:

    train_data = None
    test_data = None
    transformed_data = None
    custom_func = False
    #labels = None
    #predictions = None
    dim_red = "PCA"
    metrics = []
    x_axis = []
    y_axis = []
    density_map = []
    columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density']

    def __init__(self,train_data,test_data,test_labels,test_predictions,dim_red):
        """

        :param train_data:
        :param test_data:
        :param test_labels:
        :param test_predictions:
        :param dim_red:
        """

        #self.predictions = test_predictions
        #self.labels = test_labels
        self.dim_red = dim_red


        train_len = len(train_data)

        data = pd.concat([train_data, test_data], axis=0)

        if dim_red == "PCA":
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(data)
            self.transformed_data = pd.DataFrame(data=principalComponents, columns=['comp1', 'comp2'])
        if dim_red == "TSNE":
            self.transformed_data = pd.DataFrame(data=TSNE(n_components=2).fit_transform(data),columns=['comp1','comp2'])

        self.train_data, self.test_data = principalDf.iloc[:train_len, :], principalDf.iloc[train_len:, :]
        self.test_data["labels"] = test_labels
        self.test_data["predictions"] = test_predictions

    def metrics(self,custom_func,metrics=["accuracy"],custom = False):
        """
        Function to store metric function list
        :param metrics:
        :param custom:
        :return:
        """
        if not custom:
            for i in metrics:
                self.columns.append(i)
            self.metrics = metrics
        else:
            self.columns.append("custom")
            self.metrics.push(custom_func)
            self.custom_func = True

    def split_manual_grid(self,h=-1):
        """

        :param h:
        :return:
        """
        if h == -1:
            max_1 = max(self.test_data['comp1'])
            max_2 = max(self.test_data['comp2'])
            min_1 = min(self.test_data['comp1'])
            min_2 = min(self.test_data['comp2'])

            xaxis = round((max_1 - min_1) / 10)
            yaxis = round((max_2 - min_2) / 10)
            xaxis = xaxis.as_integer_ratio()[0]
            yaxis = yaxis.as_integer_ratio()[0]

            for i in range(0, 11):
                self.x_axis.append(round(min_1 + xaxis * i))

            for i in range(0, 11):
                self.y_axis.append(round(min_2 + yaxis * i))
        else:
            minx = min(data['principal component 1']) - h
            maxx = max(data['principal component 1']) + h
            numDivX = round((maxx - minx) / h)
            print(numDivX)

            self.x_axis.append(minx)
            for i in range(1, numDivX + 1):
                self.x_axis.append((i * h) + minx)

            miny = min(data['principal component 2']) - h
            maxy = max(data['principal component 2']) + h
            numDivY = round((maxy - miny) / h)
            print(numDivY)

            self.y_axis.append(miny)
            for i in range(1, numDivY + 1):
                self.y_axis.append((i * h) + miny)

        __gen_density_matrix()
        

    def __in_windows(self,x1, x2, y1, y2, x, y):
        """

        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param x:
        :param y:
        :return:
        """
        if (x >= x1 and x < x2) and (y >= y1 and y < y2):
            return True
        return False

    def __gen_density_matrix(self):
        """

        :return:
        """
        count = len(self.test_data)
        den_map = []
        for x in range(len(self.x_axis) - 1):
            for y in range(len(self.y_axis) - 1):
                dmrow = [self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1]]
                results = []
                preds = []
                density = 0
                for i in self.test_data.values:
                    if __in_windows(self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1], i[0], i[1]):
                        results.append(i[3])
                        preds.append(i[2])
                        density = density + 1
                dmrow.append(density)
                dmrow.append(density/count)
                if density != 0:
                    if not self.custom_func:
                        for function in self.metrics:
                            if function == "f1":
                                f1 = f1_score(results, preds, average='micro')
                                dmrow.append(f1)
                            elif function == "accuracy":
                                accuracy = accuracy_score(results, preds)
                                dmrow.append(accuracy)
                            elif function == "recall":
                                recall = recall_score(results, preds, average='micro')
                                dmrow.append(recall)
                            elif function == "precision":
                                precision = precision_score(results, preds, average='micro')
                                dmrow.append(precision)
                    den_map.append(dmrow)

        self.density_map = pd.DataFrame(data=den_map,
                          columns=self.columns)




