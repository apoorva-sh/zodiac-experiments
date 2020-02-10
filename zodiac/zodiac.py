import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math
import colorsys

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

    def __init__(self,train_data,test_data,test_labels,test_predictions,dim_red="PCA"):
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
            self.transformed_data = pd.DataFrame(data=pca.fit_transform(data), columns=['comp1', 'comp2'])
        if dim_red == "TSNE":
            self.transformed_data = pd.DataFrame(data=TSNE(n_components=2).fit_transform(data),columns=['comp1','comp2'])

        self.train_data, self.test_data = self.transformed_data.iloc[:train_len, :], self.transformed_data.iloc[train_len:, :]
        self.test_data["labels"] = test_labels
        self.test_data["predictions"] = test_predictions

    def setmetrics(self,custom_func= None,metrics=["accuracy"],custom = False):
        """
        Function to store metric function list
        :param metrics:
        :param custom:
        :return:
        """
        self.columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density']
        self.metrics = []
        print("Setting metrics..")
        if not custom:
            for i in metrics:
                self.columns.append(i)
            self.metrics = metrics
        else:
            self.columns.append("custom")
            self.metrics.push(custom_func)
            self.custom_func = True
        print("Metrics set")

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
        print("Generating density matrix...")
        count = len(self.test_data)
        den_map = []
        for x in range(len(self.x_axis) - 1):
            for y in range(len(self.y_axis) - 1):
                dmrow = [self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1]]
                results = []
                preds = []
                density = 0
                for i in self.test_data.values:
                    if self.__in_windows(self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1], i[0], i[1]):
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


    def split_manual_grid(self,h=-1):
        """

        :param h:
        :return:
        """

        print("Splitting the data into grids...")
        self.x_axis = []
        self.y_axis = []
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
            minx = min(self.test_data['comp1']) - h
            maxx = max(self.test_data['comp1']) + h
            numDivX = round((maxx - minx) / h)


            self.x_axis.append(minx)
            for i in range(1, numDivX + 1):
                self.x_axis.append((i * h) + minx)

            miny = min(self.test_data['comp2']) - h
            maxy = max(self.test_data['comp2']) + h
            numDivY = round((maxy - miny) / h)


            self.y_axis.append(miny)
            for i in range(1, numDivY + 1):
                self.y_axis.append((i * h) + miny)

        self.__gen_density_matrix()
        print("Completed")

    def split_plot(self,metric,colormap = "viridis"):
        """

        :param metric:
        :param colormap:
        :return:
        """
        if metric not in self.columns:
            raise Exception("Chosen metric was not initialized. check the metric initialization function.")
        self.test_data["color"] = self.test_data["labels"] == self.test_data["predictions"]

        green = self.test_data.color == True
        plt.clf()
        plt.figure(figsize=(16, 20))
        plt.subplot(2, 1, 1)
        plt.legend(title = "Data Classification spread")
        plt.xticks(self.x_axis)
        plt.yticks(self.y_axis)
        plt.scatter(self.test_data.loc[green,'comp1'], self.test_data.loc[green,'comp2'], c=[0,0.5,0,0.3], s=50)
        plt.scatter(self.test_data.loc[~green, 'comp1'], self.test_data.loc[~green, 'comp2'], c=[0.9, 0.2, 0, 1.0], s=50)
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.legend(title=metric + " spread")
        plt.xticks(self.x_axis)
        plt.yticks(self.y_axis)
        plt.scatter(self.test_data['comp1'], self.test_data['comp2'], c=self.__gen_color(metric),cmap=colormap, s=50)
        plt.grid()
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        del self.test_data["color"]

    def __gen_color(self, metric):
        """

        :param metric:
        :return:
        """
        c = []

        for k in self.test_data.values:
            metric_val = self.density_map.loc[(self.density_map['x1'] <= k[0]) & (self.density_map['x2'] > k[0]) & (self.density_map['y1'] <= k[1]) & (self.density_map['y2'] > k[1])][
                metric].values[0]
            c.append(metric_val)
        return c

    def metric_plot(self,metric,colormap="viridis"):
        if metric not in self.columns:
            raise Exception("Chosen metric was not initialized. check the metric initialization function.")

        plt.clf()
        plt.figure(figsize=(16, 8))
        plt.legend(title=metric + " spread")
        plt.xticks(self.x_axis)
        plt.yticks(self.y_axis)
        plt.scatter(self.test_data['comp1'], self.test_data['comp2'], c=self.__gen_color(metric), cmap=colormap, s=50)
        plt.grid()
        plt.colorbar()
        plt.tight_layout()
        plt.show()







        





