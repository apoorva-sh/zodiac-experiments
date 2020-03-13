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
    # labels = None
    # predictions = None
    dim_red = "PCA"
    metrics = []
    x_axis = []
    y_axis = []
    density_map = []
    parzen_map = []
    average = None
    model_type = ""
    columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density']
    pcolumns = ['component 1', 'component 2', 'num points']

    def __init__(self, train_data, test_data, test_labels, test_predictions, model_type, dim_red="PCA"):
        """

        :param train_data:
        :param test_data:
        :param test_labels:
        :param test_predictions:
        :param dim_red:
        """

        # self.predictions = test_predictions
        # self.labels = test_labels
        self.dim_red = dim_red

        train_len = len(train_data)

        data = pd.concat([train_data, test_data], axis=0)

        if dim_red == "PCA":
            pca = PCA(n_components=2)
            self.transformed_data = pd.DataFrame(data=pca.fit_transform(data), columns=['comp1', 'comp2'])
        if dim_red == "TSNE":
            self.transformed_data = pd.DataFrame(data=TSNE(n_components=2).fit_transform(data),
                                                 columns=['comp1', 'comp2'])

        self.train_data, self.test_data = self.transformed_data.iloc[:train_len, :], self.transformed_data.iloc[
                                                                                     train_len:, :]
        self.test_data["labels"] = test_labels
        self.test_data["predictions"] = test_predictions
        self.model_type = model_type

    def set_metrics(self, custom_func=None, metrics=["accuracy"], average=None, custom=False):
        """
        Function to set metrics
        :param custom_func:
        :param metrics:
        :param average:
        :param custom:
        :return:
        """

        self.columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density']
        self.metrics = []
        self.pcolumns = ['component 1', 'component 2', 'num points']
        print("Setting metrics..")
        if not custom:
            for i in metrics:
                self.columns.append(i)
                self.pcolumns.append(i)
            self.metrics = metrics
        else:
            self.columns.append("custom")
            self.pcolumns.append("custom")
            self.metrics.push(custom_func)
            self.custom_func = True

        self.average = average
        if self.model_type == "multiclass" and (self.average is None) and (not custom):
            if ("recall" in metrics) or ("precision" in metrics) or ("f1" in metrics):
                raise Exception("for the set metrics using multiclass model, average type cannot be None. "
                                "Check sklearn documentation for metrics for more information")

        print("Metrics set")

    def __in_windows(self, x1, x2, y1, y2, x, y):
        """

        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param x:
        :param y:
        :return:
        """
        if (x1 <= x < x2) and (y1 <= y < y2):
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
                groundtruth = []
                predictions = []
                density = 0
                for i in self.test_data.values:
                    if self.__in_windows(self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1], i[0],
                                         i[1]):
                        predictions.append(i[3])
                        groundtruth.append(i[2])
                        density = density + 1
                dmrow.append(density)
                dmrow.append(density / count)
                if density != 0:
                    if not self.custom_func:
                        for function in self.metrics:
                            if function == "f1":
                                if self.average is None:
                                    dmrow.append(f1_score(y_true=groundtruth, y_pred=predictions))
                                else:
                                    dmrow.append(f1_score(y_true=groundtruth, y_pred=predictions, average=self.average))
                            elif function == "accuracy":
                                dmrow.append(accuracy_score(y_true=groundtruth, y_pred=predictions))
                            elif function == "recall":
                                if self.average is None:
                                    dmrow.append(recall_score(y_true=groundtruth, y_pred=predictions))
                                else:
                                    dmrow.append(
                                        recall_score(y_true=groundtruth, y_pred=predictions, average=self.average))
                            elif function == "precision":
                                if self.average is None:
                                    dmrow.append(precision_score(y_true=groundtruth, y_pred=predictions))
                                else:
                                    dmrow.append(
                                        precision_score(y_true=groundtruth, y_pred=predictions, average=self.average))
                    else:
                        for function in self.metrics:
                            dmrow.append(function(groundtruth, predictions))
                    den_map.append(dmrow)

        self.density_map = pd.DataFrame(data=den_map,
                                        columns=self.columns)

    def split_manual_grid(self, h=-1):
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

    def split_plot(self, metrics, colormap="viridis"):
        """

        :param metric:
        :param colormap:
        :return:
        """

        num = len(metrics)

        self.test_data["color"] = self.test_data["labels"] == self.test_data["predictions"]
        green = self.test_data.color == True
        plt.clf()
        plt.figure(figsize=(16, 8*(num+1) + 2*(num+1)))
        plt.subplot(num+1, 1, 1)

        plt.legend(title="Data Classification spread")
        plt.xticks(self.x_axis)
        plt.yticks(self.y_axis)
        plt.scatter(self.test_data.loc[green, 'comp1'], self.test_data.loc[green, 'comp2'], c=[0, 0.5, 0, 0.3], s=50)
        plt.scatter(self.test_data.loc[~green, 'comp1'], self.test_data.loc[~green, 'comp2'], c=[0.9, 0.2, 0, 1.0],
                    s=50)
        plt.grid()

        fignum = 2
        for i in range(num):
            metric = metrics[i]
            if metric not in self.columns:
                raise Exception("Chosen metric was not initialized. check the metric initialization function.")
            plt.subplot(num+1, 1, fignum)
            fignum += 1
            plt.legend(title=metric + " spread")
            plt.xticks(self.x_axis)
            plt.yticks(self.y_axis)
            plt.scatter(self.test_data['comp1'], self.test_data['comp2'], c=self.__gen_color(metric), cmap=colormap,
                        s=50)
            plt.grid()
            plt.colorbar()
        plt.tight_layout()

        del self.test_data["color"]

    def __gen_color(self, metric):
        """

        :param metric:
        :return:
        """
        c = []

        for k in self.test_data.values:
            metric_val = self.density_map.loc[(self.density_map['x1'] <= k[0]) & (self.density_map['x2'] > k[0]) & (
                    self.density_map['y1'] <= k[1]) & (self.density_map['y2'] > k[1])][
                metric].values[0]
            c.append(metric_val)
        return c

    def metric_plot(self, metric, colormap="viridis"):
        """

        :param metric:
        :param colormap:
        :return:
        """
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

    def gen_parzen(self, radius):

        pmat = []
        for i in self.test_data.values:
            pmrow = []
            h = i[0]
            k = i[1]
            pmrow.append(h)
            pmrow.append(k)
            groundtruth = []
            predictions = []
            numpoints = 0
            for j in self.test_data.values:
                x = j[0] - h
                y = j[1] - k
                rval = radius * radius
                lval = (x * x) + (y * y)
                if lval <= rval:
                    predictions.append(j[3])
                    groundtruth.append(j[2])
                    numpoints = numpoints + 1
            pmrow.append(numpoints)

            if not self.custom_func:
                for metric in self.metrics:
                    if metric == "f1":
                        if self.average is None:
                            pmrow.append(f1_score(y_true=groundtruth, y_pred=predictions))
                        else:
                            pmrow.append(f1_score(y_true=groundtruth, y_pred=predictions, average=self.average))
                    elif metric == "accuracy":
                        pmrow.append(accuracy_score(y_true=groundtruth, y_pred=predictions))
                    elif metric == "recall":
                        if self.average is None:
                            pmrow.append(recall_score(y_true=groundtruth, y_pred=predictions))
                        else:
                            pmrow.append(recall_score(y_true=groundtruth, y_pred=predictions, average=self.average))
                    elif metric == "precision":
                        if self.average is None:
                            pmrow.append(precision_score(y_true=groundtruth, y_pred=predictions))
                        else:
                            pmrow.append(precision_score(y_true=groundtruth, y_pred=predictions, average=self.average))
            else:
                for function in self.metrics:
                    pmrow.append(function(groundtruth, predictions))
            pmat.append(pmrow)

        self.parzen_map = pd.DataFrame(data=pmat,
                                       columns=self.pcolumns)

    def parzen_plot(self, metrics, colormap="viridis"):
        """

        :param radius:
        :param metric:
        :param colormap:
        :return:
        """

        num = len(metrics)
        plt.clf()
        plt.figure(figsize=(16, (8*num) + (2*num)))
        for i in range(num):
            metric = metrics[i]
            if metric not in self.metrics:
                raise Exception("Chosen metric was not initialized. check the metric initialization function.")

            plt.subplot(num, 1, i+1)
            plt.legend(title=metric + " parzen window")
            if self.x_axis != [] and self.y_axis != []:
                plt.xticks(self.x_axis)
                plt.yticks(self.y_axis)
            plt.scatter(self.parzen_map['component 1'], self.parzen_map['component 2'], c=self.parzen_map[metric],
                        cmap=colormap, s=50)
            plt.grid(b=False)
            plt.colorbar()
        plt.tight_layout()

