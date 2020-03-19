import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.manifold import TSNE
from matplotlib.axes._axes import _log as matplotlib_axes_logger

class Zodiac:
    """
    Package to evaluate models on a granular level and visualize them
    """

    # Initialized data
    train_data = None
    test_data = None
    model_type = ""
    dim_red = "PCA"

    # Dimension reduced data corpus (test and train)
    transformed_data = None

    # Metric variables
    has_custom = False # Boolean for custom metric function
    average = None # Averaging method for f1, recall, and precision in multiclass classification
    metrics = [] # List of pre-set metric names
    custom = [] # Custom function list

    x_axis = [] # x axis ticks
    y_axis = [] # y axis ticks

    # Manual grid variables
    density_map = [] # dataframe with manual grid information
    columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density'] # column names for density_map

    # Parzen window variables
    parzen_map = [] # dataframe with parzen window information
    pcolumns = ['component 1', 'component 2', 'num points'] # column names for parzen_map

    def __init__(self, train_data, test_data, test_labels, test_predictions, model_type, dim_red="PCA"):
        """

        Constructor to initialize data

        :param train_data: training dataset as a dataframe
        :param test_data: test dataset as a dataframe
        :param test_labels: test labels as a numpy array
        :param test_predictions: test predictions as a numpy array
        :param model_type: model type ["multiclass" | "binaryclass" | "regression"]
        :param dim_red: dimension reduction technique ["PCA" | "TSNE"]
        :return None
        """
        self.dim_red = dim_red

        train_len = len(train_data) # store train length for splitting data

        data = pd.concat([train_data, test_data], axis=0) #concatenate data into one giant corpus

        # Dimension reduction using PCA or TSNE
        if dim_red == "PCA":
            pca = PCA(n_components=2)
            self.transformed_data = pd.DataFrame(data=pca.fit_transform(data), columns=['comp1', 'comp2'])
        if dim_red == "TSNE":
            self.transformed_data = pd.DataFrame(data=TSNE(n_components=2).fit_transform(data),
                                                 columns=['comp1', 'comp2'])

        # split dataset back into train and test
        self.train_data, self.test_data = self.transformed_data.iloc[:train_len, :], self.transformed_data.iloc[
                                                                                     train_len:, :]
        self.test_data["labels"] = test_labels # Store test labels
        self.test_data["predictions"] = test_predictions # Store test predictions
        self.model_type = model_type # Store model type

        # Checking for null values
        if self.train_data.isnull().values.any():
            raise Exception("Null values in training data")
        if self.test_data.labels.isnull().values.any():
            raise Exception("Null values in labels, make sure to pass an np.array ")
        if self.test_data.predictions.isnull().values.any():
            raise Exception("Null values in predictions, make sure to pass an np.array ")
        if self.test_data.isnull().values.any():
            raise Exception("Null values in test_data")

    def set_metrics(self, custom_func=None, metrics=["accuracy"], average=None):
        """

        Function to set metrics that are to be calculated

        :param custom_func: custom metric function
        :param metrics: list of pre-set metrics ["accuracy" | "recall" | "precision" | "f1"]
        :param average: multiclass averaging technique for f1, recall, precision
        :return: None
        """

        self.columns = ['x1', 'x2', 'y1', 'y2', 'num points', 'density'] # Reset column names
        self.metrics = [] # Reset metric names
        self.pcolumns = ['component 1', 'component 2', 'num points'] # Reset parzen column names

        # Check for custom function and set flag
        if custom_func is None:
            self.has_custom = False
        else:
            self.has_custom = True

        print("Setting metrics..")

        #  Store metric names in metric values and column values for manual grid DataFrame and parzen DataFrame
        if not self.has_custom:
            for i in metrics:
                self.columns.append(i)
                self.pcolumns.append(i)
            self.metrics = metrics
        else:
            self.columns.append("custom")
            self.pcolumns.append("custom")
            self.custom.push(custom_func)

        # Set average type
        self.average = average

        # Verify that evrage type is set for multiclass classification
        if self.model_type == "multiclass" and (self.average is None) and (not self.has_custom):
            if ("recall" in metrics) or ("precision" in metrics) or ("f1" in metrics):
                raise Exception("for the set metrics using multiclass model, average type cannot be None. "
                                "Check sklearn documentation for metrics for more information")

        print("Metrics set")

    def __in_windows(self, x1, x2, y1, y2, x, y):
        """
        Private function to check if test point is in a grid window
        here x and y axis are the two components in teh transformed data

        :param x1: x-axis min value of grid
        :param x2: x-axis max value of grid
        :param y1: y-axis min value of grid
        :param y2: y-axis max value of grid
        :param x: test x-axis value
        :param y: test y-axis value
        :return: boolean True if value is in grid, else False
        """
        if (x1 <= x < x2) and (y1 <= y < y2):
            return True
        return False

    def __gen_density_matrix(self):
        """

        Private function that populates the density matrix for manual grid

        :return: None

        """
        print("Generating density matrix...")

        # Get total number of test points
        count = len(self.test_data)
        den_map = [] # Reset density map

        # Running a loop through every co-ordinate value in dimension reduced dataset

        for x in range(len(self.x_axis) - 1):
            for y in range(len(self.y_axis) - 1):

                # Calculating each row of the density matrix
                # Each row contains the following values:
                # minimum value of grid on x-axis
                # maximum value of grid on x-axis
                # minimum value of grid on y-axis
                # maximum value of grid on y-axis
                # number of points in teh grid
                # density calculation for the grid: number of points in the grid/ total number of points
                # metric values

                # Setting first four values of density_map row
                dmrow = [self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1]]
                groundtruth = [] # Reset ground truth
                predictions = [] # Reset predictions
                density = 0 # Reset density

                for i in self.test_data.values:
                    if self.__in_windows(self.x_axis[x], self.x_axis[x + 1], self.y_axis[y], self.y_axis[y + 1], i[0],
                                         i[1]):
                        # If a test point is in a grid, add the label and prediction values to our list
                        predictions.append(i[3])
                        groundtruth.append(i[2])
                        density = density + 1

                # Adding fifth and sixth value to density map
                dmrow.append(density)
                dmrow.append(density / count)

                # If there are points in the grid then calculate metric value using only the
                # points in that grid and add them to the density map row
                if density != 0:
                    if not self.has_custom:
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
                        for function in self.custom:
                            dmrow.append(function(groundtruth, predictions))
                    den_map.append(dmrow)

        self.density_map = pd.DataFrame(data=den_map,
                                        columns=self.columns)

    def split_manual_grid(self, h=-1):
        """
        Public function to define and create manual grids in the dataset
        :param h: height of grid
        :return: None
        """

        print("Splitting the data into grids...")
        self.x_axis = [] # Reset x axis values
        self.y_axis = [] # Reset y axis values

        # If height is not passed, find default max height that will
        # fit the dataset and divide it into 10 blocks along each axis
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

        # Else divide dataset into grids of height h along each axis
        else:
            minx = min(self.test_data['comp1']) - h
            maxx = max(self.test_data['comp1']) + h
            num_div_x = round((maxx - minx) / h)

            self.x_axis.append(minx)
            for i in range(1, num_div_x + 1):
                self.x_axis.append((i * h) + minx)

            miny = min(self.test_data['comp2']) - h
            maxy = max(self.test_data['comp2']) + h
            num_div_y = round((maxy - miny) / h)

            self.y_axis.append(miny)
            for i in range(1, num_div_y + 1):
                self.y_axis.append((i * h) + miny)

        self.__gen_density_matrix() # Generate the density matrix using gridlines generated above
        print("Completed")

    def split_plot(self, metrics, colormap="viridis",gen_class_spread = True):
        """
        Public function to plot the manually set grid metrics for the test data set
        :param metric: metric list to plot (string)
        :param colormap: string value of colormap to use
        :param gen_class_spread: boolean value to generate classification spread plot
        :return: metric plots and the classification plot
        """
        matplotlib_axes_logger.setLevel('ERROR')
        num = len(metrics) # Check number of plots to generate
        plt.clf() # Reset plot

        if gen_class_spread:
            tot_num = num + 1
        else:
            tot_num = num

        # Set plot size
        plt.figure(figsize=(16, 10 * tot_num))

        if gen_class_spread:
            # If classification spread should be plotted

            # Plot 1: All test data with green points marking correct classification
            # and red points marking incorrect classification

            # Check which points were correctly classified
            self.test_data["color"] = self.test_data["labels"] == self.test_data["predictions"]
            green = self.test_data.color == True

            plt.subplot(tot_num, 1, 1)

            plt.title("Data Classification spread")
            plt.xticks(self.x_axis)
            plt.yticks(self.y_axis)
            plt.scatter(self.test_data.loc[green, 'comp1'], self.test_data.loc[green, 'comp2'], c=[0, 0.5, 0, 0.3], s=50)
            plt.scatter(self.test_data.loc[~green, 'comp1'], self.test_data.loc[~green, 'comp2'], c=[0.9, 0.2, 0, 1.0],
                    s=50)
            plt.grid()

            # Remove temporary columns
            del self.test_data["color"]

            fig_num = 2 # Set figure counter to two
        else:
            # If classification spread should not be plotted set figure counter to 1
            fig_num = 1

        for i in range(num):

            # Plot each metric graph with the colors mapped to metric value for each point
            # using the gen_color function
            metric = metrics[i]

            # Check if the metric was set and calculated
            if metric not in self.columns:
                raise Exception("Chosen metric was not initialized. check the metric initialization function.")

            plt.subplot(tot_num, 1, fig_num)
            plt.title(metric + " spread")
            plt.xticks(self.x_axis)
            plt.yticks(self.y_axis)
            plt.scatter(self.test_data['comp1'], self.test_data['comp2'], c=self.__gen_color(metric), cmap=colormap,
                        s=50)
            plt.grid()
            plt.colorbar()

            fig_num += 1

        plt.tight_layout()

    def __gen_color(self, metric):
        """
        Private function to generate colors for the test dataset for specific metric
        :param metric: metric for which test color values must be mapped
        :return: metric value to map color to
        """
        c = [] # Reset values

        # Check if a point falls in a grid and set c value to that metric value
        for k in self.test_data.values:
            metric_val = self.density_map.loc[(self.density_map['x1'] <= k[0]) & (self.density_map['x2'] > k[0]) & (
                    self.density_map['y1'] <= k[1]) & (self.density_map['y2'] > k[1])][
                metric].values[0]
            c.append(metric_val)
        return c

    def gen_parzen(self, radius):
        """
        Public function to generate parzen windows of a radius
        :param radius: radius for the parzen window
        :return: None
        """
        print("Generating parzen windows...")
        pmat = [] # Reset parzen matrix values
        for i in self.test_data.values:

            # Generate each row of the parzen matrix where each row contains the following:
            # test data point component - 1 value (x axis)
            # test data point component - 2 value (y axis)
            # number of points in the parzen window for each point
            # metric value for each point calculated for it's parzen window

            pmrow = []
            h = i[0]
            k = i[1]

            # Setting first two values of the parzen matrix
            pmrow.append(h)
            pmrow.append(k)

            # Reset all values
            groundtruth = []
            predictions = []
            numpoints = 0

            # For each test point, check which points fall in a circle of radius set by user
            # Add these test point prediction and label values to our list for this point metric calculation
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

            # Calculate metric value for all points in the parzen window of a test point
            if not self.has_custom:
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
                for function in self.custom:
                    pmrow.append(function(groundtruth, predictions))
            pmat.append(pmrow)

        self.parzen_map = pd.DataFrame(data=pmat,
                                       columns=self.pcolumns)
        print("Completed.")

    def parzen_plot(self, metrics, colormap="viridis"):
        """

        Public function to plot the metric values for parzen windows of test data
        :param metrics: Metric list to plot
        :param colormap: colormap to use
        :return: metric plots for parzen windows
        """

        matplotlib_axes_logger.setLevel('ERROR')

        # Check number of plots to generate
        num = len(metrics)

        # Reset plot
        plt.clf()
        # Set plot size
        plt.figure(figsize=(16, 10*num))

        for i in range(num):
            # Plot each metric plot by mapping the color value to the metric value
            metric = metrics[i]

            # Check if the metric was set and calculated
            if metric not in self.metrics:
                raise Exception("Chosen metric was not initialized. check the metric initialization function.")

            plt.subplot(num, 1, i+1)
            plt.title(metric + " parzen window")
            if self.x_axis != [] and self.y_axis != []:
                plt.xticks(self.x_axis)
                plt.yticks(self.y_axis)
            plt.scatter(self.parzen_map['component 1'], self.parzen_map['component 2'], c=self.parzen_map[metric],
                        cmap=colormap, s=50)
            plt.grid(b=False)
            plt.colorbar()
        plt.tight_layout()

