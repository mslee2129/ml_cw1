import matplotlib as plt
import numpy as np

def examine_dataset (x ,y, classes, dataset_name = ""):
    print("############", dataset_name, "############")

    print("---- Shapes ---- ")
    print("The shape of x is: ", x.shape)
    print("The shape of y is: ", y.shape)
    #print("The shape of classes is: ", classes.shape)
    print()

    print(classes, "\n")

    print("---- Min, Max, Mean ----")
    print("Minimum of x :", x.min(axis=0))
    print("Maximum of x :", x.max(axis=0))
    print("Mean of x :", x.mean(axis=0), "\n")

    print("Minimum of y:", y.min(axis=0)) 
    print("Maximum of y:", y.max(axis=0)) 
    print("Mean of y :", y.mean(axis=0), "\n") 

    """ ------------  Number of instances  ---------------- """
    # Finding the number of instances of each class
    num_classes = classes.size
    l = [0] *  num_classes
    for i in range(len(y)):
        l[y[i]] += 1

    # Printing number of obs in each class
    print("Num observations in each class: \n")
    for i in range(num_classes):
        print("Class", i, ":", round(l[i]))
    
    # Showing a graphical representation of the distribution (#) of observations by classes
    plt.bar(range(num_classes), l, )
    title_str = "Number of observation per classes. Dataset: " + dataset_name
    plt.title(title_str)
    plt.xlabel('Class number')
    plt.ylabel('Number of observations')
    # plt.show()
    file_name = "graphs/numObs_" + dataset_name
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

    """ ------------  RATIO  ---------------- """
    # Getting the ratio of each class
    ratio = [0] * num_classes
    print("Ratio of each class: \n")
    for i in range(num_classes):
        ratio[i] = float(l[i]/len(y)) * 100
        print("Class", i, ":", round(ratio[i]), "%")

    # Showing a graphical representation of the distribution (%) of observations by classes
    plt.bar(range(num_classes), ratio, )
    title_str = "Ratio of dataset: " + dataset_name
    plt.title(title_str)
    plt.xlabel('Class number')
    plt.ylabel('Percentage of values (weight of the class)')
    # plt.show()
    file_name = "graphs/ratio_" + dataset_name
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file


def graph_compare_full_noisy(x_full, y_full, classes, x_noisy, y_noisy):

    """ ------------  Number of instances  ---------------- """
    # Finding the number of instances of each class
    num_classes = classes.size
    full = [0] *  num_classes
    for i in range(len(y_full)):
        full[y_full[i]] += 1

    noisy = [0] *  num_classes
    for i in range(len(y_noisy)):
        noisy[y_noisy[i]] += 1

    # Showing a graphical representation of the distribution (#) of observations by classes
    X = np.arange(6)
    fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    ax = fig.add_subplot(111)
    ax.bar(X+0.0, full, color = 'b', width = 0.25)
    ax.bar(X+0.25, noisy, color = 'g', width = 0.25)
    ax.set_yticks(np.arange(0,1000,100))
    ax.set_ylabel('Observation count')
    ax.set_xlabel('Class')
    ax.legend(labels=['train_full', 'train_noisy'])
    title_str = "Number of observation per classes."
    # plt.show()
    file_name = "graphs/FULLNOISYCOMPARISONCOUNT" 
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

    """ ------------  RATIO  ---------------- """
    # Getting the ratio of each class
    ratio_FULL = [0] * num_classes
    print("Ratio of each class: \n")
    for i in range(num_classes):
        ratio_FULL[i] = float(full[i]/len(y_full)) * 100

    ratio_NOISY = [0] * num_classes
    print("Ratio of each class: \n")
    for i in range(num_classes):
        ratio_NOISY[i] = float(noisy[i]/len(y_noisy)) * 100

    X = np.arange(6)
    fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    ax = fig.add_subplot(111)
    ax.bar(X+0.0, ratio_FULL, color = 'b', width = 0.25)
    ax.bar(X+0.25, ratio_NOISY, color = 'g', width = 0.25)
    ax.set_yticks(np.arange(0,30,5))
    ax.set_ylabel('Ratio (%)')
    ax.set_xlabel('Class')
    ax.legend(labels=['train_full', 'train_noisy'])
    title_str = "Ratio of observation per classes."
    # plt.show()
    file_name = "graphs/FULLNOISYCOMPARISONRATIO" 
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file


def noisy_data_comparison(clean_data, noisy_data):
    # sort both arrays by attributes, not the label
    # we then iterate through the classes to see which observations differ

    # iteratively sort each attribute, see: 
    # https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted,an%20array%20in%20ascending%20value.
    # print('Num attributes: ', len(clean_data[0])-1)
    num_attributes = len(clean_data[0])-1
    for i in reversed(range(num_attributes)):
        if i == num_attributes-1:
            clean_data = clean_data[clean_data[:,i].argsort()]
            noisy_data = noisy_data[noisy_data[:,i].argsort()]
        else:
            clean_data = clean_data[clean_data[:,i].argsort(kind='mergesort')]
            noisy_data = noisy_data[noisy_data[:,i].argsort(kind='mergesort')]
    
    num_observation = len(clean_data[:,0])
    matches = 0

    # print('clean sorted data: ', clean_data[:10, :], '\n')
    # print('noisy sorted data: ', noisy_data[:10, :], '\n')

    for i in range(num_observation):
        # print('clean: ', clean_data[i,-1], ' noisy: ', noisy_data[i,-1], '\n')
        if clean_data[i,-1] == noisy_data[i,-1]:
            matches += 1
    print('Matches (%): ', matches / num_observation)
    print('Incorrect observations in noisy data (%): ', 1 - (matches / num_observation), '\n')