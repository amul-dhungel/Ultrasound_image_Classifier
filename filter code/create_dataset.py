
import glob
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Reads all files in the positive and negative directories
positive_images = glob.glob('/home/amul/Downloads/radscholors/current code/dataset/Thickened/*.*')
negative_images = glob.glob('/home/amul/Downloads/radscholors/current code/dataset/Not_Thickened/*.*')

# Merge positive and negative images
images_ = positive_images + negative_images
labels_ = [1]*len(positive_images) + [0]*len(negative_images)

# Shuffle the final dataset
images, labels = shuffle(images_, labels_)
print(len(images))

# Save the dataset
df = pd.DataFrame({'ID_IMG':images_, 'LABEL': labels_}, columns=['ID_IMG', 'LABEL'])
df.to_csv('/home/amul/Downloads/radscholors/current code/csv/dataframe')
print(labels)

#Creates a training (80%) and testing (20%) split and save the files
# removed stratify = label for now in below function
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=123, stratify=labels)
print(X_test)

df_train = pd.DataFrame({'ID_IMG':X_train, 'LABEL': y_train}, columns=['ID_IMG', 'LABEL'])
df_train.to_csv('/home/amul/Downloads/radscholors/current code/csv/train')

df_test = pd.DataFrame({'ID_IMG':X_test, 'LABEL': y_test}, columns=['ID_IMG', 'LABEL'])
df_test.to_csv('/home/amul/Downloads/radscholors/current code/csv/test')

